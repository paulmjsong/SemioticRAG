import asyncio, json, re
from typing import Dict, List

from neo4j import Driver
from neo4j_graphrag.indexes import create_vector_index, upsert_vectors
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.types import EntityType
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
    FuzzyMatchResolver,
)


# ---------------- ADD ENTITIES TO DB ----------------
def build_database(driver: Driver, json_path: str, embedder: OpenAILLM, embed_dims: int, shared_label: str, index_name: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ensure_vector_index(driver, embed_dims, shared_label, index_name)
    
    node_ids: List[int] = []
    vectors: List[List[float]] = []
    
    with driver.session() as session:
        # Insert nodes
        for entity in data.get("entities", []):
            node_id = session.execute_write(create_node, entity)
            node_ids.append(node_id)

            text = f"{entity['entity_name']} :: {entity.get('entity_description', '')}"
            vectors.append(embedder.embed_query(text))

        # Insert relationships
        for rel in data.get("relationships", []):
            session.execute_write(create_relationship, rel)

        # One batch upsert for all nodes
        if node_ids:
            print(f"⬆️  Upserting {len(node_ids)} vectors to 'embedding' property...")
            upsert_vectors(
                driver=driver,
                ids=node_ids,
                embedding_property="embedding",
                embeddings=vectors,
                entity_type=EntityType.NODE,
            )
    
    asyncio.run(resolve_duplicates(driver))


# ---------------- NEO4J OPERATIONS ----------------
def create_node(tx, entity: Dict, shared_label: str) -> str:
    label = sanitize_label(entity["entity_type"])
    query = f"""
    MERGE (n:{shared_label}:{label} {{name: $name}})
    ON CREATE SET
        n.description = $description
    ON MATCH SET
        n.description = coalesce(n.description, $description)
    RETURN elementId(n) AS eid
    """
    rec = tx.run(
        query, 
        name=entity["entity_name"], 
        description=entity.get("entity_description"),
    ).single()
    return rec["eid"]


def create_relationship(tx, rel: Dict, shared_label: str) -> None:
    rel_type = rel["relationship_type"].upper().replace(" ", "_")
    query = f"""
    MATCH (a:{shared_label} {{name: $source}})
    MATCH (b:{shared_label} {{name: $target}})
    MERGE (a)-[r:{rel_type}]->(b)
    ON CREATE SET r.description = $description
    ON MATCH  SET r.description = coalesce(r.description, $description)
    """
    tx.run(
        query,
        source=rel["source_entity"],
        target=rel["target_entity"],
        description=rel.get("relationship_description"),
    )


def ensure_vector_index(driver: Driver, embed_dims: int, shared_label: str, index_name: str) -> None:
    create_vector_index(
        driver=driver,
        name=index_name,
        label=shared_label,
        embedding_property="embedding",
        dimensions=embed_dims,
        similarity_fn="cosine",
    )


async def resolve_duplicates(driver: Driver, shared_label: str) -> None:
    if not apoc_available(driver):
        print("⚠️ APOC not available; skipping entity resolution.")
        return
    
    # Exact match first
    exact = SinglePropertyExactMatchResolver(driver=driver)
    await exact.run()

    # Fuzzy match on :__Entity__ by 'name' property
    fuzzy = FuzzyMatchResolver(
        driver=driver,
        filter_query=f"WHERE entity:`{shared_label}`",
        resolve_properties=["name"],
        similarity_threshold=0.95,
    )
    await fuzzy.run()


def clear_database(driver: Driver) -> None:
    with driver.session() as session:
        # Drop constraints
        cons = session.run("SHOW CONSTRAINTS YIELD name RETURN name").value()
        for name in cons:
            session.run(f"DROP CONSTRAINT {name} IF EXISTS")
        # Drop indexes
        idxs = session.run("SHOW INDEXES YIELD name RETURN name").value()
        for name in idxs:
            session.run(f"DROP INDEX {name} IF EXISTS")
        # Delete nodes/relationships
        session.run("MATCH (n) DETACH DELETE n")
    print("🗑️  Dropped all constraints, indexes, and data.")


# ---------------- UTILS ----------------
def sanitize_label(raw: str) -> str:
    tokens = re.split(r'[^A-Za-z0-9]+', raw)
    tokens = [t for t in tokens if t]
    label = ''.join(t[:1].upper() + t[1:] for t in tokens)
    if not label or not label[0].isalpha():
        label = "Entity" + label  # ensure starts with a letter
    return label


def apoc_available(driver: Driver) -> bool:
    with driver.session() as session:
        try:
            rec = session.run("RETURN apoc.version() AS v").single()
            return rec and rec["v"]
        except Exception:
            return False