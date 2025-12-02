import asyncio, json, re
from tqdm import tqdm
from typing import Dict, List, Optional

from neo4j import Driver
from neo4j_graphrag.indexes import create_vector_index, upsert_vectors
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.types import EntityType
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
    FuzzyMatchResolver,
)


# ---------------- ADD ENTITIES TO DB ----------------
def add_to_database(driver: Driver, dst_path: str, embedder: OpenAILLM, embed_dims: int, index_name: str) -> None:
    with open(dst_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    ensure_vector_index(driver, embed_dims, index_name)
    
    form_ids: List[str] = []
    form_embeds: List[List[float]] = []
    
    with driver.session() as session:
        # Upsert nodes
        for entity in tqdm(data["entities"], total=len(data["entities"]), desc="⬆️  Upserting entities"):
            node_id = session.execute_write(create_node, entity)
            if node_id and entity["type"] == "Form":
                form_ids.append(node_id)

            embed_text = entity['name']
            if entity.get("aliases"):
                embed_text += ". " + ". ".join(entity['aliases'])
            embed_text += ". " + entity['description']
            form_embeds.append(embedder.embed_query(embed_text))
        
        # Batch upsert vectors for Forms
        if form_ids:
            upsert_vectors(
                driver=driver,
                ids=form_ids,
                embedding_property="embedding",
                embeddings=form_embeds,
                entity_type=EntityType.NODE,
            )
        
        # Upsert edges
        for rel in tqdm(data["relations"], total=len(data["relations"]), desc="⬆️  Upserting relationships"):
            joint_node_id = session.execute_write(create_edges, rel)
    
    print("🔍 Resolving duplicate entities...")
    asyncio.run(resolve_duplicates(driver))
    
    print("✅ Database population complete.")


# ---------------- NEO4J OPERATIONS ----------------
def ensure_vector_index(driver: Driver, embed_dims: int, index_name: str) -> None:
    create_vector_index(
        driver=driver,
        name=index_name,
        label="Form",
        embedding_property="embedding",
        dimensions=embed_dims,
        similarity_fn="cosine",
    )

def create_node(tx, entity: Dict) -> Optional[str]:
    entity_type = entity["type"]
    if entity_type not in {"Form", "Concept", "Myth", "JointConcept"}:
        raise ValueError(f"Unsupported entity type: {entity_type}")
    
    query = f"""
    MERGE (n:{entity_type} {{name: $name}})
    ON CREATE SET n.description = $description
    ON MATCH  SET n.description = coalesce(n.description, $description)
    RETURN elementId(n) AS eid
    """
    rec = tx.run(
        query, 
        name=sanitize_label(entity["name"]),
        description=entity["description"],
    ).single()
    
    return rec["eid"]

def create_edges(tx, rel: Dict) -> Optional[str]:
    def run_query(source: str, source_type: str, target: str, target_type: str, rel_type: str) -> None:
        query = f"""
        MATCH (a:{source_type} {{name: $source}})
        MATCH (b:{target_type} {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET r.description = $description
        ON MATCH  SET r.description = coalesce(r.description, $description)
        RETURN count(a) AS a_count, count(b) AS b_count
        """
        result = tx.run(
            query,
            source=source,
            target=target,
            description=rel.get("description"),
        ).single()
        if result["a_count"] == 0 or result["b_count"] == 0:
            print(f"⚠️ Warning: could not create edge {source}-[{rel_type}]->{target}")
    
    rel_type = rel["type"].upper().replace(" ", "_")
    node_id = None

    if rel_type == "CONNOTES":
        # Create edge from Form to Concept
        run_query(
            source=sanitize_label(rel["source"]),
            source_type="Form",
            target=sanitize_label(rel["target"]),
            target_type="Concept",
            rel_type=rel_type,
        )
    elif rel_type == "GENERATES_MYTH":
        if len(rel["source_concepts"]) == 1:
            # Create edge from Concept to Myth
            run_query(
                source=sanitize_label(rel["source_concepts"][0]),
                source_type="Concept",
                target=sanitize_label(rel["target"]),
                target_type="Myth",
                rel_type=rel_type,
            )
        else:
            # Create intermediate JointConcept node
            joint_concept = {
                "type": "JointConcept",
                "name": "+".join(sorted(rel["source_concepts"])),
                "description": f"Joint form of concepts: {', '.join(rel['source_concepts'])}",
            }
            node_id = create_node(tx, joint_concept)
            # Create edges from Concepts to JointConcept
            for source in rel["source_concepts"]:
                run_query(
                    source=sanitize_label(source),
                    source_type="Concept",
                    target=joint_concept["name"],
                    target_type="JointConcept",
                    rel_type="PART_OF",
                )
            # Create edge from JointConcept to Myth
            run_query(
                source=joint_concept["name"],
                source_type="JointConcept",
                target=sanitize_label(rel["target"]),
                target_type="Myth",
                rel_type=rel_type,
            )
    else:
        raise ValueError(f"Unsupported relationship type: {rel_type}")
    
    return node_id

async def resolve_duplicates(driver: Driver) -> None:
    if not apoc_available(driver):
        print("⚠️ APOC not available; skipping entity resolution.")
        return
    # Exact match first
    exact = SinglePropertyExactMatchResolver(driver=driver)
    await exact.run()
    # Fuzzy match by label and name
    for label in ["Form", "Concept", "Myth", "JointConcept"]:
        fuzzy = FuzzyMatchResolver(
            driver=driver,
            filter_query=f"WHERE entity:`{label}`",
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
    print("🧹 Database cleared.")


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