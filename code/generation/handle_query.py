import base64
from neo4j import GraphDatabase, Record
from neo4j_graphrag.generation.prompts import PromptTemplate
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from utils.llm import BaseLLM, BaseEmbedder
from utils.prompts import CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT, RETRIEVAL_CYPHER, GENERATE_SYSTEM_PROMPT, GENERATE_USER_PROMPT


# ---------------- RETRIEVER ----------------
def create_retriever(driver: GraphDatabase.driver, index_name: str) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver=driver,
        index_name=index_name,
        retrieval_query=RETRIEVAL_CYPHER,
        result_formatter=formatter,
    )

def formatter(rec: Record) -> RetrieverResultItem:
    def clean_text(s):
        return "" if not s else str(s).replace("\n", " ").replace("\r", " ").strip()

    nodes = rec["nodes"]
    rels  = rec["rels"]
    id2name, id2type = {}, {}

    data = {
        "entities": [],
        "relations": [],
    }

    for n in nodes:
        nid     = n["id"]
        ntype   = n["labels"][-1]
        nname   = n["name"]
        ndesc   = clean_text(n["description"])

        data["entities"].append({
            "type": ntype,
            "name": nname,
            "description": ndesc,
        })
        id2name[nid] = nname
        id2type[nid] = ntype

    for r in rels:
        rtype       = r["type"]
        src_node    = id2name[r["start"]]
        tgt_node    = id2name[r["end"]]
        rdesc       = clean_text(r["description"])

        data["relations"].append({
            "type": rtype,
            "source": src_node,
            "target": tgt_node,
            "description": rdesc,
        })

    return RetrieverResultItem(content=str(data), metadata=data)

def retrieve_context(retriever: VectorCypherRetriever, query_vector: list[float], top_k: int=5, per_seed_limit: int=10) -> list[str]:
    results = retriever.search(
        query_vector=query_vector,
        top_k=top_k,
        query_params={
            "per_seed_limit": per_seed_limit,
        },
    )
    # print(f"Retrieved {len(results.items)} items from retriever.")

    combined = {
        "entities": [],
        "relations": [],
    }
    for item in results.items:
        item_data = item.metadata
        combined["entities"].extend(item_data.get("entities", []))
        combined["relations"].extend(item_data.get("relations", []))
        
    return combined


# ---------------- GENERATION ----------------
def generate_response(query: str, image_path: str, cap_model: BaseLLM, gen_model: BaseLLM, embedder: BaseEmbedder, retriever: VectorCypherRetriever=None) -> tuple[str, str, dict]:
    if retriever is None:
        # print("Generating response without retrieval.")
        caption = ""
        context_graph = ""
    else:
        # print("Generating response with retrieval.")
        caption = cap_model.generate(CAPTION_USER_PROMPT, CAPTION_SYSTEM_PROMPT, image_path)
        # print(f"Generated caption: {caption}")
        caption_vector = embedder.embed(caption)
        context_graph = retrieve_context(retriever, caption_vector)
        # print(f"Retrieved context: {context_graph}")
    
    prompt = PromptTemplate(
        template=GENERATE_USER_PROMPT,
        expected_inputs=["context", "query"],
    ).format(context=str(context_graph), query=query)

    response = gen_model.generate(prompt, GENERATE_SYSTEM_PROMPT, image_path)
    return (response, caption, context_graph)


# ---------------- UTILS ----------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"