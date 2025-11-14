import base64, json, os
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Tuple

from neo4j import GraphDatabase, Record
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
# from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from prompts import IMG2GRAPH_PROMPT, IMG2TEXT_PROMPT, RETRIEVAL_CYPHER, GENERATION_PROMPT


# ---------------- CREATE RETRIEVER ----------------
def create_retriever(driver: GraphDatabase.driver, embedder: OpenAIEmbeddings, seed_label: str, index_name: str) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver=driver,
        index_name=index_name,
        retrieval_query=build_retriever_query(seed_label),
        embedder=embedder,
        result_formatter=formatter,
    )

def build_retriever_query(seed_label: str, directed: bool=True) -> str:
    arrow = "->" if directed else "-"
    pattern = f"(node:{seed_label})-[*1..]{arrow}(nbr:Myth)"

    # Use a marker __PATTERN__ we replace below (avoid f-strings to keep { } intact)
    cypher = RETRIEVAL_CYPHER
    return cypher.replace("__PATTERN__", pattern)


def formatter(rec: Record) -> RetrieverResultItem:
    def clean_text(s):
        # do all escaping outside f-strings
        return "" if not s else str(s).replace("\n", " ").replace("\r", " ").strip()

    nodes = rec["nodes"]
    rels  = rec["rels"]
    id2name  = {}
    id2label = {}

    text_nodes = []
    for n in nodes:
        nid     = n["id"]
        labels  = ":".join(n["labels"])
        name    = n["name"]
        desc    = clean_text(n["description"])
        rank    = float(n["rank"])

        text_nodes.append(
            f"{labels}: {name}"
            f" (rank={rank})"
            f"\n- {desc}"
        )
        id2name[nid] = name
        id2label[nid] = labels

    text_rels = []
    for r in rels:
        # rid         = r["id"]
        rtype       = r["type"]
        start_name  = id2name[r["start"]]
        end_name    = id2name[r["end"]]
        rdesc       = clean_text(r["description"])
        rrank       = float(r["rank"])

        text_rels.append(
            f"{start_name} -[{rtype}]-> {end_name}"
            f' (rank={rrank})'
            f"\n- {rdesc}"
        )

    text_context = (
        "GRAPH NODES:\n" + "\n".join(text_nodes) +
        "\n\nGRAPH RELATIONSHIPS:\n" + "\n".join(text_rels)
    )

    return RetrieverResultItem(
        content=text_context,
        # metadata={
        #     "node_count": len(nodes),
        #     "rel_count": len(rels),
        #     "nodes": nodes,
        #     "relations": rels,
        # },
    )


# ---------------- IMG TO CAPTION ----------------
def img2caption(llm: OpenAI, model: str, image_path: str) -> str:
    content = [
        {
            "type": "text",
            "text": "Generate a factual caption for the following painting based on the visual content.",
        },
        {
            "type": "image_url",
            "image_url": {"url": encode_image(image_path)},
        },
    ]
    result = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": IMG2TEXT_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return result.choices[0].message.content.strip()


# ---------------- RETRIEVAL & GENERATION ----------------
def retrieve_context(retriever: VectorCypherRetriever, query: str, query_emb: list[float], top_k: int=1, per_seed_limit: int=10) -> list[str]:
    results = retriever.search(
        query_text=query,
        top_k=top_k,
        query_params={
            "lambda": 0.5,                      # similarity vs. seed score
            "per_seed_limit": per_seed_limit,   # limit kept paths per seed
            "q_embed": query_emb,               # query embedding
        },
    )
    # print(f"Retrieved {len(results.items)} context items.")
    return [item.content for item in results.items]

def generate_response(llm: OpenAI, gen_model: str, cap_model: str, embedder: OpenAIEmbeddings, retriever: VectorCypherRetriever, query: str, image_path: str) -> Tuple[str, List[str], str]:
    caption = img2caption(llm, cap_model, image_path)
    # print("Caption:\n", caption)

    r_query = f"Context: {caption}\n\nQuery: {query}"
    r_query_emb = embedder.embed_query(r_query)
    context_list = retrieve_context(retriever, r_query, r_query_emb)
    context_text = "\n\n".join(item for item in context_list)
    # print("Retrieved:\n", context_text)

    content = [
        {
            "type": "text",
            "text": f"Context:\n{context_text}\n\nQuery:\n{query}\n\nAnswer:",
        },
        {
            "type": "image_url",
            "image_url": {"url": encode_image(image_path)},
        },
    ]
    result = llm.chat.completions.create(
        model=gen_model,
        messages=[
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    response = result.choices[0].message.content.strip()
    return (response, context_list, caption)


# ---------------- UTILS ----------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"