import asyncio, json, re
from typing import Dict, List

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from prompts import IMG2GRAPH_PROMPT, RETRIEVAL_CYPHER, SYSTEM_PROMPT


# ---------------- CREATE RETRIEVER ----------------
def create_retriever(driver: GraphDatabase.driver, embedder: OpenAIEmbeddings, INDEX_NAME: str):
    return VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        retrieval_query=build_retriever_query(),
        embedder=embedder,
        result_formatter=formatter,
    )

def build_retriever_query(max_hops=2, directed=False):
    hops = max(1, min(max_hops, 10))
    arrow = "->" if directed else "-"
    pattern = f"-[r*1..{hops}]{arrow}"

    # Use a marker __PATTERN__ we replace below (avoid f-strings to keep { } intact)
    cypher = RETRIEVAL_CYPHER
    return cypher.replace("__PATTERN__", pattern)


def formatter(rec):
    def _clean_text(s):
        if not s:
            return ""
        # do all escaping outside f-strings
        return str(s).replace("\n", " ").replace("\r", " ").strip()

    nodes = rec["nodes"]
    rels  = rec["rels"]

    id2name = {}
    id2lbls = {}

    node_lines = []
    for i, n in enumerate(nodes):
        nid      = n["id"]
        labels   = ":".join(n.get("labels", []))
        name     = n.get("name") or "(unnamed)"
        desc     = _clean_text(n.get("description", ""))
        degree   = n.get("degree", 0)
        rank     = float(n.get("rank", 0.0))
        is_seed  = bool(n.get("isSeed", False))

        id2name[nid] = name
        id2lbls[nid] = labels

        node_lines.append(
            f'NODE {i+1} [{labels}] name="{name}" '
            f'degree={degree} rank={rank} isSeed={is_seed}\n'
            f'desc="{desc}"'
        )

    rel_lines = []
    for r in rels:
        start_id    = r["start"]
        end_id      = r["end"]
        start_name  = id2name.get(start_id, f'NODE({start_id})')
        end_name    = id2name.get(end_id,   f'NODE({end_id})')
        rtype       = r["type"]
        # edge_degree = r.get("edge_degree", 0)
        # rrank       = float(r.get("rank", 0.0))

        rel_lines.append(
            f'{start_name} -[{rtype}]-> {end_name}'
            # f'\n(rank={rrank})'
        )

    text_context = (
        "GRAPH NODES:\n\n" + "\n\n".join(node_lines) +
        "\n\nGRAPH RELATIONSHIPS:\n\n" + "\n".join(rel_lines)
    )

    graph_payload = {"nodes": nodes, "relationships": rels}

    return RetrieverResultItem(
        content=text_context,
        metadata={
            "type": "graph",
            "node_count": len(nodes),
            "rel_count": len(rels),
            "graph": graph_payload,
        },
    )


# ---------------- IMG TO GRAPH ----------------
def img2graph(llm: OpenAILLM, image_path: str) -> None:
    content = [
        {
            "type": "text",
            "text": IMG2GRAPH_PROMPT,
        },
        {
            "type": "image_url",
            "image_url": {"url": image_path},
            # "image_url": {"url": encode_ndarray_to_base64(image_path)},
        },
    ]
    result = llm.invoke(content)
    return result.content


# ---------------- RETRIEVAL & GENERATION ----------------
def retrieve_context(retriever, labels, query, query_emb, top_k=1, per_seed_limit=10):
    results = retriever.search(
        query_text=query,
        top_k=top_k,
        query_params={
            "allowedNodeLabels": labels,
            "lambda": 0.5,                      # similarity vs. seed score
            "hop_decay": 1.0,                   # 1.0 disables decay; 0.6–0.85 favors closer hops
            "per_seed_limit": per_seed_limit,   # limit kept paths per seed
            "q_embed": query_emb,               # query embedding
        },
    )
    return [item.content for item in results.items]

def generate_response(llm: OpenAILLM, retriever: VectorCypherRetriever, embedder: OpenAIEmbeddings, labels: List[str], query: str, image_path: str) -> str:
    # TODO: translate image to G1
    g1 = img2graph(llm, image_path)
    # print("\G1:\n", g1)

    # TODO: retrieve G2 from G1 + query
    r_query = f"CONTEXT: {g1}\n\nQUERY: {query}"
    r_query_emb = embedder.embed_query(r_query)
    context = retrieve_context(retriever, labels, r_query, r_query_emb)
    g2 = "\n".join(item for item in context)
    # print("\G2:\n", g2)
    
    # TODO: create G3 from G1 + G2
    g3 = g1 + "\n" + g2

    content = [
        {
            "type": "text",
            "text": f"Context:\n{g3}\n\nQuestion:\n{query}\n\nAnswer:",
        },
        {
            "type": "image_url",
            "image_url": {"url": image_path},
            # "image_url": {"url": encode_ndarray_to_base64(image_path)},
        },
    ]
    result = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return result.choices[0].message.content.strip()


# ---------------- UTILS ----------------
# def encode_ndarray_to_base64(image):
#     pil_img = Image.fromarray(image.astype("uint8"))
#     buffered = BytesIO()
#     pil_img.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return f"data:image/png;base64,{img_str}"