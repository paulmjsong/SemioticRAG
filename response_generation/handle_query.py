import base64, json
from openai import OpenAI
from typing import Dict, List, Tuple

from neo4j import GraphDatabase, Record
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation.prompts import PromptTemplate
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from prompts import CAP_SYSTEM_PROMPT, CAP_USER_PROMPT, RETRIEVAL_CYPHER, GEN_SYSTEM_PROMPT, GEN_USER_PROMPT


# ---------------- RETRIEVER ----------------
def create_retriever(driver: GraphDatabase.driver, embedder: OpenAIEmbeddings, index_name: str) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver=driver,
        index_name=index_name,
        retrieval_query=RETRIEVAL_CYPHER,
        embedder=embedder,
        result_formatter=formatter,
    )

def formatter(rec: Record) -> RetrieverResultItem:
    def clean_text(s):
        return "" if not s else str(s).replace("\n", " ").replace("\r", " ").strip()

    nodes = rec["nodes"]
    rels  = rec["rels"]
    id2name, id2type = {}, {}

    text_nodes = []
    for n in nodes:
        nid     = n["id"]
        ntype   = n["labels"][-1]
        nname   = n["name"]
        ndesc   = clean_text(n["description"])

        text_nodes.append(f"{ntype}: {nname}\n- {ndesc}")
        # text_nodes.append({
        #     "type": ntype,
        #     "name": nname,
        #     "description": ndesc,
        # })

        id2name[nid] = nname
        id2type[nid] = ntype

    text_rels = []
    for r in rels:
        rtype       = r["type"]
        src_node    = id2name[r["start"]]
        tgt_node    = id2name[r["end"]]
        rdesc       = clean_text(r["description"])

        text_rels.append(f"{src_node} -[{rtype}]-> {tgt_node}\n- {rdesc}")
        # text_rels.append({
        #     "type": rtype,
        #     "source": src_node,
        #     "target": tgt_node,
        #     "description": rdesc,
        # })

    text_context = (
        "NODES:\n" + "\n".join(text_nodes) +
        "\n\nRELATIONSHIPS:\n" + "\n".join(text_rels)
    )
    # text_context = str({
    #     "nodes": text_nodes,
    #     "relationships": text_rels,
    # })

    return RetrieverResultItem(text_context)

def retrieve_context(retriever: VectorCypherRetriever, query: str, top_k: int=5, per_seed_limit: int=10) -> list[str]:
    results = retriever.search(
        query_text=query,
        top_k=top_k,
        query_params={
            "per_seed_limit": per_seed_limit,
        },
    )
    # print(f"Retrieved {len(results.items)} context items.")
    return [item.content for item in results.items]


# ---------------- IMG TO CAPTION ----------------
def img2caption(llm: OpenAI, model: str, image_path: str) -> str:
    content = [
        {"type": "text", "text": CAP_USER_PROMPT},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]
    result = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CAP_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return result.choices[0].message.content.strip()


# ---------------- GENERATION ----------------
def generate_response(query: str, image_path: str, llm: OpenAI, gen_model: str, cap_model: str, retriever: VectorCypherRetriever=None) -> Tuple[str, List[str], str]:
    prompt = PromptTemplate(
        template=GEN_USER_PROMPT,
        expected_inputs=["context", "query"],
    )

    if retriever is None:
        caption = ""
        context_list = []
    else:
        caption = img2caption(llm, cap_model, image_path)
        context_list = retrieve_context(retriever, caption)
    
    context_text = "\n\n".join(item for item in context_list)
    text = prompt.format(context=context_text, query=query)

    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]
    result = llm.chat.completions.create(
        model=gen_model,
        messages=[
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=100,
    )
    response = result.choices[0].message.content.strip()
    return (response, context_list, caption)


# ---------------- UTILS ----------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"