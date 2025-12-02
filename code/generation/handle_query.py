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

def retrieve_context(retriever: VectorCypherRetriever, query_vector: list[float], top_k: int=5, per_seed_limit: int=10) -> list[str]:
    results = retriever.search(
        query_vector=query_vector,
        top_k=top_k,
        query_params={
            "per_seed_limit": per_seed_limit,
        },
    )
    # print(f"Retrieved {len(results.items)} context items.")
    return [item.content for item in results.items]


# ---------------- GENERATION ----------------
def generate_response(query: str, image_path: str, caption_llm: BaseLLM, generate_llm: BaseLLM, embedder: BaseEmbedder, retriever: VectorCypherRetriever=None) -> tuple[str, list[str], str]:
    if retriever is None:
        caption = ""
        context_list = []
        context_text = ""
    else:
        caption = caption_llm.generate(CAPTION_USER_PROMPT, CAPTION_SYSTEM_PROMPT, image_path)
        caption_vector = embedder.embed(caption)
        context_list = retrieve_context(retriever, caption_vector)
        context_text = "\n\n".join(item for item in context_list)
    
    prompt = PromptTemplate(
        template=GENERATE_USER_PROMPT,
        expected_inputs=["context", "query"],
    ).format(context=context_text, query=query)

    response = generate_llm.generate(prompt, GENERATE_SYSTEM_PROMPT, image_path)
    return (response, context_list, caption)


# ---------------- UTILS ----------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"