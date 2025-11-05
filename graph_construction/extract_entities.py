import json
from tqdm import tqdm

from neo4j_graphrag.llm import OpenAILLM

from prompts import CONSTRUCTION_PROMPT


# ---------------- EXTRACT ENTITIES ----------------
def extract_data(llm: OpenAILLM, src_path: str, dst_path: str) -> None:
    with open(src_path, "r") as src_file:
        articles = json.load(src_file)

    extracted = {
        "entities": [],
        "relations": [],
    }
    for article in tqdm(articles, total=len(articles), desc="Extracting entities and relationships"):
        result = llm.invoke(
            f"Input data:\n{article}\n\n{CONSTRUCTION_PROMPT}",
        )
        content = json.loads(result.content)
        extracted['entities'].extend(content['entities'])
        extracted['relations'].extend(content['relations'])
    
    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(extracted, dst_file, ensure_ascii=False, indent=4)