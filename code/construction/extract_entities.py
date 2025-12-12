import json, re
from tqdm import tqdm
from neo4j_graphrag.generation.prompts import PromptTemplate

from utils.llm import BaseLLM
from utils.prompts import EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_PROMPT
from utils.utils import load_json_file


# ---------------- EXTRACT ENTITIES ----------------
def extract_data(gen_model: BaseLLM, src_path: str, dst_path: str, chunk_size: int = 512) -> None:
    prompt = PromptTemplate(
        template=EXTRACT_USER_PROMPT,
        expected_inputs=["passage"],
    )
    if not (entries := load_json_file(src_path)):
        return
    if not (data := load_json_file(dst_file)):
        data = {
            "entities": [],
            "relations": [],
        }
    with open(dst_path, "w", encoding="utf-8") as dst_file:
        for i, entry in enumerate(entries):
            text = entry["body"]
            
            # TODO: REPLACE WITH SEMANTIC CHUNKING
            chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            # END TODO

            for chunk in tqdm(chunks, desc=f"🔄 Processing entry {i+1}/{len(entries)}", leave=False):
                if len(chunk.strip()) == 0:
                    continue
                response = gen_model.generate(
                    prompt.format(passage=chunk),
                    EXTRACT_SYSTEM_PROMPT,
                )
                content = json.loads(response.content)
                data['entities'].extend(content['entities'])
                data['relations'].extend(content['relations'])
                # break # Only process first chunk for now
            
            dst_file.seek(0)
            json.dump(data, dst_file, ensure_ascii=False, indent=4)
            dst_file.truncate()