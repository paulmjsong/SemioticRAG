import json, re
from tqdm import tqdm

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation.prompts import PromptTemplate

from prompts import SYSTEM_PROMPT, USER_PROMPT


# ---------------- EXTRACT ENTITIES ----------------
def extract_data(llm: OpenAILLM, src_path: str, dst_path: str, checkpoint: int, chunk_size: int = 512) -> None:
    prompt = PromptTemplate(
        template=USER_PROMPT,
        expected_inputs=["passage"],
    )
    with open(src_path, "r") as src_file:
        entries = json.load(src_file)
    with open(dst_path, "w+", encoding="utf-8") as dst_file:
        try:
            data = json.load(dst_file)
        except json.JSONDecodeError:
            data = {
                "entities": [],
                "relations": [],
            }
        for i, entry in enumerate(entries):
            # TODO: Apply semantic chunking or sliding window
            text = entry["body"]
            chunks = [
                text[i : i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]
            for chunk in tqdm(chunks, desc=f"Processing entry {i+1}/{len(entries)}", leave=False):
                if len(chunk.strip()) == 0:
                    continue
                result = llm.invoke(
                    input=prompt.format(passage=chunk),
                    system_instruction=SYSTEM_PROMPT,
                )
                # cleaned_string = clean_llm_output(result.content)
                # content = json.loads(cleaned_string)
                content = json.loads(result.content)
                data['entities'].extend(content['entities'])
                data['relations'].extend(content['relations'])
                # break # Only process first chunk for now
            
            if i % checkpoint == 0:
                dst_file.seek(0)
                json.dump(data, dst_file, ensure_ascii=False, indent=4)
                dst_file.truncate()
        
        dst_file.seek(0)
        json.dump(data, dst_file, ensure_ascii=False, indent=4)
        dst_file.truncate()


# ---------------- UTILS ----------------
def clean_llm_output(output: str) -> str:
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', output, flags=re.UNICODE)
    return cleaned