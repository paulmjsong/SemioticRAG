from tqdm import tqdm

from neo4j_graphrag.llm import OpenAILLM

from prompts import CONSTRUCTION_PROMPT


# ---------------- EXTRACT ENTITIES ----------------
def extract_data(llm: OpenAILLM, src_path: str, dst_path: str) -> None:
    with open(src_path, "r") as src_file, open(dst_path, "w", encoding="utf-8") as dst_file:
        for item in tqdm(src_file.readlines(), desc="Extracting entities and relationships"):
            result = llm.invoke(
                f"Input data:\n{item}\n\n{CONSTRUCTION_PROMPT}",
            )
            entities, relations = parse_extracted_data(result.content)
            record = {
                "entities": entities,
                "relations": relations,
            }
            dst_file.write(f"{record}\n")

# TODO: Rewrite this function to properly parse the LLM output
def parse_extracted_data(data: str) -> tuple[list[str], list[str]]:
    entities = []
    relations = []
    lines = data.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if line == "Entities:":
            current_section = "entities"
            continue
        elif line == "Relationships:":
            current_section = "relations"
            continue
        
        if current_section == "entities" and line:
            entities.append(line)
        elif current_section == "relations" and line:
            relations.append(line)
    
    return entities, relations