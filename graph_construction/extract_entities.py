from tqdm import tqdm

from neo4j_graphrag.llm import OpenAILLM

from prompts import CONSTRUCTION_PROMPT


# ---------------- EXTRACT ENTITIES ----------------
def extract_data(llm: OpenAILLM, save_path: str, srcs_path: str) -> None:
    with open(srcs_path, "r") as srcs_file:
        with open(save_path, "w", encoding="utf-8") as save_file:
            for item in tqdm(srcs_file.readlines(), desc="Extracting entities and relationships"):
                result = llm.invoke(
                    f"Input data:\n{item}\n\n{CONSTRUCTION_PROMPT}",
                )
                entities, relations = parse_extracted_data(result.content)
                record = {
                    "entities": entities,
                    "relations": relations,
                }
                save_file.write(f"{record}\n")


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