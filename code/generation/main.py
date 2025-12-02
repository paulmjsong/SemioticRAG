import json, os, time
from dotenv import load_dotenv
from tqdm import tqdm
from neo4j import GraphDatabase

from generation.handle_query import create_retriever, generate_response
from utils.llm import OpenAILLM, HuggingFaceLLM, OllamaLLM, LocalLLM
from utils.llm import OpenAIEmbedder, HuggingFaceEmbedder, OllamaEmbedder, LocalEmbedder


# ---------------- CONFIG ----------------
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 3072
CAPTION_MODEL = "gpt-4o-mini"
GENERATE_MODEL = "gpt-4o"


# ---------------- NEO4J SETUP ----------------
load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
INDEX = "Index"


# ---------------- MAIN ----------------
def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    embedder = OpenAIEmbedder(EMBED_MODEL)
    retriever = create_retriever(driver, INDEX)
    
    # TODO: REPLACE MODELS WITH LLAVA OR QWEN
    caption_llm = OpenAILLM(CAPTION_MODEL)
    generate_llm = OpenAILLM(GENERATE_MODEL)
    # END TODO
    
    src_path = "../example/input.json"
    dst_path = "../example/output.json"
    
    if not os.path.exists(src_path):
        print(f"❗ Source file {src_path} not found. Please provide a valid source file.")
        return
    
    with open(src_path, "r", encoding="utf-8") as src_file:
        all_input = json.load(src_file)
        all_output = []
    
    total_generations = sum(len(input["query"]) for input in all_input) * 2
    pbar = tqdm(total=total_generations, desc="Processing generations")

    for input in all_input:
        img_path = input["image"]
        qa_pairs = []

        for query in input["query"]:
            for i in range(2):  # with and without retrieval
                start_time = time.time()
                if i == 0:
                    pbar.set_postfix_str("with retrieval")
                    response, retrieved, caption = generate_response(query, img_path, caption_llm, generate_llm, retriever)
                else:
                    pbar.set_postfix_str("without retrieval")
                    response, retrieved, caption = generate_response(query, img_path, caption_llm, generate_llm, None)
                elapsed_time = time.time() - start_time

                qa_pairs.append({
                    "query": query,
                    "caption": caption,
                    "response": response,
                    "retrieved": retrieved,
                    "time": elapsed_time,
                })
                pbar.update(1)
                break  # TEMP: ONLY WITH RETRIEVAL
            break  # TEMP: ONLY FIRST QUERY
        
        all_output.append({
            "image": img_path,
            "output": qa_pairs,
        })
        break  # TEMP: ONLY FIRST INPUT
    
    pbar.close()
    driver.close()
    
    with open(dst_path, "w", encoding="utf-8") as dst_file:
        json.dump(all_output, dst_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()