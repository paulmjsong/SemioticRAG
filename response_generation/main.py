import json, os, time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
# from neo4j_graphrag.llm import OpenAILLM

from handle_query import create_retriever, generate_response


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 3072
CAP_MODEL = "gpt-4o-mini"
GEN_MODEL = "gpt-4o"


# ---------------- NEO4J SETUP ----------------
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
INDEX = "index"


# ---------------- MAIN ----------------
def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    retriever = create_retriever(driver, embedder, INDEX)
    
    # TODO: REPLACE MODELS WITH LLAVA OR QWEN
    llm = OpenAI(api_key=OPENAI_API_KEY)
    # END TODO
    
    src_path = "example/input.json"
    dst_path = "example/output.json"
    
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
                    response, retrieved, caption = generate_response(query, img_path, llm, GEN_MODEL, CAP_MODEL, retriever)
                else:
                    pbar.set_postfix_str("without retrieval")
                    response, retrieved, caption = generate_response(query, img_path, llm, GEN_MODEL, CAP_MODEL, None)
                elapsed_time = time.time() - start_time

                qa_pairs.append({
                    "query": query,
                    "caption": caption,
                    "response": response,
                    "retrieved": retrieved,
                    "time": elapsed_time,
                })
                pbar.update(1)
        
        all_output.append({
            "image": img_path,
            "output": qa_pairs,
        })
    
    pbar.close()
    driver.close()
    
    with open(dst_path, "w", encoding="utf-8") as dst_file:
        json.dump(all_output, dst_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()