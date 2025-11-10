import json, os, time
from dotenv import load_dotenv

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from handle_query import create_retriever, generate_response


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------- NEO4J SETUP ----------------
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "compress")

SHARED_LABEL = "__Entity__"
SHARED_INDEX = "__Entity__index"
# SEED_LABEL = "Form"
# SEED_INDEX = "Form_index"

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 3072
GENERATION_MODEL = "gpt-4o"


# ---------------- UTIL ----------------
def close_driver(driver: Driver) -> None:
    if driver is not None:
        driver.close()


# ---------------- MAIN ----------------
def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    llm = OpenAILLM(model_name=GENERATION_MODEL, api_key=OPENAI_API_KEY)
    retriever = create_retriever(driver, embedder, SHARED_INDEX)
    
    input_examples = [
        {
            "query": "이 그림에 묘사된 전통 한국 민화의 주요 요소들은 무엇이며, 각각의 상징적 의미는 무엇인가요?",
            "image": "painting_examples/담배피우는호랑이.png",
        },
        {
            "query": "이 그림에 묘사된 전통 한국 민화의 주요 요소들은 무엇이며, 각각의 상징적 의미는 무엇인가요?",
            "image": "painting_examples/일월오봉도.webp",
        },
        {
            "query": "이 그림에 묘사된 전통 한국 민화의 주요 요소들은 무엇이며, 각각의 상징적 의미는 무엇인가요?",
            "image": "painting_examples/작호도.jpg",
        },
    ]
    output_examples = []
    
    for input in input_examples:
        start_time = time.time()
        response = generate_response(llm, embedder, retriever, [SHARED_LABEL], input["query"], input["image"])
        elapsed_time = time.time() - start_time

        output_examples.append({
            "input" : input,
            "output": response,
            "time"  : elapsed_time,
        })
    
    dst_path = "output_examples/output.json"
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(output_examples, f, ensure_ascii=False, indent=4)
    
    close_driver(driver)


if __name__ == "__main__":
    main()