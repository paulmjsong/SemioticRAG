import os, time
from dotenv import load_dotenv
from tqdm import tqdm

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from extract_entities import extract_entities
from construct_database import clear_database, build_database
from handle_query import create_retriever, generate_answer


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENCYKOREA_API_KEY = os.getenv("ENCYKOREA_API_KEY")
ENCYKOREA_ENDPOINT = os.getenv("ENCYKOREA_ENDPOINT")


# ---------------- NEO4J SETUP ----------------
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "compress")

SHARED_LABEL = "__Entity__"
INDEX_NAME = "entity_index"

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 3072
GENERATION_MODEL = "gpt-4o-mini"


# ---------------- UTIL ----------------
def close_driver(driver: Driver) -> None:
    if driver is not None:
        driver.close()


# ---------------- MAIN ----------------
def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    llm = OpenAILLM(model_name=GENERATION_MODEL, api_key=OPENAI_API_KEY)
    
    save_path = "using_graph/entities.json"
    qlist_path = "using_graph/qlist.json"
    
    if os.path.exists(save_path):
        print("🔄 Extracting entities and relationships...")
        extract_entities(llm, save_path, qlist_path, ENCYKOREA_API_KEY, ENCYKOREA_ENDPOINT)

        print("🧹 Clearing existing database...")
        clear_database(driver)

        print("🔄 Building database from extracted entities...")
        build_database(driver, save_path, embedder, EMBED_DIMS, SHARED_LABEL, INDEX_NAME)

        print("✅ Graph built, single vector index populated, and deduplicated.")
    
    # retriever = create_retriever(driver, embedder, INDEX_NAME)
    # image_path = "https://mblogthumb-phinf.pstatic.net/MjAyMTA4MjRfMjQ5/MDAxNjI5NzkwMzI0NzAx.1F0swz3TLDYa929hy5gq1YKlhpRHuUKmaG62K10Trl0g.FBk65xo5T9h5zeh3RirPMwO3ohpXnGVr3VwHbES6vaAg.PNG.dbs1769/Untitled-1-01.png?type=w966"

    # print("🤖 Ready to answer questions about Korean folk art! (type 'exit' to quit)")
    # while True:
    #     query = input("\nQUERY: ")
    #     if query.lower() in ["exit", "quit"]:
    #         print("Goodbye!")
    #         break

    #     start_time = time.time()                 # ⏱️ TIMER

    #     response = generate_answer(llm, embedder, retriever, [SHARED_LABEL], query, image_path)

    #     elapsed_time = time.time() - start_time  # ⏱️ TIMER

    #     print("\nANSWER:\n", response)
    #     print(f"\nResponded to user query in {elapsed_time:.2f} seconds")
    
    close_driver(driver)


if __name__ == "__main__":
    main()