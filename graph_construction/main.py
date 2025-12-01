import os
from dotenv import load_dotenv

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from extract_entities import extract_data
from manage_database import clear_database, add_to_database


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------- NEO4J SETUP ----------------
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

INDEX_NAME = "Index"

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 3072
GENERATION_MODEL = "gpt-4o"


# ---------------- MAIN ----------------
def main():
    src_path = "example/data/fetched.json"
    dst_path = "example/data/extracted.json"
    
    if not os.path.exists(src_path):
        print(f"❗ Source file {src_path} not found. Please provide a valid source file.")
        return
    
    llm = OpenAILLM(model_name=GENERATION_MODEL, api_key=OPENAI_API_KEY)
    extract_data(llm, src_path, dst_path, 1)

    driver = GraphDatabase.driver(URI, auth=AUTH)
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    # clear_database(driver)
    add_to_database(driver, dst_path, embedder, EMBED_DIMS, INDEX_NAME)

    driver.close()


if __name__ == "__main__":
    main()