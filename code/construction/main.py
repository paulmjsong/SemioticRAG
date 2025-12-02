import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from construction.extract_entities import extract_data
from construction.manage_database import clear_database, add_to_database
from utils.llm import OpenAILLM, HuggingFaceLLM, OllamaLLM, LocalLLM
from utils.llm import OpenAIEmbedder, HuggingFaceEmbedder, OllamaEmbedder, LocalEmbedder


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
# HUGGING_FACE_BILLING_ADDRESS = os.getenv("HUGGING_FACE_BILLING_ADDRESS")

EXTRACT_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 3072


# ---------------- NEO4J SETUP ----------------
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
INDEX = "Index"


# ---------------- MAIN ----------------
def main():
    src_path = "../example/fetched.json"
    dst_path = "../example/extracted.json"
    
    if not os.path.exists(src_path):
        print(f"❗ Source file {src_path} not found. Please provide a valid source file.")
        return
    
    # TODO: REPLACE MODELS WITH LLAVA OR QWEN
    extract_llm = OpenAILLM(EXTRACT_MODEL)
    # END TODO
    
    llm = OpenAILLM(EXTRACT_MODEL)
    extract_data(extract_llm, src_path, dst_path)

    driver = GraphDatabase.driver(URI, auth=AUTH)
    embedder = OpenAIEmbedder(EMBED_MODEL)
    # clear_database(driver)
    add_to_database(driver, dst_path, embedder, EMBED_DIMS, INDEX)

    driver.close()


if __name__ == "__main__":
    main()