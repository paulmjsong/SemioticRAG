import os
from dotenv import load_dotenv

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from get_data import get_from_encykorea, get_from_heritage


# ---------------- CONFIG ----------------
load_dotenv()
ENCYKOREA_API_KEY = os.getenv("ENCYKOREA_API_KEY")
ENCYKOREA_ENDPOINT = os.getenv("ENCYKOREA_ENDPOINT")

HERITAGE_API_KEY = os.getenv("HERITAGE_API_KEY")
HERITAGE_ENDPOINT = os.getenv("HERITAGE_ENDPOINT")


# ---------------- MAIN ----------------
def main():
    srcs_path = "graph_construction/srcs_list.json"
    
    if not os.path.exists(srcs_path):
        os.mkdir(srcs_path)
    
    print("🔄 Fetching data from Encyclopedia of Korean Culture...")
    get_from_encykorea(srcs_path, ENCYKOREA_API_KEY, ENCYKOREA_ENDPOINT)

    # print("🔄 Fetching data from Korea Heritage Service...")
    # get_from_heritage(srcs_path, HERITAGE_API_KEY, HERITAGE_ENDPOINT)

    print("✅ Data fetching complete.")


if __name__ == "__main__":
    main()