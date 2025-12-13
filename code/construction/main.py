import argparse, os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from construction.extract_entities import extract_data
from construction.manage_database import clear_database, add_to_database
from utils.llm import OpenAILLM, LocalLLM, OpenAIEmbedder


# ---------------- NEO4J SETUP ----------------
load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
INDEX = "Index"


# ---------------- MAIN ----------------
def main(args):
    if args.extract == "y":
        match args.model:
            case "gpt-4o-mini" | "gpt-4o":
                gen_model = OpenAILLM(model=args.model, api_key=os.getenv("OPENAI_API_KEY"))
            case "qwen2.5-vl":
                gen_model = LocalLLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
            case "qwen3-vl":
                gen_model = LocalLLM(model="Qwen/Qwen3-VL-8B-Instruct")
        extract_data(gen_model, args.src, args.dst)
    
    if args.clear == "y":
        clear_database(driver)
    
    if args.upsert == "y":
        driver = GraphDatabase.driver(URI, auth=AUTH)
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            model_dim=3072,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        add_to_database(driver, args.dst, embedder, INDEX)

        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Extraction and Ingestion into Neo4j")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "qwen2.5", "qwen3"])
    parser.add_argument("--src", type=str, default="../example/construction/fetched.json")
    parser.add_argument("--dst", type=str, default="../example/construction/extracted.json")
    args = parser.parse_args()
    main(args)