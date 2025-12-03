import argparse, os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from construction.extract_entities import extract_data
from construction.manage_database import clear_database, add_to_database
from utils.llm import OpenAILLM, HuggingFaceLLM, OllamaLLM, LocalLLM
from utils.llm import OpenAIEmbedder, HuggingFaceEmbedder, OllamaEmbedder, LocalEmbedder


# ---------------- NEO4J SETUP ----------------
load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
INDEX = "Index"


# ---------------- MAIN ----------------
def main(args):
    if not os.path.exists(args.src):
        print(f"❗ Source file {args.src} not found. Please provide a valid source file.")
        return
    
    match args.model:
        case "gpt-4o-mini" | "gpt-4o":
            extract_llm = OpenAILLM(
                model=args.model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        case "qwen2.5":
            extract_llm = LocalLLM(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
            )
        case "qwen3":
            extract_llm = LocalLLM(
                model="Qwen/Qwen3-4B-Instruct-2507",
            )
    extract_data(extract_llm, args.src, args.dst)

    if args.clear == "true":
        clear_database(driver)

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
    parser.add_argument("--clear", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "qwen2.5", "qwen3"])
    parser.add_argument("--src", type=str, default="../example/fetched.json")
    parser.add_argument("--dst", type=str, default="../example/extracted.json")
    args = parser.parse_args()
    main(args)