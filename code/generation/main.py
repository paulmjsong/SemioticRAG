import argparse, json, os, time
from dotenv import load_dotenv
from tqdm import tqdm
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever

from generation.handle_query import create_retriever, generate_response
from utils.llm import OpenAILLM, LocalLLM, OpenAIEmbedder
from utils.utils import load_json_file


# ---------------- NEO4J SETUP ----------------
load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
INDEX = "Index"


# ---------------- MAIN ----------------
def main(args):
    if not (all_input := load_json_file(args.src)):
        print(f"❗ File {args.src} is invalid.")
        return
    all_output = []
    
    driver = GraphDatabase.driver(URI, auth=AUTH)
    if not (retriever := create_retriever(driver, INDEX)):
        print("❗ Retriever creation failed.")
        return
    
    match args.model:
        case "gpt-4o-mini" | "gpt-4o":
            gen_model = OpenAILLM(model=args.model, api_key=os.getenv("OPENAI_API_KEY"))
        case "qwen2.5-vl":
            gen_model = LocalLLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
        case "qwen3-vl":
            gen_model = LocalLLM(model="Qwen/Qwen3-VL-8B-Instruct")
    
    # TODO: REPLACE MODELS WITH LLAVA OR QWEN
    cap_model = OpenAILLM(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    # END TODO

    embedder = OpenAIEmbedder(
        model="text-embedding-3-large",
        model_dim=3072,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    query_generations = [args.with_retrieval, args.without_retrieval].count("y")
    total_generations = sum(len(input["query"]) for input in all_input) * query_generations
    pbar = tqdm(total=total_generations, desc="Processing generations")

    for input in all_input:
        img_path = input["image"]
        qa_pairs = []

        for query in input["query"]:
            def generate(retriever: VectorCypherRetriever | None):
                start_time = time.time()
                response, caption, context_graph = generate_response(query, img_path, cap_model, gen_model, embedder, retriever)
                elapsed_time = time.time() - start_time

                qa_pairs.append({
                    "query": query,
                    "caption": caption,
                    "response": response,
                    "retrieved": context_graph,
                    "time": elapsed_time,
                })
                pbar.update(1)
            
            if args.with_retriever:
                pbar.set_postfix_str("with retrieval")
                generate(retriever)
            if args.without_retriever:
                pbar.set_postfix_str("without retrieval")
                generate(None)
            break  # TEMP: ONLY FIRST QUERY
        
        all_output.append({
            "image": img_path,
            "output": qa_pairs,
        })
        break  # TEMP: ONLY FIRST INPUT
    
    pbar.close()
    driver.close()
    
    with open(args.dst, "w", encoding="utf-8") as dst_file:
        json.dump(all_output, dst_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Extraction and Ingestion into Neo4j")
    parser.add_argument("--with_retrieval", action="store_true")
    parser.add_argument("--without_retrieval", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "qwen2.5", "qwen3"])
    parser.add_argument("--src", type=str, default="../example/generation/input.json")
    parser.add_argument("--dst", type=str, default="../example/generation/output.json")
    args = parser.parse_args()
    main(args)