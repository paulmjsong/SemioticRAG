import argparse, json, os, time
from dotenv import load_dotenv
from tqdm import tqdm
from neo4j import GraphDatabase

from generation.handle_query import create_retriever, generate_response
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
    
    driver = GraphDatabase.driver(URI, auth=AUTH)
    retriever = create_retriever(driver, INDEX)
    if not retriever:
        print("❗ Retriever creation failed.")
        return
    
    match args.model:
        case "gpt-4o-mini" | "gpt-4o":
            generate_llm = OpenAILLM(
                model=args.model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        case "qwen2.5-vl":
            generate_llm = LocalLLM(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
            )
        case "qwen3-vl":
            generate_llm = LocalLLM(
                model="Qwen/Qwen3-VL-8B-Instruct",
            )
    
    # TODO: REPLACE MODELS WITH LLAVA OR QWEN
    caption_llm = OpenAILLM(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    # END TODO

    embedder = OpenAIEmbedder(
        model="text-embedding-3-large",
        model_dim=3072,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    with open(args.src, "r", encoding="utf-8") as src_file:
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
                    response, caption, context_graph = generate_response(query, img_path, caption_llm, generate_llm, embedder, retriever)
                else:
                    pbar.set_postfix_str("without retrieval")
                    response, caption, context_graph = generate_response(query, img_path, caption_llm, generate_llm, embedder, None)
                elapsed_time = time.time() - start_time

                qa_pairs.append({
                    "query": query,
                    "caption": caption,
                    "response": response,
                    "retrieved": context_graph,
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
    
    with open(args.dst, "w", encoding="utf-8") as dst_file:
        json.dump(all_output, dst_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Extraction and Ingestion into Neo4j")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "qwen2.5", "qwen3"])
    parser.add_argument("--src", type=str, default="../example/fetched.json")
    parser.add_argument("--dst", type=str, default="../example/extracted.json")
    args = parser.parse_args()
    main(args)