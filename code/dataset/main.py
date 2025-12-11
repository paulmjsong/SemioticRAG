import argparse, os
from dotenv import load_dotenv

from dataset.fetch_documents import fetch_from_encykorea, fetch_from_heritage, fetch_from_emuseum
from dataset.create_dataset import create_dataset
from utils.llm import OpenAILLM, LocalLLM
from utils.llm import OpenAIEmbedder


# ---------------- MAIN ----------------
def main(args):
    # --- FETCH DATA FROM SOURCES ---
    if args.encykorea or args.heritage or args.emuseum:  # or args.folkency
        load_dotenv()
        if args.encykorea:
            print("🔄 Fetching data from EncyKorea...")
            fetch_from_encykorea(
                args.labels,
                args.save_dir,
                os.getenv("ENCYKOREA_API_KEY"),
                os.getenv("ENCYKOREA_ENDPOINT_DETAIL"),
            )
        if args.heritage:
            print("🔄 Fetching data from Heritage...")
            fetch_from_heritage(
                args.save_dir,
                os.getenv("HERITAGE_ENDPOINT_SEARCH"),
                os.getenv("HERITAGE_ENDPOINT_DETAIL"),
            )
        if args.emuseum:
            print("🔄 Fetching data from eMuseum...")
            fetch_from_emuseum(
                args.save_dir,
                os.getenv("EMUSEUM_WEBPAGE_URL"),
                os.getenv("DATA_ENDPOINT_SEARCH"),
                os.getenv("DATA_API_KEY"),
            )
        print("✅ Data fetching complete.")
    
    # --- CREATE DATASET FROM FETCHED DATA ---
    file_paths = []
    if args.create:
        for filename in os.listdir(args.save_dir):
            if filename.startswith("fetched_") and filename.endswith(".json"):
                file_path = os.path.join(args.save_dir, filename)
                file_paths.append(file_path)
        if not file_paths:
            print("No JSON file starting with 'fetched_' found.")
            return
        match args.model:
            case "gpt-4o-mini" | "gpt-4o":
                llm = OpenAILLM(model=args.model, api_key=os.getenv("OPENAI_API_KEY"))
            case "qwen2.5-vl":
                llm = LocalLLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
            case "qwen3-vl":
                llm = LocalLLM(model="Qwen/Qwen3-VL-8B-Instruct")
            case None:
                llm = None
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            model_dim=3072,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        save_path = os.path.join(args.save_dir, "dataset.json")
        create_dataset(file_paths, save_path, embedder, llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Fetching from Open Sources")
    parser.add_argument("--encykorea", action="store_true")
    parser.add_argument("--heritage", action="store_true")
    parser.add_argument("--emuseum", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--save_dir", type=str, default="../example/dataset/")
    # for encykorea only
    parser.add_argument("--labels", type=str, default=None)
    # TODO: maybe remove --model later
    parser.add_argument("--model", type=str, default=None, choices=["gpt-4o-mini", "gpt-4o", "qwen2.5", "qwen3"])
    args = parser.parse_args()
    main(args)