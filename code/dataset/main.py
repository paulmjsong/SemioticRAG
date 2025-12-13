import argparse, os
from dotenv import load_dotenv

from dataset.fetch_documents import fetch_from_encykorea, fetch_from_heritage, fetch_from_emuseum
from dataset.create_dataset import create_dataset
from utils.llm import LocalClassifier


# ---------------- MAIN ----------------
def main(args):
    load_dotenv()

    # --- FETCH DATA FROM SOURCES ---
    if args.encykorea or args.heritage or args.emuseum:  # or args.folkency
        if args.encykorea:
            print("🔄 Fetching data from EncyKorea...")
            fetch_from_encykorea(
                args.encykorea_file,
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
    if args.create:
        file_paths = []
        for filename in os.listdir(args.save_dir):
            if filename.startswith("fetched_emuseum") and filename.endswith(".json"):
                file_path = os.path.join(args.save_dir, filename)
                file_paths.append(file_path)
        if not file_paths:
            print("No JSON file starting with 'fetched_' found.")
            return
        
        # classifier = LocalClassifier(model="joeddav/xlm-roberta-large-xnli")
        classifier = LocalClassifier(model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
        save_path = os.path.join(args.save_dir, "dataset.json")

        print("🔄 Parsing fetched data to create dataset...")
        create_dataset(file_paths, save_path, classifier)
        print("✅ Dataset created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Fetching from Open Sources")
    parser.add_argument("--encykorea", action="store_true")
    parser.add_argument("--heritage", action="store_true")
    parser.add_argument("--emuseum", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--save_dir", type=str, default="../example/dataset/")
    parser.add_argument("--encykorea_file", type=str, default=None)  # For EncyKorea only
    args = parser.parse_args()
    main(args)