import argparse, os
from dotenv import load_dotenv

from fetch_documents import fetch_from_encykorea, fetch_from_heritage, fetch_from_folkency
from create_dataset import create_dataset
from utils.llm import OpenAILLM, HuggingFaceLLM, OllamaLLM, LocalLLM
from utils.llm import OpenAIEmbedder, HuggingFaceEmbedder, OllamaEmbedder, LocalEmbedder


# ---------------- MAIN ----------------
def main(args):
    src_path = args.src
    dst_path = args.dst
    
    if args.encykorea or args.heritage or args.folkency:
        load_dotenv()
        if args.encykorea == "y":
            print("🔄 Fetching data from EncyKorea...")
            fetch_from_encykorea(
                src_path, dst_path,
                os.getenv("ENCYKOREA_API_KEY"),
                os.getenv("ENCYKOREA_ENDPOINT_ARTICLE")
            )
        if args.heritage == "y":
            print("🔄 Fetching data from Heritage...")
            fetch_from_heritage(
                src_path, dst_path,
                os.getenv("HERITAGE_API_KEY"),
                os.getenv("HERITAGE_ENDPOINT_ARTICLE")
            )
        if args.folkency == "y":
            print("🔄 Fetching data from FolkEncy...")
            fetch_from_folkency(
                src_path, dst_path,
                os.getenv("FOLKENCY_API_KEY"),
                os.getenv("FOLKENCY_ENDPOINT_IMAGES")
            )
        print("✅ Data fetching complete.")
    
    if args.create == "y":
        match args.model:
            case "gpt-4o-mini" | "gpt-4o":
                llm = OpenAILLM(model=args.model, api_key=os.getenv("OPENAI_API_KEY"))
            case "qwen2.5-vl":
                llm = LocalLLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
            case "qwen3-vl":
                llm = LocalLLM(model="Qwen/Qwen3-VL-8B-Instruct")
            case "none":
                llm = None
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            model_dim=3072,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        create_dataset(args.arc, args.dst, embedder, llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Fetching from Open Sources")
    parser.add_argument("--encykorea", type=str, default="n", choices=["y", "n"])
    parser.add_argument("--heritage", type=str, default="n", choices=["y", "n"])
    parser.add_argument("--folkency", type=str, default="n", choices=["y", "n"])
    parser.add_argument("--create", type=str, default="y", choices=["y", "n"])
    # maybe remove --model later
    parser.add_argument("--model", type=str, default="none", choices=["gpt-4o-mini", "gpt-4o", "qwen2.5", "qwen3", "none"])
    parser.add_argument("--src", type=str, default="dataset/한국민족문화대백과사전_3차.csv")
    parser.add_argument("--dst", type=str, default="../example/fetched.json")
    args = parser.parse_args()
    main(args)