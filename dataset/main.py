import os
from dotenv import load_dotenv

from fetch_documents import fetch_from_encykorea, fetch_from_heritage, fetch_from_folkency


# ---------------- MAIN ----------------
def main():
    src_path = "dataset/한국민족문화대백과사전_3차.csv"
    dst_path = "example/fetched.json"
    
    if not os.path.exists(src_path):
        print(f"❗ Source file {src_path} not found. Please provide a valid source file.")
        return
    
    load_dotenv()
    
    print("🔄 Fetching data from EncyKorea...")
    fetch_from_encykorea(
        src_path, dst_path,
        os.getenv("ENCYKOREA_API_KEY"),
        os.getenv("ENCYKOREA_ENDPOINT_ARTICLE")
    )

    # print("🔄 Fetching data from Heritage...")
    # fetch_from_heritage(
    #     src_path, dst_path,
    #     os.getenv("HERITAGE_API_KEY"),
    #     os.getenv("HERITAGE_ENDPOINT_ARTICLE")
    # )

    # print("🔄 Fetching data from FolkEncy...")
    # fetch_from_folkency(
    #     src_path, dst_path,
    #     os.getenv("FOLKENCY_API_KEY"),
    #     os.getenv("FOLKENCY_ENDPOINT_IMAGES")
    # )

    print("✅ Data fetching complete.")


if __name__ == "__main__":
    main()