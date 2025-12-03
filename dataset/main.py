import json, os, requests, pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# ---------------- CONFIG ----------------
load_dotenv()
ENCYKOREA_API_KEY = os.getenv("ENCYKOREA_API_KEY")
ENCYKOREA_ENDPOINT = os.getenv("ENCYKOREA_ENDPOINT_ARTICLE")


# ---------------- FETCH DATA ----------------
def fetch_from_encykorea(src_path: str, dst_path: str, exclude: list[str], API_KEY: str, ENDPOINT_URL: str) -> None:
    headers = { "X-API-Key": API_KEY }
    fetched = []

    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from Encyclopedia of Korean Culture"):
        line = ",".join(map(str, row.values))
        eid = get_eid_from_line(line, exclude)
        if eid is None:
            continue
        response = requests.get(url=ENDPOINT_URL+eid, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        article = data.get("article")
        fetched.append({
            "headword": article.get("headword"),
            "body": article.get("body").replace('\r', '').split('\n', 1)[1].strip(),
        })

    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


# ---------------- UTILS ----------------
def get_eid_from_line(line: str, exclude: list[str]) -> str:
    cells = line.strip().split(",")
    url = cells[2]
    tag = cells[3]
    has_img = cells[4]
    if "작품" in tag:
        if all(keyword not in tag for keyword in exclude) and "O" in has_img:
            return url.strip().split("/")[-1]
    elif "장르" in tag:
        return url.strip().split("/")[-1]
    return None


# ---------------- MAIN ----------------
def main():
    exclude = ['초상', '추상', '기록']
    src_path = "example/data/한국학중앙연구원_한국민족문화대백과사전_20240130.csv"
    dst_path = "example/data/fetched.json"
    
    if not os.path.exists(src_path):
        print(f"❗ Source file {src_path} not found. Please provide a valid source file.")
        return
    
    print("🔄 Fetching data from Encyclopedia of Korean Culture...")
    fetch_from_encykorea(src_path, dst_path, exclude, ENCYKOREA_API_KEY, ENCYKOREA_ENDPOINT)

    print("✅ Data fetching complete.")


if __name__ == "__main__":
    main()