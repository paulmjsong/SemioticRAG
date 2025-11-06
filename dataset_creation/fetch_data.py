import json, requests, pandas as pd
from tqdm import tqdm


# ---------------- SCRAPE DATA ----------------
def fetch_from_encykorea(src_path: str, dst_path: str, keywords: list[str], API_KEY: str, ENDPOINT_URL: str) -> None:
    headers = { "X-API-Key": API_KEY }
    fetched = []

    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from Encyclopedia of Korean Culture"):
        line = ",".join(map(str, row.values))
        eid = get_eid_from_line(line, keywords)
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

# TODO: Check the parameters needed for the KHS API
def fetch_from_heritage(src_path: str, dst_path:str, keywords: list[str], API_KEY: str, ENDPOINT_URL: str) -> None:
    headers = { "X-API-Key": API_KEY }
    fetched = []

    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from Korea Heritage Service"):
        line = ",".join(map(str, row.values))
        eid = get_eid_from_line(line, keywords)
        if eid is None:
            continue
        params = {
            "ccbaKdcd": "[Enter field id]",
            "ccbaAsno": eid,
            "ccbaCtcd": "[Enter region id]",
        }
        response = requests.get(url=ENDPOINT_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        fetched.append({
            "headword": data.get("ccbaAsno"),
            "body": data.get("content").replace('\r', '').split('\n', 1)[1].strip(),
        })

    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


# ---------------- UTILS ----------------
def get_eid_from_line(line: str, keywords: list[str]) -> str:
    parts = line.strip().split(",")
    field = parts[1]
    url = parts[-1]
    if any(keyword in field for keyword in keywords):
        return url.strip().split("/")[-1]
    return None