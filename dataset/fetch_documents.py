import json, requests, pandas as pd
from tqdm import tqdm


# ---------------- FETCH DATA ----------------
def fetch_from_encykorea(src_path: str, dst_path: str, api_key: str, endpoint_url: str) -> None:
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    
    headers = { "X-API-Key": api_key }
    fetched = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from EncyKorea"):
        eid = get_eid_from_row(row.values, ["기록유산", "변상도", "불화", "지도", "초상", "추상", "화첩", "현대"])
        if eid is None:
            continue
        response = requests.get(url=endpoint_url+eid, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        article = data.get("article")
        image = article.get("headMedia", article.get("relatedMedias"))
        if not image:
            continue
        fetched.append({
            "title":   f"{article.get('headword')} ({article.get('origin')})",
            "content": article.get("body").replace('\r', '').split('\n', 1)[1].strip(),
            "image":   image.get("url"),
        })

    with open(dst_path, "w", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


def fetch_from_heritage(src_path: str, dst_path: str, api_key: str, endpoint_url: str) -> None:
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    
    fetched = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from FolkEncy"):
        params = {
            "ccbaKdcd": row.values[1],
            "ccbaAsno": row.values[2],
            "ccbaCtcd": row.values[3],
        }
        response = requests.get(url=endpoint_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        fetched.append({
            "title":   row.values[0],
            "content": data.get("content"),
            "image":   data.get("imageUrl"),
        })

    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


def fetch_from_folkency(src_path: str, dst_path: str, api_key: str, endpoint_url: str) -> None:
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    
    fetched = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from FolkEncy"):
        params = {
            "tit_idx": row.values[1],
            "korname": row.values[0],
            "serviceKey": api_key,
        }
        response = requests.get(url=endpoint_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        fetched.append({
            "title":   data.get("summary"),
            "content": "",
            "image":   data.get("img_url_l"),
        })

    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


# ---------------- UTILS ----------------
def get_eid_from_row(row: list, exclude: list[str] = []) -> str:
    url      = str(row[1])
    tag      = str(row[2])
    # has_img  = row[3]

    # if "X" in has_img:
    #     return None
    if any(key in tag for key in exclude):
        return None
    return url.strip().split("/")[-1]