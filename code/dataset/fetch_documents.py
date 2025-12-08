import json, requests, pandas as pd
from tqdm import tqdm

from utils.utils import load_json_file


# ---------------- FETCH DATA ----------------
def fetch_from_encykorea(src_path: str, dst_path: str, api_key: str, endpoint_url: str) -> None:
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    elif src_path.endswith((".xls", "xlsx")):
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        print(f"❗ File {src_path} is invalid.")
        return
    
    if not (fetched := load_json_file(dst_path)):
        fetched = {}
    counter = len(fetched) + 1
    headers = { "X-API-Key": api_key }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from EncyKorea"):
        eid = get_eid_from_row(
            row=row.values,
            # include=["불화"],
            exclude=["기록유산", "변상도", "불화", "지도", "초상", "추상", "화첩", "현대"]
        )
        if eid is None:
            continue
        response = requests.get(url=endpoint_url+eid, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        article = data.get("article")
        if not (image := article.get("headMedia") or (article.get("relatedMedias") or [None])[0]):
            continue
        if not (content := article.get("body")):
            continue
        fetched[counter] = {
            "title":   f"{article.get('headword')} ({article.get('origin')})",
            "img_url": image.get("url"),
            "era":     article.get("era"),
            "content": content.replace('\r', '').split('\n', 1)[1].strip(),
        }
        counter += 1

    with open(dst_path, "w", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


def fetch_from_heritage(src_path: str, dst_path: str, api_key: str, endpoint_url: str) -> None:
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    elif src_path.endswith((".xls", "xlsx")):
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        print(f"❗ File {src_path} is invalid.")
        return
    
    if not (fetched := load_json_file(dst_path)):
        fetched = {}
    counter = len(fetched) + 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from FolkEncy"):
        params = {
            "ccbaKdcd": row.values[1],
            "ccbaAsno": row.values[2],
            "ccbaCtcd": row.values[3],
        }
        response = requests.get(url=endpoint_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        fetched[counter] = {
            "title":   row.values[0],
            "img_url": data.get("imageUrl"),
            "era":     data.get("ccceName"),
            "content": data.get("content"),
        }
        counter += 1

    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


def fetch_from_folkency(src_path: str, dst_path: str, api_key: str, endpoint_url: str) -> None:
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path, header=None, dtype=str, encoding="utf-8")
    elif src_path.endswith((".xls", "xlsx")):
        df = pd.read_excel(src_path, header=None, dtype=str, encoding="utf-8")
    else:
        print(f"❗ File {src_path} is invalid.")
        return
    
    if not (fetched := load_json_file(dst_path)):
        fetched = {}
    counter = len(fetched) + 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from FolkEncy"):
        params = {
            "tit_idx": row.values[1],
            "korname": row.values[0],
            "serviceKey": api_key,
        }
        response = requests.get(url=endpoint_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        fetched[counter] = {
            "title":   data.get("summary"),
            "img_url": data.get("img_url_l"),
            "era":     None,
            "content": None,
        }
        counter += 1

    with open(dst_path, "a", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


# ---------------- UTILS ----------------
def get_eid_from_row(row: list, include: list[str] = [], exclude: list[str] = []) -> str:
    url = str(row[1])
    tag = str(row[2])
    if include != []:
        if any(key in tag for key in include):
            return url.strip().split("/")[-1]
    if exclude != [] and all(key not in tag for key in exclude):
        return url.strip().split("/")[-1]
    return None