import json, os, requests, time, pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.utils import load_json_file


# ---------------- FETCH FROM ENCYKOREA ----------------
def fetch_from_encykorea(labels_path: str, save_dir: str, api_key: str, endpoint_url: str) -> None:
    if labels_path.endswith(".csv"):
        df = pd.read_csv(labels_path, header=None, dtype=str, encoding="utf-8")
    elif labels_path.endswith((".xls", "xlsx")):
        df = pd.read_excel(labels_path, header=None, dtype=str, encoding="utf-8")
    else:
        print(f"❗ File {labels_path} is invalid.")
        return
    
    # img_dir = os.path.join(save_dir, "images_encykorea")
    # if not os.path.exists(img_dir):
    #     os.makedirs(img_dir)
    
    save_path = os.path.join(save_dir, "fetched_encykorea.json")
    if not (fetched := load_json_file(save_path)):
        fetched = {}

    headers = {"X-API-Key": api_key}
    idx = len(fetched) + 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching data from EncyKorea"):
        eid = get_eid_from_row(
            row=row.values,
            # include=["불화"],
            exclude=["기록유산", "변상도", "불화", "지도", "초상", "추상", "화첩", "현대"]
        )
        if eid is None:
            continue
        response = requests.get(url=endpoint_url+eid, headers=headers, timeout=30)
        data = response.json()
        
        if not (article := data.get("article")):
            continue
        if not (desc := article.get("body")):
            continue
        if not (image := article.get("headMedia") or (article.get("relatedMedias") or [None])[0]):
            continue
        # TODO: download images
        fetched[idx] = {
            "title":   article.get('headword'),
            "img_url": image.get("url"),
            "era":     article.get("era"),
            "desc":    desc.replace('\r', '').split('\n', 1)[1].strip(),
        }
        idx += 1

    with open(save_path, "w", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


# ---------------- FETCH FROM HERITAGE ----------------
def fetch_from_heritage(save_dir: str, search_url: str, detail_url: str) -> None:
    ccbaKdcd = ""  # TODO
    page_unit = 10000
    page_idx = 1
    items = []
    pbar1 = tqdm(total=0, desc="Fetching relic ids")

    # --- Fetch list pages ---
    while True:
        pbar1.update(1)
        search_params = {
            "ccbaKdcd":  ccbaKdcd,                              # 종목코드
            "pageUnit":  page_unit,                             # 한 페이지 결과 수
            "pageIndex": page_idx,                              # 페이지 번호
        }
        if not (data := fetch_json(search_url, search_params)):
            continue
        if pbar1.total == 0:
            pbar1.total = data.get("totalCnt")
            pbar1.refresh()
        entries = data.get("item")
        pbar1.update(len(entries))

        for entry in entries:
            items.append({
                # "ccbaMnm1": entry.get("ccbaMnm1"),            # 명칭(국문)
                "ccbaAsno": entry.get("ccbaAsno"),              # 관리번호
                "ccbaCtcd": entry.get("ccbaCtcd"),              # 시도코드
            })
        if len(entries) < page_unit:
            break
        page_idx += 1
        time.sleep(0.3)

    # --- Fetch detail pages ---
    # img_dir = os.path.join(save_dir, "images_heritage")
    # if not os.path.exists(img_dir):
    #     os.makedirs(img_dir)
    
    save_path = os.path.join(save_dir, "fetched_heritage.json")
    if not (fetched := load_json_file(save_path)):
        fetched = {}
    
    idx = len(fetched) + 1
    
    for item in tqdm(items, total=len(items), desc="Fetching relic details"):
        detail_params = {
            "ccbaKdcd": ccbaKdcd,                               # 종목코드
            "ccbaAsno": item["ccbaAsno"],                       # 관리번호
            "ccbaCtcd": item["ccbaCtcd"],                       # 시도코드
        }
        if not (data := fetch_json(detail_url, detail_params)):
            continue
        entries = data["result"].get("item")
        
        for entry in entries:
            # TODO: download images
            fetched[idx] = {
                "title":   entry.get("ccbaMnm1"),               # 명칭(국문)
                "img_url": entry.get("imageUrl"),               # 대표이미지 URL
                "era":     entry.get("ccceName"),               # 시대
                "desc":    entry.get("content"),                # 설명
            }
            idx += 1
        time.sleep(0.3)

    with open(save_path, "w", encoding="utf-8") as dst_file:
        json.dump(fetched, dst_file, ensure_ascii=False, indent=4)


# ---------------- FETCH FROM EMUSEUM ----------------
def fetch_from_emuseum(save_dir: str, webpage_url: str, endpoint_url: str, api_key: str) -> None:
    # --- Fetch items ---
    total = 1836
    page_idx = 0
    items = {}
    pbar1 = tqdm(total=total, desc="Fetching relic ids")

    while page_idx <= total:
        pbar1.update(1)
        page_idx += 1
        web_params = {
            "pageNum":    page_idx,                             # 페이지 번호
            "sort":       "relicId",                            # 소장품 번호 순
            "detailFlag": "true",                               # 
            "facet3Lv1":  "PS06001",                            # 국가 코드 ("한국")
            "facet5Lv1":  "PS09009",                            # 용도 분류 코드 ("문화예술")
            "facet5Lv2":  "PS09009003",                         # 용도 분류 코드 ("서화")
            "facet5Lv3":  "PS09009003002",                      # 용도 분류 코드 ("회화")
            "facet5Lv4":  "PS09009003002002",                   # 용도 분류 코드 ("민화")
        }
        if not (soup := fetch_html(webpage_url, web_params)):
            break
        
        if (tag := soup.find("input", {"name": "relicId"})):
            relic_id = tag["value"]
        else:
            continue

        if (desc_tag := soup.find("span", class_="float-left wc110 lh35 mt3")):
            desc_raw = desc_tag.get_text(separator="\n", strip=True)
            desc = ". ".join([line.lstrip("- ").strip(". ") for line in desc_raw.split("\n")]) + "."
        else:
            continue

        if (tit_tag := soup.find("p", id="relicTitle")):
            title = tit_tag.get_text(strip=True)
        else:
            title = None

        if (era_tag := soup.find("em", string="국적/시대")):
            era_raw = era_tag.find_next("span").get_text(strip=True)
            era = era_raw.split("-")[-1].strip()
        else:
            era = None
        
        items[relic_id] ={
            "title": title,
            "era":   era,
            "desc":  desc,
        }
    pbar1.close()
    
    # --- Fetch images ---
    img_dir = os.path.join(save_dir, "images_emuseum")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    save_path = os.path.join(save_dir, "fetched_emuseum.json")
    if not (fetched := load_json_file(save_path)):
        fetched = {}
    
    page_unit = 10000
    page_idx = 0
    idx = len(fetched) + 1
    pbar2 = tqdm(total=0, desc="Fetching relic images")
    
    while items:
        page_idx += 1
        end_params = {
            "serviceKey":  api_key,                             # 인증키
            "numOfRows":   page_unit,                           # 한 페이지 결과 수
            "pageNo":      page_idx,                            # 페이지 번호
            "purposeCode": "PS09009",                           # 용도 분류 코드 ("문화예술")
        }
        if not (data := fetch_json(endpoint_url, end_params)):
            continue
        if pbar2.total == 0:
            pbar2.total = data.get("totalCount")
            pbar2.refresh()
        entries = data.get("list")
        pbar2.update(len(entries))

        for entry in entries:
            if (relic_id := entry.get("id")) not in items:
                continue

            item = items.pop(relic_id)
            if not (img_url := entry.get("imgUri")):
                continue
            
            img_path = download_img(img_url, img_dir, relic_id)
            title_kr = entry.get("nameKr") or entry.get("name") or item["title"]
            fetched[idx] = {
                "title": title_kr,                  # 명칭(국문)
                "image": img_path,                  # 이미지 경로
                "era":   item["era"],               # 시대
                "desc":  item["desc"],              # 설명
            }
            idx += 1
            
            if not items:
                break
        
        if len(entries) < page_unit:
            break
        time.sleep(0.3)
    
    pbar2.close()

    with open(save_path, "w", encoding="utf-8") as dst_file:
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

def fetch_json(url: str, params: dict, max_attempts: int=5, delay: float=1.0) -> dict | None:
    headers = {"Accept": "application/json"}
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            # --- Try parsing JSON ---
            try:
                data = response.json()
            except ValueError:
                print(f"[WARN] JSON parse failure (attempt {attempt}/{max_attempts})")
                time.sleep(delay)
                continue
            # --- Success ---
            return data
        except Exception as e:
            print(f"[ERROR] Request failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(delay)

    raise RuntimeError("Failed to fetch valid JSON after multiple attempts")

def fetch_html(url: str, params: dict, max_attempts: int=5, delay: float=1.0) -> dict | None:
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            # --- Try parsing HTML ---
            try:
                soup = BeautifulSoup(response.text, "html.parser")
            except ValueError:
                print(f"[WARN] HTML parse failure (attempt {attempt}/{max_attempts})")
                time.sleep(delay)
                continue
            # --- Success ---
            return soup
        except Exception as e:
            print(f"[ERROR] Request failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(delay)

    raise RuntimeError("Failed to fetch valid HTML after multiple attempts")

def download_img(url: str, save_dir: dict, save_name: str, max_attempts: int=5, delay: float=1.0) -> str | None:
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=30)
            path = os.path.join(save_dir, f"{save_name}.jpg")
            with open(path, "wb") as f:
                f.write(response.content)
            return path
        except Exception as e:
            print(f"[ERROR] Request failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(delay)

    raise RuntimeError("Failed to fetch image after multiple attempts")