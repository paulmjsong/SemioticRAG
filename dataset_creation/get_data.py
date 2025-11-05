import requests
from tqdm import tqdm


# ---------------- GET DATA ----------------
def get_from_encykorea(src_path: str, dst_path:str, API_KEY: str, ENDPOINT_URL: str) -> None:
    headers = {
        "X-API-Key": API_KEY,
    }
    with open(src_path, "r") as src_file, open(dst_path, "a") as dst_file:
        for line in tqdm(src_file.readlines(), desc="Fetching data from Encyclopedia of Korean Culture"):
            eid = get_eid_from_line(line)
            response = requests.get(url=ENDPOINT_URL+eid, headers=headers, timeout=30)
            
            response.raise_for_status()
            data = response.json()
            
            article = data.get("article")
            print(article.get("headword"))
            dst_file.write(f"{article.get("body")}\n")

# TODO: Implement this function
def get_from_heritage(src_path: str, dst_path:str, API_KEY: str, ENDPOINT_URL: str) -> None:
    return


# ---------------- UTILS ----------------
# TODO: Implement this function
def get_eid_from_line(line: str) -> str:
    return "E0048152"