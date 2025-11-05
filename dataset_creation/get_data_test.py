import os, requests, urllib.parse
from dotenv import load_dotenv


# ---------------- CONFIG ----------------
load_dotenv()
ENCYKOREA_API_KEY = os.getenv("ENCYKOREA_API_KEY")
ENCYKOREA_ENDPOINT_SEARCH = os.getenv("ENCYKOREA_ENDPOINT_SEARCH")
ENCYKOREA_ENDPOINT_FIELD = os.getenv("ENCYKOREA_ENDPOINT_FIELD")
ENCYKOREA_ENDPOINT_ARTICLE = os.getenv("ENCYKOREA_ENDPOINT_ARTICLE")


# ---------------- GET DATA ----------------
def get_from_encykorea(API_KEY: str) -> None:
    headers = {
        "X-API-Key": API_KEY,
        "Accept": 'application/json; charset=UTF-8',
    }
    # params = {
    #     "keyword": "민화",
    #     "field": "예술·체육",
    #     "pageNo": 1,
    # }

    # response = requests.get(url=ENCYKOREA_ENDPOINT_SEARCH, params=params, headers=headers, timeout=30)
    # response = requests.get(url=ENCYKOREA_ENDPOINT_FIELD+"예술·체육", headers=headers, timeout=30)
    eid="E0048152"
    response = requests.get(url=ENCYKOREA_ENDPOINT_ARTICLE+eid, headers=headers, timeout=30)
    
    response.raise_for_status()
    data = response.json()

    # articles = data.get("articles")
    # print(articles[0].get("headword"))
    article = data.get("article")
    print(article.get("headword"))


# ---------------- MAIN ----------------
def main():
    get_from_encykorea(ENCYKOREA_API_KEY)


if __name__ == "__main__":
    main()