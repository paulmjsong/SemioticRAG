import os, requests
from dotenv import load_dotenv

def main():
    load_dotenv()
    headers = { "X-API-Key": os.getenv("ENCYKOREA_API_KEY") }
    endpoint_url = os.getenv("ENCYKOREA_ENDPOINT_ARTICLE")
    eids = ["E0059590"]

    for eid in eids:
        response = requests.get(url=endpoint_url+eid, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        article = data.get("article")
        # print(f"{article.get('headword')} ({article.get('origin')})\n\n" + article.get("body"))
        # print(article.get('headMedia'))
        # print(article.get('relatedMedias'))
        print(article.get('headMedia') or article.get('relatedMedias'))

if __name__ == "__main__":
    main()