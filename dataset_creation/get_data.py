import requests


# ---------------- EXTRACT ENTITIES ----------------
def get_from_encykorea(srcs_path: str, api_key: str, enpoint_url: str) -> None:
    headers = {
        "X-API-Key": api_key,
        "Accept": 'application/json',
    }
    query = "민화"
    with open(srcs_path, "a") as srcs_file:
        params = {
            "q": query,
            "page": 1,
            "size": 10,
        }
        response = requests.get(enpoint_url, params=params)
        response.raise_for_status()
        data = response.json()
        for item in data.get("data", []):
            if "content" in item:
                srcs_file.write(item.content + "\n")


def get_from_heritage(srcs_path: str, api_key: str, enpoint_url: str) -> None:
    headers = {
        "X-API-Key": api_key,
        "Accept": 'application/json',
    }
    query = "민화"
    with open(srcs_path, "a") as srcs_file:
        params = {
            "q": query,
            "page": 1,
            "size": 10,
        }
        response = requests.get(enpoint_url, params=params)
        response.raise_for_status()
        data = response.json()
        for item in data.get("data", []):
            if "content" in item:
                srcs_file.write(item.content + "\n")