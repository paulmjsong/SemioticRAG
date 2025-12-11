import json, os, requests, time
from tqdm import tqdm


def download_img(url: str, save_dir: dict, relic_id: str, max_attempts: int=5, delay: float=1.0) -> str | None:
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=30)
            path = os.path.join(save_dir, f"{relic_id}.jpg")
            with open(path, "wb") as f:
                f.write(response.content)
            return path
        except Exception as e:
            print(f"[ERROR] Request failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(delay)

    raise RuntimeError("Failed to fetch image after multiple attempts")


def main():
    file_path = "../example/dataset/fetched_emuseum.json"
    with open(file_path, "r", encoding="utf-8") as file:
        fetched = json.load(file)
    
    save_dir = "../example/dataset/images/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for item in tqdm(fetched.values(), total=len(fetched), desc="Downloading relic images"):
        img_url = item.pop("img_url")
        relic_id = item.pop("id")
        item["image"] = download_img(img_url, save_dir, relic_id)
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(fetched, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()