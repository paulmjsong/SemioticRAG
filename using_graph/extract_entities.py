import requests

from neo4j_graphrag.llm import OpenAILLM

from prompts import CONSTRUCTION_PROMPT


# ---------------- EXTRACT ENTITIES ----------------
def extract_entities(llm: OpenAILLM, save_path: str, qlist_path: str, api_key: str, enpoint_url: str) -> None:
    headers = {
        "X-API-Key": api_key,
        "Accept": 'application/json',
    }
    with open(qlist_path, "r") as file:
        for line in file:
            q = line.strip()
            # page = wikipedia.page(src, auto_suggest=False)  # TODO: wikipedia -> actual source
            params = {
                "q": q,
                "page": 1,
                "size": 10,
            }
            response = requests.get(enpoint_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results=[]
            for item in data.get("data", []):
                if "content" in item:
                    result = llm.invoke(
                        f"Input data:\n{item.content}\n\n{CONSTRUCTION_PROMPT}",
                    )
                    results.append(result.content)
            
            with open(save_path, "a", encoding="utf-8") as file:
                for result in results:
                    file.write(result + "\n")
            file.close()
    file.close()