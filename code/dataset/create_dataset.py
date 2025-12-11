import json, nltk, regex, numpy as np
from numpy.linalg import norm

from utils.llm import BaseLLM, BaseEmbedder
from utils.utils import load_json_file


def create_dataset(src_paths: list[str], dst_path: str, embedder: BaseEmbedder, llm: BaseLLM | None = None) -> None:
    for src_path in src_paths:
        if not (fetched := load_json_file(src_path)):
            print(f"❗ File {src_path} is not valid JSON.")
            continue
        if not (dataset := load_json_file(dst_path)):
            dataset = {}

        for i, item in enumerate(fetched):
            text = item.get("desc")
            sents = nltk.sent_tokenize(text)
            good_sents = []

            for sent in sents:
                if is_good_sent(sent, embedder, llm):
                    good_sents.append(sent)
            dataset[i] = {
                "title":     item.get("title"),
                "img_url":   item.get("img_url"),
                "era":       item.get("era"),
                "sentences": good_sents,
            }

        with open(dst_path, "w", encoding="utf-8") as dst_file:
            json.dump(dataset, dst_file, ensure_ascii=False, indent=2)


def is_good_sent(sentence: str, embedder: BaseEmbedder) -> bool:
    # TODO: filtering logic
    # Step 1
    regex_patterns = ["의미", "해석", "상징", "뜻", "염원", "교훈"]  # but not "상징성" -> how?
    if any(r in sentence for r in regex_patterns):
        return True
    # Step 2
    emb_patterns = [
        "특히 풍속화들은 조영석의 서민 삶에 대한 관심을 사실주의적 창작 태도로 구체화시킨 것이다",  # 사제첩 (E0025955)
        "전체 경관이 몇 개로 따로따로 떨어져 있으면서 조화를 이루는 경군(景群)들로 짜여 있다는 점이다.",  # 몽유도원도 (E0018824)
    ]
    embs = [embedder.embed(e) for e in emb_patterns]
    sentence_emb = embedder.embed(sentence)
    for emb in embs:
        dot_product = np.dot(sentence_emb, emb)
        magnitude1 = norm(sentence_emb)
        magnitude2 = norm(emb)
        similarity = dot_product / (magnitude1 * magnitude2)
        if similarity >= 0.7:
            return True
    # END TODO
    return False


def is_good_sent_llm(text: str, llm: BaseLLM) -> list:
    # Step 3
    if llm:
        system_prompt = "From the given text, return all sentences that contain information about the symbolic meaning of an artwork or artistic genre. Return the sentences in their original form. The output must be in valid JSON format as an array of strings. If no sentences satisfy the condition, return an empty JSON array ([]). Do not output any other text."
        user_prompt = f"Given: {text}"
        response = llm.generate(user_prompt=user_prompt, system_prompt=system_prompt)
        return response.json(response)


def is_hangul(string: str) -> bool:
    return bool(regex.search(r'\p{IsHangul}', string))

def is_hanja(string: str) -> bool:
    return bool(regex.search(r'\p{IsHan}', string))