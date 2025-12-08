import json, nltk, numpy as np
from numpy.linalg import norm

from utils.llm import BaseLLM, BaseEmbedder
from utils.utils import load_json_file

# docs_format = {
#     "title":   str,
#     "img_url": str,
#     "era":     str,
#     "content": str,
# }

# data_format = {
#     "title":     str,
#     "img_url":   str,
#     "era":       str,
#     "sentences": list[str],
# }

def create_dataset(src_path: str, dst_path: str, embedder: BaseEmbedder, llm: BaseLLM | None = None) -> None:
    if not (docs := load_json_file(src_path)):
        print(f"❗ File {src_path} is not valid JSON.")
        return
    if not (data := load_json_file(dst_path)):
        data = {}

    for i, doc in enumerate(docs):
        text = doc.get("content")
        sentences = nltk.sent_tokenize(text)
        meaningful = []

        for sentence in sentences:
            if is_meaningful(sentence, embedder, llm):
                meaningful.append(sentence)
        data[i] = {
            "title": doc.get("title"),
            "img_url": doc.get("img_url"),
            "era": doc.get("era"),
            "sentences": meaningful,
        }

    with open(dst_path, "w", encoding="utf-8") as dst_file:
        json.dump(data, dst_file, ensure_ascii=False, indent=2)


def is_meaningful(sentence: str, embedder: BaseEmbedder, llm: BaseLLM | None = None) -> bool:
    # TODO: filtering logic
    # Step 1
    regex_patterns = ["의미", "상징", "뜻", ""]  # but not "상징성" -> how?
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
    # Step 3
    if llm:
        system_prompt = "Determine if the given sentence contains information about symbolic meaning of a certain artwork or genre of artworks. Answer with either True or False. Do not output any other text."
        user_prompt = f"Given: {sentence}\n\nTrue or False?"
        response = llm.generate(user_prompt=user_prompt, system_prompt=system_prompt)
        return bool(response)
    # END TODO
    return False