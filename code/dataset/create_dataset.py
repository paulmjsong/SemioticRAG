import json, nltk, re
from tqdm import tqdm

from utils.llm import BaseClassifier
from utils.utils import load_json_file


# ---------------- CREATE DATASET ----------------
def create_dataset(src_paths: list[str], dst_path: str, classifier: BaseClassifier) -> None:
    idx = 1
    for src_path in src_paths:
        if not (fetched := load_json_file(src_path)):
            print(f"❗ File {src_path} is not valid JSON.")
            continue
        if not (dataset := load_json_file(dst_path)):
            dataset = {}

        for key, item in tqdm(fetched.items(), total=len(fetched), desc=f"Creating dataset from {src_path}"):
            if not(sents := item.get("sentences")):
                sents = nltk.sent_tokenize(item.get("desc"))
            sents_new = _get_sents_analyzed(sents, classifier)
            if not sents_new:
                continue
            dataset[idx] = {
                "title":     item.get("title"),
                "image":     item.get("image"),
                "era":       item.get("era"),
                "sentences": sents_new,
            }
            idx += 1

        with open(dst_path, "w", encoding="utf-8") as dst_file:
            json.dump(dataset, dst_file, ensure_ascii=False, indent=4)


# ---------------- CLASSIFY SENTENCES ----------------
candidate_labels = [
    "subject and composition",      # descriptive
    "symbolism and meaning",        # symbolic
    "artistic technique",           # ???
    "historical context",           # contextual
    "physical metadata",            # x
]

META_RE = re.compile(r"(제목|작품명|작가|소장|전시|출처|제작연도|연도|기증)", re.IGNORECASE)
SIZE_RE = re.compile(r"(센티|cm|밀리|가로|세로|높이|길이|크기)", re.IGNORECASE)

def _is_obvious_non_symbolic(sent: str, min_chars: int=12, min_tokens: int=3) -> bool:
    # if len(re.findall(r'[가-힣]', sent)) < min_chars:
    #     return True
    if len(sent.split()) < min_tokens:
        return True
    if META_RE.search(sent): return True
    if SIZE_RE.search(sent): return True
    return False

def _get_sents_analyzed(sents: str | list[str], classifier: BaseClassifier) -> list[str]:
    sents = [sent for sent in sents if not _is_obvious_non_symbolic(sent)]
    template = "This text is about {} of an artwork."
    results = classifier.classify(sents, candidate_labels, template)

    good_sents = []
    for result in results:
        simple_labels = []
        for label in result["labels"]:
            simple_labels.append(label.split()[-1])
        good_sents.append({
            "sequence": result["sequence"],
            "labels": simple_labels,
            "scores": [round(score, 10) for score in result["scores"]],
        })
    return good_sents