import json, nltk, re
from tqdm import tqdm

from utils.llm import BaseLLM, BaseClassifier, BaseEmbedder
from utils.utils import load_json_file


# ---------------- CREATE DATASET ----------------
def create_dataset(src_paths: list[str], dst_path: str, classifier: BaseClassifier, is_analysis: bool=False) -> None:
    for src_path in src_paths:
        if not (fetched := load_json_file(src_path)):
            print(f"❗ File {src_path} is not valid JSON.")
            continue
        if not (dataset := load_json_file(dst_path)):
            dataset = {}

        for key, item in tqdm(fetched.items(), total=len(fetched), desc=f"Creating dataset from {src_path}"):
            if not(sents := item.get("sentences")):
                sents = nltk.sent_tokenize(item.get("desc"))
            if is_analysis:
                sents_new = get_sents_analyzed(sents, classifier)
            else:
                sents_new = get_sents_filtered(sents, classifier)
            dataset[key] = {
                "title":     item.get("title"),
                "image":     item.get("image"),
                "era":       item.get("era"),
                "sentences": sents_new,
            }

        with open(dst_path, "w", encoding="utf-8") as dst_file:
            json.dump(dataset, dst_file, ensure_ascii=False, indent=4)


# ---------------- CLASSIFY SENTENCES ----------------
candidate_labels = [
    ## Used in datasets 1-2
    # "visual description",                 # 
    # "symbolic meaning",                   # keep
    # "physical attributes",                # 
    # "metadata/title",                     # 
    # "contextual or external information", # (not in 1)

    ## Used in datasets 3-5
    # "visual description",             # 
    # "symbolic meaning",               # keep
    # "social function",                # keep (not in 4,5)
    # "historical background",          # keep?
    # "style and technique",            # (not in 4,5)
    # "metadata",                       # 
    # "irrelevant or noise",            # (not in 5)

    ## Used in datasets 6-7
    # "visual composition",             # 
    # "symbolic meaning",               # keep
    # "cultural background",            # keep? ("historical" in 6, "cultural" in 7)
    # "metadata",                       # 
    
    ## Used in dataset 8
    "subject and composition",          # 
    "symbolism and meaning",            # keep
    "artistic technique",               # 
    "historical context",               # keep?
    "physical metadata",                # 
]

META_RE = re.compile(r"(제목|작품명|작가|소장|전시|출처|제작연도|연도|기증)", re.IGNORECASE)
SIZE_RE = re.compile(r"(센티|cm|밀리|가로|세로|높이|길이|크기)", re.IGNORECASE)

def is_obvious_non_symbolic(sent):
    if META_RE.search(sent): return True
    if SIZE_RE.search(sent): return True
    return False

def get_sents_filtered(sents: str | list[str], classifier: BaseClassifier, min_score=0.60, min_margin=0.15) -> list[str]:
    sents = [sent for sent in sents if not is_obvious_non_symbolic(sent)]
    results = classifier.classify(sents, candidate_labels)

    good_sents = []
    for result in results:
        sequence = result["sequence"]
        labels = result["labels"]
        # scores = result["scores"]
        top_label = labels[0]
        # top_score = scores[0]
        # second_score = result["scores"][1]
        # if top_score >= min_score and (top_score - second_score) >= min_margin:
        if "symbolism" in top_label or "context" in top_label:
            good_sents.append(sequence)
    
    return good_sents

def get_sents_analyzed(sents: str | list[str], classifier: BaseClassifier) -> list[str]:
    sents = [sent for sent in sents if not is_obvious_non_symbolic(sent)]
    template = "This text is about {} of an artwork."
    results = classifier.classify(sents, candidate_labels, template)

    good_sents = []
    for result in results:
        simple_labels = []
        for label in result["labels"]:
            simple_labels.append(label.split()[-1])
        # for label in result["labels"]:
        #     if "describes" in label:
        #         simple = "descriptive"
        #     else:
        #         simple = "symbolic"
        #     simple_labels.append(simple)
        good_sents.append({
            "sequence": result["sequence"],
            "labels": simple_labels,
            "scores": [round(score, 10) for score in result["scores"]],
        })
    
    return good_sents