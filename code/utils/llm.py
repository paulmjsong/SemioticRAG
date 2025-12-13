import base64, ollama, torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ---------------- BASE CLASSES ----------------
class BaseLLM:
    def generate(self, user_prompt: str, system_prompt: str|None=None, img_path: str|None=None, **kwargs) -> str:
        raise NotImplementedError

class BaseClassifier:
    def classify(self, sequences: list[str], labels: list[str], template: str|None=None) -> list[str]:
        raise NotImplementedError

class BaseEmbedder:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        return NotImplementedError


# ---------------- LLM WRAPPER ----------------
class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, user_prompt: str, system_prompt: str|None=None, img_path: str|None=None, **kwargs) -> str:
        messages = _build_messages(user_prompt, system_prompt, img_path)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

class LocalLLM(BaseLLM):
    def __init__(self, model: str):
        self.pipe = pipeline(
            task="image-text-to-text",
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, user_prompt: str, system_prompt: str|None=None, img_path: str|None=None, **kwargs) -> str:
        messages = _build_messages(user_prompt, system_prompt, img_path)
        response = self.pipe(
            text=messages,
            return_full_text=False,
            **kwargs,
        )
        return response[0]['generated_text'].strip()


# ---------------- CLASSIFIER WRAPPER ----------------
class LocalClassifier(BaseClassifier):
    def __init__(self, model: str):
        self.pipe = pipeline(
            task="zero-shot-classification",
            model=model,
        )

    def classify(self, sequences: list[str], candidate_labels: list[str], hypothesis_template: str|None=None) -> list[str]:
        if hypothesis_template:
            return self.pipe(
                sequences=sequences,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False,
            )
        return self.pipe(
            sequences=sequences,
            candidate_labels=candidate_labels,
            multi_label=False,
        )


# ---------------- EMBEDDER WRAPPER ----------------
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str, model_dim: int, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = model_dim

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
    
    def get_dimension(self) -> int:
        return self.dimension

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model: str):
        self.embedder = SentenceTransformer(model_name_or_path=model)

    def embed(self, text: str) -> list[float]:
        return self.embedder.encode(text)
    
    def get_dimension(self) -> int:
        return self.embedder.get_sentence_embedding_dimension()


# ---------------- UTILS ----------------
def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

def _build_messages(user_prompt: str, system_prompt: str|None=None, img_path: str|None=None) -> list:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if not img_path:
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": _encode_image(img_path)}}
        ]})
    return messages