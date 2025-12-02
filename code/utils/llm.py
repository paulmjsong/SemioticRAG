import base64, ollama, os, torch
from openai import OpenAI
from huggingface_hub import InferenceClient
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings


# ---------------- BASE CLASSES ----------------
class BaseLLM:
    def generate(self, user_prompt: str, system_prompt: str=None, img_path: str=None, **kwargs) -> str:
        raise NotImplementedError

class BaseEmbedder:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


# ---------------- LLM WRAPPER ----------------
class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name

    def generate(self, user_prompt: str, system_prompt: str=None, img_path: str=None, **kwargs) -> str:
        messages = build_messages(user_prompt, system_prompt, img_path)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

class OllamaLLM:
    def __init__(self, model_name: str):
        self.client = ollama.Client()
        self.model = model_name

    def generate(self, user_prompt: str, system_prompt: str=None, img_path: str=None, **kwargs) -> str:
        messages = build_messages(user_prompt, system_prompt, img_path)
        response = ollama.chat(
            messages=messages,
            model=self.model,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

class HuggingFaceLLM:
    def __init__(self, model_name: str):
        billing_address = os.getenv("HUGGING_FACE_BILLING_ADDRESS")
        self.client = InferenceClient(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            headers={"X-HF-Bill-To": billing_address} if billing_address else None,
        )

    def generate(self, user_prompt: str, system_prompt: str=None, img_path: str=None, **kwargs) -> str:
        messages = build_messages(user_prompt, system_prompt, img_path)
        response = self.client.chat.completions.create(
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

class LocalLLM:
    def __init__(self, model_name: str):
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, user_prompt: str, system_prompt: str=None, img_path: str=None, **kwargs) -> str:
        messages = build_messages(user_prompt, system_prompt, img_path)
        response = self.pipe(
            text=messages,
            return_full_text=False,
            **kwargs,
        )
        return response[0]['generated_text'].strip()


# ---------------- EMBEDDER WRAPPER ----------------
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.client = ollama.Client()
        self.model = model_name

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings(
            prompt=text,
            model=self.model,
        )
        return response["embedding"]

class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        billing_address = os.getenv("HUGGING_FACE_BILLING_ADDRESS")
        self.embedder = HuggingFaceInferenceAPIEmbeddings(
            model_name=model_name,
            api_key=os.getenv("HUGGING_FACE_API_KEY"),
            headers={"X-HF-Bill-To": billing_address} if billing_address else None,
        )

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)


# ---------------- UTILS ----------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

def build_messages(user_prompt: str, system_prompt: str=None, img_path: str=None) -> list:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if not img_path:
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": encode_image(img_path)}}
        ]})
    return messages