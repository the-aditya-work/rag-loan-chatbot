from .config import settings

def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
    return (
        "You are a helpful assistant. Answer the question using ONLY the given contexts. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"{context_block}\n\nQuestion: {question}\nAnswer:"
    )

# --- Local (free) FLAN-T5 ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LocalGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.GEN_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.GEN_MODEL)
    def generate(self, prompt: str, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- Optional OpenAI ---
def openai_generate(prompt: str):
    import openai
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not provided")
    openai.api_key = settings.OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content
