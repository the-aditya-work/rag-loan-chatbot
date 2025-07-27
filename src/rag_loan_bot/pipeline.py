import numpy as np
from .config import settings
from .embed import Embedder
from .vectorstore import FAISSStore
from .generator import LocalGenerator, openai_generate, build_prompt
from .pipeline import RagPipeline  # or the correct name

class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = FAISSStore.load()
        self.generator = LocalGenerator() if not settings.USE_OPENAI else None

    def query(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or settings.TOP_K
        q_emb = self.embedder.encode([question])
        results = self.store.search(q_emb, k=top_k)
        contexts = [r[0] for r in results]
        prompt = build_prompt(question, contexts)
        if settings.USE_OPENAI:
            answer = openai_generate(prompt)
        else:
            answer = self.generator.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }
