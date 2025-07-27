import faiss
import numpy as np
from pathlib import Path
import pickle
from .config import settings

class FAISSStore:
    def __init__(self, dim: int, store_path: Path = settings.VECTORSTORE_PATH):
        self.store_path = store_path
        self.index = faiss.IndexFlatIP(dim)  # using cosine-like via normalized vectors
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: list[str]):
        if len(self.texts) == 0:
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)
        self.texts.extend(texts)

    def save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.store_path.with_suffix(".faiss")))
        with open(self.store_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.texts, f)

    @classmethod
    def load(cls):
        path = settings.VECTORSTORE_PATH
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".pkl"), "rb") as f:
            texts = pickle.load(f)
        store = cls(index.d, path)
        store.index = index
        store.texts = texts
        return store

    def search(self, query_emb: np.ndarray, k: int):
        D, I = self.index.search(query_emb, k)
        results = [(self.texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results
