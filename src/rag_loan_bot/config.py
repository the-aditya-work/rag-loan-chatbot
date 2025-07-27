from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    DATA_RAW_DIR: Path = Path("data/raw")
    DATA_PROCESSED_DIR: Path = Path("data/processed")
    VECTORSTORE_PATH: Path = DATA_PROCESSED_DIR / "faiss_index"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    EMBEDDING_MODEL: str = "intfloat/e5-small-v2"
    GEN_MODEL: str = "google/flan-t5-base"  # light & free
    TOP_K: int = 4

    # Optionally use an API LLM (OpenAI, Claude, Gemini) by setting these
    OPENAI_API_KEY: str | None = None
    USE_OPENAI: bool = False
    OPENAI_MODEL: str = "gpt-4o-mini"

settings = Settings()
