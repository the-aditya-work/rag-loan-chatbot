from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import RAGPipeline

app = FastAPI(title="RAG Loan Chatbot")
pipeline = RAGPipeline()

class Question(BaseModel):
    question: str
    top_k: int | None = None

@app.post("/ask")
def ask(q: Question):
    return pipeline.query(q.question, top_k=q.top_k)
