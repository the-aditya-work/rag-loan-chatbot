from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import settings

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    documents = []
    for t in texts:
        documents.extend(splitter.split_text(t))
    return documents
