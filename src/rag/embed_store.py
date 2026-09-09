"""Embeddings lokal CPU + pgvector. DeepSeek tidak punya embedding API, jadi pakai HF lokal."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from src.config import settings

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore():
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.PG_COLLECTION,
        connection=settings.DATABASE_URL,
        use_jsonb=True,
    )
