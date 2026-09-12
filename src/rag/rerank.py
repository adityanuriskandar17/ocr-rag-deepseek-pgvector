"""Reranker Fase 2 v2: cross-encoder CPU untuk saring kandidat hybrid.

Alur: hybrid top-20 (recall) -> cross-encoder (precision) -> top-5.
Model default `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual,
termasuk Indonesia — dipilih lewat eksperimen: varian English-only
`ms-marco-MiniLM` menggeser chunk ID yang benar). CPU ±100-150ms/doc.
Fallback: kalau model gagal load (misal offline), kembalikan urutan awal.
"""

import logging

from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(
            settings.RERANK_MODEL,
            device="cpu",
            max_length=512,
        )
    return _reranker


def rerank(question: str, docs: list[Document], top_k: int | None = None) -> list[Document]:
    top_k = top_k or settings.TOP_K
    if not docs:
        return []
    if not settings.RERANK_ENABLED or len(docs) <= 1:
        return docs[:top_k]
    try:
        model = get_reranker()
    except Exception as e:  # offline / download gagal -> jangan matikan RAG
        logger.warning("reranker load gagal (%s), pakai urutan hybrid", e)
        return docs[:top_k]

    # Batasi panjang per doc biar inferensi CPU cepat + stabil.
    pairs = [(question, (d.page_content or "")[:2000]) for d in docs]
    scores = model.predict(pairs, batch_size=settings.RERANK_BATCH_SIZE, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_k]]


def rerank_with_scores(
    question: str, docs: list[Document], top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Varian debug: kembalikan (doc, score) buat eval/analisa."""
    top_k = top_k or settings.TOP_K
    if not docs:
        return []
    if not settings.RERANK_ENABLED or len(docs) <= 1:
        return [(d, 0.0) for d in docs[:top_k]]
    model = get_reranker()
    pairs = [(question, (d.page_content or "")[:2000]) for d in docs]
    scores = model.predict(pairs, batch_size=settings.RERANK_BATCH_SIZE, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_k]]
