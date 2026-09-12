"""Hybrid retrieval Fase 1 v2: vector (pgvector) + keyword (Postgres FTS) + RRF.

Kenapa:
- Vector bagus untuk semantik ("total belanja berapa?").
- FTS bagus untuk exact ("INV-2024-0091", "Alodokter").
- RRF menggabung keduanya tanpa perlu tuning bobot.

Tabel langchain-postgres:
- langchain_pg_collection(uuid, name, cmetadata)
- langchain_pg_embedding(id, collection_id, embedding, document, cmetadata jsonb)
"""

import psycopg
from langchain_core.documents import Document

from src.config import settings
from src.rag.embed_store import get_vectorstore

RRF_K = 60


def _pg_url() -> str:
    return settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


def get_collection_id() -> str | None:
    with psycopg.connect(_pg_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
                (settings.PG_COLLECTION,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None


def vector_search(query: str, k: int = 20) -> list[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)


def keyword_search(query: str, k: int = 20) -> list[Document]:
    """FTS dengan config 'simple' + fallback ILIKE untuk kode pendek.

    'simple' dipilih karena:
    - tidak stemming (aman untuk nomor/kode),
    - lowercase otomatis (selesaikan masalah kapital 'Alodokter'),
    - tersedia di pg16 tanpa extension tambahan.
    """
    q = query.strip()
    if not q:
        return []
    coll_id = get_collection_id()
    if coll_id is None:
        return []

    with psycopg.connect(_pg_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document, cmetadata,
                       ts_rank(to_tsvector('simple', document),
                               plainto_tsquery('simple', %s)) AS rank
                FROM langchain_pg_embedding
                WHERE collection_id = %s
                  AND to_tsvector('simple', document) @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (q, coll_id, q, k),
            )
            rows = cur.fetchall()

            # Fallback: kalau tsquery tidak kena sama sekali (misal kode
            # dengan tanda baca aneh), pakai ILIKE substring.
            if not rows:
                cur.execute(
                    """
                    SELECT document, cmetadata, 0.0 AS rank
                    FROM langchain_pg_embedding
                    WHERE collection_id = %s AND document ILIKE %s
                    LIMIT %s
                    """,
                    (coll_id, f"%{q[:80]}%", k),
                )
                rows = cur.fetchall()

    return [
        Document(page_content=r[0], metadata=r[1] or {})
        for r in rows
    ]


def _doc_key(d: Document) -> tuple:
    m = d.metadata or {}
    # Pakai full content: chunk overlap 120 char bikin [:120] tabrakan
    # untuk chunk berurutan dari halaman yang sama.
    return (m.get("source"), m.get("page"), d.page_content)


def rrf_fusion(*doc_lists: list[Document], top_k: int = 5, rrf_k: int = RRF_K) -> list[Document]:
    """Reciprocal Rank Fusion: score(d) = sum 1/(rrf_k + rank)."""
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for docs_list in doc_lists:
        for rank, d in enumerate(docs_list, start=1):
            key = _doc_key(d)
            docs[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[k] for k, _ in ranked[:top_k]]


def hybrid_search(
    query: str,
    vector_k: int = 20,
    keyword_k: int = 20,
    top_k: int | None = None,
) -> list[Document]:
    top_k = top_k or settings.TOP_K
    vec_docs = vector_search(query, k=vector_k)
    kw_docs = keyword_search(query, k=keyword_k)
    if not kw_docs:
        return vec_docs[:top_k]
    if not vec_docs:
        return kw_docs[:top_k]
    return rrf_fusion(vec_docs, kw_docs, top_k=top_k)
