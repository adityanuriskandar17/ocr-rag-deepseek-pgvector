"""RAG chain: hybrid retriever (vector + FTS + RRF) + DeepSeek via LangChain."""

import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings
from src.rag.embed_store import get_vectorstore
from src.rag.hybrid import hybrid_search
from src.rag.rerank import rerank

SYSTEM = """Kamu asisten dokumen Indonesia/Inggris.
Jawab HANYA dari konteks. Kalau tidak ada di konteks, katakan tidak tahu.
Selalu sertakan sitasi [source hal. X].
Bahasa jawab ikuti bahasa pertanyaan."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "human",
            "Konteks:\n{context}\n\nPertanyaan: {question}",
        ),
    ]
)


def get_llm():
    # DeepSeek API kompatibel OpenAI, jadi pakai ChatOpenAI + base_url.
    # DEEPSEEK_MODEL bisa diisi "deepseek-chat" atau string "v4 flash" dari dashboard kamu.
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0,
    )


def format_docs(docs):
    out = []
    for d in docs:
        src = d.metadata.get("source", "?")
        pg = d.metadata.get("page", "?")
        out.append(f"[{src} hal. {pg}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)


def get_chain():
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": settings.TOP_K})
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    # return chain + retriever biar API bisa tampilkan sources
    return chain, retriever


def query(question: str) -> dict:
    """Query v2: agent (router/retrieve/rewrite/grade) dengan fallback.

    - AGENT_ENABLED=true -> LangGraph di src/rag/graph.py.
    - Kalau agent gagal (misal graph error), fallback ke hybrid+rerank langsung
      biar API tidak pernah 500 untuk masalah agent.
    """
    if settings.AGENT_ENABLED:
        try:
            from src.rag.graph import agent_answer

            return agent_answer(question)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("agent gagal (%s), fallback hybrid", e)
    return _hybrid_query(question)


def _hybrid_query(question: str) -> dict:
    """Hybrid (vector + FTS + RRF) -> rerank -> generate, tanpa agent."""
    t0 = time.perf_counter()
    candidates = hybrid_search(
        question,
        vector_k=settings.HYBRID_VECTOR_K,
        keyword_k=settings.HYBRID_KEYWORD_K,
        top_k=settings.RERANK_CANDIDATES,
    )
    retrieve_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    docs = rerank(question, candidates, top_k=settings.TOP_K)
    rerank_ms = (time.perf_counter() - t1) * 1000

    # Generate dengan pertanyaan ASLI (biar bahasa & kapital jawaban natural).
    context = format_docs(docs)
    t2 = time.perf_counter()
    answer = (PROMPT | get_llm() | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    llm_ms = (time.perf_counter() - t2) * 1000
    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ]
    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "retrieve_ms": round(retrieve_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "llm_ms": round(llm_ms, 1),
        },
        "route": "hybrid_rerank_rag",
    }
