"""Agent Fase 3 v2: router -> retrieve (hybrid+rerank) -> grade -> generate/rewrite.

Graf LangGraph (tanpa checkpointer, stateless per request):

  START -> router -> [direct_answer -> END | retrieve]
  retrieve -> grade -> [generate -> END | rewrite -> retrieve | no_answer -> END]

Kenapa tiap node ada:
- router: sapaan/obrolan umum tidak perlu bayar retrieval+embedding.
- grade: verifikasi sitasi *sebelum* generate. Kalau konteks tidak mendukung,
  jawab jujur "tidak tahu" daripada halusinasi (masalah v1/v2 awal).
- rewrite (max 1x): query vague ("berapa totalnya?") ditulis ulang jadi
  keyword pencarian ("total invoice") lalu retrieve ulang.
"""

import logging
import time
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from src.config import settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    docs: list[Any]
    rewrites: int
    route: str
    answer: str
    sources: list[dict]
    timings: dict


ROUTER_PROMPT = """Klasifikasikan pertanyaan pengguna ke SATU kata:
- DIRECT: sapaan, basa-basi, atau pengetahuan umum yang TIDAK butuh dokumen (misal "halo", "siapa kamu", "apa itu python").
- RAG: pertanyaan tentang isi dokumen/data yang di-upload (ciri: menyebut nama, angka, invoice, perusahaan, "dokumen ini", "berapa", "di mana", "siapa").

Pertanyaan: {question}
Jawab hanya: DIRECT atau RAG"""

REWRITE_PROMPT = """Tulis ulang pertanyaan menjadi kata kunci pencarian dokumen yang spesifik.
Buang kata ganti vague ("nya", "itu", "tersebut"), pertahankan nama/angka/peran penting.
Jawab hanya dengan query pencarian, tanpa penjelasan.

Pertanyaan asli: {question}"""

GRADER_PROMPT = """Apakah KONTEKS di bawah cukup untuk menjawab PERTANYAAN secara faktual?
Cukup = ada potongan teks yang secara langsung mendukung jawaban (nama/angka/fakta).
Jawab hanya: YA atau TIDAK.

PERTANYAAN: {question}

KONTEKS:
{context}"""


def _llm():
    # Import lokal biar graph.py tidak menarik chain.py (hindari circular import).
    from src.rag.chain import get_llm

    return get_llm()


def router_node(state: AgentState) -> dict:
    q = state["question"]
    try:
        out = (_llm() | StrOutputParser()).invoke(
            ROUTER_PROMPT.format(question=q[:1000])
        )
        label = out.strip().upper()
    except Exception as e:
        logger.warning("router gagal (%s), default RAG", e)
        label = "RAG"
    route = "direct" if "DIRECT" in label else "rag"
    return {"route": route, "search_query": q, "rewrites": 0}


def direct_answer_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    answer = (_llm() | StrOutputParser()).invoke(
        "Kamu asisten ramah berbahasa Indonesia. Jawab singkat (maks 3 kalimat). "
        "Kalau ditanya soal dokumen, arahkan user untuk bertanya spesifik.\n\n"
        f"Pertanyaan: {state['question']}"
    )
    return {
        "answer": answer,
        "sources": [],
        "timings": {"llm_ms": round((time.perf_counter() - t0) * 1000, 1)},
    }


def retrieve_node(state: AgentState) -> dict:
    from src.rag.hybrid import hybrid_search
    from src.rag.rerank import rerank

    q = state.get("search_query") or state["question"]
    timings = dict(state.get("timings") or {})
    t0 = time.perf_counter()
    candidates = hybrid_search(
        q,
        vector_k=settings.HYBRID_VECTOR_K,
        keyword_k=settings.HYBRID_KEYWORD_K,
        top_k=settings.RERANK_CANDIDATES,
    )
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t1 = time.perf_counter()
    docs = rerank(q, candidates, top_k=settings.TOP_K)
    timings["rerank_ms"] = timings.get("rerank_ms", 0) + round(
        (time.perf_counter() - t1) * 1000, 1
    )
    return {"docs": docs, "timings": timings}


def _grade(question: str, docs: list) -> bool:
    """True = konteks cukup, boleh generate. Heuristik dulu, LLM kedua."""
    if not docs:
        return False
    from src.rag.chain import format_docs

    context = format_docs(docs)[:6000]
    try:
        out = (_llm() | StrOutputParser()).invoke(
            GRADER_PROMPT.format(question=question[:1000], context=context)
        )
        return "YA" in out.strip().upper()
    except Exception as e:
        logger.warning("grader gagal (%s), default boleh generate", e)
        return True


def rewrite_node(state: AgentState) -> dict:
    try:
        new_q = (_llm() | StrOutputParser()).invoke(
            REWRITE_PROMPT.format(question=state["question"][:1000])
        )
        new_q = new_q.strip().strip('"') or state["question"]
    except Exception as e:
        logger.warning("rewrite gagal (%s), pakai query asli", e)
        new_q = state["question"]
    return {
        "search_query": new_q,
        "rewrites": state.get("rewrites", 0) + 1,
        "route": "rewrite_rag",
    }


def generate_node(state: AgentState) -> dict:
    from src.rag.chain import PROMPT, format_docs

    docs = state.get("docs", [])
    t0 = time.perf_counter()
    answer = (PROMPT | _llm() | StrOutputParser()).invoke(
        {"context": format_docs(docs), "question": state["question"]}
    )
    timings = dict(state.get("timings") or {})
    timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ]
    route = state.get("route", "rag")
    if route == "rag":
        route = "rewrite_rag" if state.get("rewrites", 0) else "hybrid_rerank_rag"
    return {"answer": answer, "sources": sources, "timings": timings, "route": route}


def no_answer_node(state: AgentState) -> dict:
    return {
        "answer": (
            "Saya tidak menemukan jawaban di dokumen yang tersedia. "
            "Coba tanyakan dengan kata kunci lain (misal nama perusahaan, "
            "nomor invoice, atau judul dokumen)."
        ),
        "sources": [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
            for d in state.get("docs", [])[:2]
        ],
        "route": "no_answer",
    }


def _route_after_router(state: AgentState) -> str:
    return "direct" if state.get("route") == "direct" else "retrieve"


def _route_after_grade(state: AgentState) -> str:
    if _grade(state["question"], state.get("docs", [])):
        return "generate"
    if state.get("rewrites", 0) < settings.AGENT_MAX_REWRITE:
        return "rewrite"
    return "no_answer"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("no_answer", no_answer_node)
    g.set_entry_point("router")
    g.add_conditional_edges(
        "router", _route_after_router, {"direct": "direct_answer", "retrieve": "retrieve"}
    )
    g.add_edge("direct_answer", END)
    # grade sebagai conditional edge langsung (tanpa node, hemat 1 hop)
    g.add_conditional_edges(
        "retrieve",
        _route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("no_answer", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def agent_answer(question: str) -> dict:
    """Jalankan agent. Return shape sama seperti chain.query()."""
    result = get_graph().invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "timings": result.get("timings", {}),
        "route": result.get("route", "rag"),
    }
