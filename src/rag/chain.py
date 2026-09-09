"""RAG chain: pgvector retriever + DeepSeek (OpenAI-compatible) via LangChain."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings
from src.rag.embed_store import get_vectorstore

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
    chain, retriever = get_chain()
    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ]
    return {"answer": answer, "sources": sources}
