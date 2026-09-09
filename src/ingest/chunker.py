from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, metadata: dict) -> list[dict]:
    splitter = get_splitter()
    chunks = splitter.split_text(text)
    return [{"page_content": c, "metadata": metadata} for c in chunks if c.strip()]
