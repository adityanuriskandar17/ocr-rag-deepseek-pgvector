"""Pipeline: file scan/foto/PDF -> OCR -> chunk. Dipakai API + CLI."""

from pathlib import Path
from src.ingest.pdf_loader import pdf_to_images, is_pdf
from src.ingest.ocr import ocr_image
from src.ingest.chunker import chunk_text
from src.config import settings


def ingest_path(path: str | Path) -> list[dict]:
    path = Path(path)
    if is_pdf(path):
        images = pdf_to_images(path, dpi=settings.OCR_DPI)
    else:
        images = [str(path)]

    all_chunks = []
    for idx, img in enumerate(images):
        r = ocr_image(img)
        if not r["text"].strip():
            continue
        meta = {"source": path.name, "page": idx + 1, "ocr_conf": round(r["conf"], 3)}
        all_chunks.extend(chunk_text(r["text"], meta))
    return all_chunks


def ingest_and_store(path: str | Path) -> int:
    from src.rag.embed_store import get_vectorstore

    chunks = ingest_path(path)
    if not chunks:
        return 0
    vs = get_vectorstore()
    vs.add_documents(
        [
            __import__("langchain_core.documents", fromlist=["Document"]).Document(
                page_content=c["page_content"], metadata=c["metadata"]
            )
            for c in chunks
        ]
    )
    return len(chunks)
