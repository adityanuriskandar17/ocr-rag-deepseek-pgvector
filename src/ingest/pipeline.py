"""Pipeline: file scan/foto/PDF -> OCR (paralel) -> chunk. Dipakai API + CLI."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import settings
from src.ingest.chunker import chunk_text
from src.ingest.ocr import ocr_image_threadsafe
from src.ingest.pdf_loader import is_pdf, pdf_to_images

logger = logging.getLogger(__name__)


def _ocr_one(args: tuple[int, str]) -> tuple[int, dict, float]:
    idx, img = args
    t0 = time.perf_counter()
    r = ocr_image_threadsafe(img)
    dt = time.perf_counter() - t0
    return idx, r, dt


def ingest_path(path: str | Path) -> list[dict]:
    total_t0 = time.perf_counter()
    path = Path(path)

    t0 = time.perf_counter()
    if is_pdf(path):
        images = pdf_to_images(path, dpi=settings.OCR_DPI)
    else:
        images = [str(path)]
    logger.info("render %s: %d hal dalam %.1fs", path.name, len(images), time.perf_counter() - t0)

    # OCR paralel per halaman (ONNX me-release GIL saat inferensi,
    # tiap thread pakai engine sendiri -> aman).
    workers = max(1, min(settings.OCR_WORKERS, len(images)))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_ocr_one, enumerate(images)))
    ocr_time = time.perf_counter() - t0
    # kembalikan urutan halaman
    results.sort(key=lambda x: x[0])
    for idx, r, dt in results:
        logger.info("ocr hal %d/%d: %.1fs conf=%.2f", idx + 1, len(images), dt, r["conf"])
    logger.info("ocr total: %d hal dalam %.1fs (%d workers)", len(images), ocr_time, workers)

    t0 = time.perf_counter()
    all_chunks = []
    for idx, r, _ in results:
        if not r["text"].strip():
            continue
        meta = {"source": path.name, "page": idx + 1, "ocr_conf": round(r["conf"], 3)}
        all_chunks.extend(chunk_text(r["text"], meta))
    logger.info(
        "ingest %s selesai: %d chunk dalam %.1fs",
        path.name, len(all_chunks), time.perf_counter() - total_t0,
    )
    return all_chunks


def ingest_and_store(path: str | Path) -> int:
    from src.rag.embed_store import get_vectorstore

    chunks = ingest_path(path)
    if not chunks:
        return 0
    t0 = time.perf_counter()
    vs = get_vectorstore()
    vs.add_documents(
        [
            __import__("langchain_core.documents", fromlist=["Document"]).Document(
                page_content=c["page_content"], metadata=c["metadata"]
            )
            for c in chunks
        ]
    )
    logger.info("embed+store: %d chunk dalam %.1fs", len(chunks), time.perf_counter() - t0)
    return len(chunks)
