"""Pipeline: PDF digital (teks langsung) / scan/foto (OCR) -> chunk. Dipakai API + CLI."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import settings
from src.ingest.chunker import chunk_text
from src.ingest.ocr import ocr_image_threadsafe
from src.ingest.pdf_loader import is_pdf, pdf_page_texts, pdf_pages_to_images

logger = logging.getLogger(__name__)


def _ocr_one(args: tuple[int, str]) -> tuple[int, dict, float]:
    idx, img = args
    t0 = time.perf_counter()
    r = ocr_image_threadsafe(img)
    dt = time.perf_counter() - t0
    return idx, r, dt


def ingest_path(path: str | Path) -> tuple[list[dict], dict]:
    """Ingest file -> (chunks, stats). stats = {pages_text, pages_ocr}."""
    total_t0 = time.perf_counter()
    path = Path(path)
    stats = {"pages_text": 0, "pages_ocr": 0}

    if is_pdf(path):
        # Jalur cepat: ambil teks embedded dulu (milis/detik, tanpa render).
        # Hanya halaman tanpa teks (= hasil scan/foto) yang di-render + OCR.
        embedded = pdf_page_texts(path)
        results: list[tuple[int, dict]] = []
        ocr_pages = []
        for idx, text in enumerate(embedded):
            if len(text.strip()) >= settings.PDF_TEXT_MIN_CHARS:
                results.append(
                    (idx, {"text": text, "conf": 1.0, "extract": "text"})
                )
                stats["pages_text"] += 1
            else:
                ocr_pages.append(idx)
        logger.info(
            "pdf %s: %d hal teks langsung, %d hal perlu OCR",
            path.name, stats["pages_text"], len(ocr_pages),
        )
        if ocr_pages:
            t0 = time.perf_counter()
            images = pdf_pages_to_images(path, pages=ocr_pages, dpi=settings.OCR_DPI)
            logger.info(
                "render %s: %d hal dalam %.1fs",
                path.name, len(ocr_pages), time.perf_counter() - t0,
            )
            # OCR paralel per halaman (ONNX me-release GIL saat inferensi,
            # tiap thread pakai engine sendiri -> aman).
            workers = max(1, min(settings.OCR_WORKERS, len(ocr_pages)))
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as ex:
                ocr_out = list(ex.map(_ocr_one, [(i, images[i]) for i in ocr_pages]))
            ocr_time = time.perf_counter() - t0
            for idx, r, dt in ocr_out:
                r["extract"] = "ocr"
                results.append((idx, r))
                logger.info("ocr hal %d: %.1fs conf=%.2f", idx + 1, dt, r["conf"])
                stats["pages_ocr"] += 1
            logger.info(
                "ocr total: %d hal dalam %.1fs (%d workers)",
                len(ocr_pages), ocr_time, workers,
            )
        # kembalikan urutan halaman
        results.sort(key=lambda x: x[0])
    else:
        # Foto mentah tidak punya teks embedded -> selalu OCR.
        t0 = time.perf_counter()
        idx, r, dt = _ocr_one((0, str(path)))
        r["extract"] = "ocr"
        results = [(idx, r)]
        stats["pages_ocr"] += 1
        logger.info("ocr %s: %.1fs conf=%.2f", path.name, dt, r["conf"])

    all_chunks = []
    for idx, r in results:
        if not r["text"].strip():
            continue
        meta = {
            "source": path.name,
            "page": idx + 1,
            "ocr_conf": round(r["conf"], 3),
            "extract": r.get("extract", "ocr"),
        }
        all_chunks.extend(chunk_text(r["text"], meta))
    logger.info(
        "ingest %s selesai: %d chunk dalam %.1fs (teks=%d hal, ocr=%d hal)",
        path.name, len(all_chunks), time.perf_counter() - total_t0,
        stats["pages_text"], stats["pages_ocr"],
    )
    return all_chunks, stats


def ingest_and_store(path: str | Path) -> dict:
    from src.rag.embed_store import get_vectorstore

    chunks, stats = ingest_path(path)
    if not chunks:
        return {"chunks": 0, **stats}
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
    return {"chunks": len(chunks), **stats}
