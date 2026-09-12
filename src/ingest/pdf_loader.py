import tempfile
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[str]:
    """Render PDF scan -> list PNG paths. CPU only."""
    return pdf_pages_to_images(pdf_path, pages=None, dpi=dpi)


def pdf_pages_to_images(
    pdf_path: str | Path, pages: list[int] | None, dpi: int = 300
) -> dict[int, str] | list[str]:
    """Render halaman tertentu (index 0-based) -> {idx: png_path}.

    pages=None berarti semua halaman (kompatibel pdf_to_images lama).
    Dipakai pipeline agar hanya halaman scan yang di-render/OCR.
    """
    doc = fitz.open(pdf_path)
    tmpdir = tempfile.mkdtemp(prefix="porto_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out_paths = {}
    targets = range(len(doc)) if pages is None else pages
    for i in targets:
        out = f"{tmpdir}/page_{i+1:04d}.png"
        doc[i].get_pixmap(matrix=mat).save(out)
        out_paths[i] = out
    doc.close()
    if pages is None:
        return [out_paths[i] for i in sorted(out_paths)]
    return out_paths


def pdf_page_texts(pdf_path: str | Path) -> list[str]:
    """Ambil teks embedded per halaman (gratis, tanpa render/OCR).

    Halaman scan/foto mengembalikan string kosong / sangat pendek —
    itu sinyal untuk fallback OCR di pipeline.
    """
    doc = fitz.open(pdf_path)
    texts = [page.get_text() or "" for page in doc]
    doc.close()
    return texts


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"
