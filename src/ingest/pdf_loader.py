import tempfile
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[str]:
    """Render PDF scan -> list PNG paths. CPU only."""
    doc = fitz.open(pdf_path)
    tmpdir = tempfile.mkdtemp(prefix="porto_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out_paths = []
    for i, page in enumerate(doc):
        out = f"{tmpdir}/page_{i+1:04d}.png"
        page.get_pixmap(matrix=mat).save(out)
        out_paths.append(out)
    doc.close()
    return out_paths


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"
