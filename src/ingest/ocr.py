"""OCR CPU dengan RapidOCR (ONNX). Tanpa GPU, cocok untuk scan/foto kertas."""

from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def ocr_image(image_path: str | Path) -> dict:
    """
    Returns: {text: str, boxes: [...], conf: float}
    """
    engine = get_engine()
    result, _ = engine(str(image_path))
    # result: list [box, txt, conf]
    if not result:
        return {"text": "", "boxes": [], "conf": 0.0}

    lines, boxes, confs = [], [], []
    for box, txt, conf in result:
        lines.append(txt)
        boxes.append(box)
        confs.append(float(conf))

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return {"text": "\n".join(lines), "boxes": boxes, "conf": avg_conf}
