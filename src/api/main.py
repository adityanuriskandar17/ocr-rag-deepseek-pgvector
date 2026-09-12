from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import logging
import shutil
from src.ingest.pipeline import ingest_and_store
from src.rag.chain import query

logger = logging.getLogger(__name__)

app = FastAPI(title="Porto-AI OCR+RAG")


class AskReq(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = ingest_and_store(tmp)
    except Exception as e:
        logger.exception("ingest gagal: %s", file.filename)
        return JSONResponse(status_code=500, content={"error": f"Ingest gagal: {e}"})
    return {"file": file.filename, **result}


@app.post("/query")
def ask(req: AskReq):
    try:
        return query(req.question)
    except Exception as e:
        # Selalu balas JSON agar UI tidak crash JSONDecodeError.
        # Penyebab umum: DEEPSEEK_API_KEY salah/habis (401) atau DB mati.
        logger.exception("query gagal")
        return JSONResponse(status_code=500, content={"error": f"Query gagal: {e}"})
