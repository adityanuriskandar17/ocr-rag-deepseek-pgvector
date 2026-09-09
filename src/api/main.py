from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import shutil
from src.ingest.pipeline import ingest_and_store
from src.rag.chain import query

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
    n = ingest_and_store(tmp)
    return {"chunks": n, "file": file.filename}


@app.post("/query")
def ask(req: AskReq):
    return query(req.question)
