# Porto-AI — OCR + RAG (CPU-friendly) | AI Engineer Portfolio

Arsitektur yang nunjukin skill AI Engineer end-to-end, **tanpa butuh GPU**:

```
PDF scan / foto kertas
  -> PyMuPDF 300 DPI -> RapidOCR (ONNX CPU) -> teks + bbox/conf
  -> Recursive chunk (800/120)
  -> HF embeddings lokal CPU (multilingual ID+EN)
  -> pgvector (Postgres)
  -> Retriever top-5 -> DeepSeek (OpenAI-compatible via LangChain) -> jawaban + sitasi
```

Kenapa ini bagus buat portofolio:
- **Ingestion nyata:** bukan PDF teks, tapi scan/foto -> OCR. Recruiter suka.
- **RAG beneran:** chunking, embeddings, vector DB, prompt grounding, sitasi `[source hal. X]`.
- **Stack hireable:** LangChain LCEL, FastAPI, pgvector, Docker, eval.
- **Murah:** OCR + embedding full CPU lokal, LLM cuma pakai DeepSeek API yang kamu punya.

## Quickstart

```bash
cp .env.example .env  # isi DEEPSEEK_API_KEY + DEEPSEEK_MODEL
docker compose up -d db
pip install -r requirements.txt

# API
uvicorn src.api.main:app --reload --port 8000
# UI (terminal baru)
streamlit run ui_app.py
```

Ingest via UI atau curl:
```bash
curl -F "file=@scan.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"Berapa total pada invoice?"}'
```

## Config penting (.env)
- `DEEPSEEK_MODEL=deepseek-v4-flash` — terverifikasi via `/v1/models` (alternatif: `deepseek-v4-pro`, `deepseek-v4-flash-vision-exp`).
- `EMBEDDING_MODEL` — lokal CPU, tidak pakai API DeepSeek (DeepSeek tidak ada embedding API).
- `DATABASE_URL` — `...@localhost:5433/...` (container pgvector; 5432 dipakai Postgres bawaan OS).
- `PG_COLLECTION`, `TOP_K`, `CHUNK_SIZE` — tunables buat bahan cerita interview.

## Dokumentasi
- `docs/01-arsitektur.md` — diagram + alur data + keputusan desain
- `docs/02-teknik-rag.md` — chunking, embeddings, pgvector, retrieval, grounding/sitasi
- `docs/03-ocr-pipeline.md` — kenapa RapidOCR CPU, bukan VLM GPU
- `docs/04-api.md` — referensi endpoint + contoh curl
- `docs/05-setup-operasi.md` — instalasi + troubleshooting (port, auth, cache model)
- `docs/06-evaluasi-roadmap.md` — eval sekarang + next (reranker, hybrid, LangGraph)

## Yang bisa kamu pamerkan di CV / interview
1. Tradeoff OCR: kenapa RapidOCR CPU bukan Unlimited-OCR GPU 3B.
2. Chunking + eval: coba 400 vs 800, ukur hit-rate.
3. Grounding: prompt anti-halusinasi + sitasi wajib.
4. Next: reranker, hybrid BM25+vector, LangGraph agent, observability.

## Struktur
- `src/ingest/` : pdf_loader, ocr, chunker, pipeline
- `src/rag/` : embed_store (pgvector), chain (DeepSeek)
- `src/api/main.py` : FastAPI
- `ui_app.py` : Streamlit demo
- `eval/eval_rag.py` : eval minimal
