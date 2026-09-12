# Porto-AI — OCR + Agentic RAG (CPU-friendly) | AI Engineer Portfolio

Arsitektur yang nunjukin skill AI Engineer end-to-end, **tanpa butuh GPU**:

```
PDF scan / foto kertas
  -> PyMuPDF: teks digital langsung, atau render 300 DPI -> RapidOCR (ONNX CPU)
  -> teks + bbox/conf
  -> Recursive chunk (800/120)
  -> HF embeddings lokal CPU (multilingual ID+EN)
  -> pgvector (Postgres) + index FTS
  -> Agent (LangGraph): router -> hybrid retrieve top-20 -> rerank top-5
     -> grade -> DeepSeek (OpenAI-compatible) -> jawaban + sitasi
```

Kenapa ini bagus buat portofolio:
- **Ingestion nyata:** bukan PDF teks, tapi scan/foto -> OCR. Recruiter suka.
- **RAG beneran:** chunking, embeddings, vector DB, prompt grounding, sitasi `[source hal. X]`.
- **Agentic:** router (direct vs RAG), hybrid vektor+FTS dengan RRF, reranker multilingual CPU, grade sebelum generate, rewrite 1x, jujur "tidak tahu" kalau konteks tidak mendukung.
- **Stack hireable:** LangChain LCEL, LangGraph, FastAPI, pgvector, Docker, eval.
- **Murah:** OCR + embedding + rerank full CPU lokal, LLM cuma pakai DeepSeek API yang kamu punya.

## Quickstart

```bash
cp .env.example .env  # isi DEEPSEEK_API_KEY + DEEPSEEK_MODEL
docker compose up -d db
docker exec -i portoai-pgvector psql -U portoai -d portoai < scripts/add_fts_index.sql  # index FTS v2
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
- v2: `HYBRID_VECTOR_K/HYBRID_KEYWORD_K`, `RERANK_ENABLED/MODEL/CANDIDATES`, `AGENT_ENABLED/MAX_REWRITE` (lihat `docs/05-setup-operasi.md`).

## Dokumentasi
- `docs/01-arsitektur.md` — diagram + alur data + keputusan desain
- `docs/02-teknik-rag.md` — chunking, embeddings, pgvector, retrieval, grounding/sitasi
- `docs/03-ocr-pipeline.md` — kenapa RapidOCR CPU, bukan VLM GPU
- `docs/04-api.md` — referensi endpoint + contoh curl
- `docs/05-setup-operasi.md` — instalasi + troubleshooting (port, auth, cache model)
- `docs/06-evaluasi-roadmap.md` — eval sekarang + hasil eksperimen v2 + sisa roadmap

## Yang bisa kamu pamerkan di CV / interview
1. Tradeoff OCR: kenapa RapidOCR CPU bukan Unlimited-OCR GPU 3B.
2. Hybrid retrieval: FTS `simple` + vektor + RRF — exact-match tanpa hack lowercase.
3. Reranker: English-only gagal di dokumen Indonesia → ganti multilingual (eksperimen nyata).
4. Agent anti-halusinasi: grade sebelum generate, query di luar dokumen dijawab jujur (`route=no_answer`).
5. Next: hit-rate/faithfulness eval, ablasi chunking 400/800/1200, enrichment chunk.

## Struktur
- `src/ingest/` : pdf_loader, ocr, chunker, pipeline
- `src/rag/` : embed_store (pgvector), hybrid (FTS+RRF), rerank (cross-encoder), graph (LangGraph agent), chain (DeepSeek)
- `scripts/add_fts_index.sql` : migrasi index full-text
- `src/api/main.py` : FastAPI
- `ui_app.py` : Streamlit demo
- `eval/eval_rag.py` : eval sitasi + route/timings
