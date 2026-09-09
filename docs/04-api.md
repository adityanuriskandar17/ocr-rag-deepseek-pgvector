# API & UI

Base URL lokal: `http://localhost:8000` (lihat `src/api/main.py`).

## GET /health
```bash
curl http://localhost:8000/health
# {"ok": true}
```

## POST /ingest — upload scan/foto
```bash
curl -F "file=@scan.pdf" http://localhost:8000/ingest
# {"chunks": 42, "file": "scan.pdf"}
```
- Menerima `pdf/jpg/jpeg/png` (validasi di UI Streamlit; API menerima apa saja lalu diproses pipeline).
- Mengembalikan jumlah chunk yang masuk pgvector.
- Error 500 + `password authentication failed` → DB salah (lihat `docs/05-setup-operasi.md`).

## POST /query — tanya dokumen
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Berapa total pada invoice?"}'
```
Respons:
```json
{
  "answer": "... [invoice.pdf hal. 2] ...",
  "sources": [{"source": "invoice.pdf", "page": 2}]
}
```

## UI Streamlit (`ui_app.py`)
- Jalankan: `streamlit run ui_app.py` → biasanya `http://localhost:8501`.
- Ada 2 aksi: **Ingest** (upload) dan **Tanya** (input teks) — memanggil API di atas via `requests`.
- `API = "http://localhost:8000"` di baris atas file — ubah kalau API di port lain.
