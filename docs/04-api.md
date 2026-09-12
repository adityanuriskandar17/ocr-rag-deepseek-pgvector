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
# {"file": "scan.pdf", "chunks": 42, "pages_text": 10, "pages_ocr": 0}
```
- Menerima `pdf/jpg/jpeg/png` (validasi di UI Streamlit; API menerima apa saja lalu diproses pipeline).
- PDF digital diambil teksnya langsung (cepat); hanya halaman scan yang di-OCR — lihat `docs/03-ocr-pipeline.md`.
- Mengembalikan jumlah chunk + perincian halaman teks vs OCR.
- Error 500 + `password authentication failed` → DB salah (lihat `docs/05-setup-operasi.md`).

## POST /query — tanya dokumen (agent v2)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Berapa total pada invoice?"}'
```
Respons:
```json
{
  "answer": "... [invoice.pdf hal. 2] ...",
  "sources": [{"source": "invoice.pdf", "page": 2}],
  "timings": {"retrieve_ms": 75.1, "rerank_ms": 3100.5, "llm_ms": 2731.0},
  "route": "hybrid_rerank_rag"
}
```
- `route` — jalur agent yang dipakai:
  - `direct`: HANYA sapaan/basa-basi/tanya kemampuan aplikasi, tanpa retrieval (cepat, `sources: []`). Persona direct dibatasi: tidak memberi tutorial/kode/pengetahuan umum.
  - `hybrid_rerank_rag`: retrieval normal (hybrid + rerank + generate).
  - `rewrite_rag`: query ditulis ulang 1x sebelum retrieve ulang.
  - `no_answer`: konteks tidak mendukung — jawaban jujur "tidak tahu". Pertanyaan umum (misal "buatkan kode ocr") masuk sini, bukan dijawab dari memori LLM.
- `timings` — latensi per tahap (ms). `retrieve_ms` besar di query pertama dingin (load model embedding), hangat berikutnya ±75ms.

## Error (selalu JSON)
`/ingest` dan `/query` tidak pernah balas badan kosong — gagal pun tetap JSON status 500:
```json
{"error": "Query gagal: Error code: 401 - ..."}
```
Penyebab umum: `DEEPSEEK_API_KEY` salah/habis (401) atau DB mati. UI menampilkan `error` sebagai pesan merah, bukan traceback.

## UI Streamlit (`ui_app.py`)
- Jalankan: `streamlit run ui_app.py` → biasanya `http://localhost:8501`.
- Ada 2 aksi: **Ingest** (upload) dan **Tanya** (input teks) — memanggil API di atas via `requests`.
- `API = "http://localhost:8000"` di baris atas file — ubah kalau API di port lain.
