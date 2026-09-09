# Evaluasi & Roadmap

## Evaluasi sekarang (minimal)
- `eval/eval_rag.py`: menjalankan daftar pertanyaan uji, mengecek jawaban mengandung sitasi (`hal.`), mencetak PASS/FAIL.
- Cara pakai (setelah ingest contoh dokumen):
```bash
source .venv/bin/activate
python eval/eval_rag.py
```

## Yang layak ditambah untuk portofolio
1. **Retrieval hit-rate.** Simpan 10-20 pasang pertanyaan → halaman jawaban yang benar, ukur berapa yang retriever-nya kena top-5.
2. **Faithfulness.** Minta LLM kedua menilai apakah setiap klaim jawaban didukung konteks (skala 0-1).
3. **Ablasi chunking.** Bandingkan `CHUNK_SIZE` 400 vs 800 vs 1200 terhadap hit-rate; jadikan grafik di README.
4. **Hybrid search.** Tambah BM25 (misal `rank_bm25` di Postgres full-text) + reciprocal rank fusion dengan skor vektor — bagus untuk nomor invoice/nama exact.
5. **Reranker.** `cross-encoder/ms-marco-MiniLM` untuk menyaring top-20 → top-5.
6. **Agentic (LangGraph).** Router (butuh retrieval atau jawab langsung?) → retriever → verifier sitasi.
7. **Observability.** Log latency per tahap (OCR ms, embed ms, retrieve ms, LLM ms) + `ocr_conf` rata-rata.

## Definisi selesai (DoD) tiap fitur
- Ada test/eval yang bisa diulang (`python eval/...`).
- Ada contoh input/output di `examples/`.
- Ada 1 paragraf di docs menjelaskan tradeoff yang diambil.
