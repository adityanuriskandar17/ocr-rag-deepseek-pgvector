# Evaluasi & Roadmap

## Evaluasi sekarang (minimal)
- `eval/eval_rag.py`: menjalankan daftar pertanyaan uji, mengecek jawaban mengandung sitasi (`hal.`), mencetak PASS/FAIL + `route`/`timings`.
- Cara pakai (setelah ingest contoh dokumen):
```bash
source .venv/bin/activate
python eval/eval_rag.py
```

## Selesai di v2 (dengan hasil eksperimen)
1. **Hybrid search.** `src/rag/hybrid.py`: BM25-like Postgres FTS (`ts_rank`, config `simple`) + vektor + RRF. Menyelesaikan exact-match nomor invoice/nama dan masalah kapitalisasi (`Alodokter`) tanpa hack lowercase.
2. **Reranker.** `src/rag/rerank.py`: `mmarco-mMiniLMv2-L12-H384-v1` multilingual, 20 → 5. Eksperimen: varian English-only menggeser chunk Indonesia yang benar → diganti multilingual.
3. **Agentic (LangGraph).** `src/rag/graph.py`: router (direct vs RAG) → retrieve → grade → generate/rewrite 1x/no_answer. Query di luar dokumen dijawab jujur "tidak tahu" (`route=no_answer`), bukan dihalusinasi.
4. **Observability awal.** `timings {retrieve_ms, rerank_ms, llm_ms}` + `route` di setiap respons `/query`. Ditemukan: retrieve dingin ~6.4s (load embedding) vs hangat ~75ms.

## Sisa untuk portofolio
1. **Retrieval hit-rate.** Simpan 10-20 pasang pertanyaan → halaman jawaban yang benar, ukur berapa yang retriever-nya kena top-5 (sebelum vs sesudah rerank).
2. **Faithfulness.** Minta LLM kedua menilai apakah setiap klaim jawaban didukung konteks (skala 0-1).
3. **Ablasi chunking.** Bandingkan `CHUNK_SIZE` 400 vs 800 vs 1200 terhadap hit-rate; jadikan grafik di README.
4. **Enrichment chunk.** Prefix judul dokumen per chunk untuk query komposisional lintas halaman.
5. **Contoh & observability penuh.** Folder `examples/` + Langfuse/LangSmith.

## Definisi selesai (DoD) tiap fitur
- Ada test/eval yang bisa diulang (`python eval/...`).
- Ada contoh input/output di `examples/`.
- Ada 1 paragraf di docs menjelaskan tradeoff yang diambil.
