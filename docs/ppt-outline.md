# Outline PPT — OCR + RAG tanpa GPU (10 slide)

> Aturan: 1 slide 1 pesan, max 5 baris, visual > teks. Angka di slide 7 isi dari log timing kamu.

## 1. Cover
- Judul: OCR + RAG tanpa GPU
- Sub: Scan kertas → tanya jawab berdasar dokumen (dengan sitasi)
- Nama + link GitHub: github.com/adityanuriskandar17/ocr-rag-deepseek-pgvector

## 2. Masalah
- Dokumen masih scan/foto → teks tidak searchable
- LLM saja: halusinasi + tidak tahu dokumen privat + knowledge cutoff
- GPU mahal → butuh solusi CPU-friendly

## 3. Solusi (diagram)
- Scan PDF/foto → PyMuPDF 300 DPI → RapidOCR ONNX CPU
- → Chunk 800/120 → embedding MiniLM lokal → pgvector + index FTS
- → Agent: router → hybrid retrieve top-20 → rerank top-5 → grade → DeepSeek v4-flash → jawaban + sitasi [source hal. X]

## 4. Arsitektur & Stack
- RapidOCR (ONNX CPU): deteksi + recognition, tanpa GPU
- MiniLM multilingual lokal: embedding ID/EN, hemat API
- pgvector (Docker :5433) + FTS GIN: hybrid vektor + keyword, RRF fusion
- Reranker mmarco multilingual CPU: 20 kandidat → 5 terbaik
- LangGraph agent: router → retrieve → grade → generate/rewrite/jujur-tidak-tahu
- DeepSeek v4-flash (OpenAI-compatible): generasi grounded, temp 0
- FastAPI + Streamlit: API /ingest /query + demo UI

## 5. Alur Agent
- Ingest (sekali): OCR → chunk → embed → simpan (+ index FTS)
- Router (tiap tanya): sapaan/umum → jawab langsung; sisanya → RAG
- Retrieve: hybrid vektor+FTS top-20 → rerank → top-5
- Grade: konteks cukup → generate + sitasi; tidak → rewrite 1x atau "tidak tahu"

## 6. Demo
- Screenshot 1: upload scan → "42 chunks"
- Screenshot 2: tanya "Berapa total invoice?" → jawaban + sources
- (Ganti placeholder ini dengan screenshot/GIF asli 10 detik)

## 7. Hasil / Angka (isi dari log kamu)
- Ingest 10 hal: __ mnt → __ mnt (paralel 4 workers + batch 32)
- Rata-rata OCR: __ dtk/hal, conf __%
- Retrieve: dingin ~6.4 dtk (load embedding) vs hangat ~75ms; rerank CPU ~3 dtk/20 docs; LLM ~3 dtk
- Q&A: selalu ada sitasi (eval PASS); query di luar dokumen → jujur "tidak tahu" (route=no_answer)
- Biaya: OCR+embed+rerank Rp0 (CPU lokal), LLM hanya per query DeepSeek; sapaan tidak kena retrieval (route=direct)

## 8. Tradeoff jujur
- Bukan VLM GPU 3B (Unlimited-OCR): overkill & mahal untuk teks polos
- Reranker English-only gagal di dokumen Indonesia → ganti multilingual (eksperimen nyata)
- Lemah: tulisan tangan jelek, stempel tindih, tabel kompleks, query komposisional lintas halaman
- Mitigasi: simpan ocr_conf + bbox + sitasi halaman → bisa diaudit; grade sebelum generate → tidak mengarang

## 9. Roadmap
- Eval hit-rate + faithfulness otomatis, ablasi chunking 400/800/1200
- Enrichment chunk (prefix judul dokumen)
- Folder examples + observability penuh (Langfuse/LangSmith)

## 10. Penutup
- Link: github.com/adityanuriskandar17/ocr-rag-deepseek-pgvector
- Yang dipelajari: ingestion nyata > demo mainan; grounding > prompt panjang
- Terima kasih — Q&A
