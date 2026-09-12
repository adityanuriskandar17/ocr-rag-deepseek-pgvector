# Teknik RAG yang dipakai

## 1. Chunking — RecursiveCharacterTextSplitter
- Lokasi: `src/ingest/chunker.py`
- Config: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120` (lihat `.env`)
- Separator berurutan: `\n\n → \n → ". " → " " → ""`
- Kenapa: 800 char ≈ 1-2 paragraf dokumen Indonesia — cukup konteks tanpa melebihi window retriever. Overlap 120 menjaga kalimat terpotong di batas chunk.
- Metadata per chunk: `{source, page, ocr_conf}` — dipakai untuk sitasi dan filtering.

## 2. Embeddings lokal multilingual
- Lokasi: `src/rag/embed_store.py::get_embeddings`
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensi, ~471 MB safetensors)
- Jalan di CPU (`model_kwargs={"device":"cpu"}`), normalisasi L2 (`normalize_embeddings=True`) agar cosine-similarity stabil.
- Kenapa lokal, bukan API: DeepSeek tidak menyediakan embedding API. Selain hemat, ini juga bahan cerita portofolio (hybrid: retrieval lokal + generasi API).
- Cache: `~/.cache/huggingface` — download sekali, ingest berikutnya offline.

## 3. Vector store — pgvector
- Lokasi: `src/rag/embed_store.py::get_vectorstore`
- Koleksi: `PG_COLLECTION=porto_docs`, kolom JSONB untuk metadata.
- Extension `vector 0.8.6` wajib aktif (`CREATE EXTENSION IF NOT EXISTS vector`).
- Koneksi: `DATABASE_URL` di `.env` → container `portoai-pgvector` di port **5433** (bukan 5432 yang dipakai Postgres bawaan OS).

## 4. Retrieval — hybrid vektor + FTS + RRF (v2)
- Lokasi: `src/rag/hybrid.py::hybrid_search`
- `vector_search(k=20)` via `get_vectorstore().similarity_search` (semantik: "total belanja berapa?").
- `keyword_search(k=20)` via SQL `ts_rank(to_tsvector('simple', document), plainto_tsquery('simple', q))` + fallback `ILIKE` untuk kode aneh. Config `simple` dipilih karena tidak stemming (aman untuk nomor/kode) dan lowercase otomatis — ini yang menghapus hack multi-query lowercase di v1.
- `rrf_fusion` (`1/(60+rank)`) menggabung keduanya tanpa tuning bobot → `top_k=RERANK_CANDIDATES=20`.
- Index: `scripts/add_fts_index.sql` (GIN `tsvector('simple', document)` + trigram `document`). Idempoten, aman dijalankan ulang.
- Dedup pakai **full content** sebagai key — `content[:120]` tabrakan karena chunk overlap 120 (bug yang sempat ketemu saat implementasi).

## 5. Rerank — cross-encoder multilingual (v2)
- Lokasi: `src/rag/rerank.py::rerank` — kandidat 20 → 5.
- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, termasuk Indonesia), CPU ±100-150ms/doc. Hasil eksperimen: varian English-only `ms-marco-MiniLM` menggeser chunk bahasa Indonesia yang benar, jadi diganti multilingual.
- Batasan jujur: query komposisional lintas halaman (chunk hal. 2 tidak menyebut nama subjek) tetap sulit untuk scorer apapun — diatasi di layer agent (rewrite + grade), bukan dengan model rerank lebih besar.

## 6. Generation — agent grounded + sitasi (v2)
- LLM: `ChatOpenAI(model=deepseek-v4-flash, base_url=https://api.deepseek.com/v1, temperature=0)` — lihat `src/rag/chain.py::get_llm`.
- System prompt: jawab **hanya** dari konteks, bahasa mengikuti pertanyaan, kalau tidak ada → katakan tidak tahu, wajib sitasi `[source hal. X]`.
- Graf LangGraph (`src/rag/graph.py`, `langgraph==0.2.28`): `router → retrieve → grade → generate/rewrite/no_answer`.
  - `router`: HANYA sapaan/basa-basi/tanya kemampuan aplikasi → jawab langsung tanpa retrieval (hemat cost); SEMUA yang lain → RAG (kalau ragu, RAG). Pengetahuan umum/tutorial/kode yang tidak ada di dokumen berakhir di `no_answer`, bukan dijawab dari memori LLM.
  - `grade`: LLM menilai konteks cukup/tidak **sebelum** generate. Tidak cukup + rewrite tersisa → `rewrite` tulis ulang query jadi keyword lalu retrieve ulang (max `AGENT_MAX_REWRITE=1`). Tetap tidak cukup → pesan jujur "tidak tahu" + max 2 sumber.
  - `query()` di `chain.py` memanggil `agent_answer()` dengan fallback hybrid langsung kalau graph error.
- `query()` mengembalikan `{"answer", "sources": [{source, page}], "timings": {retrieve_ms, rerank_ms, llm_ms}, "route": "direct|hybrid_rerank_rag|rewrite_rag|no_answer"}` agar UI/API bisa menampilkan bukti + latensi.

## 7. Yang belum (roadmap lanjutan)
- Evaluasi faithfulness + hit-rate otomatis (`eval/eval_rag.py` saat ini masih cek sitasi).
- Ablasi chunking 400 vs 800 vs 1200 dengan grafik di README.
- Enrichment chunk (prefix judul dokumen) untuk query komposisional lintas halaman.
- Observability penuh (Langfuse/LangSmith) + `examples/` input/output.
