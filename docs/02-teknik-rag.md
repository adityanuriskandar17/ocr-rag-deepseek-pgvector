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

## 4. Retrieval — top-K similarity
- Lokasi: `src/rag/chain.py::get_chain`
- `as_retriever(search_kwargs={"k": TOP_K=5})` — cosine distance bawaan pgvector.
- `format_docs()` menyusun konteks sebagai `[source hal. X]\n<isi>` dipisah `---` agar LLM bisa menyitir.

## 5. Generation — grounded + sitasi
- LLM: `ChatOpenAI(model=deepseek-v4-flash, base_url=https://api.deepseek.com/v1, temperature=0)` — lihat `src/rag/chain.py::get_llm`.
- System prompt: jawab **hanya** dari konteks, bahasa mengikuti pertanyaan, kalau tidak ada → katakan tidak tahu, wajib sitasi `[source hal. X]`.
- Pattern LangChain LCEL: `{"context": retriever|format_docs, "question": Passthrough} | Prompt | LLM | StrOutputParser`.
- `query()` mengembalikan `{"answer", "sources": [{source, page}]}` agar UI/API bisa menampilkan bukti.

## 6. Yang belum (roadmap portofolio)
- Reranker (cross-encoder) setelah top-20 → top-5.
- Hybrid BM25 + vektor untuk istilah exact (nomor invoice, nama).
- Evaluasi faithfulness + hit-rate otomatis (`eval/eval_rag.py` saat ini masih minimal).
- LangGraph agent: router → retriever → verifier.
