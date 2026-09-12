# Arsitektur Porto-AI — OCR + Agentic RAG (v2)

```mermaid
flowchart LR
    subgraph Ingest["1. Ingest (sekali)"]
        direction TB
        A[PDF scan / foto] --> B[Render 300 DPI]
        B --> C[RapidOCR CPU]
        C --> D[Chunk 800/120]
        D --> E[Embed MiniLM]
        E --> F[(pgvector + FTS)]
    end
    subgraph Query["2. Tanya - agent (tiap query)"]
        direction TB
        G[Pertanyaan] --> R{Router}
        R -->|sapaan/cara pakai| Z[Direct answer]
        R -->|semua yg lain| H[Hybrid retrieve top-20]
        H --> RR[Rerank → top-5]
        RR --> GR{Grade cukup?}
        GR -->|ya| I[Generate + sitasi]
        GR -->|tidak, coba 1x| RW[Rewrite query]
        RW --> H
        GR -->|tetap tidak| N[Jujur: tidak tahu]
        I --> J[DeepSeek v4-flash]
    end
    F -.-> H
```

## Alur data

| Tahap | Input → Output | Kode |
|---|---|---|
| Ingest | `scan.pdf` → teks digital langsung, atau render 300 DPI → PNG per halaman scan | `src/ingest/pdf_loader.py::pdf_page_texts`, `pdf_pages_to_images` |
| OCR | PNG → teks + bbox + confidence | `src/ingest/ocr.py::ocr_image` |
| Chunk | teks panjang → potongan 800 char | `src/ingest/chunker.py` |
| Embed+Store | chunk → vektor 384-dim → pgvector | `src/rag/embed_store.py`, `src/ingest/pipeline.py` |
| FTS index | kolom `document` → GIN `tsvector('simple')` + trigram | `scripts/add_fts_index.sql` |
| Router | pertanyaan → `direct` / `rag` | `src/rag/graph.py::router_node` |
| Retrieve | pertanyaan → 20 kandidat (hybrid RRF) | `src/rag/hybrid.py::hybrid_search` |
| Rerank | 20 kandidat → 5 terbaik (cross-encoder) | `src/rag/rerank.py::rerank` |
| Grade/Rewrite | cukup? → generate / tulis ulang 1x / jujur tidak tahu | `src/rag/graph.py::_grade`, `rewrite_node` |
| Generate | konteks + pertanyaan → jawaban + sitasi | `src/rag/chain.py::query` via `graph.agent_answer` |
| Serve | HTTP `/ingest`, `/query` + UI Streamlit | `src/api/main.py`, `ui_app.py` |

## Peta konsep: mana RAG, mana LangChain, mana LangGraph

Diagram di atas sengaja tidak melabeli library. Tiga istilah ini perannya beda:

- **RAG** = nama *pola* keseluruhan ("ambil konteks dari dokumen → tempel ke prompt → LLM menjawab"). Di diagram: `Embed MiniLM` + `pgvector + FTS` (menyiapkan bahan) dan `Hybrid retrieve` → `Rerank` → `Grade` → `Generate + sitasi` (memakai bahan). Bukan RAG: `Render`, `RapidOCR`, `Chunk` (persiapan data), `Router`/`Direct answer` (jalur pintas tanpa dokumen), `Rewrite` (manipulasi query).
- **LangChain** = *perkakas* (library) yang dipakai di dalam node-node itu:
  | Node diagram | Komponen LangChain | File |
  |---|---|---|
  | `Chunk 800/120` | `RecursiveCharacterTextSplitter` | `src/ingest/chunker.py` |
  | `Embed MiniLM` | `HuggingFaceEmbeddings` (CPU, normalized) | `src/rag/embed_store.py` |
  | `pgvector + FTS` | `PGVector` vectorstore (+ SQL FTS mentah via `psycopg`) | `src/rag/embed_store.py`, `src/rag/hybrid.py` |
  | `Hybrid retrieve` | `similarity_search` + RRF manual | `src/rag/hybrid.py` |
  | `Router/Grade/Rewrite/Generate` | `ChatOpenAI` (wrapper DeepSeek) + prompt + `StrOutputParser`, pola LCEL | `src/rag/graph.py`, `src/rag/chain.py` |
- **LangGraph** = *otak/pengatur alur* (khusus kotak "Tanya - agent"): node `Router`, decision `Grade cukup?`, dan loop `Rewrite query → retrieve ulang`. Hidup di `src/rag/graph.py` (`langgraph==0.2.28`).
- **Bukan LangChain sama sekali:** `Render 300 DPI` (PyMuPDF), `RapidOCR CPU` (ONNX Runtime), index GIN/trigram (Postgres murni), cross-encoder rerank (`sentence-transformers` langsung), FastAPI, Streamlit.

Satu kalimat: LangChain menyediakan perkakasnya, LangGraph mengatur alurnya, RAG adalah nama pola keseluruhannya.

## Keputusan desain penting

1. **CPU-first.** OCR (RapidOCR ONNX) + embedding (MiniLM) jalan di CPU. Cuma LLM yang via API. Alasannya: murah, mudah didemo, tanpa GPU.
2. **DeepSeek OpenAI-compatible.** `ChatOpenAI` + `base_url=https://api.deepseek.com/v1`, model `deepseek-v4-flash` dikonfig via `.env`. Ganti model tanpa ubah kode.
3. **pgvector di Docker port 5433.** Laptop sudah ada Postgres bawaan di 5432, jadi container dipindah agar tidak bentrok. Lihat `docs/05-setup-operasi.md`.
4. **Grounding wajib + verifikasi pre-generate.** Prompt memaksa jawab hanya dari konteks + sitasi `[source hal. X]`, dan node `grade` menolak generate kalau konteks tidak mendukung → jawab jujur "tidak tahu". Lihat `docs/02-teknik-rag.md`.
5. **Degradasi aman.** Reranker gagal load (offline) → pakai urutan hybrid. Agent error → fallback hybrid+rerank langsung. Endpoint selalu balas JSON (termasuk saat error 500) agar UI tidak crash.
