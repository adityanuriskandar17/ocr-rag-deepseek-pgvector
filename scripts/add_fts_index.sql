-- Fase 1 v2: Full-text (BM25-like via ts_rank) untuk hybrid retrieval.
-- Kenapa 'simple': dokumen ID+EN campur + nomor invoice/kode.
-- Stemmer 'english' merusak kode exact, 'indonesian' tidak ada di pg16 default.
-- Jadi 'simple' = lowercase + tokenisasi tanpa stemming, cocok untuk exact match.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- GIN index untuk to_tsvector('simple', document) di tabel langchain.
-- IF NOT EXISTS biar idempotent, aman dijalankan ulang.
CREATE INDEX IF NOT EXISTS idx_embedding_fts_simple
ON langchain_pg_embedding
USING GIN (to_tsvector('simple', document));

-- Index trigram untuk fallback ILIKE (kode pendek / nomor yang lolos tsquery).
CREATE INDEX IF NOT EXISTS idx_embedding_doc_trgm
ON langchain_pg_embedding
USING GIN (document gin_trgm_ops);
