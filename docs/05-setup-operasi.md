# Setup & Operasi

## Prasyarat
- Python 3.12, Docker + Docker Compose, key DeepSeek.

## 1. Database
```bash
docker compose up -d db
docker exec portoai-pgvector psql -U portoai -d portoai \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
- Container: `portoai-pgvector`, image `pgvector/pgvector:pg16`, port host **5433** → container 5432.
- Kenapa 5433? Laptop ini sudah ada Postgres bawaan di 5432. Kalau pakai 5432 akan bentrok (`bind: address already in use`) atau API nyambung ke DB yang salah (`FATAL: password authentication failed for user "portoai"`).
- `DATABASE_URL` di `.env` wajib `...@localhost:5433/...` agar cocok.

## 2. Python
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
- Catatan: `requirements.txt` memakai pin lentur (`>=`) untuk paket LangChain agar cocok di Python 3.12. Error `No matching distribution found for langchain-huggingface==0.0.4` sudah diperbaiki dengan cara ini.

## 3. Konfigurasi (.env)
- Dibuat dari `.env.example`. Jangan commit `.env` (sudah di `.gitignore`).
- Wajib: `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL=deepseek-v4-flash`, `DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`.
- Model terverifikasi dari `/v1/models`: `deepseek-v4-flash`, `deepseek-v4-pro`, `deepseek-v4-flash-vision-exp`.

## 4. Jalan
```bash
uvicorn src.api.main:app --reload --port 8000   # terminal 1
streamlit run ui_app.py                          # terminal 2
```
- Download `model.safetensors` (~471 MB) hanya sekali ke `~/.cache/huggingface`. Restart berikutnya offline.
- Produksi: hilangkan `--reload` agar model tidak di-load ganda.

## Troubleshooting cepat
| Gejala | Penyebab | Fix |
|---|---|---|
| `bind: address already in use` port 5432 | Postgres bawaan OS | Pakai 5433 (sudah default repo ini) |
| `password authentication failed for user "portoai"` | API nyambung ke DB salah / container mati | `docker compose ps`, samakan port `.env` ↔ compose |
| `extension vector does not exist` | Lupa create extension | Ulangi step DB di atas |
| `401 deepseek` | Key salah / kuota habis | Cek `platform.deepseek.com`, rotate key, update `.env` lalu restart API |
| Download model berulang | Cache terhapus / ganti `EMBEDDING_MODEL` | Jangan hapus `~/.cache/huggingface` |
