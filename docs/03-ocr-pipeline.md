# Pipeline OCR (CPU, tanpa GPU)

## Jalur cepat: PDF digital tidak di-OCR
Sejak update ini, `ingest_path` memeriksa teks embedded per halaman dulu (`pdf_page_texts`):
- Halaman dengan teks ≥ `PDF_TEXT_MIN_CHARS=100` → dipakai langsung (`extract=text`, `ocr_conf=1.0`). 5 halaman digital selesai dalam <1 detik.
- Hanya halaman tanpa teks (= hasil scan/foto) yang di-render 300 DPI + OCR (`extract=ocr`). PDF campuran didukung per halaman.
- Foto JPG/PNG mentah selalu lewat OCR (tidak punya teks embedded).
- Respons `/ingest` melaporkan perincian: `{"chunks", "pages_text", "pages_ocr"}`.

## Kenapa bukan Unlimited-OCR / VLM 3B?
VLM seperti `baidu/Unlimited-OCR` butuh GPU NVIDIA + CUDA (bfloat16, context 32k, vLLM/SGLang). Untuk use-case **scan PDF + foto kertas → teks**, OCR klasik CPU sudah cukup dan jauh lebih murah. VLM baru layak kalau butuh parsing tabel/rumus/layout markdown yang kompleks.

## Tahapan

1. **Render PDF → gambar** (`src/ingest/pdf_loader.py`)
   - `PyMuPDF` + `fitz.Matrix(dpi/72)` dengan `OCR_DPI=300`.
   - 300 DPI adalah sweet spot: cukup tajam untuk OCR, tidak terlalu besar di memori.
   - Foto JPG/PNG langsung dipakai tanpa render ulang.

2. **OCR per gambar** (`src/ingest/ocr.py`)
   - Engine: `RapidOCR` (ONNX Runtime, CPU) — turunan PaddleOCR yang dioptimasi deployment.
   - Pipeline internal: deteksi teks (DBNet) → klasifikasi arah → recognition (CRNN).
   - Output dinormalisasi ke `{text, boxes, conf}`; `conf` rata-rata dipakai sebagai metadata `ocr_conf` per halaman.
   - Model ONNX di-download sekali lalu di-cache; ingest berikutnya offline.

3. **Tips kualitas foto kertas**
   - Sejajarkan kamera, cahaya rata, hindari bayangan.
    - `use_angle_cls` (klasifikasi arah) sudah built-in di RapidOCR untuk rotasi 0/90/180/270.
   - Kalau hasil jelek: naikkan DPI ke 400 atau crop margin sebelum ingest.

## Keterbatasan jujur (bagus untuk interview)
- Tulisan tangan jelek, stempel tindih teks, dan tabel kompleks → akurasi turun.
- Tidak menghasilkan markdown/tabel seperti VLM GPU.
- Mitigasi: simpan `ocr_conf` + bbox sehingga jawaban RAG bisa ditelusur ke halaman sumber.
