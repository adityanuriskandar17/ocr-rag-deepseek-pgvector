import streamlit as st
import requests

API = "http://localhost:8000"

st.title("Porto-AI — OCR + RAG (DeepSeek + pgvector)")

up = st.file_uploader("Upload PDF scan / foto kertas", type=["pdf", "jpg", "jpeg", "png"])
if up and st.button("Ingest"):
    r = requests.post(f"{API}/ingest", files={"file": (up.name, up.getvalue())})
    try:
        st.json(r.json())
    except Exception:
        st.error(f"API error {r.status_code}: {r.text[:300]}")

q = st.text_input("Tanya dokumen:")
if q and st.button("Tanya"):
    r = requests.post(f"{API}/query", json={"question": q})
    try:
        res = r.json()
    except Exception:
        st.error(f"API error {r.status_code}: {r.text[:300]}")
        st.stop()
    if r.status_code != 200 or "error" in res:
        st.error(res.get("error", f"API error {r.status_code}"))
        st.stop()
    st.markdown(res["answer"])
    st.json(res["sources"])
