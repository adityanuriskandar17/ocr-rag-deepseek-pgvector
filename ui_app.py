import streamlit as st
import requests

API = "http://localhost:8000"

st.title("Porto-AI — OCR + RAG (DeepSeek + pgvector)")

up = st.file_uploader("Upload PDF scan / foto kertas", type=["pdf", "jpg", "jpeg", "png"])
if up and st.button("Ingest"):
    r = requests.post(f"{API}/ingest", files={"file": (up.name, up.getvalue())})
    st.json(r.json())

q = st.text_input("Tanya dokumen:")
if q and st.button("Tanya"):
    r = requests.post(f"{API}/query", json={"question": q}).json()
    st.markdown(r["answer"])
    st.json(r["sources"])
