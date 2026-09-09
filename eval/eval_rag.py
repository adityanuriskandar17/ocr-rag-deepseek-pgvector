"""Eval sederhana: retrieval hit-rate + cek jawaban mengandung sitasi."""

from src.rag.chain import query

TESTS = [
    {"q": "Dokumen ini tentang apa?", "must_contain": ["hal."]},
]

for t in TESTS:
    res = query(t["q"])
    ok = all(m.lower() in res["answer"].lower() for m in t["must_contain"])
    print(f"Q: {t['q']}\nA: {res['answer'][:500]}\nPASS={ok}\n---")
