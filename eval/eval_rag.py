"""Eval sederhana: cek jawaban mengandung sitasi + tampilkan route/timings agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chain import query

TESTS = [
    {"q": "Dokumen ini tentang apa?", "must_contain": ["hal."]},
]

for t in TESTS:
    res = query(t["q"])
    ok = all(m.lower() in res["answer"].lower() for m in t["must_contain"])
    print(f"Q: {t['q']}\nA: {res['answer'][:500]}")
    print(f"route={res.get('route')} timings={res.get('timings')}")
    print(f"PASS={ok}\n---")
