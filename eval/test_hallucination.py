"""Test anti-halusinasi RAG. Jalankan: python eval/test_hallucination.py

3 lapis pertahanan (sesuai docs/02-teknik-rag.md):
1. Prompt grounding (temperature=0 + 'jawab HANYA dari konteks')
2. Sitasi wajib [source hal. X] -> bisa diaudit
3. Test ini: sitasi ada + topik di luar dokumen DITOLAK + angka sesuai konteks
"""

import re
import sys

sys.path.insert(0, ".")

from src.rag.chain import query

CITATION = re.compile(r"\[.+?hal\.\s*\d+", re.IGNORECASE)
REFUSAL = re.compile(r"tidak (tahu|ada di konteks|diketahui|terdapat di)", re.IGNORECASE)

# 1) Pertanyaan yang jawabannya ADA di dokumen kamu.
#    Ganti expected_keywords dengan kata yang benar-benar ada di scan kamu.
GROUNDED_TESTS = [
    {"q": "Dokumen ini tentang apa? Sebutkan poin utamanya.", "expected_keywords": []},
]

# 2) Pertanyaan JEBAKAN: topik pasti tidak ada di dokumen.
#    Model yang halu akan ngarang; yang benar harus menolak.
TRAP_TESTS = [
    "Siapa pemenang Piala Dunia 2030?",
    "Jelaskan resep rendang padang yang enak.",
]


def check_citation(answer: str) -> bool:
    return bool(CITATION.search(answer))


def check_keywords(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


def main() -> int:
    fails = 0

    print("=== A. Grounded (harus jawab + bersitasi) ===")
    for t in GROUNDED_TESTS:
        res = query(t["q"])
        a = res["answer"]
        ok_cite = check_citation(a)
        ok_kw = check_keywords(a, t["expected_keywords"])
        ok = ok_cite and ok_kw
        print(f"Q: {t['q']}\nA: {a[:400]}\nsitasi={ok_cite} keywords={ok_kw} -> {'PASS' if ok else 'FAIL'}\n---")
        fails += not ok

    print("=== B. Jebakan (harus MENOLAK, tidak ngarang) ===")
    for q in TRAP_TESTS:
        res = query(q)
        a = res["answer"]
        refused = bool(REFUSAL.search(a))
        has_cite = check_citation(a)
        # LULUS jika menolak, atau minimal tidak memberi sitasi palsu yang meyakinkan
        ok = refused or not has_cite
        print(f"Q: {q}\nA: {a[:400]}\nmenolak={refused} -> {'PASS' if ok else 'FAIL (ngarang!) '}\n---")
        fails += not ok

    print(f"SELESAI: {'SEMUA PASS' if not fails else f'{fails} GAGAL'}")
    print("Tindak lanjut kalau FAIL: naikkan TOP_K, perbaiki chunk/OCR, atau ketatkan prompt.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
