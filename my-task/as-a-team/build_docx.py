# -*- coding: utf-8 -*-
"""01-team-proposal.md, 01-team-proposal-v2.md -> .docx. 원본은 .md 다. .docx 는 손으로 고치지 않고 다시 만든다.
    python my-task/as-a-team/build_docx.py
변환기는 arc-tf/deliverables/build_docx.py 의 것을 그대로 쓴다. 필요: python-docx.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "arc-tf" / "deliverables"))
from build_docx import Document, cover, render, setup  # noqa: E402

for SRC in ("01-team-proposal.md", "01-team-proposal-v2.md"):
    doc = Document()
    setup(doc)
    cover(doc, "AX Team 승격 기획안")
    render(doc, (HERE / SRC).read_text(encoding="utf-8").replace("⟨", "<").replace("⟩", ">"), None)
    out = HERE / SRC.replace(".md", ".docx")
    doc.save(out)
    print(f"wrote {out.name}")
