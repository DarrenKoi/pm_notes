# -*- coding: utf-8 -*-
"""advanced-course-vendor-request.md -> .docx. 원본은 .md 다. .docx 는 손으로 고치지 않고 다시 만든다.
    python my-task/ai-dt-curriculum/build_docx.py
변환기는 arc-tf/deliverables/build_docx.py 의 것을 그대로 쓴다. 필요: python-docx.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "arc-tf" / "deliverables"))
from build_docx import Document, cover, render, setup  # noqa: E402

SRC = "advanced-course-vendor-request.md"
doc = Document()
setup(doc)
cover(doc, "기반기술센터 AI Agent 고급 과정 커리큘럼 요청서")
render(doc, (HERE / SRC).read_text(encoding="utf-8").replace("⟨", "<").replace("⟩", ">"), None)
out = HERE / SRC.replace(".md", ".docx")
doc.save(out)
print(f"wrote {out.name}")
