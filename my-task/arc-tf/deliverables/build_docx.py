# -*- coding: utf-8 -*-
"""ARC TF 기획안 .md -> .docx.

원본은 .md 다. 이 .docx 는 손으로 고치지 않고 다시 만든다.
    python my-task/arc-tf/deliverables/build_docx.py
필요: python-docx. 한글 폰트: 맑은 고딕 (회사 PC 기준, Mac 에서는 대체 폰트로 보인다).

지원하는 마크다운: frontmatter, # ~ ####, 문단, - / 1. 목록, > 인용, 표, ``` 코드, **굵게**, `코드`, [링크](...).
"""
import re
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "arc-tf-proposal.docx"
# (파일, 별첨 제목 또는 None)
SOURCES = [("01-tf-proposal.md", None), ("00-problem-statement.md", "별첨 A")]
SKIP_SECTIONS = {"관련 문서"}

FONT = "맑은 고딕"
MONO = "Consolas"
ACCENT = RGBColor(0x1F, 0x49, 0x7D)
GREY = RGBColor(0x59, 0x59, 0x59)
TABLE_HEAD_FILL = "DCE6F1"

INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def set_font(run_or_style, name=FONT):
    run_or_style.font.name = name
    rpr = run_or_style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), name)


def add_inline(par, text, bold=False, size=None, color=None):
    for part in INLINE.split(text):
        if not part:
            continue
        is_bold, is_code = bold, False
        if part.startswith("**") and part.endswith("**"):
            part, is_bold = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, is_code = part[1:-1], True
        elif part.startswith("["):
            part = re.match(r"\[([^\]]+)\]", part).group(1)
        run = par.add_run(part)
        run.bold = is_bold
        set_font(run, MONO if is_code else FONT)
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # ponytail: column width by longest text (clamped), not true text measurement
    weights = [min(max(len(r[j]) if j < len(r) else 0 for r in cells[:1] + body), 40) + 4 for j in range(len(header))]
    widths = [Cm(16.6 * w / sum(weights)) for w in weights]
    for j, col in enumerate(table.columns):
        col.width = widths[j]  # gridCol: LibreOffice, Google Docs read this, not cell width
    for i, row in enumerate([header] + body):
        for j in range(len(header)):
            cell = table.cell(i, j)
            cell.width = widths[j]
            par = cell.paragraphs[0]
            add_inline(par, row[j] if j < len(row) else "", bold=(i == 0), size=9.5)
            if i == 0:
                shade(cell, TABLE_HEAD_FILL)
    doc.add_paragraph()


def add_code(doc, lines):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.5)
    for k, line in enumerate(lines):
        run = par.add_run(line)
        set_font(run, MONO)
        run.font.size = Pt(8.5)
        if k < len(lines) - 1:
            run.add_break(WD_BREAK.LINE)


def strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        return text[end + 4:]
    return text


def render(doc, md, appendix):
    lines = strip_frontmatter(md).splitlines()
    i, skipping = 0, False
    while i < len(lines):
        line = lines[i]
        heading = re.match(r"^(#{1,4}) (.+)", line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            if level == 2:
                skipping = title in SKIP_SECTIONS
            if not skipping:
                if level == 1 and appendix:
                    title = f"{appendix}. {title}"
                doc.add_heading(title, level=level)
            i += 1
            continue
        if skipping:
            i += 1
            continue
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            add_code(doc, block)
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        elif line.startswith(">"):
            par = doc.add_paragraph(style="Quote")
            add_inline(par, line.lstrip("> ").strip(), color=GREY)
        elif re.match(r"^\s*- ", line):
            par = doc.add_paragraph(style="List Bullet 2" if line.startswith("  ") else "List Bullet")
            add_inline(par, re.sub(r"^\s*- ", "", line))
        elif re.match(r"^\d+\. ", line):
            par = doc.add_paragraph(style="List Number")
            add_inline(par, re.sub(r"^\d+\. ", "", line))
        elif line.strip() == "---":
            pass
        elif line.strip():
            add_inline(doc.add_paragraph(), line.strip())
        i += 1


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.2))
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    styles = doc.styles
    set_font(styles["Normal"])
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"].paragraph_format.space_after = Pt(4)
    styles["Normal"].paragraph_format.line_spacing = 1.25
    for name, size in (("Title", 24), ("Heading 1", 18), ("Heading 2", 14), ("Heading 3", 12), ("Heading 4", 11)):
        st = styles[name]
        set_font(st)
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = ACCENT
        st.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
    for name in ("Quote", "List Bullet", "List Bullet 2", "List Number"):
        set_font(styles[name])


def cover(doc):
    for _ in range(6):
        doc.add_paragraph()
    doc.add_paragraph("ARC TF 구성 기획안", style="Title")
    add_inline(doc.add_paragraph(), "레시피(Recipe) 셋업 자동화 태스크포스", size=14, color=GREY)
    doc.add_paragraph()
    add_inline(doc.add_paragraph(), "보고 대상: AI TF 담당 임원(1차 승인) → 기반기술센터장(2차 최종 승인)", size=11)
    add_inline(doc.add_paragraph(), f"작성: TF 리드 · {date.today():%Y-%m-%d} · 초안", size=11)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def main():
    doc = Document()
    setup(doc)
    cover(doc)
    for k, (name, appendix) in enumerate(SOURCES):
        if k:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        render(doc, (ROOT / name).read_text(encoding="utf-8"), appendix)
    doc.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    sys.exit(main())
