# -*- coding: utf-8 -*-
"""ARC TF .md -> 독자별 .docx 3종.

원본은 .md 다. 이 .docx 는 손으로 고치지 않고 다시 만든다.
    python my-task/arc-tf/deliverables/build_docx.py          # 3종 모두
    python my-task/arc-tf/deliverables/build_docx.py boss     # team | boss | exec 중 하나

  team  팀원용 전체본: 문제 정의, 기획안, PRD, 기술 설계, 운영 문서, 위험, 결정 기록
  boss  담당 임원용 기획안: 기획안 본문 + 별첨 A 문제 정의
  exec  최고 임원용 한 장 요약
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
# 대상: (출력 파일, 표지 제목, [(원본 .md, 별첨 제목 또는 None)])
TARGETS = {
    "team": ("arc-tf-team-full.docx", "ARC TF 팀원용 전체 문서", [
        ("00-problem-statement.md", None), ("01-tf-proposal.md", None), ("06-platform-prd.md", None),
        ("07-platform-architecture.md", None), ("02-roles-and-teams.md", None), ("03-operating-rhythm.md", None),
        ("04-onboarding.md", None), ("05-poc-assets.md", None), ("risk-register.md", None),
        ("decision-log.md", None)]),
    "boss": ("arc-tf-proposal.docx", "Auto Recipe Creation TF 구성 기획안", [
        ("01-tf-proposal.md", None), ("00-problem-statement.md", "별첨 A")]),
    "exec": ("arc-tf-exec-summary.docx", None, [("08-exec-summary.md", None)]),
}
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


KANBAN_LANES = ["FBE3CC", "DCE6F1", "EDEDED"]  # 트랙별 칸 색 (Align Fail, 관찰과 지식화, 2차)


def add_kanban(doc, header, body):
    """'트랙'으로 시작하는 표: 행 = 트랙(스윔레인), 열 = 기간, 칸 = 작업 카드."""
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    first = Cm(2.6)
    rest = Cm((16.6 - 2.6) / (len(header) - 1))
    for j, col in enumerate(table.columns):
        col.width = first if j == 0 else rest
    for i, row in enumerate([header] + body):
        for j in range(len(header)):
            cell = table.cell(i, j)
            cell.width = first if j == 0 else rest
            items = (row[j] if j < len(row) else "").split("<br>")
            for k, item in enumerate(items):
                par = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
                par.paragraph_format.space_after = Pt(2)
                prefix = "" if i == 0 or j == 0 or item.strip() in ("", "-") else "▪ "
                add_inline(par, prefix + item.strip(), bold=(i == 0 or j == 0), size=9,
                           color=RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else None)
            if i == 0:
                shade(cell, "1F497D")
            else:
                shade(cell, KANBAN_LANES[(i - 1) % len(KANBAN_LANES)])
    doc.add_paragraph()


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]
    if header[0] == "트랙":
        return add_kanban(doc, header, body)
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
            add_inline(par, (row[j] if j < len(row) else "").replace("<br>", ", "), bold=(i == 0), size=9.5)
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
            add_inline(par, re.sub(r"^\s*- (\[ \] )?", lambda m: "☐ " if m.group(1) else "", line))
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


def cover(doc, title_text):
    for _ in range(8):
        doc.add_paragraph()
    title = doc.add_paragraph(title_text, style="Title")
    title.alignment = 1  # center
    sub = doc.add_paragraph()
    sub.alignment = 1
    add_inline(sub, f"{date.today():%Y-%m}", size=12, color=GREY)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build(target):
    out_name, cover_title, sources = TARGETS[target]
    doc = Document()
    setup(doc)
    if cover_title:
        cover(doc, cover_title)
    else:  # exec 한 장 요약: 표지 없이, 한 쪽에 들어가게 촘촘히
        doc.styles["Normal"].font.size = Pt(10)
        doc.styles["Normal"].paragraph_format.line_spacing = 1.1
    for k, (name, appendix) in enumerate(sources):
        if k:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        render(doc, (ROOT / name).read_text(encoding="utf-8"), appendix)
    out = HERE / out_name
    doc.save(out)
    print(f"wrote {out.relative_to(ROOT.parent.parent)}")


def main():
    for target in sys.argv[1:] or TARGETS:
        build(target)


if __name__ == "__main__":
    sys.exit(main())
