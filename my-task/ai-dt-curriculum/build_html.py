# -*- coding: utf-8 -*-
"""advanced-course-vendor-request.md -> 미팅용 .html 본문 갱신. 원본은 .md 다.
    python my-task/ai-dt-curriculum/build_html.py
.html 의 <head>(스타일)와 끝의 탭 <script>는 그대로 두고 <body> 본문만 다시 만든다.
"## N. 제목" 하나가 탭 하나다 (탭 스크립트가 한 자리 번호만 읽으므로 9개까지).
"""
import html
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "advanced-course-vendor-request.md"
OUT = HERE / "advanced-course-vendor-request.html"


def inline(text):
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html.escape(text, quote=False))


def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def block(lines):
    """한 절의 줄들을 HTML로. 표, 목록, 문단만 다룬다."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            head, body = rows[0], rows[2:]
            cls = ' class="modules"' if len(head) >= 4 else ""
            out.append(f'<div class="tbl"><table{cls}><thead><tr>'
                       + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table></div>")
            continue
        for pat, tag in ((r"^- ", "ul"), (r"^\d+\. ", "ol")):
            if re.match(pat, line):
                items = []
                while i < len(lines) and re.match(pat, lines[i]):
                    items.append(f"  <li>{inline(re.sub(pat, '', lines[i]))}</li>")
                    i += 1
                out.append(f"<{tag}>\n" + "\n".join(items) + f"\n</{tag}>")
                break
        else:
            if line.strip():
                out.append(f"<p>{inline(line.strip())}</p>")
            i += 1
    return "\n".join(out)


md = SRC.read_text(encoding="utf-8")
md = md[md.find("\n---", 3) + 4:]  # frontmatter 제거
title = re.search(r"^# (.+)$", md, re.M).group(1)
lead = re.search(r"^> (.+)$", md, re.M).group(1)
sections = [(m.group(1), m.group(2).strip().splitlines())
            for m in re.finditer(r"^## \d+\. (.+)\n((?:(?!^## ).*\n?)*)", md, re.M)]
assert 0 < len(sections) <= 9, len(sections)

# 머리의 요약 칸: "과정 개요" 표에서 대상 조직을 뺀 행
overview = next(body for name, body in sections if name == "과정 개요")
facts = [cells(r) for r in overview if r.startswith("|")][2:]
facts = [f for f in facts if f[0] != "대상 조직"][:5]

tabs = "".join(
    f'<button role="tab" id="t{k}" aria-controls="s{k}" aria-selected="{"true" if k == 1 else "false"}">'
    f"<span>{k:02d}</span>{html.escape(name)}</button>"
    for k, (name, _) in enumerate(sections, 1))
panels = "\n".join(
    f'<section id="s{k}" role="tabpanel" aria-labelledby="t{k}"{"" if k == 1 else " hidden"}>\n'
    f"<h2>{html.escape(name)}</h2>\n{block(body)}\n</section>"
    for k, (name, body) in enumerate(sections, 1))
body = f"""<body>
<header><div class="wrap">
<p class="eyebrow">업체 미팅 자료, {date.today():%Y-%m-%d}</p>
<h1>{html.escape(title)}</h1>
<p class="lead"><strong>{inline(lead)}</strong></p>
<dl class="facts">
{chr(10).join(f"  <div><dt>{inline(k)}</dt><dd>{inline(v)}</dd></div>" for k, v in facts)}
</dl>
</div></header>
<nav class="tabs"><div class="wrap" role="tablist">{tabs}</div></nav>
<main><div class="wrap">
{panels}
</div></main>
<footer><div class="wrap"><p>세부 커리큘럼과 시간 배분은 업체 제안 기준</p></div></footer>
"""

page = OUT.read_text(encoding="utf-8")
a, b = page.index("<body>"), page.index("<script>")
page = page[:a] + body + page[b:]
page = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", page, count=1)
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT.name}: {len(sections)} tabs")
