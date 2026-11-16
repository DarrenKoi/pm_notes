---
tags: [pptx, python-pptx, template, report]
level: intermediate
last_updated: 2026-11-17
---

# 회사 보고용 pptx 템플릿

> 회사 마스터 슬라이드에 옮겨 쓰기 좋은 배치와 색 규칙, 그리고 그대로 실행되는 python-pptx 시작 코드. 이 파일 하나만 다른 저장소에 복사하면 같은 형태의 deck 을 만들 수 있다.

## 왜 필요한가? (Why)

- 회사에는 별도 마스터 슬라이드가 있고, 만든 장표를 거기에 옮겨 쓴다. 마스터가 이미 가진 것(제목 아래 선, 로고, 슬로건, 부서명, 쪽 번호)을 deck 에 또 그리면 겹친다
- 장표마다 손으로 위치를 맞추면 제목, 요약문, 본문 높이가 장마다 달라진다. 좌표와 색을 코드 한 곳에 모아 두면 deck 전체가 같은 형태가 된다
- 코딩 에이전트에게 "이 템플릿으로 만들어 줘"라고 이 파일을 주면, 배치와 색을 다시 설명하지 않아도 된다

## 핵심 개념 (What)

### 배치

슬라이드는 16:9, 13.33 x 7.5 inch, Windows PowerPoint 기준이다. 글꼴은 맑은 고딕이다.

| 요소 | 위치 (inch) | 글자 | 규칙 |
|------|-------------|------|------|
| 제목 | 왼쪽 0.3, 위 0.15, 높이 0.46 | 26pt 볼드, 검정 | 제목 아래 가로선을 그리지 않음 |
| 요약문 | 위 0.75, 높이 0.45, 좌우 0.8 | 20pt 볼드, 가운데 정렬 | 장표마다 한 줄. 검정 글자에 중요 문구와 단어만 파랑. 배경색 없음. 지우지 않음 |
| 본문 | 위 1.55 ~ 아래 6.95, 좌우 0.5 | 13~17pt | 요약문과 본문 사이 0.35. 아래 0.55 는 마스터 footer 자리라 비움 |
| footer | 없음 | 없음 | 로고, 슬로건, 부서명, 경계선, 쪽 번호를 그리지 않음 |
| 표지 제목 | 위 2.95, 가로 가운데 | 40pt 볼드 | 날짜와 가로선 없음 |
| 표지 부제 | 위 3.85, 가로 가운데 | 18pt, 짙은 회색 | 제목 바로 아래 |

### 색

상자는 회색 바탕에 검정 글자가 기본이다. 색은 몇 개만 쓴다.

| 이름 | hex | 쓰는 곳 |
|------|-----|---------|
| 검정 `INK` | 262626 | 본문 글자, 글머리 기호, 강조 상자 바탕(흰 글자와 함께) |
| 짙은 회색 `SUB` | 595959 | 보조 글자, 화살표 |
| 진회색 `DARK` | 404040 | 연결선 (0.75pt) |
| 회색 `GRAY` | BFBFBF | 컬럼 사이 화살표 도형 |
| 테두리 `LINE` | D0D0D0 | 상자 테두리 |
| 밝은 회색 `SILVER` | D9D9D9 | 표 머리줄, 컬럼 머리줄, 일반 상자 |
| 연회색 `MIST` | F2F2F2 | 표 줄무늬, 텍스트 상자 바탕 |
| 짙은 파랑 `BLUE` | 0B3C8F | 요약문의 중요 문구, "운영" 같은 메모 머리말. 상자 채움에는 쓰지 않음 |
| 연한 파랑 `BLUE_PALE` | E8F0FA | 표 강조 행 |
| 주황 `ORANGE` | F38B1F | 관문, 최종 목표, `⟨확인⟩` 자리에만 |

- 강조 상자는 장표에 한두 개만 둔다. 검정 바탕에 흰 글자다
- 전과 후를 나누는 2 컬럼에서는 "후" 컬럼 머리줄만 검정으로 강조한다
- 연결선은 얇고 진하게 그린다. 두껍고 희미한 회색 선은 쓰지 않는다

### 내용 구성

- 장표 하나에 제목 하나, 요약문 한 줄, 본문 덩어리 하나에서 셋
- 2 컬럼으로 전과 후를 나누는 구성을 먼저 고려한다. 좌는 이슈, 문제, 기존이고 우는 해결, 개선이다
- 본문 문자열에 `**볼드**`, `__밑줄__` 표기를 직접 쓴다. 요약문에서는 `**문구**` 가 파랑이 된다
- 근거 수치가 없으면 지어내지 않고 `⟨확인⟩` 으로 둔다. 이 글자는 자동으로 주황이 된다
- 목록과 표 칸은 개조식(~임, ~함, 명사)으로 쓰고 마침표를 찍지 않는다

## 어떻게 사용하는가? (How)

### 1. 준비

```bash
pip install python-pptx
```

### 2. 시작 코드

아래를 `build_deck.py` 로 저장하고 실행하면 `deck.pptx` 가 나온다. 장표 하나가 함수 하나이고, 맨 아래 `SLIDES` 목록이 순서다. 예시 장표 네 개를 지우고 자기 내용으로 바꾼다.

```python
# -*- coding: utf-8 -*-
"""회사 보고용 pptx 템플릿. 실행: python build_deck.py  (필요: python-pptx, 글꼴: 맑은 고딕)"""
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FONT = "맑은 고딕"
INK = RGBColor(0x26, 0x26, 0x26)        # 검정: 본문, 글머리 기호, 강조 상자 바탕
SUB = RGBColor(0x59, 0x59, 0x59)        # 짙은 회색: 보조 글자, 화살표
DARK = RGBColor(0x40, 0x40, 0x40)       # 진회색: 연결선
GRAY = RGBColor(0xBF, 0xBF, 0xBF)       # 컬럼 사이 화살표 도형
LINE = RGBColor(0xD0, 0xD0, 0xD0)       # 상자 테두리
SILVER = RGBColor(0xD9, 0xD9, 0xD9)     # 밝은 회색: 머리줄, 일반 상자
MIST = RGBColor(0xF2, 0xF2, 0xF2)       # 연회색: 줄무늬, 텍스트 상자 바탕
BLUE = RGBColor(0x0B, 0x3C, 0x8F)       # 짙은 파랑: 요약문의 중요 문구, 메모 머리말
BLUE_PALE = RGBColor(0xE8, 0xF0, 0xFA)  # 표 강조 행
ORANGE = RGBColor(0xF3, 0x8B, 0x1F)     # 관문, 최종 목표, ⟨확인⟩
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Emu(12192000), Emu(6858000)      # 16:9, 13.33 x 7.5 inch
M = Inches(0.5)                         # 본문 좌우 여백
TITLE_X, TITLE_Y, TITLE_H = Inches(0.3), Inches(0.15), Inches(0.46)
MSG_Y, MSG_H = Inches(0.75), Inches(0.45)
BODY_TOP, BODY_BOTTOM = Inches(1.55), Inches(6.95)  # 아래 0.55 inch 는 마스터 footer 자리
BODY_H = BODY_BOTTOM - BODY_TOP
HEAD_H = Inches(0.55)                   # 컬럼 머리줄 높이

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
MARK = re.compile(r"(\*\*.+?\*\*|__.+?__)")


# ---------- 기본 도구 ----------
def runs_of(t):
    """'**볼드**', '__밑줄__' 표기를 (글자, 옵션) 조각으로 나눈다."""
    out = []
    for part in MARK.split(t):
        if part.startswith("**"):
            out.append((part[2:-2], {"bold": True}))
        elif part.startswith("__"):
            out.append((part[2:-2], {"underline": True}))
        elif part:
            out.append((part, {}))
    return out


def text(s, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.15, gap=0.45):
    """paras: 문자열 또는 문단 목록. 문단은 문자열(표기 해석) 또는 (글자, 옵션) 목록."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate([paras] if isinstance(paras, str) else paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.line_spacing = align, spacing
        if i:
            p.space_before = Pt(size * gap)
        for t, opt in (runs_of(para) if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = FONT, Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.underline = opt.get("underline", False)
            f.color.rgb = opt.get("color", ORANGE if "⟨" in t else color)
    return tb


def box(s, x, y, w, h, fill=MIST, line=None, shape=MSO_SHAPE.RECTANGLE):
    b = s.shapes.add_shape(shape, x, y, w, h)
    b.fill.solid()
    b.fill.fore_color.rgb = fill
    b.line.color.rgb = line or fill  # 선 없는 상자도 채움색 선을 둬서 선 있는 상자와 폭을 맞춘다
    b.line.width = Pt(1)
    b.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = 0.12
    return b


# ---------- 장표 틀 ----------
def slide(title, msg):
    """제목 + 요약문. 요약문은 검정 글자이고 **표시한 문구** 만 파랑이다."""
    s = prs.slides.add_slide(BLANK)
    text(s, TITLE_X, TITLE_Y, W - 2 * M, TITLE_H, title, size=26, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    runs = [(part, {"color": BLUE} if opt else {}) for part, opt in runs_of(msg)]
    text(s, M + Inches(0.3), MSG_Y, W - 2 * M - Inches(0.6), MSG_H, [runs], size=20, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return s


def cover(title, sub):
    s = prs.slides.add_slide(BLANK)
    text(s, Inches(1.0), Inches(2.95), W - Inches(2.0), Inches(0.8), title, size=40, bold=True, align=PP_ALIGN.CENTER)
    text(s, Inches(1.0), Inches(3.85), W - Inches(2.0), Inches(0.5), sub, size=18, color=SUB, align=PP_ALIGN.CENTER)
    return s


# ---------- 본문 부품 ----------
def column(s, x, y, w, h, head, items, size=15, strong=False):
    """머리줄 + 글머리 목록. strong=True 면 머리줄이 검정 바탕 + 흰 글자 (전/후 구성의 '후')."""
    box(s, x, y, w, HEAD_H, fill=INK if strong else SILVER)
    text(s, x + Inches(0.25), y, w - Inches(0.5), HEAD_H, head, size=17, bold=True,
         color=WHITE if strong else INK, anchor=MSO_ANCHOR.MIDDLE)
    box(s, x, y + HEAD_H, w, h - HEAD_H, fill=WHITE, line=LINE)
    text(s, x + Inches(0.3), y + HEAD_H + Inches(0.28), w - Inches(0.6), h - HEAD_H - Inches(0.4),
         [[("▪ ", {"bold": True})] + runs_of(t) for t in items], size=size, gap=0.6)


def two_col(s, left_head, left, right_head, right, y=BODY_TOP, h=BODY_H, size=16, before=False, arrow=True):
    """2 컬럼. before=True 면 전/후 구성으로 보고 오른쪽 머리줄만 검정으로 강조한다."""
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    column(s, M, y, cw, h, left_head, left, size)
    column(s, M + cw + gap, y, cw, h, right_head, right, size, strong=before)
    if arrow:
        a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, M + cw + Inches(0.17), y + h / 2 - Inches(0.25), Inches(0.26), Inches(0.5))
        a.fill.solid()
        a.fill.fore_color.rgb = GRAY
        a.line.fill.background()
        a.shadow.inherit = False


def table(s, rows, y, widths, size=14, row_h=Inches(0.55), bold_col0=False, shade=None):
    """rows[0] 이 머리줄. widths 합은 W - 2 * M. shade(row) 가 참인 행은 연한 파랑으로 강조한다."""
    t = s.shapes.add_table(len(rows), len(rows[0]), M, y, sum(widths, Inches(0)), row_h * len(rows)).table
    for j, wd in enumerate(widths):
        t.columns[j].width = wd
    for i, row in enumerate(rows):
        t.rows[i].height = row_h
        hl = bool(shade and i and shade(row))
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = SILVER if i == 0 else (BLUE_PALE if hl else (MIST if i % 2 == 0 else WHITE))
            c.margin_left = c.margin_right = Inches(0.12)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = val
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0 or (bold_col0 and j == 0) or hl
            r.font.color.rgb = ORANGE if (i and "⟨" in val) else INK
    return t


def flow(s, items, y, h=Inches(1.4), strong=()):
    """가로 흐름 상자. items = [(제목, 설명)]. strong 에 적은 순번(0부터)만 검정 바탕으로 강조한다."""
    n, gap = len(items), Inches(0.3)
    fw = (W - 2 * M - gap * (n - 1)) / n
    for i, (head, body) in enumerate(items):
        x = M + i * (fw + gap)
        on = i in strong
        box(s, x, y, fw, h, fill=INK if on else SILVER)
        text(s, x + Inches(0.08), y + Inches(0.12), fw - Inches(0.16), Inches(0.45), head, size=16, bold=True,
             color=WHITE if on else INK, align=PP_ALIGN.CENTER, spacing=1.0)
        text(s, x + Inches(0.08), y + Inches(0.62), fw - Inches(0.16), h - Inches(0.7), body.split("\n"), size=12,
             color=WHITE if on else SUB, align=PP_ALIGN.CENTER, spacing=1.0, gap=0)
        if i < n - 1:
            a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, x + fw + gap / 2 - Inches(0.08), y + h / 2 - Inches(0.15),
                                   Inches(0.16), Inches(0.3))
            a.fill.solid()
            a.fill.fore_color.rgb = GRAY
            a.line.fill.background()
            a.shadow.inherit = False


def node(s, x, y, w, h, title, sub=None, strong=False, fill=MIST):
    """그림용 둥근 상자: 제목 + 작은 설명. strong=True 면 검정 바탕 + 흰 글자."""
    fc = WHITE if (strong or fill == ORANGE) else INK
    box(s, x, y, w, h, fill=INK if strong else fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    paras = [[(title, {"bold": True, "color": fc})]]
    if sub:
        paras += [[(t, {"size": 11, "color": fc if fc == WHITE else SUB})] for t in sub.split("\n")]
    text(s, x + Inches(0.05), y, w - Inches(0.1), h, paras, size=15, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.0, gap=0.15)


def arrow(s, x0, y0, x1, y1, label=None, color=SUB, label_w=Inches(2.4)):
    """화살표 (끝점에 삼각 머리). label 은 선 위 가운데."""
    c = s.shapes.add_connector(1, x0, y0, x1, y1)
    c.line.color.rgb, c.line.width = color, Pt(1.75)
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle")
    if label:
        text(s, (x0 + x1) / 2 - label_w / 2, min(y0, y1) - Inches(0.34), label_w, Inches(0.3), label, size=12,
             bold=True, color=color, align=PP_ALIGN.CENTER)


def connector(s, x0, y0, x1, y1):
    """조직도 연결선: 얇고(0.75pt) 진한 회색."""
    ln = s.shapes.add_connector(1, x0, y0, x1, y1).line
    ln.color.rgb, ln.width = DARK, Pt(0.75)


def note_line(s, y, label, body, size=13):
    """본문 아래 한 줄 메모: 파란 머리말 + 내용."""
    text(s, M, y, W - 2 * M, Inches(0.4), [[(label + "  ", {"bold": True, "color": BLUE})] + runs_of(body)], size=size)


def save(path):
    # Windows PowerPoint 는 소수 좌표(W / 2 같은 나눗셈 결과)를 "복구 필요"로 본다. 저장 전에 한 번 반올림한다.
    for sl in prs.slides:
        for el in sl._element.iter():
            for k in ("x", "y", "cx", "cy", "w", "h"):
                v = el.get(k)
                if v and "." in v:
                    el.set(k, str(round(float(v))))
        assert all(sh.top + sh.height <= BODY_BOTTOM + Emu(1) for sh in sl.shapes), "본문이 footer 자리를 침범함"
    prs.save(path)
    print(f"wrote {path}, {len(prs.slides)} slides")


# ---------- 예시 장표 (자기 내용으로 바꾼다) ----------
def s_cover():
    cover("과제 이름 기획안", "한 줄로 쓴 과제 설명")


def s_before_after():  # 전/후 2 컬럼
    s = slide("추진 방향", "기존 방식의 **반복 작업**을 줄이고 **공통 도구**로 전환")
    two_col(s, "기존", [
                "담당자마다 **수작업 반복**, 건당 ⟨확인⟩분",
                "판단 기준이 사람에게만 있어 __담당자가 바뀌면 사라짐__",
                "대상이 늘면 인력도 같은 비율로 증가"],
            "변경", [
                "작업 절차를 **공통 도구**로 수행",
                "판단 기준을 문서와 데이터로 축적",
                "위험한 조작은 __담당자 승인__ 후 실행"],
            h=BODY_H - Inches(0.6), before=True)
    note_line(s, BODY_BOTTOM - Inches(0.4), "운영", "**매주 미팅**으로 진행사항 확인, 월간 보고")


def s_flow_table():  # 흐름 상자 + 표
    s = slide("진행 단계", "**4단계**로 진행, 3단계에서 **담당자 확인**")
    flow(s, [("① 수집", "작업 기록 모으기"), ("② 정리", "절차와 판단 기준 정리"),
             ("③ 확인", "담당자 검토와 승인"), ("④ 적용", "현업 사용, 피드백 반영")],
         BODY_TOP, strong=(2,))
    rows = [("단계", "산출물", "시점"),
            ("수집", "작업 기록 ⟨확인⟩건", "1분기"),
            ("정리", "절차 문서, 판단 기준표", "2분기"),
            ("확인", "승인된 절차", "2분기"),
            ("적용", "현업 사용 결과와 개선 목록", "3분기")]
    table(s, rows, BODY_TOP + Inches(1.8), [Inches(2.4), Inches(6.9), Emu(W - 2 * M - Inches(9.3))],
          bold_col0=True, shade=lambda r: r[0] == "확인")


def s_diagram():  # 노드와 화살표
    s = slide("구성", "**공통 도구**가 두 경로로 대상 시스템을 다룸")
    nw, nh = Inches(2.6), Inches(1.6)
    y = BODY_TOP + Inches(0.4)
    mid = y + nh / 2
    node(s, M, y, nw, nh, "공통 도구", "절차 실행\n결과 확인", strong=True)
    node(s, (W - nw) / 2, y, nw, nh, "대상 시스템", "화면과 파일")
    node(s, W - M - nw, y, nw, nh, "결과물", "담당자 승인 후 적용", fill=ORANGE)
    arrow(s, M + nw + Inches(0.1), mid, (W - nw) / 2 - Inches(0.1), mid, "① 조작")
    arrow(s, (W + nw) / 2 + Inches(0.1), mid, W - M - nw - Inches(0.1), mid, "② 생성")
    two_col(s, "공통 도구가 하는 일", ["절차 실행과 기록", "결과 확인, 이상하면 **중단**"],
            "담당자가 하는 일", ["절차 검토와 __승인__", "사용 후 피드백"],
            y=BODY_TOP + Inches(2.5), h=BODY_H - Inches(2.5), size=14, arrow=False)


SLIDES = [s_cover, s_before_after, s_flow_table, s_diagram]

if __name__ == "__main__":
    for make in SLIDES:
        make()
    save(Path(__file__).resolve().parent / "deck.pptx")
```

### 3. 장표를 추가하는 방법

1. `s_이름()` 함수를 하나 만들고 첫 줄에서 `slide("제목", "요약문")` 을 부른다. 요약문의 중요 문구는 `**` 로 감싼다
2. 본문은 `BODY_TOP` 과 `BODY_BOTTOM` 을 기준으로 놓는다. 숫자 좌표를 직접 쓰지 않으면 본문 영역을 바꿀 때 두 값만 고치면 된다
3. 함수를 `SLIDES` 목록에 넣는다
4. 실행한다. 본문이 6.95 inch 아래로 내려가면 `save` 가 오류를 낸다

### 4. 마스터 슬라이드가 다를 때 고칠 값

| 바꾸고 싶은 것 | 고칠 값 |
|----------------|---------|
| 제목 위치 | `TITLE_X`, `TITLE_Y` |
| 요약문 높이 | `MSG_Y` |
| 본문이 시작하는 높이 | `BODY_TOP` |
| footer 와 겹침 | `BODY_BOTTOM` 을 줄임 |
| 파랑 색 값 | `BLUE` |

### 5. 확인

- Windows PowerPoint 에서 열어 "복구" 안내가 뜨지 않는지 본다
- 마스터 슬라이드에 붙여 넣고 제목, footer 와 겹치지 않는지 본다
- LibreOffice 로 PDF 를 만들어 한 장씩 훑어볼 수도 있다: `soffice --headless --convert-to pdf deck.pptx`. 글꼴이 달라서 줄바꿈은 Windows 와 조금 다르게 나온다

### 6. 에이전트에게 맡길 때

다른 저장소에서 코딩 에이전트에게 이렇게 요청한다.

```text
pptx-template.md 의 배치와 색 규칙, 시작 코드를 그대로 써서 <문서 경로> 내용을 pptx 로 만들어 줘.
장표마다 요약문 한 줄을 넣고 중요 문구는 ** 로 표시해 줘. footer, 제목 아래 선, 쪽 번호는 넣지 마.
```

## 참고 자료 (References)

- [python-pptx 문서](https://python-pptx.readthedocs.io/)
- 이 템플릿을 실제로 쓴 예: 같은 폴더의 `build_pptx.py` (deck 3종을 만드는 스크립트)

## 관련 문서

- [ARC TF 작업공간 README](../README.md)
