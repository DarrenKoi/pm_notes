# -*- coding: utf-8 -*-
"""Auto Recipe Creation TF 독자별 deck (.pptx) 3종 생성.

    python my-task/arc-tf/deliverables/build_pptx.py          # 3종 모두
    python my-task/arc-tf/deliverables/build_pptx.py boss     # team | boss | exec 중 하나

  boss  담당 임원 보고 (arc-tf-proposal.pptx): 01 기획안 요약 (PoC 시사점부터) + 공통 플랫폼 구조 1장
  team  팀원용 (arc-tf-team-full.pptx): boss deck + 기술 설계(07) 상세 5장
  exec  최고 임원 보고 (arc-tf-exec-summary.pptx): 08 한 장 요약, 5장 (PoC 시사점 포함)

deck 은 01, 07, 08 .md 의 요약이다. 그 문서를 고치면 아래 슬라이드 내용도 같이 고친다
(자동 동기화 안 됨). 슬라이드 하나 = 함수 하나, 맨 아래 DECKS 가 deck 별 순서다. 필요: python-pptx. 한글 폰트: 맑은 고딕. 16:9, Windows PowerPoint 기준.

회사 스타일:
  Header  제목(좌측) + 2색 구분선(회색 + 빨강/주황 그라데이션)
  Body    중앙 상단 핵심 메시지(파란색 강조) → 2 컬럼 박스(좌: 문제, 기존 / 우: 해결 방향, 개선)
  Footer  좌측 로고 | 중앙 슬로건 | 우측 부서 이름
  강조    **볼드**, __밑줄__ 표기를 본문 문자열에 직접 쓴다

로고, 슬로건, 부서 이름은 회사 정보라 git 밖의 brand_사내용.json 에서 읽는다 (없으면 자리 표시):
  {"slogan": "...", "dept": "...", "logo": "logo_사내용.png"}
"""
import json
from lxml import etree
from pptx.oxml.ns import qn
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
BRAND_FILE = HERE / "brand_사내용.json"
BRAND = {"slogan": "⟨슬로건⟩", "dept": "⟨부서명⟩", "logo": None}
if BRAND_FILE.exists():
    BRAND.update(json.loads(BRAND_FILE.read_text(encoding="utf-8")))

FONT = "맑은 고딕"
INK = RGBColor(0x26, 0x26, 0x26)
SUB = RGBColor(0x59, 0x59, 0x59)
GRAY = RGBColor(0xBF, 0xBF, 0xBF)
LINE = RGBColor(0xD0, 0xD0, 0xD0)
MIST = RGBColor(0xF4, 0xF5, 0xF7)
BLUE = RGBColor(0x1F, 0x5A, 0xA6)
BLUE_PALE = RGBColor(0xE8, 0xF0, 0xFA)
RED = RGBColor(0xD7, 0x26, 0x3D)
ORANGE = RGBColor(0xF3, 0x8B, 0x1F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TBD = ORANGE  # ⟨확인⟩ 표시 색
LANE = [MIST, WHITE, MIST]  # 칸반 레인: 무채색 교대

W, H = Emu(12192000), Emu(6858000)  # exact 16:9 widescreen
M = Inches(0.5)
MSG_Y, MSG_H = Inches(1.18), Inches(0.68)
COL_Y = Inches(2.08)
COL_H = Inches(4.35)
FOOT_Y = Inches(6.88)

prs = BLANK = None  # new_deck() 가 deck 마다 다시 만든다


def new_deck():
    global prs, BLANK
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    prs.part._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldSz').attrib.pop('type', None)
    BLANK = prs.slide_layouts[6]

MARK = re.compile(r"(\*\*.+?\*\*|__.+?__)")


# ---------- helpers ----------
def runs_of(t):
    """'**볼드**', '__밑줄__' 표기를 (text, opts) 조각으로 나눈다."""
    out = []
    for part in MARK.split(t):
        if not part:
            continue
        if part.startswith("**"):
            out.append((part[2:-2], {"bold": True}))
        elif part.startswith("__"):
            out.append((part[2:-2], {"underline": True}))
        else:
            out.append((part, {}))
    return out


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.15, gap=0.45, name=None):
    """paras: str 또는 문단 목록. 문단은 str(표기 해석) 또는 (text, opts) 목록."""
    box = slide.shapes.add_textbox(x, y, w, h)
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [paras] if isinstance(paras, str) else paras
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if i:
            p.space_before = Pt(size * gap)
        for t, opt in (runs_of(para) if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.underline = opt.get("underline", False)
            f.color.rgb = opt.get("color", TBD if "⟨" in t else color)
    return box


def box(slide, x, y, w, h, fill=MIST, line=None, shape=MSO_SHAPE.RECTANGLE, radius=0.08):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def gradient_bar(slide, x, y, w, h):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.gradient()
    s.fill.gradient_angle = 0
    st = s.fill.gradient_stops
    st[0].color.rgb, st[0].position = RED, 0
    st[1].color.rgb, st[1].position = ORANGE, 1.0
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def header(s, title):
    text(s, M, Inches(0.3), W - 2 * M, Inches(0.62), title, size=26, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    box(s, M, Inches(1.0), W - 2 * M, Inches(0.03), fill=GRAY)
    gradient_bar(s, M, Inches(0.98), Inches(3.4), Inches(0.07))


def footer(s):
    n = len(prs.slides)
    box(s, M, FOOT_Y - Inches(0.08), W - 2 * M, Emu(9525), fill=LINE)
    if BRAND["logo"] and (HERE / BRAND["logo"]).exists():
        s.shapes.add_picture(str(HERE / BRAND["logo"]), M, FOOT_Y, height=Inches(0.36))
    else:
        text(s, M, FOOT_Y, Inches(2), Inches(0.36), "⟨로고⟩", size=11, bold=True, color=SUB, anchor=MSO_ANCHOR.MIDDLE)
    text(s, M + Inches(3), FOOT_Y, W - 2 * M - Inches(6), Inches(0.36), BRAND["slogan"], size=11, color=SUB,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, W - M - Inches(3), FOOT_Y, Inches(3), Inches(0.36),
         [[(BRAND["dept"], {"color": SUB}), (f"   {n}", {"color": GRAY})]], size=11,
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def message(s, msg, y=MSG_Y, h=MSG_H):
    """중앙 상단 핵심 메시지 (파란색 강조)."""
    box(s, M, y, W - 2 * M, h, fill=BLUE_PALE)
    text(s, M + Inches(0.3), y, W - 2 * M - Inches(0.6), h, msg, size=20, bold=True, color=BLUE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide(title, msg):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    header(s, title)
    if msg:
        message(s, msg)
    return s


def column(s, x, y, w, h, head, items, accent, size=15, mark="▪ "):
    box(s, x, y, w, Inches(0.5), fill=accent)
    text(s, x + Inches(0.25), y, w - Inches(0.5), Inches(0.5), head, size=17, bold=True, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)
    box(s, x, y + Inches(0.5), w, h - Inches(0.5), fill=WHITE, line=LINE)
    text(s, x + Inches(0.3), y + Inches(0.75), w - Inches(0.6), h - Inches(0.95),
         [[(mark, {"color": accent, "bold": True})] + runs_of(t) for t in items], size=size, gap=0.6)


def two_col(s, left_head, left, right_head, right, y=COL_Y, h=COL_H, size=17, arrow=True, before=False):
    """두 컬럼 모두 파랑. before=True면 좌: 기존, 문제 (회색) → 우: 변경, 해결 (파랑)."""
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    column(s, M, y, cw, h, left_head, left, SUB if before else BLUE, size)
    column(s, M + cw + gap, y, cw, h, right_head, right, BLUE, size)
    if not arrow:
        return
    a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, M + cw + Inches(0.17), y + h / 2 - Inches(0.25), Inches(0.26), Inches(0.5))
    a.fill.solid()
    a.fill.fore_color.rgb = GRAY
    a.line.fill.background()
    a.shadow.inherit = False


def node(s, x, y, w, h, title, sub=None, fill=MIST, fc=INK, size=14):
    """그림용 둥근 상자: 제목 + 작은 설명."""
    box(s, x, y, w, h, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    paras = [[(title, {"bold": True, "size": size, "color": fc})]]
    if sub:
        paras += [[(t, {"size": 11, "color": fc if fc == WHITE else SUB})] for t in sub.split("\n")]
    text(s, x + Inches(0.05), y, w - Inches(0.1), h, paras, size=size, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.0, gap=0.15)


def arrow(s, x0, y0, x1, y1, label=None, color=SUB, head=True, label_w=Inches(2.4)):
    """그림용 화살표 (끝점에 삼각 머리). label은 선 위 가운데."""
    c = s.shapes.add_connector(1, x0, y0, x1, y1)
    c.line.color.rgb = color
    c.line.width = Pt(1.75)
    if head:
        etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle")
    if label:
        text(s, (x0 + x1) / 2 - label_w / 2, min(y0, y1) - Inches(0.34), label_w, Inches(0.3), label, size=12,
             bold=True, color=color, align=PP_ALIGN.CENTER)


def note_line(s, y, label, body, size=14):
    text(s, M, y, W - 2 * M, Inches(0.4), [[(label + "  ", {"bold": True, "color": BLUE})] + runs_of(body)], size=size)


def table(s, rows, x, y, widths, size=14, row_h=Inches(0.45), bold_col0=False, shade=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, sum(widths, Inches(0)), row_h * len(rows)).table
    for j, wd in enumerate(widths):
        t.columns[j].width = wd
    for i, row in enumerate(rows):
        hl = shade(row) if (shade and i) else False
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = BLUE if i == 0 else (BLUE_PALE if hl else (MIST if i % 2 == 0 else WHITE))
            c.margin_left = c.margin_right = Inches(0.12)
            tf = c.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = val
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0 or (bold_col0 and j == 0) or hl
            r.font.color.rgb = WHITE if i == 0 else (TBD if "⟨" in val else INK)
    return t


def s_cover(title="Auto Recipe Creation TF 구성 기획안",
            sub="R&D MI 현업 엔지니어들과 만드는 레시피 셋업 자동화 공통 플랫폼"):  # 표지
    s = prs.slides.add_slide(BLANK)
    text(s, Inches(1.0), Inches(2.45), W - Inches(2.0), Inches(1.0), title, size=40, bold=True)
    box(s, Inches(1.0), Inches(3.62), W - Inches(2.0), Inches(0.03), fill=GRAY)
    gradient_bar(s, Inches(1.0), Inches(3.6), Inches(4.2), Inches(0.07))
    text(s, Inches(1.0), Inches(3.95), W - Inches(2.0), Inches(0.5), sub, size=18, color=SUB)
    text(s, Inches(1.0), Inches(4.55), W - Inches(2.0), Inches(0.4), "2026-12", size=16, color=BLUE, bold=True)
    footer(s)

def s_effects():  # 효과
    s = slide("기대 효과와 측정 지표",
              "합류 첫 달에 기준선 측정, 측정 전에는 목표를 숫자로 정하지 않음")
    metrics = [("효과", "측정 지표", "현재", "목표", "타겟 시점"),
               ("지식 자산화", "부분 작업 리스트로 정리한 Recipe 셋업을 Agent가 그대로 수행해 성공한 건수", "0건", "⟨X⟩건", "2027 연말"),
               ("ARC 적용 비율", "Recipe 의뢰 수 대비 ARC를 적용한 Recipe 비율", "⟨확인⟩", "⟨확인⟩%", "2028"),
               ("Agent 개입 범위", "Recipe당 셋업 단계 중 ARC Agent가 진행한 비율", "0%", "Recipe별 ⟨확인⟩%", "2028"),
               ("셋업 시간 단축", "Recipe 1건 셋업 경과 시간", "대상 구간 실측 예정", "⟨확인⟩% 단축", "2028")]
    table(s, metrics, M, COL_Y, [Inches(2.3), Inches(5.3), Inches(1.9), Inches(1.8), Emu(W - 2 * M - Inches(11.3))],
          size=14, row_h=Inches(0.46), bold_col0=True, shade=lambda r: r[4].startswith("2027"))
    text(s, M, Inches(4.75), W - 2 * M, Inches(0.8),
         [[("확정 시점  ", {"bold": True, "color": BLUE})] + runs_of("기준선 선발대 2027-02, 2차 2027-08 / 장비군별 판정 기준 2027-01 말 / 지식화 목표는 시범 실측 후 확정")],
         size=13, color=SUB, spacing=1.0)
    footer(s)

def s_status():  # 2026 추진 현황
    s = slide("2026 ARC 추진 현황",
              "세 장비군이 각자 ARC 추진 중, 공통으로 얻은 Lesson Learned 세 가지")
    gap = Inches(0.3)
    cw = (W - 2 * M - 2 * gap) / 3
    ch = Inches(2.8)
    cols = [("CD 추진 현황", [
                "**Align Fail 자동 대응 Agent** 개발 진행 (Recipe Tuning 포함), 현재 __Offline 테스트 중__",
                "**소형 VLM**(Vision Language Model) 기반 장비 컨트롤 (Computer-use) 기능 구현"]),
            ("Thickness 추진 현황", [
                "의뢰부터 Recipe 생성까지 **ARC Pipeline** 진행",
                "DL 기반 Align Fail 대응 __Offline 테스트 중__",
                "MI 장비군 중 __가장 높은 자동화__ 구현"]),
            ("Inspection 추진 현황", [
                "**DRB(Design Recipe Builder)**를 __Tuning하는 방식__으로 ARC 진행 중 (NAND Tech 1개 대상)"])]
    for i, (head, items) in enumerate(cols):
        column(s, M + i * (cw + gap), COL_Y, cw, ch, head, items, BLUE, size=14)
    ly = COL_Y + ch + Inches(0.12)
    box(s, M, ly, W - 2 * M, Inches(1.7), fill=MIST)
    text(s, M + Inches(0.3), ly + Inches(0.1), W - 2 * M - Inches(0.6), Inches(1.4), [
        [("Lesson Learned", {"bold": True, "color": BLUE})],
        runs_of("1. **생성형 AI 모델만으로 ARC 구현 어려움** (정밀 좌표 클릭, 셋업 절차, 결과 판단, 화면 조작 등의 문제)"),
        runs_of("2. Rule 기반 자동화, 컴퓨터 비전(CV), ML/DL을 **생성형 AI 모델과 함께 쓰는 구조** __필수__"),
        runs_of("3. 생성형 AI 모델에게 **Recipe 작업 방식을 어떻게 학습**시킬지 __구체화 필요__")],
        size=15, gap=0.3)
    footer(s)


def s_assets():  # 업무 전략 1: Agent가 Recipe를 다루는 방식과 공통 과제
    s = slide("업무 전략 1: Recipe 작업 방식",
              "화면 조작과 외부 파일 편집으로 Recipe 작업, 공통 과제는 공통 플랫폼으로 해결")
    y0, nh = COL_Y, Inches(0.68)
    ax, aw = M, Inches(2.5)
    mx, mw = Inches(5.4), Inches(2.9)
    rx, rw = Inches(10.5), W - M - Inches(10.5)
    y1, y2 = y0, y0 + Inches(0.92)
    node(s, ax, y0, aw, Inches(1.6), "ARC Agent", "생성형 AI\n+ 최적화된 소형 모델", fill=BLUE, fc=WHITE, size=17)
    node(s, mx, y1, mw, nh, "장비 화면", "API 없음, 마우스와 키보드로 조작")
    node(s, mx, y2, mw, nh, "Recipe 파일", "장비 밖에서 직접 편집")
    arrow(s, ax + aw + Inches(0.1), y1 + nh / 2, mx - Inches(0.1), y1 + nh / 2, "① 화면 조작 (Computer Use)", color=BLUE)
    arrow(s, ax + aw + Inches(0.1), y2 + nh / 2, mx - Inches(0.1), y2 + nh / 2, "② 외부 파일 편집", color=BLUE)
    node(s, rx, y0, rw, Inches(1.6), "Recipe 생성, 수정", "저장과 적용은\n엔지니어 승인", fill=ORANGE, fc=WHITE, size=16)
    arrow(s, mx + mw + Inches(0.1), y1 + nh / 2, rx - Inches(0.1), y1 + nh / 2, "공통 플랫폼의 기반", label_w=Inches(2.0))
    arrow(s, mx + mw + Inches(0.1), y2 + nh / 2, rx - Inches(0.1), y2 + nh / 2, "장비군별 도구", label_w=Inches(2.0))
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    cy, chh = Inches(3.9), Inches(2.75)
    column(s, M, cy, cw, chh, "Agent가 Recipe를 다루는 방식", [
               "**Computer Use**: VLM(화면 판단), PyAutoGUI(키보드, 마우스 입력), Win32(창 단위 조작)를 조합해 구현",
               "**외부 파일 편집**: 장비 화면을 거치지 않아 __조작 단계 축소__, 장비별 Recipe 파일 구조 해석 필요",
               "**소형 모델**: 기반기술센터에서 __최적화된 소형 모델__을 운영해 자동화 작업에 함께 사용"], BLUE, size=13)
    column(s, M + cw + gap, cy, cw, chh, "공통 플랫폼으로 해결할 공통 과제", [
               "**Agent의 마우스, 키보드 조작**, 최적화된 소형 모델 운영",
               "**녹화 DB 추출**: 엔지니어 작업 녹화를 Agent가 쓰는 지식으로",
               "**기술 부채 해소**: Vibe Coding으로 __원리 확인 없이 진행한 사항__ 정리",
               "장비군마다 따로 하면 __같은 작업 반복__"], BLUE, size=13)
    footer(s)

def s_lessons():  # 시사점
    s = slide("업무 전략",
              "현업 노하우를 AI가 쓰는 구조로 개발자가 구현, 공통 과제는 공통 플랫폼으로 해결")
    two_col(s, "1. 두 역할의 유기적인 협업", [
                "**개발자**: 화면 인식, 조작 엔진, 안전 규칙, 모델 서빙",
                "**현업 엔지니어**: Recipe 셋업 절차, 성공 판정 기준, 실장비 검증, 결과 책임",
                "현업 MI 엔지니어의 **업무 노하우와 Recipe 셋업 방식**을 AI가 활용할 수 있는 구조로 __개발자가 구현__",
                "한쪽만으로는 자동화 대상 선정이나 구현 불가"],
            "2. ARC 공통 과제는 공통 플랫폼과 TF로", [
                "공통 플랫폼으로: **Agent의 마우스, 키보드 조작**",
                "함께 필요한 업무: **최적화된 소형 모델 운영** (기반기술센터), **녹화 DB 추출**",
                "**기술 부채 해소**: Vibe Coding으로 __원리 확인 없이 진행한 사항__ 대부분",
                "TF로 공유: **주간 미팅**에서 개발 노하우 공유 (자동화 방법, 사내 DB 테이블 등)",
                "장비군마다 따로 하면 __같은 작업 반복__"],
            h=Inches(3.7), size=14, arrow=False)
    note_line(s, Inches(6.0), "결론", "ARC 필수 요소(조작, 소형 모델 운영, 녹화 추출)는 **공통 플랫폼으로 해결**, 개발 노하우는 **주간 미팅에서 공유**, 장비 노하우와 판정은 __장비군 담당자의 역할__", size=13)
    footer(s)

def s_why_not_llm():  # LLM 모델만으로는 안 되는 이유
    s = slide("별첨: LLM 모델만으로 안 되는 이유",
              "모델은 \"화면을 보고 다음 할 일을 판단\"하는 한 부분, 나머지는 플랫폼이 채워야 함")
    rows = [("항목", "모델만 쓸 때의 한계", "플랫폼이 채우는 것"),
            ("좌표 정밀도", "같은 화면도 물을 때마다 위치가 다름", "CV 정밀 좌표, OCR 결과 확인"),
            ("장비 지식(노하우)", "사내 계측 화면, 셋업 절차, 판정 기준을 모름", "녹화에서 뽑고 엔지니어가 승인한 절차"),
            ("성공/실패 여부", "조작 결과를 스스로 확인 못 함, 실패를 놓치면 오측정", "규칙 확인, 결과 검증, 불확실하면 멈춤"),
            ("진행 상태", "수십 번의 조작 중 어디까지 했는지 놓침", "절차 상태, 체크포인트, 실행 로그"),
            ("화면 조작", "장비에 조작 API가 없음", "화면 캡처, 입력 실행기, 즉시 중단")]
    table(s, rows, M, COL_Y, [Inches(2.4), Inches(5.5), Emu(W - 2 * M - Inches(7.9))], size=14, row_h=Inches(0.55),
          bold_col0=True)
    note_line(s, Inches(5.65), "모델 교체", "최신 VLM은 __공통 평가 세트__로 재서 판단 부분만 교체. 오른쪽 칸은 모델이 좋아져도 그대로 필요", size=13)
    footer(s)

def s_where():  # 별도 PC 대신 엔지니어 PC 설치형
    s = slide("업무 전략 2: 엔지니어 PC 설치",
              "별도 운영 PC 대신 엔지니어가 이미 쓰는 PC에 공통 플랫폼 설치")
    two_col(s, "처음 계획: 별도 운영 PC", [
                "장비군, 엔지니어가 늘 때마다 **PC 추가 확보 절차** (자산 신청, 자리 확보 등)",
                "PC마다 __장비 접속 권한__ 별도 발급",
                "운영 PC **별도 관리**로 인한 부담",
                "__아무도 보지 않는 화면__에서 Agent 운영, 결과 확인 지연"],
            "변경: 엔지니어 PC에 설치형 공통 플랫폼", [
                "이미 사용 중인 PC에 **S/W 설치**하는 형태",
                "기존 계정과 접속 경로 그대로 사용, __사내 배포로 일괄 갱신__",
                "**엔지니어가 원할 때 Agent 실행**, 보는 화면에서 즉시 확인, 중단 (__Human on Command__)",
                "부작용: 입력 때만 Agent가 화면을 __일시 제어__ (몇 초)"],
            h=Inches(2.85), size=14, before=True)
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    dy, dh = Inches(5.45), Inches(0.95)
    my = dy + dh / 2
    sw, bw = Inches(1.0), Inches(1.95)
    ag = (cw - 2 * sw - bw) / 2  # 화살표 구간
    pad = Inches(0.08)
    # 처음 계획: 엔지니어 → 별도 운영 PC → 장비
    x = M
    node(s, x, dy, sw, dh, "엔지니어", "자기 PC")
    arrow(s, x + sw + pad, my, x + sw + ag - pad, my, "확인 지연", label_w=ag)
    node(s, x + sw + ag, dy, bw, dh, "별도 운영 PC", "Agent 실행\n아무도 보지 않는 화면", fill=GRAY)
    arrow(s, x + sw + ag + bw + pad, my, x + cw - sw - pad, my, "조작", label_w=ag)
    node(s, x + cw - sw, dy, sw, dh, "장비")
    # 변경: 사내 서버 → 엔지니어 PC(Agent 설치) → 장비
    x = M + cw + gap
    node(s, x, dy, sw, dh, "사내 서버", "모델 판단")
    arrow(s, x + sw + pad, my, x + sw + ag - pad, my, "판단", color=BLUE, label_w=ag)
    node(s, x + sw + ag, dy, bw, dh, "엔지니어 PC", "Agent 설치\n보면서 실행, 즉시 중단", fill=BLUE, fc=WHITE)
    arrow(s, x + sw + ag + bw + pad, my, x + cw - sw - pad, my, "조작", color=BLUE, label_w=ag)
    node(s, x + cw - sw, dy, sw, dh, "장비")
    footer(s)

def s_team():  # TF 구성 (조직도)
    s = slide("TF 구성",
              "엔진은 공통 플랫폼 팀, 절차는 장비군 담당 엔지니어")
    top_w = Inches(6.4)
    tx = (W - top_w) / 2
    box(s, tx, COL_Y, top_w, Inches(0.72), fill=BLUE)
    text(s, tx, COL_Y + Inches(0.06), top_w, Inches(0.36), "ARC TF (TF 리드 총괄)", size=17, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER)
    text(s, tx, COL_Y + Inches(0.4), top_w, Inches(0.3), "명단 12월 지정 / 월간 임원 보고", size=12,
         color=WHITE, align=PP_ALIGN.CENTER)
    goal = lambda t: [[("2027 상반기 목표", {"bold": True, "color": ORANGE})], t]
    groups = [("공통 플랫폼 팀: 개발 약 5명 규모", BLUE, [
                  ("백엔드 (2명)", ["서버, 판단, 규칙 확인", "녹화 DB 추출", "LLMOps"]),
                  ("프론트엔드 (2명)", ["설치형 에이전트", "대화형 서비스 구현", "Workflow 모니터링", "승인 화면"]),
                  ("현업 엔지니어", ["1차 MI팀 소속", "개발 함께 진행"])]),
              ("장비군 담당: 장비군마다 1~2명", BLUE, [
                  ("CD-SEM\n(1차)", goal("Align Fail 자동 대응 및 Recipe Tuning(이미지 변경) 구현")),
                  ("Thickness\n(1차)", goal("Film 10 Layer 이하 의뢰 Recipe 90% 이상 생성 자동화")),
                  ("Inspection\n(1차)", goal("Optical Inspection 장비군 전체 디바이스향 DRB 기반 ARC 확장")),
                  ("다른 MI 장비군\n(2차)", [ "상반기 인터뷰, Workflow 조사", "7월 본격 참여"])])]
    gg = Inches(0.4)
    gw = (W - 2 * M - gg) / 2
    bus_y, gy = COL_Y + Inches(0.92), COL_Y + Inches(1.12)
    bw_g = Inches(0.12)
    line = lambda x0, y0, x1, y1: setattr(s.shapes.add_connector(1, x0, y0, x1, y1).line.color, "rgb", GRAY)
    line(W / 2, COL_Y + Inches(0.72), W / 2, bus_y)
    line(M + gw / 2, bus_y, W - M - gw / 2, bus_y)
    for gi, (head, accent, members) in enumerate(groups):
        gx = M + gi * (gw + gg)
        line(gx + gw / 2, bus_y, gx + gw / 2, gy)
        box(s, gx, gy, gw, Inches(0.42), fill=accent)
        text(s, gx, gy, gw, Inches(0.42), head, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        bw = (gw - (len(members) - 1) * bw_g) / len(members)
        for k, (name, lines) in enumerate(members):
            x, y = gx + k * (bw + bw_g), gy + Inches(0.52)
            dim = gi == 1 and k == 3  # 2차 준비 단계
            box(s, x, y, bw, Inches(2.15), fill=MIST if dim else WHITE, line=LINE)
            box(s, x, y, bw, Inches(0.06), fill=GRAY if dim else accent)
            text(s, x + Inches(0.06), y + Inches(0.12), bw - Inches(0.12), Inches(0.5), name.split("\n"), size=12, bold=True, gap=0,
                 align=PP_ALIGN.CENTER)
            text(s, x + Inches(0.06), y + Inches(0.6 if gi == 0 else 0.72), bw - Inches(0.12), Inches(1.4), lines, size=12, color=INK,
                 align=PP_ALIGN.CENTER, spacing=1.0, gap=0.25)
    note_line(s, Inches(6.2), "운영", "**매주 미팅**에서 __개발 노하우 공유__, 현재와 앞으로의 __진행사항 체크__ (2차 장비군도 합류 시점부터 참여)", size=13)
    footer(s)

def s_kanban():  # 일정 칸반
    s = slide("2027 일정",
              "상반기 1차 장비군 목표와 플랫폼 베타, 하반기 영역 확장")
    periods = ["2026-12", "2027 1분기", "2027 2분기", "2027 3분기", "2027 4분기"]
    lanes = [("1차 장비군\n목표", [["TF 승인", "녹화 DB 접근 확인"],
                             ["CD-SEM Align Fail 시험", "Thickness 대상 Recipe", "Inspection DRB 시험"],
                             ["장비군별 자동화 운영", "★ 목표 달성 확인"],
                             ["자동 운영 유지", "요소 목록 반영"], ["자동 운영 유지", "2차에 노하우 적용"]]),
             ("공통\n플랫폼", [["최소 녹화기 준비"], ["플랫폼 설계", "모델 서빙, 평가 세트", "주간 미팅 시작"],
                              ["녹화 추출, 라벨링", "RAG 구축, 베타 배포"], ["Workflow 생성, 실행", "준수 확인, 수정", "수정 내용 RAG 보완"],
                              ["2차 장비군 RAG 반영", "★ 목표 달성 확인"]]),
             ("영역 확장\n(2차)", [["2차 명단 지정"], ["인터뷰, Workflow 조사"], ["요구사항 보완"], ["본격 참여, 주간 미팅 합류"], ["화면 영역 지도"]])]
    lw = Inches(1.45)
    colw = (W - 2 * M - lw) / len(periods)
    hy = COL_Y - Inches(0.05)
    for j, p in enumerate(periods):
        x = M + lw + j * colw
        box(s, x + Inches(0.03), hy, colw - Inches(0.06), Inches(0.38), fill=BLUE)
        text(s, x, hy, colw, Inches(0.38), p, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    card_h, card_g = Inches(0.33), Inches(0.05)
    lane_hs = [Inches(1.3), Inches(1.3), Inches(0.5)]
    y = hy + Inches(0.45)
    for li, ((name, cols_), lh) in enumerate(zip(lanes, lane_hs)):
        box(s, M, y, W - 2 * M, lh, fill=LANE[li])
        text(s, M + Inches(0.12), y, lw - Inches(0.2), lh, name.split("\n"), size=13, bold=True,
             anchor=MSO_ANCHOR.MIDDLE, spacing=1.0, gap=0)
        for j, items in enumerate(cols_):
            x = M + lw + j * colw + Inches(0.07)
            for k, it in enumerate(items):
                cy = y + Inches(0.08) + k * (card_h + card_g)
                gate = it.startswith("★")
                box(s, x, cy, colw - Inches(0.14), card_h, fill=ORANGE if gate else WHITE, line=None if gate else LINE)
                text(s, x + Inches(0.1), cy, colw - Inches(0.3), card_h, it, size=12, bold=gate,
                     color=WHITE if gate else INK, anchor=MSO_ANCHOR.MIDDLE)
        y += lh + Inches(0.06)
    text(s, M, y + Inches(0.04), W - 2 * M, Inches(0.85),
         [[("6월 목표  ", {"bold": True, "color": BLUE})] + runs_of("장비군별 목표 달성, 공통 플랫폼 베타 배포 (1차 장비군 PC에서 무장애 운영)"),
          [("12월 목표  ", {"bold": True, "color": BLUE})] + runs_of("부분 작업 리스트대로 Agent가 수행에 성공한 Recipe 셋업 **⟨X⟩건**, 2차 장비군 합류"),
          [("2028 진입  ", {"bold": True, "color": BLUE})] + runs_of("12월 목표 달성과 화면 입력 안전 기준 충족")],
         size=12, color=SUB, spacing=1.0, gap=0.3)
    footer(s)

def s_safety():  # 안전과 리스크
    s = slide("안전과 리스크",
              "위험 조작은 엔지니어 승인, 자동 운영은 승인된 조작만")
    two_col(s, "리스크", [
                "**오조작**: Agent가 생산 레시피를 잘못 변경",
                "**시스템 변경**: 장비 S/W, 화면이 바뀌면 기존 절차가 맞지 않음",
                "**방화벽 해결 불가**: 필요한 접속 경로가 열리지 않을 수 있음",
                "**장비군마다 다른 화면과 절차**: 공통 부분을 찾기 어려울 수 있음"],
            "대응", [
                "위험 조작, 저장 전 __엔지니어 승인__, 즉시 중단 수단, 자동 전환은 장비별 승인",
                "화면, 절차가 바뀌면 이전 검증 결과를 쓰지 않고 __재검증__",
                "필요한 접속 경로를 __사전 확인__, 불가하면 가능한 경로 안에서 범위 조정 **⟨확인⟩**",
                "공통 엔진과 장비별 절차를 분리, __1차 3개 장비군으로 먼저 다지고__ 베타 뒤 확장"],
            h=Inches(3.65), size=14)
    footer(s)

def s_decision():  # 결정 요청
    s = slide("결정 요청",
              "ARC TF 구성(조직도) 승인, 2027-01 운영 시작")
    two_col(s, "12월 결정 요청: TF 구성 승인", [
                "**2027-01~12 ARC TF 운영 승인** (1차, 2차 명단 함께 지정)",
                "**1차 CD-SEM, Thickness, Inspection 담당 지정**: 장비군마다 1~2명",
                "**개발 인력 4명 지정**: 백엔드 2, 프론트엔드 2",
                "**2차 다른 MI 장비군 담당 지정**: 장비군마다 1~2명, __7월부터 본격 참여__"],
            "함께 필요한 자원", [
                "2027-01 전: 공통 프로젝트를 위한 **HCP 환경 및 자원** (현 H200 2장, 2027 하반기 4장 이상 확보)",
                "2027-01 전: **RCS 녹화 스토리지 접근** 여부 확인, 영상 적재를 위한 **별도 스토리지 DB** 확보"],
            h=Inches(3.4), size=16, arrow=False)
    footer(s)

# ---------- 공통 플랫폼 구조 (07 기술 설계 요약) ----------
def flow(s, items, y, h=Inches(1.1), fills=None, head_size=15, body_size=11):
    """가로 흐름 상자: items = [(제목, 설명)], 상자 사이에 화살표."""
    n = len(items)
    gap = Inches(0.3)
    fw = (W - 2 * M - gap * (n - 1)) / n
    for i, (h_, b_) in enumerate(items):
        x = M + i * (fw + gap)
        fill, fc = (fills[i] if fills else (MIST, INK))
        box(s, x, y, fw, h, fill=fill)
        text(s, x + Inches(0.08), y + Inches(0.1), fw - Inches(0.16), Inches(0.45), h_, size=head_size, bold=True,
             color=fc, align=PP_ALIGN.CENTER, spacing=1.0)
        text(s, x + Inches(0.08), y + Inches(0.55), fw - Inches(0.16), h - Inches(0.6), b_.split("\n"), size=body_size,
             color=fc if fc == WHITE else SUB, align=PP_ALIGN.CENTER, spacing=1.0, gap=0)
        if i < n - 1:
            a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, x + fw + gap / 2 - Inches(0.08), y + h / 2 - Inches(0.15),
                                   Inches(0.16), Inches(0.3))
            a.fill.solid(); a.fill.fore_color.rgb = GRAY; a.line.fill.background(); a.shadow.inherit = False


def s_platform():  # 공통 플랫폼 구조 (boss, team)
    s = slide("핵심 개발 방향",
              "Computer Use 기반 RAG Workflow 순환 플랫폼")
    flow(s, [("① 녹화 추출, 라벨링", "녹화 비디오에서 작업 내용 추출\n현업 엔지니어가 라벨링"),
             ("② RAG 저장", "Recipe 셋업 지식 DB\n(절차, 노하우, 결정 지점)"),
             ("③ Workflow 생성, 실행", "RAG에서 셋업 Workflow 생성\nAgent가 Computer Use로 실행"),
             ("④ 준수 확인, 수정", "Workflow를 지키는지 확인\n틀리면 엔지니어가 개입, 수정")],
         COL_Y, h=Inches(1.3), fills=[(BLUE, WHITE)] * 4, head_size=16, body_size=12)
    fgap = Inches(0.3)
    fw = (W - 2 * M - fgap * 3) / 4
    c2, c4 = M + (fw + fgap) + fw / 2, M + 3 * (fw + fgap) + fw / 2
    fb, ry = COL_Y + Inches(1.3), COL_Y + Inches(1.5)
    arrow(s, c4, fb, c4, ry, color=ORANGE, head=False)
    arrow(s, c4, ry, c2, ry, color=ORANGE, head=False)
    arrow(s, c2, ry, c2, fb + Inches(0.03), color=ORANGE)
    text(s, M, COL_Y + Inches(1.54), W - 2 * M, Inches(0.36),
         [[("순환  ", {"bold": True})] + runs_of("④에서 엔지니어가 고친 내용은 __②RAG에 다시 저장__, 다음 Workflow에 반영")], size=14, color=BLUE,
         align=PP_ALIGN.CENTER)
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    cy, chh = Inches(3.95), Inches(2.0)
    column(s, M, cy, cw, chh, "공통 플랫폼 팀이 만드는 것", [
               "녹화 내용 추출, **RAG DB**, Workflow 생성",
               "**Computer Use** 실행 엔진 (VLM, PyAutoGUI, Win32)",
               "준수 확인 (__생성형 AI + Rule, CV, ML/DL__), 즉시 중단"], BLUE, size=13)
    column(s, M + cw + gap, cy, cw, chh, "현업 엔지니어가 하는 것", [
               "추출 내용 **라벨링**, Workflow 검토와 승인",
               "Agent 실행 중 __개입, 수정__ / 녹화본으로 미리 실험",
               "실전 사용 후 **개발자에게 피드백** → 플랫폼에 반영"], BLUE, size=13)
    note_line(s, Inches(6.1), "플랫폼의 역할", "2028 보조 (엔지니어 앞에서 반복 입력 대행)  →  2029 자율 (Agent가 셋업 진행)이 될 수 있도록 __점진적 발전__", size=13)
    footer(s)


def s_offline():  # Offline 파이프라인 (team)
    s = slide("Offline 지식화",
              "녹화에서 승인된 절차까지 6단계, AI 추출물은 검토할 절차 초안")
    flow(s, [("전처리", "1~2fps 탐색\n원본 프레임 보존"), ("변화 감지", "조작 영역만\n전후 화면, OCR"),
             ("의미 행동", "관찰 사실과\n추정 행동 분리"), ("의미 단계", "시작, 완료\n조건으로 묶음"),
             ("시연 정렬", "공통 경로와\n분기 후보"), ("분기 승인", "담당 엔지니어\n승인")],
         COL_Y, fills=[(MIST, INK)] * 5 + [(BLUE, WHITE)])
    two_col(s, "자동으로 하는 것", [
                "변화 구간 검출, 전후 화면 묶음 (라이브 영역, 커서, 시계 제외)",
                "VLM의 **관찰 사실 / 추정 행동 / 불확실 항목** 분리 기록",
                "여러 녹화 정렬로 분기 후보 제시",
                "VLM confidence 숫자는 __실행 승인 기준으로 쓰지 않음__"],
            "사람이 정하는 것", [
                "**화면 영역 지도** (라이브 영역, 조작 영역)",
                "의미 단계 경계 검토",
                "__분기 조건 승인__: 영상 하나나 빈도만으로 확정하지 않음",
                "미관찰 구간(키 입력, 뷰어 밖 팝업)은 추정으로 메우지 않음"],
            y=Inches(3.45), h=Inches(3.0), size=13)
    footer(s)


def s_online():  # Online 실행 루프와 3단 분리 (team)
    s = slide("Online 실행",
              "LLM은 제안만, 서버가 규칙으로 다시 확인, 실행은 조작 단위 1개씩")
    flow(s, [("관찰", "화면, 장비 확인"), ("상태 인식", "현재 단계"), ("검색, 판단", "호환 시연 3건\nLLM 제안"),
             ("규칙 확인", "권한, 분기, 인자"), ("실행", "조작 단위 1개"), ("검증", "사후조건\n체크포인트")],
         COL_Y, fills=[(MIST, INK)] * 3 + [(BLUE, WHITE), (MIST, INK), (MIST, INK)])
    rows = [("역할", "하는 일", "하지 않는 일"),
            ("판단 (LLM)", "다음 분기, 조작 단위, 근거 제안", "직접 조작, 사전조건 충족 선언"),
            ("규칙 확인 (서버)", "권한, 허용 분기, 인자, 값과 단위, 절차 버전 재확인. 거부 시 실행 안 함", "판단"),
            ("실행 (조작 단위)", "지정된 조작 1개 실행 후 화면 재관찰", "임의 메뉴 탐색, 추가 수정")]
    table(s, rows, M, Inches(3.45), [Inches(2.6), Inches(6.0), Emu(W - 2 * M - Inches(8.6))], size=13, row_h=Inches(0.5),
          bold_col0=True)
    note_line(s, Inches(5.65), "중단", "모름, 불일치, 시간 초과 → 멈추고 엔지니어 검토. 통신이 끊기면 재시도보다 __실제 결과 조회__ 먼저", size=13)
    note_line(s, Inches(6.1), "입력", "승인 절차서는 신뢰 저장소에서만 읽음. 화면 글자, 검색 문서는 **데이터로만** 취급", size=13)
    footer(s)


def s_gates():  # 안전 관문 (team)
    s = slide("안전 관문",
              "실행 전, 사본 저장, 운영 적용을 나눠 확인")
    rows = [("관문", "시점", "확인할 것"),
            ("G1", "실행 전", "장비, 세션, 권한, 버전 확인 / 원본 백업 / 변경 대상과 범위 확정"),
            ("G2", "작업용 사본 수정, 저장", "허용 필드, 값, 단위 확인 / 범위 밖 변경 차단 / 저장 후 다시 읽어 차이 확인"),
            ("G3", "운영 적용", "담당 엔지니어가 백업, 차이, 검증 증거를 보고 별도 승인")]
    table(s, rows, M, COL_Y, [Inches(1.3), Inches(3.0), Emu(W - 2 * M - Inches(4.3))], size=15, row_h=Inches(0.6),
          bold_col0=True)
    two_col(s, "즉시 중단", [
                "장비 불일치, **미승인 분기**",
                "상태 불명, 금지 필드 변경",
                "__검증 실패__"],
            "기본 정책", [
                "**덮어쓰기 금지, 적용 금지**",
                "같은 요청 재실행 시 사본, 저장 중복 없음 (요청 ID 연결)",
                "감사 기록: 요청, 규칙 버전, 전후 증거, 차이, 승인자"],
            y=Inches(4.75), h=Inches(1.9), size=13)
    footer(s)


def s_eval():  # 평가 지표 (team)
    s = slide("평가 방법",
              "미사용 시연으로 판단을 먼저 평가, 수치 목표는 실측 후 합의")
    rows = [("지표", "정의", "쓰는 관문"),
            ("단계 추출 F1", "담당 엔지니어 정답 대비 단계 정밀도, 재현율", "2027-12"),
            ("상태, 분기 정확도", "맞힌 상태, 분기 / 평가한 상태, 분기 (따로 집계)", "2027-12, 2028 진입"),
            ("미승인 행동 제안률", "미승인 행동 제안 / 전체 제안", "2028 진입"),
            ("안전 중단률, 자동 결정률", "모르는 상황 중 안전 중단 / 전체 중 자동 결정", "2028, 2029"),
            ("검증된 작업 성공률", "사후조건과 정확한 차이 통과 / 시도 작업", "2028 보조"),
            ("소요 시간, 사람 투입", "관찰부터 검증까지 p50, p95 / 검토 시간, 개입 횟수", "2028, 2029")]
    table(s, rows, M, COL_Y, [Inches(3.2), Inches(6.4), Emu(W - 2 * M - Inches(9.6))], size=13, row_h=Inches(0.48),
          bold_col0=True, shade=lambda r: r[2].startswith("2027"))
    note_line(s, Inches(5.65), "누수 방지", "같은 작업, 파생 영상은 한쪽에만. 시점 t 평가에는 __t 이전 관찰만__, 정답 경로는 평가자만", size=13)
    note_line(s, Inches(6.1), "검증 순서", "녹화본 평가 → **입력 없는 Shadow** → 감독하 실장비 (격리 GUI 환경 대체)", size=13)
    footer(s)


def s_stack():  # 구현 형태와 기술 스택 (team)
    s = slide("구현 형태",
              "엔지니어 PC 설치형 에이전트 + 서버 웹앱, 양쪽 Python")
    rows = [("구성 요소", "위치", "언어, 프레임워크"),
            ("화면 캡처, 변화 감지", "엔지니어 PC", "Python + WGC 창 캡처"),
            ("입력 실행, 즉시 중단", "엔지니어 PC", "Python + SendInput, pywin32"),
            ("상태, 제어권 인계 화면", "엔지니어 PC", "트레이 아이콘 + localhost 페이지"),
            ("판단, 규칙 확인", "기존 서버", "Python + 기존 Flask"),
            ("승인 화면, 절차서 검토", "서버 + 브라우저", "Vue 3 + Flask API"),
            ("배포", "엔지니어 PC", "PyInstaller + 회사 서명 + 사내 배포")]
    table(s, rows, M, COL_Y, [Inches(3.6), Inches(2.6), Emu(W - 2 * M - Inches(6.2))], size=13,
          row_h=Inches(0.42), bold_col0=True, shade=lambda r: r[1] == "엔지니어 PC")
    note_line(s, Inches(5.4), "제어권 인계", "알림 → 엔지니어가 넘겨줌 → 대상 창 확인 → 짧은 입력 → 결과 확인 → 반환. __엔지니어 조작 시 즉시 중단__", size=13)
    note_line(s, Inches(5.85), "보류", "C#/.NET은 Python 캡처가 PC 자원 한도를 못 지킬 때의 대안. Electron, Tauri는 쓰지 않음", size=13)
    note_line(s, Inches(6.3), "실측", "입력 중단 신뢰성(선발대 장비군), PC 자원, 회사 보안 프로그램 아래 설치와 롤백", size=13)
    footer(s)


# ---------- 최고 임원용 (08 한 장 요약) ----------
def s_exec_onepage():
    s = slide("한 장 요약",
              "숙련자 노하우를 회사 자산으로, 장비가 늘어도 셋업 인력은 비례해 늘리지 않음")
    two_col(s, "문제", [
                "레시피 셋업은 숙련자의 **수작업 반복**, 레시피당 평균 30~60분",
                "판단 기준이 사람에게만 있어 __담당자가 바뀌면 사라짐__",
                "장비, 제품이 늘면 셋업 인력도 같은 비율로 증가",
                "Align Fail 알람마다 엔지니어가 __하던 일을 멈추고 대응__"],
            "해결 방향", [
                "작업 녹화에서 절차, 판단 근거를 추출해 **지식 자산**으로 축적",
                "AI가 승인된 절차를 __한 단계씩__ 수행, 불명확하면 멈춤",
                "위험 조작, 저장은 **엔지니어 승인**",
                "**공통 플랫폼** 하나 위에 장비별 절차만 추가, 선발대 뒤 확장"],
            size=16, before=True)
    footer(s)


def s_exec_roadmap():
    s = slide("단계별 타겟",
              "2027 1차 장비군 목표와 플랫폼 베타부터, 2029 AI 셋업 진행까지")
    flow(s, [("2027 상반기", "1차 장비군 목표\n플랫폼 베타"), ("2027 하반기", "2차 장비군 합류\n지식 자산화"),
             ("2028", "반복 입력 대행"), ("2029", "1차 장비군\nAI가 셋업 진행")],
         COL_Y, h=Inches(1.5), fills=[(BLUE, WHITE), (BLUE, WHITE), (MIST, INK), (ORANGE, WHITE)],
         head_size=20, body_size=15)
    rows = [("시기", "회사에 남는 것"),
            ("2027 상반기", "다른 장비군이 쓸 공통 플랫폼"),
            ("2027 하반기", "장비군 확장 구조, 노하우의 조직 자산화"),
            ("2028", "Recipe별 셋업의 Agent 진행 비율 증가"),
            ("2029", "셋업 인력을 장비 수에 비례해 늘리지 않음")]
    table(s, rows, M, Inches(3.85), [Inches(3.0), Emu(W - 2 * M - Inches(3.0))], size=15, row_h=Inches(0.45),
          bold_col0=True)
    note_line(s, Inches(6.3), "측정", "Recipe별 셋업 단계 중 Agent가 진행한 비율(%). 기준선은 합류 첫 달에 실측, 측정 전 목표 숫자 없음", size=13)
    footer(s)


BOSS = [s_cover, s_status, s_team, s_assets, s_where, s_platform, s_effects,
        s_kanban, s_safety, s_decision]
TEAM = BOSS[:6] + [s_offline, s_online, s_gates, s_eval, s_stack] + BOSS[6:] + [s_why_not_llm]
EXEC = [lambda: s_cover("Auto Recipe Creation TF", "레시피 셋업 노하우를 회사 자산으로 만드는 공통 플랫폼"),
        s_lessons, s_exec_onepage, s_team, s_exec_roadmap]
DECKS = {"boss": ("arc-tf-proposal.pptx", BOSS), "team": ("arc-tf-team-full.pptx", TEAM),
         "exec": ("arc-tf-exec-summary.pptx", EXEC)}


def build(target):
    out_name, slides = DECKS[target]
    new_deck()
    for make in slides:
        make()
    # Windows PowerPoint flags non-integer EMU (e.g. x="4673498.33") as needing repair.
    # Divisions like W / 2 yield floats, so round every coordinate once here.
    for sl in prs.slides:
        for el in sl._element.iter():
            for k in ("x", "y", "cx", "cy", "w", "h"):
                v = el.get(k)
                if v and "." in v:
                    el.set(k, str(round(float(v))))
    prs.save(HERE / out_name)
    print(f"wrote {out_name}, {len(prs.slides)} slides")


if __name__ == "__main__":
    for target in sys.argv[1:] or DECKS:
        build(target)
