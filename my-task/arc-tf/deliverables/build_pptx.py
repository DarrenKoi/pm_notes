# -*- coding: utf-8 -*-
"""Auto Recipe Creation TF 담당 보고 deck (.pptx) 생성.

    python my-task/arc-tf/deliverables/build_pptx.py

deck 은 01-tf-proposal.md 의 요약이다. 그 문서를 고치면 아래 슬라이드 내용도 같이 고친다
(자동 동기화 안 됨). 필요: python-pptx. 한글 폰트: 맑은 고딕. 16:9, Windows PowerPoint 기준.

회사 스타일:
  Header  제목(좌측) + 2색 구분선(회색 + 빨강/주황 그라데이션)
  Body    중앙 상단 핵심 메시지(파란색 강조) → 2 컬럼 박스(좌: 문제, 기존 / 우: 해결 방향, 개선)
  Footer  좌측 로고 | 중앙 슬로건 | 우측 부서 이름
  강조    **볼드**, __밑줄__ 표기를 본문 문자열에 직접 쓴다

로고, 슬로건, 부서 이름은 회사 정보라 git 밖의 brand_사내용.json 에서 읽는다 (없으면 자리 표시):
  {"slogan": "...", "dept": "...", "logo": "logo_사내용.png"}
"""
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "arc-tf-proposal.pptx"
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
LANE = [RGBColor(0xFD, 0xEB, 0xD9), BLUE_PALE, RGBColor(0xEE, 0xEE, 0xEE)]

W, H = Emu(12192000), Emu(6858000)  # exact 16:9 widescreen
M = Inches(0.5)
MSG_Y, MSG_H = Inches(1.18), Inches(0.68)
COL_Y = Inches(2.08)
COL_H = Inches(4.35)
FOOT_Y = Inches(6.88)

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
    box(s, M, y, Inches(0.08), h, fill=BLUE)
    text(s, M + Inches(0.3), y, W - 2 * M - Inches(0.6), h, msg, size=20, bold=True, color=BLUE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide(title, msg, notes):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    header(s, title)
    if msg:
        message(s, msg)
    s.notes_slide.notes_text_frame.text = notes
    return s


def column(s, x, y, w, h, head, items, accent, size=15, mark="▪ "):
    box(s, x, y, w, Inches(0.5), fill=accent)
    text(s, x + Inches(0.25), y, w - Inches(0.5), Inches(0.5), head, size=17, bold=True, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)
    box(s, x, y + Inches(0.5), w, h - Inches(0.5), fill=WHITE, line=LINE)
    text(s, x + Inches(0.3), y + Inches(0.75), w - Inches(0.6), h - Inches(0.95),
         [[(mark, {"color": accent, "bold": True})] + runs_of(t) for t in items], size=size, gap=0.6)


def two_col(s, left_head, left, right_head, right, y=COL_Y, h=COL_H, size=17):
    """좌: 문제, 기존 (회색) → 우: 해결 방향, 개선 (파랑)."""
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    column(s, M, y, cw, h, left_head, left, SUB, size)
    column(s, M + cw + gap, y, cw, h, right_head, right, BLUE, size)
    a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, M + cw + Inches(0.17), y + h / 2 - Inches(0.25), Inches(0.26), Inches(0.5))
    a.fill.solid()
    a.fill.fore_color.rgb = ORANGE
    a.line.fill.background()
    a.shadow.inherit = False


def note_line(s, y, label, body, size=14):
    text(s, M, y, W - 2 * M, Inches(0.4), [[(label + "  ", {"bold": True, "color": ORANGE})] + runs_of(body)], size=size)


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


# ---------- 1. 표지 ----------
s = prs.slides.add_slide(BLANK)
s.notes_slide.notes_text_frame.text = "Auto Recipe Creation TF 구성 기획안을 보고드리겠습니다."
text(s, Inches(1.0), Inches(2.45), W - Inches(2.0), Inches(1.0), "Auto Recipe Creation TF 구성 기획안",
     size=40, bold=True)
box(s, Inches(1.0), Inches(3.62), W - Inches(2.0), Inches(0.03), fill=GRAY)
gradient_bar(s, Inches(1.0), Inches(3.6), Inches(4.2), Inches(0.07))
text(s, Inches(1.0), Inches(3.95), W - Inches(2.0), Inches(0.5),
     "R&D MI 현업 엔지니어들과 만드는 레시피 셋업 자동화 공통 플랫폼", size=18, color=SUB)
text(s, Inches(1.0), Inches(4.55), W - Inches(2.0), Inches(0.4), "2026-12", size=16, color=BLUE, bold=True)
footer(s)

# ---------- 2. 요청: 지정군 ----------
s = slide("TF 구성 요청: R&D MI 현업 엔지니어들과 만드는 자동화 공통 플랫폼",
          "1차 지정군 3개 장비군으로 공통 플랫폼의 기초 틀을 만들고, 2차 지정군에 실제로 적용함",
          "요청드리는 것은 TF 구성과 지정군 엔지니어들의 협업입니다. 1차 지정군은 CD-SEM, Thickness, VeritySEM입니다. 이미 ARC 작업이 진척 중이고 엔지니어들도 데이터 흐름과 작업 방식을 알고 있어, 공통 플랫폼의 기초 틀을 만들기에 가장 적합합니다. 2차 지정군은 Inspection 장비군으로, 선택 사항입니다. 1차에서 얻은 노하우를 적용해 공통 플랫폼이 실제로 다른 장비군에도 통하는지 확인합니다. 개발은 2명이 각자 업무 시간의 50%를 투입해 실질 1명으로 운영합니다.")
two_col(s, "1차 지정군: 공통 플랫폼의 기초 틀", [
            "**CD-SEM, Thickness, VeritySEM** 담당 엔지니어 TF 조직도 협업",
            "사유: 이미 ARC 작업이 진척 중이며 엔지니어들도 __기본적인 데이터 흐름과 작업 방식을 알고 있음__",
            "요청: 장비군마다 1~2명, **주 5시간**",
            "역할: 셋업 절차와 판정 기준 제공, 작업 녹화, 품질 확인"],
        "2차 지정군 (선택): 공통 플랫폼 실제 적용", [
            "**Inspection 장비군**",
            "1차 지정군 장비로 얻은 작업 노하우를 적용",
            "__공통 플랫폼이 다른 장비군에도 통하는지__ 확인하는 첫 대상",
            "개발 인력: **2명 × 50% = 실질 1명** (개인 과제와 병행, 교육 포함)"])
footer(s)

# ---------- 3. 문제 ----------
s = slide("레시피 셋업의 문제: 숙련자 손에 의존, Align Fail이 나면 장비 정지",
          "레시피 1건 셋업에 30~60분, 판단은 숙련자에게만 있음 (현업 경험치)",
          "레시피 셋업은 레시피 한 건에 평균 30~60분이 듭니다. 현업 경험치입니다. 특히 측정 위치와 Align을 잡는 구간에서 측정, 실패, 재조정을 되풀이합니다. 장비에 조작 API가 없어 원격 화면에서 사람이 직접 해야 하고, 그 판단은 숙련자에게만 있습니다. 정확한 구간 시간은 온보딩 기간에 담당 엔지니어가 직접 재서 확정합니다.")
two_col(s, "현재 (As-is)", [
            "측정 위치, Align을 잡을 때 **측정 → 실패 → 재조정** 반복, 그동안 화면 앞을 떠날 수 없음",
            "판단 기준이 숙련자의 눈과 손에만 있어 사람마다 품질이 다르고, 담당자가 바뀌면 사라짐",
            "장비, 제품이 늘면 셋업 인력도 같은 비율로 늘어야 함",
            "Align Fail이 나면 __사람이 올 때까지 장비가 멈춤__"],
        "개선 방향 (To-be)", [
            "시행착오가 가장 많은 **측정 위치, Align 구간**부터 자동화",
            "숙련자의 조작과 판단을 __지식 자산__으로 남겨 막힌 지점에서 바로 찾아 씀",
            "장비군이 늘어도 **공통 플랫폼** 위에 절차만 추가",
            "Align Fail은 **무인 자동 대응**으로 장비 가동 시간 회수"])
note_line(s, Inches(6.5), "측정", "자동화 대상 구간 실측은 2027-01~02 온보딩 기간에 담당 엔지니어가 직접 수행", size=13)
footer(s)

# ---------- 4. 핵심 개발 방향 ----------
s = slide("핵심 개발 방향: AI가 지식 자산으로 레시피를 셋업할 수 있는가",
          "공통 플랫폼을 통해 AI가 지식 자산화된 마우스, 키보드 조작과 셋업 방법을 수행하는 것",
          "가장 중요한 개발 방향입니다. 엔지니어의 마우스, 키보드 조작과 레시피 셋업 방법을 지식 자산으로 만들고, 공통 플랫폼을 통해 AI가 그것을 그대로 수행할 수 있느냐입니다. 2027년에는 두 가지를 타겟으로 합니다. 상반기에는 CD-SEM과 Thickness의 Align Fail 자동 대응을 완성하고, 연말까지 1차 지정군의 셋업 절차와 노하우를 지식 자산으로 쌓습니다. 셋업 시간 단축은 2028년 보조 단계, Agent가 셋업을 진행하는 자율 단계는 2029년 타겟입니다.")
flow = [("엔지니어 작업", "화면 녹화, 마우스, 키보드 조작"), ("지식 자산", "절차, 노하우, 결정 지점"),
        ("공통 플랫폼", "화면 인식, 입력 실행, 안전 규칙"), ("2029 자율 목표", "Agent가 셋업 진행, 위험 시 엔지니어 개입")]
fw = Inches(2.75)
fg = (W - 2 * M - 4 * fw) / 3
for i, (h_, b_) in enumerate(flow):
    x = M + i * (fw + fg)
    fill, fc = [(MIST, INK), (MIST, INK), (BLUE, WHITE), (ORANGE, WHITE)][i]
    box(s, x, COL_Y, fw, Inches(1.25), fill=fill)
    text(s, x + Inches(0.15), COL_Y + Inches(0.18), fw - Inches(0.3), Inches(0.45), h_, size=18, bold=True,
         color=fc, align=PP_ALIGN.CENTER)
    text(s, x + Inches(0.15), COL_Y + Inches(0.68), fw - Inches(0.3), Inches(0.5), b_, size=12,
         color=fc if i >= 2 else SUB, align=PP_ALIGN.CENTER)
    if i < 3:
        a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, x + fw + fg / 2 - Inches(0.1), COL_Y + Inches(0.45), Inches(0.2), Inches(0.35))
        a.fill.solid(); a.fill.fore_color.rgb = GRAY; a.line.fill.background(); a.shadow.inherit = False
two_col(s, "2027 상반기 타겟 (6월 판정)", [
            "**Align Fail 자동 대응 완성**",
            "CD-SEM, Thickness 실알람 __무인 대응__",
            "장비 가동 시간 회수"],
        "2027 연말 타겟 (12월 판정)", [
            "**지식 자산 축적**",
            "1차 지정군 셋업 절차와 노하우를 축적",
            "__막힌 지점에서 바로 찾아 쓰는__ 돌파구 추천"],
        y=Inches(3.55), h=Inches(2.45), size=15)
note_line(s, Inches(6.18), "이후", "2028 보조 (엔지니어 앞에서 반복 입력 대행, 셋업 시간 단축)  →  2029 자율 (Agent가 셋업 진행)", size=13)
footer(s)

# ---------- 5. 효과 ----------
s = slide("기대 효과와 측정 지표: 타겟 시점별로 확인",
          "측정하기 전에는 목표를 숫자로 정하지 않고, 온보딩 기간에 기준선을 재서 채움",
          "효과는 언제 확인할 수 있는지로 나눴습니다. 2027 상반기에는 Align Fail로 멈추던 장비 시간을 회수합니다. 연말에는 노하우가 절차서로 남고 막힌 지점에서 바로 쓰입니다. 엔지니어 투입과 셋업 시간 감소는 2028년 보조 단계부터입니다. 현재 값은 아직 비어 있는 칸이 많습니다. 측정하기 전에는 목표를 숫자로 정하지 않고, 온보딩 기간에 기준선을 재서 채우겠습니다.")
metrics = [("효과", "측정 지표", "현재", "목표", "타겟 시점"),
           ("장비 가동 시간 회수", "Align Fail 장비 정지 시간 감소분 (월)", "⟨확인⟩", "⟨확인⟩", "2027 상반기"),
           ("지식 자산화", "절차서로 정리된 레시피 유형 수", "0", "장비군마다 ⟨N⟩개", "2027 연말"),
           ("막힘 해소", "돌파구 추천 채택률", "-", "⟨확인⟩%", "2027 연말"),
           ("엔지니어 투입 감소", "건당 직접 투입 시간, 예정 밖 개입", "⟨확인⟩", "⟨확인⟩", "2028"),
           ("셋업 시간 단축", "대상 구간 셋업 경과 시간", "대상 구간 실측 예정", "⟨확인⟩% 단축", "2028"),
           ("품질 균일화", "셋업 뒤 첫 측정 성공률", "⟨확인⟩", "⟨확인⟩", "2029")]
table(s, metrics, M, COL_Y, [Inches(2.6), Inches(4.4), Inches(2.0), Inches(2.0), Emu(W - 2 * M - Inches(11.0))],
      size=14, row_h=Inches(0.5), bold_col0=True, shade=lambda r: r[4].startswith("2027"))
text(s, M, Inches(5.8), W - 2 * M, Inches(0.8),
     [[("측정 기준  ", {"bold": True, "color": ORANGE})] + runs_of("측정 위치와 Align 셋업 구간, 실패와 재작업 포함, 승인 대기 별도 기록. 회사 효과 = __월 셋업 건수 × 건당 직접 투입 감소 시간__"),
      [("확정 시점  ", {"bold": True, "color": ORANGE})] + runs_of("기준선 2027-02 / Align Fail 판정 기준 2027-01 말 / 지식화 목표는 시범 실측 후 확정")],
     size=13, color=SUB, spacing=1.0)
footer(s)

# ---------- 6. 이미 확보한 것 ----------
s = slide("이미 확보한 것: PoC 기반과 현업 자동화 재사용",
          "0에서 시작하지 않음: VLM 기반 PoC와 Thickness PyAutoGUI 자동화를 혼합해 씀",
          "1차 PoC의 확인 범위는 화면 조작과 실장비 반자동 운영입니다. 실알람 무인 대응 완성은 2027 상반기 목표입니다. API 없는 장비 화면을 사내 VLM으로 읽고 조작할 수 있다는 것을 확인했고, 최근 SecureVDI 경유 화면에도 입력할 수 있는 것을 확인했습니다. 그래서 VeritySEM과 Thickness도 같은 방식으로 다룰 수 있습니다. 또 Thickness 담당자들이 PyAutoGUI로 자동화해 둔 작업이 많습니다. 정해진 반복 입력은 이 스크립트를 재사용하고, 화면 인식과 판단이 필요한 구간은 VLM이 맡는 혼합 방식을 검토하겠습니다.")
two_col(s, "확인한 것 (1차 PoC)", [
            "사내 VLM, OCR로 **API 없는 장비 화면**을 보고 조작",
            "VLM은 \"어디를 볼지\", CV는 \"정확히 어디인지\" 역할 분리",
            "기준 이미지로 Align Key 탐색 (오프라인 평가)",
            "반자동 운영: 위치 재조정은 자동, 확정은 엔지니어",
            "__SecureVDI 경유 화면에 입력 가능__ (2026-11 확인)"],
        "혼합 방향: PyAutoGUI + VLM", [
            "Thickness 담당자들이 PyAutoGUI로 자동화해 둔 반복 작업이 많음",
            "정해진 순서의 반복 입력: **기존 스크립트 재사용**",
            "위치 확인, 예외 화면, 막힘: **VLM이 판단**",
            "스크립트 앞뒤에 VLM 화면 확인, __다르면 멈춤__",
            "온보딩 때 목록화, 2027-06 요소 목록에 반영"], size=14)
note_line(s, Inches(6.5), "추가 검증", "SecureVDI 경유 VLM 인식 정확도, PyAutoGUI 혼합 안정성 (2027 상반기 Thickness 운영)", size=13)
footer(s)

# ---------- 7. 범위 ----------
s = slide("자동화 범위: 측정 위치, Align 셋업부터 단계적으로",
          "레시피 셋업 5단계 중 시행착오가 가장 많은 구간만 Agent가 맡고, 위험 조작은 엔지니어가 승인",
          "레시피 셋업은 생성부터 저장까지 다섯 단계입니다. 자동화 대상은 가운데, 측정 위치와 Align 셋업 구간입니다. 2027년에는 이 구간을 관찰하고 지식화하고, 2028년부터 엔지니어 앞에서 Agent가 입력을 대행합니다. Agent는 엔지니어 본인 PC에서 돌고 장비 PC에는 아무것도 설치하지 않습니다. 측정 실행과 저장 같은 위험 조작은 반드시 엔지니어가 승인합니다. Align Fail 무인 대응만 예외로, 위치 재조정과 재측정 재시도만 허용합니다.")
steps = [("레시피 생성, 기본 정보", False), ("측정 위치, Align 셋업", True), ("파라미터 설정", False),
         ("검증 측정 판정", False), ("저장", False)]
sw = Inches(2.3)
sg = (W - 2 * M - 5 * sw) / 4
for i, (t_, agent) in enumerate(steps):
    x = M + i * (sw + sg)
    box(s, x, COL_Y, sw, Inches(0.75), fill=BLUE if agent else MIST)
    text(s, x + Inches(0.1), COL_Y, sw - Inches(0.2), Inches(0.75), t_, size=15, bold=True,
         color=WHITE if agent else INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1.0)
two_col(s, "기존: 다섯 단계 모두 엔지니어 수동", [
            "원격 화면(RCS, SecureVDI)에서 **사람이 직접 조작**, 장비에 조작 API 없음",
            "측정 위치, Align 구간에서 측정, 실패, 재조정 반복",
            "Align Fail이 나면 사람이 올 때까지 대기"],
        "개선: 대상 구간만 Agent, 단계적으로 이관", [
            "**2027 관찰과 지식화 → 2028 입력 대행 → 2029 자율 목표**",
            "__위험 조작(측정 실행, 삭제)과 저장 직전에 엔지니어 승인__",
            "Align Fail 무인 허용: 레시피 변경 없는 위치 재조정과 재측정만",
            "엔지니어 PC에서 실행, **장비 PC 설치 없음**"],
        y=Inches(3.05), h=Inches(3.4), size=14)
footer(s)

# ---------- 8. TF 구성 ----------
s = slide("TF 구성: 엔진은 공통 플랫폼 팀, 절차는 담당 엔지니어",
          "장비마다 따로 만들지 않고, 공통 엔진 위에 장비군별 절차와 판정 기준을 얹음",
          "장비마다 자동화를 따로 만들면 화면 인식, 조작, 안전 장치를 장비 수만큼 다시 만들게 됩니다. 반대로 개발자가 모든 장비의 레시피 절차를 알 수는 없습니다. 그래서 공통 플랫폼 팀이 엔진을 만들고, 장비군 담당 엔지니어들이 절차와 판정 기준을 맡습니다. 2차 지정군 Inspection은 선택이며, 참여하면 1월에 함께 합류해 미리 학습하고 작업을 녹화해 둡니다.")
top_w = Inches(8.4)
tx = (W - top_w) / 2
box(s, tx, COL_Y, top_w, Inches(1.0), fill=BLUE)
text(s, tx, COL_Y + Inches(0.1), top_w, Inches(0.45), "TF 리드 + 공통 플랫폼 팀 (개발 2명, 각 50%)", size=19,
     bold=True, color=WHITE, align=PP_ALIGN.CENTER)
text(s, tx, COL_Y + Inches(0.55), top_w, Inches(0.4), "화면 인식, 조작 엔진, 안전 규칙, 승인 화면, VLM 연동, 교육", size=14,
     color=WHITE, align=PP_ALIGN.CENTER)
cells = [("1차 지정군", "CD-SEM", "TF 리드 직접 담당", True), ("1차 지정군", "Thickness", "MI 엔지니어, PyAutoGUI 공유", True),
         ("1차 지정군", "VeritySEM", "MI 엔지니어", True), ("2차 지정군 (선택)", "Inspection", "학습, 작업 녹화", False)]
cw = Inches(2.85)
gap = (W - 2 * M - 4 * cw) / 3
bus_y = COL_Y + Inches(1.3)
cell_y = COL_Y + Inches(1.6)
for x0, y0, x1, y1 in ((W / 2, COL_Y + Inches(1.0), W / 2, bus_y), (M + cw / 2, bus_y, W - M - cw / 2, bus_y)):
    s.shapes.add_connector(1, x0, y0, x1, y1).line.color.rgb = GRAY
for i, (a_, b_, c_, first) in enumerate(cells):
    x = M + i * (cw + gap)
    s.shapes.add_connector(1, x + cw / 2, bus_y, x + cw / 2, cell_y).line.color.rgb = GRAY
    box(s, x, cell_y, cw, Inches(0.42), fill=SUB if first else GRAY)
    text(s, x, cell_y, cw, Inches(0.42), a_, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, x, cell_y + Inches(0.42), cw, Inches(1.0), fill=WHITE, line=LINE)
    text(s, x, cell_y + Inches(0.52), cw, Inches(0.45), b_, size=18, bold=True, align=PP_ALIGN.CENTER)
    text(s, x, cell_y + Inches(0.95), cw, Inches(0.4), c_, size=12, color=SUB, align=PP_ALIGN.CENTER)
note_line(s, Inches(5.85), "상반기", "TF 리드는 **Align Fail** (CD-SEM 겸임), 개발 1명은 녹화분 확보와 수집 기반")
note_line(s, Inches(6.25), "하반기", "TF 리드는 **절차화와 지식 자산**, 개발 1명은 상시 관찰 모드와 운영 유지")
footer(s)

# ---------- 9. 일정 칸반 ----------
s = slide("2027 일정: 상반기 Align Fail 완성, 연말 지식 자산 축적",
          "6월 관문에서 Align Fail 무인 대응을, 12월 관문에서 지식 자산을 판정받음",
          "2027년 일정을 트랙별로 보여 드립니다. 위 줄은 Align Fail입니다. 1분기에 CD-SEM 반자동 운영을 시작하고 2분기에 CD-SEM과 Thickness를 무인으로 전환해 6월에 판정받습니다. 가운데 줄은 관찰과 지식화입니다. 온보딩과 녹화로 시작해 하반기에 절차서와 지식 자산을 만들고 12월에 판정받습니다. 아래 줄은 선택 사항인 2차 지정군입니다. Align Fail은 플랫폼 부품으로 만들어, 6월에 필요했던 요소를 정리해 장기 계획에 반영합니다. Align Fail 성과가 기준에 못 미쳐도 지식화 관문과 안전 요건을 채우면 2028 진입을 막지 않고, 미달분은 보완이나 축소로 결정합니다. 관문마다 리드가 결과를 정리하고 담당 엔지니어 품질 확인을 거쳐 월간 보고에서 승인받습니다.")
periods = ["2026-12", "2027 1분기", "2027 2분기", "2027 3분기", "2027 4분기"]
lanes = [("Align Fail", [["TF 승인", "PC 자원 실측", "녹화 DB 접근 확인"],
                         ["CD-SEM 입력 시험", "CD-SEM 반자동", "Thickness 인식 구축"],
                         ["CD-SEM 무인 전환", "Thickness 무인 전환", "★ 6월 관문"],
                         ["무인 운영 유지", "요소 목록 반영"], []]),
         ("관찰과\n지식화", [["최소 녹화기 준비"], ["온보딩 8주", "기준선 측정", "녹화 시작"],
                            ["녹화 축적", "화면 영역 지도"], ["상시 관찰 모드", "절차서 초안", "노하우 질문"],
                            ["지식 자산 DB", "돌파구 추천", "★ 12월 관문"]]),
         ("2차 지정군\n(선택)", [["참여 결정"], ["온보딩"], ["작업 녹화"], ["절차서 정리"], ["플랫폼 적용 준비"]])]
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
     [[("6월 관문  ", {"bold": True, "color": ORANGE})] + runs_of("자동 해결률 기준 충족, __오측정 0건__, 최소 평가 건수와 기간 충족"),
      [("12월 관문  ", {"bold": True, "color": ORANGE})] + runs_of("장비군마다 절차서 ⟨N⟩개, 돌파구 추천 채택률, 엔지니어 PC 자원 한도 충족"),
      [("2028 진입  ", {"bold": True, "color": ORANGE})] + runs_of("지식화 관문 통과와 입력 실행기 안전 요건 충족")],
     size=12, color=SUB, spacing=1.0, gap=0.3)
footer(s)

# ---------- 10. 확인 필요 사항 ----------
s = slide("확인 필요 사항: 장비 화면 녹화분 DB 직접 접근",
          "녹화분 DB 직접 접근 가능 여부를 2026-12까지 R&D MI H/W 담당자와 확정",
          "R&D MI H/W 담당자께 확인이 필요한 사항이 하나 있습니다. 장비 화면 녹화분 DB에 직접 접근할 수 있는지입니다. 가능하면 녹화분을 바로 받아 써서 개발 속도가 빨라집니다. 불가능하면 RCS에서 GUI 자동화로 녹화분을 내려받아야 합니다. 속도가 느리고, 다운로드 받는 자동화 작업이 별도로 필요합니다. 또 스크린샷과 녹화 영상을 매우 많이 쓰기 때문에 스토리지 소모가 큽니다. 2027년에는 할당 자원 안에서 보존 기한을 정해 운영하고, 사용량을 실측해 추후 스토리지 자산 확보를 요청드리겠습니다.")
two_col(s, "불가능하면: RCS에서 GUI 자동화로 다운로드", [
            "RCS 화면에서 GUI 자동화로 녹화분을 내려받아야 함",
            "**속도가 느림**",
            "__다운로드 받는 자동화 작업 별도 필요__",
            "관찰 일정에 여유를 두고 야간에 처리"],
        "가능하면: 녹화분 DB 직접 접근", [
            "녹화분을 **바로 받아 사용**",
            "__개발 속도가 빨라짐__",
            "다운로드 자동화 작업 불필요",
            "확인 담당: R&D MI H/W 담당자, 기한 2026-12"],
        h=Inches(3.3))
box(s, M, Inches(5.6), W - 2 * M, Inches(0.85), fill=MIST)
text(s, M + Inches(0.3), Inches(5.6), W - 2 * M - Inches(0.6), Inches(0.85),
     [[("스토리지  ", {"bold": True, "color": ORANGE})] + runs_of("스크린샷, 녹화 영상으로 __소모량이 매우 큼__. 2027년은 할당 자원과 보존 기한으로 운영, 사용량 실측 후 **하반기 스토리지 자산 확보 요청**")],
     size=14, anchor=MSO_ANCHOR.MIDDLE)
footer(s)

# ---------- 11. 안전과 위험 ----------
s = slide("안전과 위험: 위험 조작은 엔지니어 승인, 무인 운영은 Align Fail 재조정만",
          "Agent가 생산 레시피를 잘못 건드리지 않도록 승인, 중단, 금지 조작을 먼저 정함",
          "가장 걱정하실 부분은 Agent가 생산 레시피를 잘못 건드리는 것이라고 생각합니다. 측정 실행, 저장, 삭제 같은 위험 조작은 반드시 엔지니어가 승인하고, 엔지니어는 언제든 즉시 멈출 수 있습니다. Align Fail 무인 대응도 위치 재조정과 재측정 재시도만 허용합니다. 운영상 큰 위험은 겸직 시간과 개발 인력입니다. 개발 인력이 실질 1명이라 상반기는 Align Fail을 우선하고, 매월 진척을 보고해 늦어지면 범위 축소를 요청드리겠습니다.")
two_col(s, "위험", [
            "**오조작**: Agent가 생산 레시피를 잘못 변경",
            "**개발 인력 실질 1명**: 일정 지연",
            "**겸직 시간 부족**: 담당 엔지니어 참여 저조",
            "**장비군마다 다른 화면, 절차**"],
        "대응", [
            "위험 조작, 저장 전 __엔지니어 승인__, 즉시 중단 수단, 무인 전환은 장비별 승인",
            "상반기 Align Fail 우선, 매월 진척 보고, __늦으면 범위 축소__",
            "주 5시간을 기획안에 명시, 월간 보고에 참여 시간 실적 보고",
            "공통 엔진과 장비별 절차를 분리, 1차는 3개 장비군으로 제한"],
        h=Inches(3.65), size=14)
note_line(s, Inches(5.95), "가드레일", "Align Fail 무인 운영에서 저장, 삭제, 모드 변경, 레시피 수정 금지. 불확실하면 자체 중단하고 엔지니어 호출. **오측정 0건**")
footer(s)

# ---------- 12. 결정 요청 ----------
s = slide("결정 요청",
          "2027-01부터 TF 운영, 지정군 엔지니어 협업, 개발 인력 배정을 승인해 주시기 바랍니다",
          "다시 한 번 요청드립니다. 2027년 1월부터 TF 운영과 지정군 엔지니어들의 협업, 개발 인력 배정을 승인해 주십시오. 승인해 주시면 1월에 온보딩을 시작하고, 기준선은 2월에 측정하며, 효과 목표는 지표별 확정 일정에 맞춰 보고드리겠습니다.")
two_col(s, "12월 결정 요청 사항", [
            "**2027-01~12 ARC TF 운영 승인**",
            "**CD-SEM, Thickness, VeritySEM 담당 지정**: 장비군별 1~2명, __주 5시간__",
            "**TF 리드 포함 개발 2명 배정**: 각 업무 시간 50%, 교육 포함",
            "선택: 2차 지정군 Inspection 장비군 참여 여부"],
        "승인 후 진행", [
            "2027-01: 온보딩 시작, 녹화 DB 접근 방식 확정",
            "2027-02: 장비군별 __기준선 측정__",
            "2027-06: Align Fail 무인 대응 관문",
            "2027-12: 지식 자산 관문, 효과 목표는 지표별 확정 일정에 맞춰 보고"])
footer(s)

# ---------- 13. 별첨 ----------
s = slide("별첨: 요청 사항 상세", None,
          "요청 사항은 일곱 가지입니다. 12월에 결정을 부탁드리는 것은 위 세 가지, TF 구성과 지정군 엔지니어 협업, 개발 인력입니다. 인프라와 서버, 검증 장비는 1월 가동 전까지 협의하고, 스토리지는 사용량을 실측한 뒤 하반기에 요청드리겠습니다.")
asks = [("시점", "요청"),
        ("2026-12", "ARC TF 구성 승인"),
        ("2026-12", "1차 지정군(CD-SEM, Thickness, VeritySEM) 담당 엔지니어 협업 (장비군마다 1~2명, 주 5시간). 2차 지정군 Inspection은 선택"),
        ("2026-12", "공통 플랫폼 팀 개발 인력 2명 (TF 리드 포함, 1인당 50%, 실질 1명, 교육 포함)"),
        ("2027-01 전", "사내 추론 인프라 사용 보장, 추가 비용 승인 주체 지정"),
        ("2027-01 전", "사내 서버 자원 할당 (저장소, 검색, DB, 캐시, 새 서버 없음)"),
        ("2027-01 전", "비생산 검증용 장비 사용 인가 (1차 지정군마다 하루 1시간)"),
        ("2027 하반기", "녹화 스토리지 자산 확보 (사용량 실측 근거, 예상 용량 ⟨확인⟩)")]
table(s, asks, M, Inches(1.3), [Inches(2.1), Emu(W - 2 * M - Inches(2.1))], size=14, row_h=Inches(0.6),
      shade=lambda r: r[0] == "2026-12")
text(s, M, Inches(6.3), W - 2 * M, Inches(0.35), "음영 행: 12월에 결정을 요청하는 항목", size=12, color=SUB)
footer(s)

# Windows PowerPoint flags non-integer EMU (e.g. x="4673498.33") as needing repair.
# Divisions like W / 2 yield floats, so round every coordinate once here.
for sl in prs.slides:
    for el in sl._element.iter():
        for k in ("x", "y", "cx", "cy", "w", "h"):
            v = el.get(k)
            if v and "." in v:
                el.set(k, str(round(float(v))))
prs.save(OUT)
print(f"wrote {OUT.name}, {len(prs.slides)} slides")
