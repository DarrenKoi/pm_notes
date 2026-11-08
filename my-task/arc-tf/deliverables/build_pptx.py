# -*- coding: utf-8 -*-
"""Auto Recipe Creation TF 독자별 deck (.pptx) 3종 생성.

    python my-task/arc-tf/deliverables/build_pptx.py          # 3종 모두
    python my-task/arc-tf/deliverables/build_pptx.py boss     # team | boss | exec 중 하나

  boss  담당 임원 보고 (arc-tf-proposal.pptx): 01 기획안 요약 + 공통 플랫폼 구조 1장
  team  팀원용 (arc-tf-team-full.pptx): boss deck + 기술 설계(07) 상세 6장
  exec  최고 임원 보고 (arc-tf-exec-summary.pptx): 08 한 장 요약, 4장

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
LANE = [RGBColor(0xFD, 0xEB, 0xD9), BLUE_PALE, RGBColor(0xEE, 0xEE, 0xEE)]

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


def s_cover(title="Auto Recipe Creation TF 구성 기획안",
            sub="R&D MI 현업 엔지니어들과 만드는 레시피 셋업 자동화 공통 플랫폼"):  # 표지
    s = prs.slides.add_slide(BLANK)
    s.notes_slide.notes_text_frame.text = f"{title}을 보고드리겠습니다."
    text(s, Inches(1.0), Inches(2.45), W - Inches(2.0), Inches(1.0), title, size=40, bold=True)
    box(s, Inches(1.0), Inches(3.62), W - Inches(2.0), Inches(0.03), fill=GRAY)
    gradient_bar(s, Inches(1.0), Inches(3.6), Inches(4.2), Inches(0.07))
    text(s, Inches(1.0), Inches(3.95), W - Inches(2.0), Inches(0.5), sub, size=18, color=SUB)
    text(s, Inches(1.0), Inches(4.55), W - Inches(2.0), Inches(0.4), "2026-12", size=16, color=BLUE, bold=True)
    footer(s)

def s_ask_groups():  # 요청: 지정군
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

def s_problem():  # 문제
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

def s_direction():  # 핵심 개발 방향
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

def s_effects():  # 효과
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

def s_assets():  # 이미 확보한 것
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

def s_scope():  # 범위
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

def s_team():  # TF 구성
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

def s_kanban():  # 일정 칸반
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

def s_check_db():  # 확인 필요 사항
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

def s_safety():  # 안전과 위험
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

def s_decision():  # 결정 요청
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

def s_appendix_asks():  # 별첨
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
    s = slide("공통 플랫폼 구조: Offline 지식화와 Online 실행",
              "녹화에서 승인된 절차를 만들고, 실제 화면을 확인하며 승인된 조작만 한 단계씩 실행함",
              "공통 플랫폼은 두 갈래로 만듭니다. 왼쪽 Offline은 엔지니어 작업 녹화에서 절차를 뽑고 담당 엔지니어가 승인해 지식 자산으로 쌓는 부분으로, 2027년 관찰과 지식화 단계입니다. 오른쪽 Online은 현재 화면을 확인하고 비슷한 사례를 찾아 AI가 다음 조작을 제안하면, 서버가 규칙으로 다시 확인한 뒤 승인된 조작 하나만 실행합니다. 2027년에는 화면에 입력하지 않고 제안만 해서 판단 정확도를 먼저 재고, 화면 입력은 2028년 보조 단계부터 엽니다. 장비와 무관한 부분은 한 번만 만들고, 장비별 절차와 판정 기준만 장비군 담당이 채웁니다.")
    two_col(s, "Offline: 지식화 (2027)", [
                "엔지니어 작업 **녹화 → 화면 변화 → 의미 단계**",
                "여러 녹화를 비교해 공통 경로와 분기 후보를 찾음",
                "__분기 조건은 담당 엔지니어가 승인__, 승인된 것만 씀",
                "절차, 조작 단위, 시연, 원본 증거를 버전별로 보관"],
            "Online: 실행 (2027 제안부터 2029 자율까지)", [
                "현재 화면 확인 → 비슷한 사례 검색 → **AI는 제안만**",
                "서버 **규칙 확인**이 권한, 허용 범위를 다시 판단",
                "승인된 조작 **1개씩** 실행하고 결과 확인 후 다음 단계",
                "__불명확하면 멈추고 엔지니어 호출__, 덮어쓰기, 적용 기본 금지"],
            h=Inches(3.9), size=15)
    note_line(s, Inches(6.2), "공통화", "장비와 무관한 녹화 처리, 저장소, 판단, 안전 관문은 플랫폼 팀이 1번 개발, 장비별 절차와 판정 기준은 장비군 담당", size=13)
    footer(s)


def s_review():  # Codex 설계 검토 결과 (team)
    s = slide("기술 설계 검토: Codex Video-to-Workflow 설계 반영",
              "구조는 채택하고, 첫 Online 대상과 검증 환경은 TF 조건에 맞춰 조정함",
              "Codex가 만든 Video-to-Workflow 설계 22장을 검토했습니다. Offline과 Online 분리, 판단과 규칙 확인과 실행의 분리, 안전 관문, 미사용 시연 평가는 그대로 채택합니다. 다만 Codex 예시의 첫 대상인 Baseline 복사와 파라미터 수정은 기술 검증 예시로만 쓰고, 첫 제품 대상은 측정 위치와 Align 셋업으로 유지합니다. 이 점은 Codex와도 확인했습니다. 별도 시뮬레이터가 없으므로 격리 GUI 환경은 녹화본 평가, 입력 없는 Shadow, 하루 1시간 감독하 실장비 시험으로 대신합니다. 검토 중에 PRD가 2027년 반자동 실행을 요구해 기획안과 어긋난 것을 찾아 PRD를 고쳤습니다.")
    two_col(s, "채택", [
                "**Offline 지식화 / Online 실행** 분리",
                "판단(LLM), __규칙 확인(서버)__, 실행(조작 단위) 3단 분리",
                "관찰 사실, 추정 행동, 승인 규칙을 다른 상태로 저장",
                "안전 관문 G1~G3, 미사용 시연 평가와 지표 6개"],
            "조정", [
                "첫 Online 대상: **측정 위치, Align 셋업 유지** (Baseline 복사는 검증 예시)",
                "용어: Workflow → **절차** (분기, 완료 조건 포함)",
                "격리 GUI 환경 → 녹화본 평가 + Shadow + __하루 1시간 감독하 실장비__",
                "PRD 일정 수정: 화면 입력은 **2028 보조**부터"],
            size=14)
    footer(s)


def s_offline():  # Offline 파이프라인 (team)
    s = slide("Offline 지식화: 녹화에서 승인된 절차까지 6단계",
              "AI 추출물은 매크로가 아니라 검토할 절차 초안이며, 분기 조건은 담당 엔지니어가 승인함",
              "Offline은 여섯 단계입니다. 녹화를 초당 1~2장으로 훑고 변화 구간만 원본으로 다시 봅니다. 라이브 영역과 커서는 빼고 조작 영역 변화만 이벤트로 묶습니다. VLM은 관찰한 사실과 추정한 행동을 나눠 적고 모든 판단을 원본 프레임에 연결합니다. 클릭 여러 개를 시작과 완료 조건이 있는 의미 단계로 묶고, 여러 녹화를 정렬해 분기 후보를 찾습니다. 분기 조건은 담당 엔지니어가 승인한 것만 씁니다. 키 입력을 기록하지 않으므로 복원할 수 없는 구간은 미관찰로 표시하고 추정으로 메우지 않습니다.")
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
    s = slide("Online 실행: 한 단계씩 확인하며 실행, 판단과 실행 분리",
              "LLM은 제안만, 서버가 규칙으로 다시 확인, 실행은 지정된 조작 단위 1개만",
              "Online 실행 루프입니다. 화면을 관찰하고 현재 상태를 인식한 뒤, 장비와 화면 버전이 맞는 시연만 골라 비슷한 사례를 찾습니다. LLM은 다음 분기와 조작 단위를 제안만 하고, 서버의 규칙 확인이 권한, 허용 분기, 인자, 값과 단위를 다시 봅니다. 통과하면 조작 단위 하나만 실행하고 사후조건을 확인한 뒤 체크포인트를 저장합니다. 모르거나 불일치하거나 시간이 초과되면 멈추고 엔지니어에게 넘깁니다. 승인 절차서와 허용 범위는 신뢰 저장소에서만 읽고, 화면 안의 글자는 데이터로만 취급합니다.")
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
    s = slide("안전 관문: 실행 전, 사본 저장, 운영 적용을 나눠 확인",
              "조작별 승인 표는 누가 승인하는가, G1~G3는 승인 전후에 무엇을 확인하는가",
              "안전 관문은 세 개입니다. G1은 실행 전에 장비, 세션, 권한, 버전을 확인하고 원본을 백업합니다. G2는 작업용 사본을 고치고 저장할 때 허용 필드와 값만 바뀌었는지, 저장 후 다시 읽어 차이를 확인합니다. G3 운영 적용은 담당 엔지니어가 백업과 차이, 검증 증거를 보고 별도로 승인합니다. 장비 불일치, 미승인 분기, 상태 불명, 금지 필드 변경, 검증 실패는 즉시 중단 조건입니다. 기본 정책은 덮어쓰기 금지, 적용 금지입니다.")
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
    s = slide("평가 방법: 미사용 시연으로 판단을 먼저, 실장비는 감독하에",
              "지표 정의만 정함, 수치 목표는 시범 녹화 실측 후 담당 엔지니어와 합의",
              "평가는 지식 자산에 쓴 시연과 평가용 시연을 나눠 정보 누수를 막습니다. 시점 t를 평가할 때는 그 이전 관찰만 Agent에 줍니다. 지표는 여섯 개이며, 단계 추출 F1과 상태, 분기 정확도는 2027년 12월 관문에, 미승인 행동 제안률은 2028 진입 판단에, 검증된 작업 성공률은 2028 보조 단계에 씁니다. 지금은 정의만 있고 실측값은 없습니다. 녹화본 재생으로 화면 조작 성공률을 주장하지 않습니다.")
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
    note_line(s, Inches(6.1), "검증 순서", "녹화본 평가 → **입력 없는 Shadow** → 하루 1시간 감독하 실장비 (격리 GUI 환경 대체)", size=13)
    footer(s)


def s_milestones():  # 단계별 구현 순서 (team)
    s = slide("단계별 구현 순서: Codex 마일스톤 M1~M5와 TF 일정",
              "기획안 일정은 그대로, 각 단계는 증거를 확인한 뒤에만 다음 범위로 확대",
              "Codex 설계의 다섯 마일스톤을 TF 일정에 맞췄습니다. M1 데이터와 정답 구축은 온보딩과 상반기 녹화 축적이며, 온보딩이 끝났다고 통과가 아니라 핵심 분기와 예외 표본을 확보해야 통과입니다. M2 Offline 추출은 3분기 절차화, M3 판단 입력과 재생 평가는 4분기 지식 자산과 돌파구 추천입니다. M4 Shadow와 감독하 실장비는 2028 보조, M5 제한 운영 검토는 2028년 말부터 2029년입니다. M5는 검토 단계이며 자율 완주를 보장하지 않습니다. 첫 범위는 장비와 화면 버전 1종, 레시피 유형 1종입니다.")
    rows = [("단계", "산출물", "TF 시기", "다음 단계 통과 조건"),
            ("M1 데이터, 정답", "녹화 목록, 메타데이터, 라벨 기준", "2027 상반기", "핵심 분기, 예외 표본 확보"),
            ("M2 Offline 추출", "의미 단계, 분기 후보", "2027 3분기", "담당 엔지니어 검수"),
            ("M3 판단, 재생 평가", "승인 절차, JSON 계약, 시연 검색", "2027 4분기", "미사용 시연 평가"),
            ("M4 Shadow, 실장비", "제안 → 작업용 사본 실행", "2028 보조", "규칙 차단, 저장 차이, 중단, 재개"),
            ("M5 제한 운영 검토", "백업, 차이, 감사, 복구 증거", "2028 말~2029", "담당 엔지니어 승인")]
    table(s, rows, M, COL_Y, [Inches(2.7), Inches(3.9), Inches(2.0), Emu(W - 2 * M - Inches(8.6))], size=13,
          row_h=Inches(0.55), bold_col0=True, shade=lambda r: r[2].startswith("2027"))
    note_line(s, Inches(5.75), "첫 범위", "장비, 화면 버전 1종 + 레시피 유형 1종, 대상 구간은 **측정 위치, Align 셋업**", size=13)
    note_line(s, Inches(6.2), "주의", "M5는 제한 운영 __검토__ 단계, 2029 자율 완주를 보장하지 않음", size=13)
    footer(s)


# ---------- 최고 임원용 (08 한 장 요약) ----------
def s_exec_onepage():
    s = slide("한 장 요약: 숙련자 노하우를 회사 자산으로",
              "장비가 늘어도 셋업 인력을 같은 비율로 늘리지 않는 구조를 만들고자 함",
              "레시피 셋업은 숙련 엔지니어가 원격 화면에서 손으로 반복하는 작업이고, 한 건에 평균 30~60분이 듭니다. 판단 기준이 사람에게만 있어 담당자가 바뀌면 노하우가 사라지고, 장비가 늘면 인력도 같이 늘어야 합니다. 그래서 엔지니어 작업 녹화에서 절차와 판단 근거를 뽑아 회사 지식 자산으로 쌓고, AI가 승인된 절차를 한 단계씩 수행하게 하려 합니다. 위험 조작과 저장은 엔지니어가 승인합니다. 장비마다 따로 만들지 않고 공통 플랫폼 하나 위에 장비별 절차만 얹습니다.")
    two_col(s, "문제", [
                "레시피 셋업은 숙련자의 **수작업 반복**, 레시피당 평균 30~60분",
                "판단 기준이 사람에게만 있어 __담당자가 바뀌면 사라짐__",
                "장비, 제품이 늘면 셋업 인력도 같은 비율로 증가",
                "Align Fail 알람 시 사람이 올 때까지 장비 정지"],
            "해결 방향", [
                "작업 녹화에서 절차, 판단 근거를 추출해 **지식 자산**으로 축적",
                "AI가 승인된 절차를 __한 단계씩__ 수행, 불명확하면 멈춤",
                "위험 조작, 저장은 **엔지니어 승인**",
                "**공통 플랫폼** 하나 위에 장비별 절차만 추가"],
            size=16)
    footer(s)


def s_exec_roadmap():
    s = slide("단계별 타겟: 검증 증거를 확인하며 단계적으로 확대",
              "2027 상반기 Align Fail 무인 대응부터, 2029 AI 셋업 진행까지",
              "단계별 타겟입니다. 2027년 상반기에는 CD-SEM과 Thickness의 Align Fail 무인 대응으로 장비 가동 시간을 회수합니다. 연말까지 1차 3개 장비군의 셋업 노하우를 지식 자산으로 쌓습니다. 2028년에는 엔지니어 앞에서 반복 입력을 대행하고, 2029년에는 AI가 셋업을 진행하고 특이 사항 때만 엔지니어가 개입합니다. 효과는 엔지니어 직접 투입 시간으로 확인하며, 기준선은 2027년 2월에 실측합니다. 단계마다 검증 증거를 확인한 뒤에만 다음 단계로 넘어갑니다.")
    flow(s, [("2027 상반기", "Align Fail\n무인 대응"), ("2027 연말", "셋업 노하우\n지식 자산화"),
             ("2028", "반복 입력 대행"), ("2029", "AI가 셋업 진행\n특이 사항만 개입")],
         COL_Y, h=Inches(1.5), fills=[(BLUE, WHITE), (BLUE, WHITE), (MIST, INK), (ORANGE, WHITE)],
         head_size=20, body_size=15)
    rows = [("시기", "회사에 남는 것"),
            ("2027 상반기", "장비 가동 시간 회수"),
            ("2027 연말", "노하우의 조직 자산화"),
            ("2028", "엔지니어 직접 투입 시간 감소"),
            ("2029", "셋업 인력을 장비 수에 비례해 늘리지 않음")]
    table(s, rows, M, Inches(3.85), [Inches(3.0), Emu(W - 2 * M - Inches(3.0))], size=15, row_h=Inches(0.45),
          bold_col0=True)
    note_line(s, Inches(6.3), "측정", "엔지니어 직접 투입 시간, 예정에 없던 개입 횟수. 기준선 2027-02 실측, 측정 전 목표 숫자 없음", size=13)
    footer(s)


BOSS = [s_cover, s_ask_groups, s_problem, s_direction, s_effects, s_assets, s_scope, s_platform, s_team,
        s_kanban, s_check_db, s_safety, s_decision, s_appendix_asks]
TEAM = BOSS[:8] + [s_review, s_offline, s_online, s_gates, s_eval, s_milestones] + BOSS[8:]
EXEC = [lambda: s_cover("Auto Recipe Creation TF", "레시피 셋업 노하우를 회사 자산으로 만드는 공통 플랫폼"),
        s_exec_onepage, s_exec_roadmap, s_decision]
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
