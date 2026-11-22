# -*- coding: utf-8 -*-
"""Auto Recipe Creation TF 독자별 deck (.pptx) 3종 생성.

    python my-task/arc-tf/deliverables/build_pptx.py          # 3종 모두
    python my-task/arc-tf/deliverables/build_pptx.py boss     # team | boss | exec | map 중 하나

  boss  담당 임원 보고 (arc-tf-proposal-rev2.pptx, 이전 보고본 arc-tf-proposal.pptx 는 보존): 01 기획안 요약 (왜 해야 하는가부터) + 핵심 개발 방향 1장 (exec와 같은 합본)
        본 장표 10장(왜 해야 하는가, 왜 TF인가 추가) + 별첨 2장(연도별 대상 장비, 단계별 목표와 업무 전환 효과)
  team  팀원용 (arc-tf-team-full.pptx): boss deck + 업무 전략 1, 2, 공통 플랫폼 구조, 기술 설계(07) 상세 5장
        + BP사 협업 검토, 현업 엔지니어의 참여 방법, 확장성 근거
  map   팀장 피드백 대응표 (arc-tf-feedback-map.pptx): 표지 + 대응표 + 팀장에게 확인할 것. feedback-response-map.md 의 요약
  exec  최고 임원 보고 (arc-tf-exec-summary.pptx): 5장 (2026 추진 현황과 Lesson Learned, 핵심 개발 방향(업무 전략 합본), TF 구성, 개발 로드맵)

deck 은 01, 07, 08 .md 의 요약이다. 그 문서를 고치면 아래 슬라이드 내용도 같이 고친다
(자동 동기화 안 됨). 슬라이드 하나 = 함수 하나, 맨 아래 DECKS 가 deck 별 순서다. 필요: python-pptx. 한글 폰트: 맑은 고딕. 16:9, Windows PowerPoint 기준.

회사 스타일:
  Header  제목(좌측)만. 구분선은 회사 마스터 슬라이드 것을 쓴다
  Body    제목 아래 핵심 메시지 한 줄 → 2 컬럼 박스(좌: 문제, 기존 / 우: 해결 방향, 개선). footer 는 넣지 않는다
          아래 좌표는 예전 배치 기준이고, build 가 본문을 BODY_TOP~BODY_BOTTOM 영역으로 옮기며 세로로 늘린다
  강조    **볼드**, __밑줄__ 표기를 본문 문자열에 직접 쓴다
"""
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

FONT = "맑은 고딕"
# 회사에서 자주 쓰는 색 (2026-11-12 사용자 제공, 색 이름만 받았고 hex 값은 근사치):
#   빨강, 주황, 연한 분홍, 연한 파랑, 짙은 파랑, 베이지, 연노랑, 크림, 흰색, 검정,
#   짙은 회색, 진회색, 밝은 회색, 연회색
# 이 deck 은 그중 몇 개만 쓴다: 표와 텍스트 박스는 연회색(MIST), 밝은 회색(SILVER) 바탕에 검정(INK),
# 강조 상자는 검정(INK) 바탕 + 흰 글자, 나머지 상자는 밝은 회색 + 검정 글자 (2026-11-16 사용자 지시: 짙은 파랑 채움 대신).
# 짙은 파랑(BLUE)은 요약문의 중요 문구와 메모 머리말 같은 작은 글자에만 (글머리 기호는 검정), 연한 파랑(BLUE_PALE)은 표 강조 행, 주황은 관문, 최종 목표, <확인> 자리에만.
# 아직 안 쓰는 색: 연한 분홍 FCE4E4, 연노랑 FFF2CC, 크림 FFF9E6, 베이지 F2E6D8
INK = RGBColor(0x26, 0x26, 0x26)     # 검정 (본문)
SUB = RGBColor(0x59, 0x59, 0x59)     # 짙은 회색 (보조 글자)
DARK = RGBColor(0x40, 0x40, 0x40)    # 진회색 (조직도 연결선)
GRAY = RGBColor(0xBF, 0xBF, 0xBF)    # 구분선, 화살표
LINE = RGBColor(0xD0, 0xD0, 0xD0)    # 박스 테두리
SILVER = RGBColor(0xD9, 0xD9, 0xD9)  # 밝은 회색 (표, 박스 머리줄)
MIST = RGBColor(0xF2, 0xF2, 0xF2)    # 연회색 (표 줄무늬, 텍스트 박스 바탕)
BLUE = RGBColor(0x0B, 0x3C, 0x8F)     # 짙은 파랑 (2026-11-16 더 진하게, 이전 1F5AA6)
BLUE_PALE = RGBColor(0xE8, 0xF0, 0xFA)
RED = RGBColor(0xD7, 0x26, 0x3D)
ORANGE = RGBColor(0xF3, 0x8B, 0x1F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TBD = RED  # <확인> 표시 색 (2026-11-18 주황에서 빨강으로)
LANE = [MIST, WHITE, MIST]  # 칸반 레인: 무채색 교대

W, H = Emu(12192000), Emu(6858000)  # exact 16:9 widescreen
M = Inches(0.5)
MSG_Y, MSG_H = Inches(0.75), Inches(0.45)  # 핵심 메시지: 제목(~0.61) 아래, 본문(BODY_TOP~) 위
COL_Y = Inches(2.08)
COL_H = Inches(4.35)
# 슬라이드 함수의 본문 좌표는 예전 배치(SRC_TOP~SRC_BOTTOM) 기준이다. build 가 저장 직전에 한 번에
# BODY_TOP~BODY_BOTTOM 으로 옮기면서 같은 비율로 늘린다. 본문 영역을 바꾸려면 이 두 값만 고친다.
SRC_TOP, SRC_BOTTOM = COL_Y, Inches(6.7)
BODY_TOP, BODY_BOTTOM = Inches(1.55), Inches(6.95)  # 슬라이드 아래 0.55 inch 는 마스터 footer 자리라 비워 둔다 (7.2 는 footer 를 침범했음)
STRETCH = (BODY_BOTTOM - BODY_TOP) / (SRC_BOTTOM - SRC_TOP)

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
            f.color.rgb = opt.get("color", TBD if "<" in t else color)
    return box


def box(slide, x, y, w, h, fill=MIST, line=None, shape=MSO_SHAPE.RECTANGLE, radius=0.08):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    # 선 없는 박스도 채움색 1pt 선을 둔다: 선은 경계 가운데에 그려져서, 선 있는 박스와 폭이 1pt 어긋나기 때문
    s.line.color.rgb = line or fill
    s.line.width = Pt(1)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def header(s, title):
    # 제목: 회사 마스터 슬라이드에 맞춰 위로 0.27 inch, 왼쪽으로 0.2 inch 옮긴 자리 (2026-11-16).
    # 처음에는 0.4 inch 올렸으나 너무 위라서 조금 내렸다.
    # 0.5 inch 를 다 올리면 26pt 글자 윗부분이 슬라이드 밖으로 나가서 0.4 inch 까지만 올렸다
    text(s, M - Inches(0.2), Inches(0.15), W - 2 * M, Inches(0.46), title, size=26, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    # 제목 아래 2색 구분선은 그리지 않는다 (회사 마스터 슬라이드에 있음, 2026-11-16)


def message(s, msg):
    """제목 바로 아래 핵심 메시지 한 줄 (바탕 없음). 검정 글자, **표시한 중요 문구** 만 파랑. 위치가 고정이라 build 의 본문 이동에서 빠진다."""
    runs = [(part, {"color": BLUE} if opt else {}) for part, opt in runs_of(msg)]  # **중요 문구** 는 파랑
    text(s, M + Inches(0.3), MSG_Y, W - 2 * M - Inches(0.6), MSG_H, [runs], size=20, bold=True, color=INK,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide(title, msg):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    header(s, title)
    if msg:
        message(s, msg)
    return s


def column(s, x, y, w, h, head, items, accent, size=15, mark="▪ ", strong=False):
    """머리줄은 밝은 회색 + 검정 글자. strong=True(전/후 구성의 '후')만 검정 + 흰 글자. 글머리 기호는 검정 (accent 인자는 이제 쓰지 않음)."""
    box(s, x, y, w, Inches(0.5), fill=INK if strong else SILVER)
    text(s, x + Inches(0.25), y, w - Inches(0.5), Inches(0.5), head, size=17, bold=True,
         color=WHITE if strong else INK, anchor=MSO_ANCHOR.MIDDLE)
    box(s, x, y + Inches(0.5), w, h - Inches(0.5), fill=WHITE, line=LINE)
    text(s, x + Inches(0.3), y + Inches(0.75), w - Inches(0.6), h - Inches(0.95),
         [[(mark, {"color": INK, "bold": True})] + runs_of(t) for t in items], size=size, gap=0.6)


def two_col(s, left_head, left, right_head, right, y=COL_Y, h=COL_H, size=17, arrow=True, before=False):
    """두 컬럼 모두 밝은 회색 머리줄. before=True면 우(변경, 해결)만 검정 머리줄로 강조."""
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    column(s, M, y, cw, h, left_head, left, SUB if before else BLUE, size)
    column(s, M + cw + gap, y, cw, h, right_head, right, BLUE, size, strong=before)
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


def table(s, rows, x, y, widths, size=14, row_h=Inches(0.45), bold_col0=False, shade=None, border=False, col0_color=None):
    """border=True 는 표 둘레에 검정 바깥선. col0_color 는 첫 칸 글자색, 이때 <확인> 자리만 주황(칸 전체 아님)."""
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, sum(widths, Inches(0)), row_h * len(rows)).table
    for j, wd in enumerate(widths):
        t.columns[j].width = wd
    for i, row in enumerate(rows):
        hl = shade(row) if (shade and i) else False
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = SILVER if i == 0 else (BLUE_PALE if hl else (MIST if i % 2 == 0 else WHITE))
            c.margin_left = c.margin_right = Inches(0.12)
            tf = c.text_frame
            tf.word_wrap = True
            if border:
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.paragraphs[0].text = ""
            for part in (re.split(r"(<[^>]+>)", val) if col0_color else [val]):
                if not part:
                    continue
                r = tf.paragraphs[0].add_run()
                r.text = part
                r.font.name, r.font.size = FONT, Pt(size)
                r.font.bold = i == 0 or (bold_col0 and j == 0) or hl or bool(col0_color and "<" in part)
                r.font.color.rgb = TBD if (i and "<" in part) else (col0_color if (col0_color and i and j == 0) else INK)
    if border:
        o = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, sum(widths, Inches(0)), row_h * len(rows))
        o.fill.background(); o.line.color.rgb = INK; o.line.width = Pt(1.25); o.shadow.inherit = False
    return t


def s_cover(title="Auto Recipe Creation TF 구성 기획안",
            sub="레시피 셋업 자동화 공통 플랫폼"):  # 표지
    s = prs.slides.add_slide(BLANK)
    text(s, Inches(1.0), Inches(2.95), W - Inches(2.0), Inches(0.8), title, size=40, bold=True, align=PP_ALIGN.CENTER)  # 가로 가운데, 세로는 중앙보다 조금 위
    if sub:
        text(s, Inches(1.0), Inches(3.85), W - Inches(2.0), Inches(0.5), sub, size=18, color=SUB, align=PP_ALIGN.CENTER)

def s_effects():  # 효과
    s = slide("기대 효과와 측정 지표",
              "장비군별 **기준선 측정** 후 **2027-07**에 2028~2029 목표 수치 확정")
    metrics = [("효과", "측정 지표", "현재", "목표", "타겟 시점"),
               ("지식 자산화", "단위 작업(파일 열기, 이미지 확인 등) 목록 중 Agent가 엔지니어 개입 없이 수행에 성공한 비율", "0%", "<X>%", "2027 연말"),
               ("ARC 적용 비율", "Recipe 의뢰 수 대비 ARC를 적용한 Recipe 비율", "<확인>", "1차 장비군 90% 이상 (지향점)", "2029"),
               ("Agent 개입 범위", "Recipe당 셋업 단계 중 ARC Agent가 진행한 비율", "0%", "기준선 대비 <N>배", "2028"),
               ("셋업 시간 단축", "Recipe 1건 셋업 경과 시간", "대상 구간 실측 예정", "<확인>% 단축", "2028")]
    table(s, metrics, M, COL_Y, [Inches(2.3), Inches(5.3), Inches(1.9), Inches(1.8), Emu(W - 2 * M - Inches(11.3))],
          size=14, row_h=Inches(0.46), bold_col0=True, shade=lambda r: r[4].startswith("2027"))
    text(s, M, Inches(4.75), W - 2 * M, Inches(0.8),
         [[("확정 시점  ", {"bold": True, "color": BLUE})] + runs_of("기준선 선발대 2027-02, 2차 2027-08 / 장비군별 판정 기준 2027-01 말 / 지식화 목표는 시범 실측 후 확정 / 2028~2029 목표 수치 2027-07")],
         size=13, color=SUB, spacing=1.0)

def s_status():  # 2026 추진 현황과 Lesson Learned
    s = slide("2026 ARC 추진 현황",
              "장비군별 ARC 추진 현황과 **Lesson Learned**")
    gap = Inches(0.3)
    cw = (W - 2 * M - 2 * gap) / 3
    ch = Inches(2.8)
    cols = [("CD 추진 현황", [
                "**Align Fail 자동 대응 Agent** 개발 진행 (Recipe Tuning 연계), 현재 __Offline 테스트 중__",
                "**소형 VLM**(Vision Language Model) 기반 장비 컨트롤 (Computer-use) 기능 구현"]),
            ("Thickness 추진 현황", [
                "**MI 장비군 중 자동화 완성도 가장 높음**: 의뢰부터 Recipe 생성까지 __ARC Pipeline__ 진행",
                "DL 기반 Align Fail 대응 __Offline 테스트 중__"]),
            ("Inspection 추진 현황", [
                "**DRB(Design Recipe Builder)**를 __Tuning하는 방식__으로 ARC 진행 중 (NAND Tech 1개 대상)",
                "DRB Tuning에 **Rule 기반 Computer-use** 사용",
                "Recipe Setup AI 활용 __DB 구축 완료__ (RAG, 마이다스)"])]
    for i, (head, items) in enumerate(cols):
        column(s, M + i * (cw + gap), COL_Y, cw, ch, head, items, BLUE, size=14)
    ly = COL_Y + ch + Inches(0.12)
    box(s, M, ly, W - 2 * M, Inches(1.7), fill=MIST)
    text(s, M + Inches(0.3), ly + Inches(0.1), W - 2 * M - Inches(0.6), Inches(1.4), [
        [("Lesson Learned", {"bold": True, "color": BLUE})],
        runs_of("1. **생성형 AI 모델만으로 ARC 구현 어려움** (정밀 좌표 클릭, 셋업 절차, 결과 판단, 화면 조작 등의 문제)"),
        runs_of("2. Rule 기반 자동화, 컴퓨터 비전(CV), ML/DL을 생성형 AI 모델과 **함께 쓰는 하이브리드 방식** __필수__"),
        runs_of("3. 엔지니어의 노하우인 **Recipe 작업 방식을 지식 자산화 (DB)**해야 __생성형 AI 모델이 활용 가능__")],
        size=15, gap=0.3)


def s_assets():  # 업무 전략 1: Agent의 Recipe 작업 방식과 공통 과제
    s = slide("업무 전략 1: Recipe 작업 방식",
              "Recipe 작업은 **화면 조작**과 **외부 파일 편집** 두 방식, 공통 과제는 **공통 플랫폼**으로 해결")
    y0, nh = COL_Y, Inches(0.68)
    ax, aw = M, Inches(2.5)
    mx, mw = Inches(5.4), Inches(2.9)
    rx, rw = Inches(10.5), W - M - Inches(10.5)
    y1, y2 = y0, y0 + Inches(0.92)
    node(s, ax, y0, aw, Inches(1.6), "ARC Agent", "생성형 AI\n+ 최적화 소형 모델", fill=INK, fc=WHITE, size=17)
    node(s, mx, y1, mw, nh, "장비 화면", "API 없음, 마우스와 키보드로 조작")
    node(s, mx, y2, mw, nh, "Recipe 파일", "장비 밖에서 직접 편집")
    arrow(s, ax + aw + Inches(0.1), y1 + nh / 2, mx - Inches(0.1), y1 + nh / 2, "① 화면 조작 (Computer Use)", color=INK)
    arrow(s, ax + aw + Inches(0.1), y2 + nh / 2, mx - Inches(0.1), y2 + nh / 2, "② 외부 파일 편집", color=INK)
    node(s, rx, y0, rw, Inches(1.6), "Recipe 생성, 수정", "저장과 적용은\n엔지니어 승인", fill=ORANGE, fc=WHITE, size=16)
    arrow(s, mx + mw + Inches(0.1), y1 + nh / 2, rx - Inches(0.1), y1 + nh / 2, "공통 플랫폼의 기반", label_w=Inches(2.0))
    arrow(s, mx + mw + Inches(0.1), y2 + nh / 2, rx - Inches(0.1), y2 + nh / 2, "장비군별 도구", label_w=Inches(2.0))
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    cy, chh = Inches(3.9), Inches(2.75)
    column(s, M, cy, cw, chh, "Agent의 Recipe 작업 방식", [
               "**RAG**: 엔지니어의 Recipe 작업 방식을 모방하도록 __Context를 받을 수 있는 DB 구축__",
               "**Computer Use**: VLM(화면 판단), PyAutoGUI(키보드, 마우스 입력), Win32(창 단위 조작)를 조합해 구현",
               "**외부 파일 편집**: Recipe 생성, 수정 단계 간소화로 __ARC 최적화__",
               "**소형 모델**: ARC 작업에 __최적화된 소형 모델__ 도입 (효율적 GPU 자원 배분과 빠른 속도)"], BLUE, size=13)
    column(s, M + cw + gap, cy, cw, chh, "공통 플랫폼 대상 공통 과제", [
               "엔지니어의 **Recipe 작업 화면을 추출**해 Agent가 활용할 수 있는 __지식으로 변환__",
               "장비군별 개별 진행 시 __같은 사이클 반복__ → **공통 플랫폼 필요**",
               "지식을 Agent가 실행할 수 있도록 **Workflow로 만드는 작업**",
               "**기술 부채 해소**: Vibe Coding 중 핵심 기술 사항을 정리하는 __문서 작업__ 필요",
               "**Agent의 마우스, 키보드 조작**, 최적화 소형 모델 운영"], BLUE, size=13)

def s_why_not_llm():  # LLM 모델만으로는 안 되는 이유
    s = slide("별첨: LLM 모델만으로 안 되는 이유",
              "모델은 \"화면을 보고 다음 할 일을 판단\"하는 한 부분, 나머지는 **플랫폼**이 채워야 함")
    rows = [("항목", "모델만 쓸 때의 한계", "플랫폼이 채우는 것"),
            ("좌표 정밀도", "같은 화면도 질의마다 위치가 달라짐", "CV 정밀 좌표, OCR 결과 확인"),
            ("장비 지식(노하우)", "사내 계측 화면, 셋업 절차, 판정 기준을 모름", "녹화에서 뽑고 엔지니어가 승인한 절차"),
            ("성공/실패 여부", "조작 결과 자체 확인 불가, 실패 누락 시 오측정", "규칙 확인, 결과 검증, 불확실하면 멈춤"),
            ("진행 상태", "수십 번 조작 중 진행 위치 누락", "절차 상태, 체크포인트, 실행 로그"),
            ("화면 조작", "장비에 조작 API가 없음", "화면 캡처, 입력 실행기, 즉시 중단")]
    table(s, rows, M, COL_Y, [Inches(2.4), Inches(5.5), Emu(W - 2 * M - Inches(7.9))], size=14, row_h=Inches(0.55),
          bold_col0=True)
    note_line(s, Inches(5.65), "모델 교체", "최신 VLM은 __공통 평가 세트__로 측정 후 판단 부분만 교체, 모델 교체 후에도 규칙 확인과 결과 검증 유지", size=13)

def s_where():  # 별도 PC 대신 엔지니어 PC 설치형
    s = slide("업무 전략 2: 엔지니어 PC 설치",
              "별도 운영 PC 대신 **엔지니어 PC**에 공통 플랫폼 설치")
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
    node(s, x, dy, sw, dh, "엔지니어", "본인 PC")
    arrow(s, x + sw + pad, my, x + sw + ag - pad, my, "확인 지연", label_w=ag)
    node(s, x + sw + ag, dy, bw, dh, "별도 운영 PC", "Agent 실행\n아무도 보지 않는 화면", fill=GRAY)
    arrow(s, x + sw + ag + bw + pad, my, x + cw - sw - pad, my, "조작", label_w=ag)
    node(s, x + cw - sw, dy, sw, dh, "장비")
    # 변경: 사내 서버 → 엔지니어 PC(Agent 설치) → 장비
    x = M + cw + gap
    node(s, x, dy, sw, dh, "사내 서버", "모델 판단")
    arrow(s, x + sw + pad, my, x + sw + ag - pad, my, "판단", color=INK, label_w=ag)
    node(s, x + sw + ag, dy, bw, dh, "엔지니어 PC", "Agent 설치\n보면서 실행, 즉시 중단", fill=INK, fc=WHITE)
    arrow(s, x + sw + ag + bw + pad, my, x + cw - sw - pad, my, "조작", color=INK, label_w=ag)
    node(s, x + cw - sw, dy, sw, dh, "장비")

def s_team():  # TF 구성 (조직도)
    s = slide("TF 구성",
              "**공통 플랫폼 팀** 안에서 **개발 인력**과 **장비군 담당 엔지니어**가 함께 개발")
    top_w = Inches(6.4)
    tx = (W - top_w) / 2
    th = Inches(0.46)
    champ_y = COL_Y + th + Inches(0.16)
    box(s, tx, COL_Y, top_w, th, fill=INK)
    text(s, tx, COL_Y, top_w, th, "기반기술센터 AI/DT TF 담당", size=16, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, tx, champ_y, top_w, th, fill=SILVER)
    # 줄 전체가 <> 자동 주황이 되지 않게 검정으로 고정
    text(s, tx, champ_y, top_w, th, [[("ARC TF Champion: <직급 확인>", {"color": INK})]], size=16, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    goal = lambda t: [[("2027 상반기 목표", {"bold": True, "color": ORANGE})], t]
    # 한 팀으로 그린다: 같은 사람은 한 번만 나온다. (머리말, 폭 비율, 구성원)
    groups = [("개발 3명: 기반기술전략 AX 파트", 0.36, [
                  ("백엔드\n(2명)", ["서버 개발/운영", "LLMOps", "녹화 DB 추출", "지식 자산화"]),
                  ("프론트엔드\n(1명)", ["설치형 에이전트", "대화형 서비스 구현", "Workflow 모니터링", "승인 화면"])]),
              ("R&D DMI 장비군 담당: 장비군마다 2명", 0.64, [
                  ("CD-SEM\n(1차, 2명)", goal("Align Fail 자동 대응 Agent 현장 배치")),
                  ("Thickness\n(1차, 2명)", goal("Film 10 layer 이하 의뢰건 90% 이상 자동화")),
                  ("Inspection\n(1차, 2명)", goal("Optical Inspection 장비군 전체로 ARC 확장")),
                  ("다른 MI 장비군\n(2차, 각 2명)", ["상반기 인터뷰, Workflow 조사", "6월 베타 확인 후 7월 본격 참여"])])]
    gg = Inches(0.3)
    team_y, team_h = champ_y + th + Inches(0.14), Inches(0.42)
    gy = team_y + team_h + Inches(0.08)
    bw_g = Inches(0.12)
    def line(x0, y0, x1, y1):  # 연결선: 얇고 진한 회색
        ln = s.shapes.add_connector(1, x0, y0, x1, y1).line
        ln.color.rgb, ln.width = DARK, Pt(0.75)
    line(W / 2, COL_Y + th, W / 2, champ_y)
    line(W / 2, champ_y + th, W / 2, team_y)
    # 팀 머리줄 하나가 두 묶음을 덮는다
    box(s, M, team_y, W - 2 * M, team_h, fill=SILVER)
    text(s, M, team_y, W - 2 * M, team_h,
         [[("공통 플랫폼 팀: ", {"color": INK}), ("1차 9명", {"color": BLUE}),
           (" (개발 3명 + 장비군 담당 6명)", {"color": INK})]],
         size=16, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    gx = M
    for gi, (head, share, members) in enumerate(groups):
        gw = (W - 2 * M - gg) * share
        box(s, gx, gy, gw, Inches(0.36), fill=MIST, line=LINE)
        text(s, gx, gy, gw, Inches(0.36), head, size=13, bold=True, color=INK, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        bw = (gw - (len(members) - 1) * bw_g) / len(members)
        for k, (name, lines) in enumerate(members):
            x, y = gx + k * (bw + bw_g), gy + Inches(0.44)
            dim = gi == 1 and k == 3  # 2차 준비 단계
            box(s, x, y, bw, Inches(1.85), fill=MIST if dim else WHITE, line=LINE)
            box(s, x, y, bw, Inches(0.06), fill=GRAY if dim else INK)
            text(s, x + Inches(0.06), y + Inches(0.12), bw - Inches(0.12), Inches(0.5), name.split("\n"), size=12, bold=True, gap=0,
                 align=PP_ALIGN.CENTER)
            text(s, x + Inches(0.06), y + Inches(0.72), bw - Inches(0.12), Inches(1.1), lines, size=12, color=INK,
                 align=PP_ALIGN.CENTER, spacing=1.0, gap=0.25)
        gx += gw + gg
    note_line(s, Inches(6.42), "운영", "**Weekly 미팅** (개발 노하우 공유, 진행사항 체크), **월간 AI/DT TF장 보고**, **분기 부문장님 보고**", size=13)

def s_kanban():  # 일정 칸반
    s = slide("2027 일정",
              "상반기 **1차 장비군 목표**와 **플랫폼 베타**, 하반기 **영역 확장**")
    periods = ["2026-12", "2027 1분기", "2027 2분기", "2027 3분기", "2027 4분기"]
    lanes = [("1차 장비군\n목표", [["TF 승인", "녹화 DB 접근 확인"],
                             ["CD-SEM Align Fail 평가", "THK Recipe 자동화\n선정 및 평가", "Inspection DRB 평가"],
                             ["Pilot 운영", "★ 목표 달성 확인"],
                             ["자동 운영 유지", "요소 목록 반영", "2028~2029 수치 확정"], ["자동 운영 유지", "2차에 노하우 적용"]]),
             ("공통\n플랫폼", [["엔지니어 Recipe 작업\n녹화 DB 구축"], ["플랫폼 설계", "모델 서빙, 평가 세트", "Weekly 미팅 운영"],
                              ["녹화 추출, 라벨링", "RAG 구축, 베타 배포"], ["Workflow 초안, 승인", "입력 없는 제안 검증", "수정 내용 RAG 보완"],
                              ["2차 장비군 RAG 반영", "★ 목표 달성 확인"]]),
             ("영역 확장\n(2차)", [["2차 명단 지정"], ["인터뷰, Workflow 조사"], ["요구사항 보완"], ["Weekly 미팅 합류"], ["workflow 구현 및\n자동화 테스트"]])]
    lw = Inches(1.45)
    colw = (W - 2 * M - lw) / len(periods)
    hy = COL_Y - Inches(0.05)
    for j, p in enumerate(periods):
        x = M + lw + j * colw
        box(s, x + Inches(0.03), hy, colw - Inches(0.06), Inches(0.38), fill=SILVER)
        text(s, x, hy, colw, Inches(0.38), p, size=13, bold=True, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    card_h, card_g = Inches(0.29), Inches(0.04)
    pitch = card_h + card_g
    rows_of = lambda it: it.count("\n") + 1  # 두 줄 카드는 두 칸을 차지
    # 레인 높이는 가장 긴 열의 줄 수에 맞춘다
    lane_hs = [Inches(0.12) + max(sum(map(rows_of, items)) for items in cols_) * pitch for _, cols_ in lanes]
    y = hy + Inches(0.45)
    for li, ((name, cols_), lh) in enumerate(zip(lanes, lane_hs)):
        box(s, M, y, W - 2 * M, lh, fill=LANE[li])
        text(s, M + Inches(0.12), y, lw - Inches(0.2), lh, name.split("\n"), size=13, bold=True,
             anchor=MSO_ANCHOR.MIDDLE, spacing=1.0, gap=0)
        for j, items in enumerate(cols_):
            x = M + lw + j * colw + Inches(0.07)
            cy = y + Inches(0.08)
            for it in items:
                ch = rows_of(it) * pitch - card_g
                gate = it.startswith("★")
                box(s, x, cy, colw - Inches(0.14), ch, fill=ORANGE if gate else WHITE, line=None if gate else LINE)
                text(s, x + Inches(0.1), cy, colw - Inches(0.3), ch, it.split("\n"), size=12, bold=gate,
                     color=WHITE if gate else INK, anchor=MSO_ANCHOR.MIDDLE, spacing=1.0, gap=0)
                cy += ch + card_g
        y += lh + Inches(0.06)
    text(s, M, y + Inches(0.04), W - 2 * M, Inches(0.85),
         [[("6월 목표  ", {"bold": True, "color": BLUE})] + runs_of("장비군별 목표 달성, 공통 플랫폼 베타 배포 (1차 장비군 PC에서 무장애 운영)"),
          [("12월 목표  ", {"bold": True, "color": BLUE})] + runs_of("단위 작업 목록 중 Agent가 엔지니어 개입 없이 수행에 성공한 비율 **<X>%**, 2차 장비군 합류"),
          [("2028 진입  ", {"bold": True, "color": BLUE})] + runs_of("12월 목표 달성과 화면 입력 안전 기준 충족")],
         size=12, color=SUB, spacing=1.0, gap=0.3)

def s_safety():  # 안전과 리스크
    s = slide("안전과 리스크",
              "위험 조작은 **엔지니어 승인**, 자동 운영은 **승인된 조작만**")
    two_col(s, "리스크", [
                "**오조작**: Agent의 Recipe Tuning 실패, 진행 정체 (같은 조작만 되풀이)",
                "**시스템 변경**: 장비 S/W, 화면이 바뀌면 기존 절차가 맞지 않음",
                "**방화벽 문제**: RCS, Radmin은 원칙상 SecureVDI 안에서 사용하게 되어 있음"],
            "대응", [
                "위험 조작, 저장 전 __엔지니어 승인__, 즉시 중단 수단, 자동 전환은 장비별 승인",
                "1차 자동 운영은 __장비군별 승인 범위__ 안에서만",
                "화면, 절차가 바뀌면 이전 검증 결과를 쓰지 않고 __재검증__",
                "RCS, Radmin을 SecureVDI를 통하지 않고 바로 사용할 수 있는지 __정책 확인__",
                "CD-SEM은 SecureVDI를 통하지 않고 사용 중"],
            h=Inches(3.65), size=14)

def s_decision():  # 결정 필요 사항
    s = slide("결정 필요 사항",
              "**ARC TF 구성**(조직도) 승인, **2027-01** 운영 시작")
    two_col(s, "12월 결정 필요 사항: TF 구성 승인", [
                "**2027-01~12 ARC TF 운영 승인** (1차, 2차 명단 함께 지정)",
                "**1차 CD-SEM, Thickness, Inspection 담당 지정**: 장비군마다 2명 (6명)",
                "**개발 인력 3명 지정**: 백엔드 2, 프론트엔드 1. 기반기술전략 AX 파트",
                "**2차 다른 MI 장비군 담당 지정**: 장비군마다 2명, __6월 베타 확인 후 7월 본격 참여__"],
            "필요 자원", [
                "2027-01 전: 공통 프로젝트를 위한 **HCP 환경 및 자원** (현 H200 2장, 2027 하반기 4장 이상 확보)",
                "2027-01 전: **RCS 녹화 스토리지 접근** 여부 확인, 영상 적재를 위한 **별도 스토리지 DB** 확보"],
            h=Inches(3.4), size=16, arrow=False)

# ---------- 공통 플랫폼 구조 (07 기술 설계 요약) ----------
def flow(s, items, y, h=Inches(1.1), fills=None, head_size=15, body_size=11):
    """가로 흐름 상자: items = [(제목, 설명)], 상자 사이에 화살표."""
    n = len(items)
    gap = Inches(0.3)
    fw = (W - 2 * M - gap * (n - 1)) / n
    for i, (h_, b_) in enumerate(items):
        x = M + i * (fw + gap)
        fill, fc = (fills[i] if fills else (MIST, INK))
        box(s, x, y, fw, h, fill=fill, line=LINE if fill == WHITE else None)
        if fill == WHITE:  # 흰 박스는 밝은 회색 머리줄 (TF 구성 장과 같은 모양)
            box(s, x, y, fw, Inches(0.55), fill=SILVER)
        text(s, x + Inches(0.08), y + Inches(0.1), fw - Inches(0.16), Inches(0.45), h_, size=head_size, bold=True,
             color=fc, align=PP_ALIGN.CENTER, spacing=1.0)
        text(s, x + Inches(0.08), y + Inches(0.55 if fill != WHITE else 0.68), fw - Inches(0.16), h - Inches(0.6), b_.split("\n"), size=body_size,
             color=fc, align=PP_ALIGN.CENTER, spacing=1.0, gap=0)
        if i < n - 1:
            a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, x + fw + gap / 2 - Inches(0.08), y + h / 2 - Inches(0.15),
                                   Inches(0.16), Inches(0.3))
            a.fill.solid(); a.fill.fore_color.rgb = GRAY; a.line.fill.background(); a.shadow.inherit = False


def s_platform(merged=False):  # 공통 플랫폼 구조. merged=True(exec)는 업무 전략 1, 2 의 내용(소형 모델, 외부 파일 편집)을 합침
    s = slide("핵심 개발 방향",
              "**Computer Use** 기반 **RAG Workflow 순환** 플랫폼")
    flow(s, [("① 녹화 추출, 라벨링", "- 녹화 DB에서 작업 내용 추출\n- 현업 엔지니어가 라벨링"),
             ("② RAG 저장", "- Recipe 셋업 지식 DB\n- 절차, 노하우, 결정 지점 저장"),
             ("③ Workflow 생성, 실행", "- RAG로 Workflow 초안 생성\n- 승인 버전만 Agent가 실행"),
             ("④ 준수 확인, 수정", "- Workflow를 지키는지 확인\n- 틀리면 엔지니어가 개입, 수정")],
         COL_Y, h=Inches(1.3), fills=[(WHITE, INK)] * 4, head_size=16, body_size=13)
    fgap = Inches(0.3)
    fw = (W - 2 * M - fgap * 3) / 4
    c2, c4 = M + (fw + fgap) + fw / 2, M + 3 * (fw + fgap) + fw / 2
    fb, ry = COL_Y + Inches(1.3), COL_Y + Inches(1.5)
    arrow(s, c4, fb, c4, ry, color=ORANGE, head=False)
    arrow(s, c4, ry, c2, ry, color=ORANGE, head=False)
    arrow(s, c2, ry, c2, fb + Inches(0.03), color=ORANGE)
    text(s, M, COL_Y + Inches(1.64), W - 2 * M, Inches(0.36),
         [[("순환  ", {"bold": True})] + runs_of("④의 엔지니어 수정 내용은 __검토, 승인 후 새 버전으로 ②RAG에 저장__, 다음 Workflow에 반영")], size=14, color=BLUE,
         align=PP_ALIGN.CENTER)
    gap = Inches(0.6)
    cw = (W - 2 * M - gap) / 2
    cy, chh = Inches(4.25), Inches(2.0)
    column(s, M, cy, cw, chh, "공통 플랫폼 팀 역할", [
               "녹화 내용 추출 (**지식 자산화**), **RAG DB** 구축, Workflow 생성",
               "**Computer Use** 실행 엔진 개발"]
           + (["**최적화 소형 모델** 발굴 및 운영"] if merged else [])
           + ["UI/UX 개발"], BLUE, size=13)
    column(s, M + cw + gap, cy, cw, chh, "현업 엔지니어 역할", [
               "추출 내용 **라벨링**, Workflow 검토와 승인",
               "Agent 실행 중 __개입, 수정__ / 녹화본으로 미리 실험",
               "실전 사용 후 **개발자에게 피드백** → 플랫폼에 반영"]
           + (["장비군별 도구로 **외부 Recipe 파일 편집** (공통 플랫폼 범위 밖)"] if merged else []), BLUE, size=13)
    note_line(s, Inches(6.5), "플랫폼의 역할", "2027 지식 자산화 (작업 녹화, Workflow 구축)  →  2028 보조 (반복 입력 대행)  →  2029 자율 (Agent가 셋업 진행)", size=13)


def s_offline():  # Offline 파이프라인 (team)
    s = slide("Offline 지식화",
              "녹화에서 승인된 절차까지 **6단계**, AI 추출물은 **검토할 절차 초안**")
    flow(s, [("전처리", "1~2fps 탐색\n원본 프레임 보존"), ("변화 감지", "조작 영역만\n전후 화면, OCR"),
             ("의미 행동", "관찰 사실과\n추정 행동 분리"), ("의미 단계", "시작, 완료\n조건으로 묶음"),
             ("시연 정렬", "공통 경로와\n분기 후보"), ("분기 승인", "담당 엔지니어\n승인")],
         COL_Y, fills=[(MIST, INK)] * 5 + [(INK, WHITE)])
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


def s_online():  # Online 실행 루프와 3단 분리 (team)
    s = slide("Online 실행",
              "LLM은 **제안만**, 서버가 **규칙으로 다시 확인**, 실행은 **조작 단위 1개씩**")
    flow(s, [("관찰", "화면, 장비 확인"), ("상태 인식", "현재 단계"), ("검색, 판단", "호환 시연 최대 3건\n없으면 검토 요청"),
             ("규칙 확인", "권한, 분기, 인자"), ("실행", "조작 단위 1개"), ("검증", "사후조건\n체크포인트")],
         COL_Y, fills=[(MIST, INK)] * 3 + [(INK, WHITE), (MIST, INK), (MIST, INK)])
    rows = [("역할", "하는 일", "하지 않는 일"),
            ("판단 (LLM)", "다음 분기, 조작 단위, 근거 제안", "직접 조작, 사전조건 충족 선언"),
            ("규칙 확인 (서버)", "권한, 허용 분기, 인자, 값과 단위, 절차 버전 재확인, 규칙 위반 시 실행 차단", "판단"),
            ("실행 (PC)", "입력 직전 대상 창, 화면 최신 여부, 세션 잠김, 승인 유효기간 재검사 후 조작 1개 실행", "임의 메뉴 탐색, 추가 수정")]
    table(s, rows, M, Inches(3.45), [Inches(2.6), Inches(6.0), Emu(W - 2 * M - Inches(8.6))], size=13, row_h=Inches(0.5),
          bold_col0=True)
    note_line(s, Inches(5.65), "중단", "모름, 불일치, 시간 초과 → 멈추고 엔지니어 검토 / 통신 단절 시 재시도보다 __실제 결과 조회__ 먼저", size=13)
    note_line(s, Inches(6.1), "입력", "승인 절차서는 신뢰 저장소에서 ID, 버전으로만 조회 / 화면 글자, 검색 문서는 **데이터로만** 취급", size=13)


def s_gates():  # 안전 관문 (team)
    s = slide("안전 관문",
              "**실행 전, 사본 저장, 운영 적용**을 나눠 확인")
    rows = [("관문", "시점", "확인할 것"),
            ("G1", "실행 전", "장비, 세션, 권한, 버전 확인 / 원본 백업 / 변경 대상과 범위 확정"),
            ("G2", "작업용 사본 수정, 저장", "허용 필드, 값, 단위 확인 / 범위 밖 변경 차단 / 저장 직전 담당 엔지니어 승인 / 저장 후 다시 읽어 차이 확인"),
            ("G3", "운영 적용", "담당 엔지니어가 백업, 차이, 검증 증거를 보고 별도 작업으로 승인")]
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


def s_eval():  # 평가 지표 (team)
    s = slide("평가 방법",
              "**미사용 시연**으로 판단을 먼저 평가, 관문 수치는 **시범 실측 후 2027-03 말 합의**")
    rows = [("지표", "정의", "쓰는 관문"),
            ("단계 추출 F1", "담당 엔지니어 정답 대비 단계 정밀도와 재현율의 조화평균", "2027-12"),
            ("상태, 분기 정확도", "맞힌 상태, 분기 / 평가한 상태, 분기 (따로 집계)", "2027-12, 2028 진입"),
            ("미승인 행동 제안률", "미승인 행동 제안 / 전체 제안", "2028 진입"),
            ("안전 중단률, 자동 결정률", "모르는 상황 중 안전 중단 / 전체 중 자동 결정", "2028, 2029"),
            ("검증된 작업 성공률", "사후조건과 정확한 차이 통과 / 시도 작업 (감독하 실장비)", "2028 보조"),
            ("소요 시간, 사람 투입", "관찰부터 검증까지 p50, p95 / 검토 시간, 개입 횟수", "2028, 2029")]
    table(s, rows, M, COL_Y, [Inches(3.2), Inches(6.4), Emu(W - 2 * M - Inches(9.6))], size=13, row_h=Inches(0.48),
          bold_col0=True, shade=lambda r: r[2].startswith("2027"))
    note_line(s, Inches(5.65), "누수 방지", "같은 작업, 파생 영상은 한쪽에만. 시점 t 평가에는 __t 이전 관찰만__, 정답 경로는 평가자만", size=13)
    note_line(s, Inches(6.1), "검증 순서", "녹화본 평가 → **입력 없는 Shadow** → 감독하 실장비 (격리 GUI 환경 대체)", size=13)


def s_stack():  # 구현 형태와 기술 스택 (team)
    s = slide("구현 형태",
              "**엔지니어 PC 설치형 에이전트** + **서버 웹앱**, 양쪽 Python (설계안, 실측 후 확정)")
    rows = [("구성 요소", "위치", "언어, 프레임워크"),
            ("화면 캡처, 변화 감지", "엔지니어 PC", "Python + WGC 창 캡처"),
            ("입력 실행, 즉시 중단", "엔지니어 PC", "Python + ctypes SendInput, pywin32 창 관리"),
            ("상태, 제어권 인계 화면", "엔지니어 PC", "트레이 아이콘 + localhost 페이지"),
            ("VLM 판단, 규칙 확인", "기존 서버", "Python + 기존 Flask"),
            ("승인 화면, 절차서 검토", "서버 + 브라우저", "Vue 3 + Flask API"),
            ("배포", "엔지니어 PC", "PyInstaller + 회사 서명 + 사내 배포")]
    table(s, rows, M, COL_Y, [Inches(3.6), Inches(2.6), Emu(W - 2 * M - Inches(6.2))], size=13,
          row_h=Inches(0.42), bold_col0=True, shade=lambda r: r[1] == "엔지니어 PC")
    note_line(s, Inches(5.4), "제어권 인계", "알림 → 엔지니어가 넘겨줌 → 대상 창 확인 → 짧은 입력 → 결과 확인 → 반환. __엔지니어 조작 시 즉시 중단__", size=13)
    note_line(s, Inches(5.85), "보류", "C#/.NET은 Python 캡처의 PC 자원 한도 초과 시 대안, Electron 및 Tauri 미채택", size=13)
    note_line(s, Inches(6.3), "실측", "입력 중단 신뢰성(선발대 장비군), PC 자원, 회사 보안 프로그램 아래 설치와 롤백", size=13)


# ---------- 최고 임원용 ----------
def s_exec_roadmap(second_line=False):  # second_line=True(rev2 본 장표): 연도별 대상 장비 장 대신 2차 장비군 한 줄
    s = slide("개발 로드맵",
              "2027 **지식 자산화**, 2028 **보조**, 2029 **자율**로 단계별 확대")
    fh = Inches(1.2) if second_line else Inches(1.5)  # 한 줄을 더 넣는 만큼 흐름 상자를 줄임
    flow(s, [("2027 상반기", "1차 장비군 목표\n플랫폼 베타"), ("2027 하반기", "2차 장비군 합류\n지식 자산화"),
             ("2028 보조", "반복 입력 대행"), ("2029 자율", "Agent가\n셋업 진행")],
         COL_Y, h=fh, fills=[(WHITE, INK)] * 3 + [(INK, WHITE)],
         head_size=20, body_size=15)
    rows = [("시기", "주요 내용", "목표"),
            ("2027 상반기", "1차 장비군 평가와 Pilot 운영, 엔지니어 Recipe 작업 녹화 DB 구축", "1차 장비군 목표 달성, 플랫폼 베타 배포"),
            ("2027 하반기", "2차 장비군 Weekly 미팅 합류, workflow 구현 및 자동화 테스트", "단위 작업 Agent 수행 성공 비율 <X>%"),
            ("2028", "보조: 엔지니어 앞에서 반복 입력 대행", "Agent 개입 범위 <N>배, 셋업 시간 <확인>% 단축"),
            ("2029", "자율: Agent가 셋업 진행", "1차 장비군 ARC 적용 비율 90% 이상")]
    table(s, rows, M, COL_Y + fh + Inches(0.14), [Inches(1.9), Inches(6.0), Emu(W - 2 * M - Inches(7.9))], size=14, row_h=Inches(0.45),
          bold_col0=True, border=True, col0_color=INK)
    if second_line:
        note_line(s, Inches(5.95), "2차 장비군", "2026-12 지정, **2027-07 합류**. 순서 기준은 월 의뢰 건수, 건당 셋업 시간, 1차 장비군과 접속 경로가 비슷한 정도", size=13)
    note_line(s, Inches(6.25 if second_line else 6.2), "확장성", "**화면만으로 조작**(API 불필요), 장비별 코드 없이 장비군을 추가하는 구조로 설계. __2차 장비군 합류 결과(2027-12 확인)로 검증__ 후 공정 장비 확장 검토", size=13)
    note_line(s, Inches(6.55), "측정", "ARC 적용 비율 = ARC 적용 Recipe 수 / 의뢰 Recipe 수 (장비군별 대상 Recipe 기준, 재작업 기준 충족 건만 집계)", size=13)


# ---------- 팀장 피드백 반영 (2026-11-23): 01 기획안의 같은 이름 절을 요약 ----------
def s_why():  # 왜 해야 하는가 (01 1장: 문제, 이 과제를 해야 하는 이유)
    # 2026-11-23 카운슬: 수치는 "2~3명분 (추정)" 한 번만. "하루 2~3건"은 30~60분과 범위를 맞춘 뒤에 넣는다
    s = slide("왜 해야 하는가",
              "각자 추진 중인 ARC의 **중복을 없애고 노하우를 합쳐** **Recipe 업무 시간**을 줄여야 함")
    two_col(s, "지금의 문제", [
                "**Recipe 셋업과 Tuning이 수작업**: 숙련 엔지니어가 원격 화면에서 __측정, 실패, 재조정을 반복__",
                "**노하우가 개인에게 있음**: 담당자가 바뀌면 사라지고 사람마다 품질이 다름",
                "**장비군별 각자 ARC 추진**: 화면 조작, 모델 운영, 녹화분 확보를 __장비군마다 반복__",
                "**의뢰가 늘면 셋업 인력도 같이 늘어야 함**: Data 분석, 고도화된 장비 관리에 __쓸 시간이 없음__"],
            "ARC TF로 얻는 것", [
                "**중복 제거, 노하우 통합**: 공통 플랫폼 하나로 공동 대응, 셋업 방식을 __회사 지식 DB로 축적__",
                "**Recipe 업무 단축**: 반자동 단계에서 __1명이 2~3명분__ 처리 (추정), 줄인 시간은 Data 분석과 장비 관리 고도화로 전환",
                "**지금 시작해야 함**: 숙련 엔지니어의 작업 데이터는 지금부터 쌓아야 함. AI 모델은 교체할 수 있지만 __노하우 데이터는 대체 불가__",
                "**단계 판정**: 2027-06, 2027-12 목표 달성 확인에서 __보완, 축소, 중단__ 판단"],
            h=Inches(3.95), size=14, before=True)
    note_line(s, Inches(6.25), "효과 측정", "플랫폼 실행 로그로 자동 집계해 **월간 보고**. 기준선 측정 2027-02, 목표 수치는 __2027-07 확정__", size=13)


def s_why_tf():  # 추진 방식 비교 (01 5장: 추진 방식 비교, BP사 협업 검토)
    s = slide("추진 방식 비교: 왜 TF인가",
              "**추진 동력**, **현업 수고의 공식 인정**, **두 조직을 잇는 보고 라인**을 위해 TF 필요")
    rows = [("추진 방식", "장점", "한계"),
            ("장비군별 각자 추진 (현재)", "승인 절차 불필요", "화면 조작, 모델 운영, 녹화분 확보를 장비군마다 반복 개발. 노하우 공유 안 됨. 개발 가능한 담당자가 있는 장비군만 진행"),
            ("AX 파트 과제 + 현업 협조 요청", "조직 변경 없음", "현업 시간이 공식 인정되지 않음. 본인 과제에 없어 참여 동력 없음"),
            ("정규 조직 신설", "전담 가능", "효과 검증 전에 조직 개편 필요"),
            ("BP사 위탁", "내부 인력 부담 감소", "BP사가 장비 API나 DB를 새로 열어 줘야 가능. 비용과 수고 대비 실익 낮음 (ARC는 화면 조작 방식이라 API 없이 진행)"),
            ("ARC TF (제안)", "추진 동력, 현업 수고 공식 인정, 두 조직 보고 라인", "겸직 운영. 2027-12에 2028 운영 형태 판단")]
    table(s, rows, M, COL_Y, [Inches(3.1), Inches(3.3), Emu(W - 2 * M - Inches(6.4))], size=13, row_h=Inches(0.52),
          bold_col0=True, shade=lambda r: r[0].startswith("ARC TF"), border=True)
    note_line(s, Inches(5.45), "BP 협업", "**RCS 업체**와는 __녹화 DB 접근 경로__ 협업 검토", size=13)
    note_line(s, Inches(5.83), "TF 운영", "2027-01~12 운영, 전원 겸직(참여 시간은 탄력적으로 운영), 현업 엔지니어의 TF 참여를 __본인 과제로 인정__", size=13)
    note_line(s, Inches(6.21), "R&D DMI 합의", "<확인: 장비군 담당 6명 참여에 대한 R&D DMI 합의 상태>", size=13)


def s_bp():  # BP사 협업 검토 상세 (team)
    s = slide("BP사 협업 검토",
              "장비 BP사 참여는 **API, DB 개방**이 전제라 실익이 낮음, **RCS 업체**와 녹화 DB 협업 검토")
    rows = [("방법", "BP사가 해야 하는 일", "판단"),
            ("장비 API 제어", "장비를 외부에서 조작할 수 있는 API 개발과 제공", "비용과 수고 대비 실익 낮음. 화면 조작 방식으로 대체"),
            ("Recipe 정보 API", "중요 Recipe 정보를 외부로 내보내는 API 제공", "비용과 수고 대비 실익 낮음"),
            ("DB 접근 권한", "장비 DB를 외부에서 읽을 수 있는 권한 제공", "비용과 수고 대비 실익 낮음"),
            ("RCS 업체 협업", "녹화 DB 접근 경로 제공", "검토 대상: 녹화 DB 접근 확인(R&D MI H/W 담당자)과 같은 건")]
    table(s, rows, M, COL_Y, [Inches(2.4), Inches(4.9), Emu(W - 2 * M - Inches(7.3))], size=13, row_h=Inches(0.48),
          bold_col0=True, shade=lambda r: r[0].startswith("RCS"), border=True)
    ly = Inches(4.7)
    box(s, M, ly, W - 2 * M, Inches(1.5), fill=MIST)
    text(s, M + Inches(0.3), ly + Inches(0.12), W - 2 * M - Inches(0.6), Inches(1.6), [
        [("개발을 BP사에 맡기기 어려운 이유", {"bold": True, "color": BLUE})],
        runs_of("1. **요구사항을 미리 명세할 수 없음**: 무엇을 자동화할지는 현업 엔지니어와 반복하면서 정해짐"),
        runs_of("2. 셋업 노하우와 PoC 자산은 __사내에 남겨야 하는 자산__"),
        runs_of("3. 장비 화면과 녹화 영상에 외부 인력이 접근하려면 **보안 절차** 필요")],
        size=14, gap=0.3)


def s_engineer_role():  # 현업 엔지니어의 참여 방법 (team, 01 6장)
    s = slide("현업 엔지니어의 참여 방법",
              "개발, AI 지식 없이 **평소 작업과 판단**으로 참여, 초안은 AI가 쓰고 엔지니어는 **확인과 수정**")
    rows = [("하는 일", "방법", "필요한 지식"),
            ("작업 녹화", "평소대로 Recipe를 셋업 (녹화 DB나 녹화기가 기록)", "없음"),
            ("라벨링", "AI가 나눈 작업 단계에 이름을 붙이고 틀린 곳을 고침", "장비, Recipe"),
            ("결정 지점 답변", "\"왜 여기서 이렇게 했는지\" 묻는 질문에 답함", "장비, Recipe"),
            ("Workflow 검토와 승인", "절차서를 읽고 맞는지 판정", "장비, Recipe"),
            ("실행 중 개입과 수정", "Agent가 틀리면 멈추고 고침. 고친 내용은 지식 DB에 다시 저장", "장비, Recipe")]
    table(s, rows, M, COL_Y, [Inches(3.0), Inches(6.9), Emu(W - 2 * M - Inches(9.9))], size=14, row_h=Inches(0.48),
          bold_col0=True, border=True)
    note_line(s, Inches(5.25), "비유", "후임에게 **인수인계서**를 쓰는 일과 같음. 코드 작성은 개발 인력이 맡음", size=13)
    note_line(s, Inches(5.65), "검증", "2026-12 시범: 개발, AI에 익숙하지 않은 엔지니어가 녹화 1건을 라벨링, __소요 시간과 수정률 측정__ <확인>", size=13)
    note_line(s, Inches(6.05), "지원", "별도 교육 과정 없이 **Weekly 미팅**에서 개발 노하우 공유", size=13)


def s_equip(backup=False):  # 연도별 대상 장비 (01 4장). backup=True 는 rev2 의 별첨
    s = slide(("별첨: " if backup else "") + "연도별 대상 장비",
              "1차 장비군은 **2029 ARC 적용 90% 이상**, 2차 장비군은 **2027 하반기 합류** 후 목표 확정")
    rows = [("장비군", "2027 상반기", "2027 하반기", "2028", "2029"),
            ("CD-SEM (1차)", "Align Fail 자동 대응 Agent 현장 배치", "셋업 절차 지식 자산화", "보조: 반복 입력 대행", "자율: ARC 적용 90% 이상"),
            ("Thickness (1차)", "Film 10 layer 이하 의뢰건 90% 이상 자동화", "셋업 절차 지식 자산화", "보조: 반복 입력 대행", "자율: ARC 적용 90% 이상"),
            ("Inspection (1차)", "Optical Inspection 장비군 전체로 ARC 확장", "셋업 절차 지식 자산화", "보조: 반복 입력 대행", "자율: ARC 적용 90% 이상"),
            ("2차: <장비군명 확인>", "인터뷰, Workflow 조사, 작업 녹화", "7월 Weekly 미팅 합류, Workflow 구현과 자동화 테스트", "<합류 후 확정>", "<합류 후 확정>"),
            ("공정 장비 (확장 검토)", "없음", "2차 합류 결과로 확장 구조 검증 (12월 확인)", "<타당성 검토 시점 확인>", "<확인>")]
    table(s, rows, M, COL_Y, [Inches(2.4), Inches(3.1), Inches(2.9), Inches(1.95), Emu(W - 2 * M - Inches(10.35))], size=12,
          row_h=Inches(0.58), bold_col0=True, border=True, col0_color=INK)
    note_line(s, Inches(5.85), "2차 순서 기준", "월 의뢰 건수, 건당 셋업 시간, 1차 장비군과 접속 경로가 비슷한 정도", size=13)
    note_line(s, Inches(6.23), "2차 명단", "2026-12에 1차와 함께 지정, 목표는 합류 후 기준선 측정(2027-08)으로 확정", size=13)


def s_effect(backup=False):  # 단계별 목표와 업무 전환 효과 (01 4장). backup=True 는 rev2 의 별첨
    s = slide(("별첨: " if backup else "") + "단계별 목표와 업무 전환 효과",
              "반자동(2028) 단계에서 **엔지니어 1명이 2~3명분** Recipe 처리, 줄인 시간은 **분석 업무로 전환**")
    rows = [("시기", "푸는 문제", "산출물", "업무 전환 효과"),
            ("2027 상반기", "단순, 반복 의뢰의 Recipe를 손으로 셋업", "1차 장비군 목표 3건, 플랫폼 베타, 기준선 측정", "Thickness Film 10 layer 이하 의뢰건 90% 이상 자동화"),
            ("2027 하반기", "셋업 노하우가 개인에게만 있음", "절차서, Recipe 셋업 지식 DB", "시간 절감 없음, 투입 단계 (라벨링, 검토). 단위 작업 성공 비율 <X>%"),
            ("2028 보조", "반복 입력 때문에 엔지니어가 화면 앞에 묶임", "Agent가 반복 입력 대행", "엔지니어 1명이 2~3명분 처리 (추정). 건수, 시간 환산은 <기준선 측정 후 확정>"),
            ("2029 자율", "의뢰가 늘면 셋업 인력도 같이 늘어야 함", "Agent가 셋업 진행", "1차 장비군 ARC 적용 90% 이상. 시간 효과 <2027-07 확정>")]
    table(s, rows, M, COL_Y, [Inches(1.6), Inches(3.3), Inches(3.3), Emu(W - 2 * M - Inches(8.2))], size=12,
          row_h=Inches(0.6), bold_col0=True, border=True, col0_color=INK)
    note_line(s, Inches(5.25), "현재 기준", "엔지니어 1명이 **하루(8시간) Recipe 2~3건** 대응 (경험치, Recipe Creation과 Tuning 기준)", size=13)
    note_line(s, Inches(5.63), "산식", "절감 시간 = 의뢰 건수 × 건당 대응 시간 × ARC 적용 비율 × Agent 개입 범위 (라벨링, 검토 투입 시간은 뺌)", size=13)
    note_line(s, Inches(6.01), "측정", "플랫폼 실행 로그로 자동 집계해 **월간 보고**. 수치는 기준선 측정(2027-02) 후 __2027-07 확정__", size=13)
    note_line(s, Inches(6.39), "전환 업무", "확보한 시간은 **Data 분석, 고도화된 장비 관리**, 더 많은 Recipe 의뢰 대응에 사용", size=13)


def s_extend():  # 확장성 근거 (team, 01 4장)
    s = slide("확장성: 공정 장비로 이어질 수 있는 근거",
              "확장되는 것은 **조작 엔진과 지식화 순서**, **2차 장비군 합류**(2027-12)가 첫 검증")
    two_col(s, "확장되는 것", [
                "**화면만으로 조작**: 장비 API가 필요 없어 __원격 화면이 있는 장비면 적용 가능__",
                "**장비별 코드 없이 추가**: 화면 영역 지도와 절차 정의만으로 새 장비군 추가",
                "**지식화 순서 공통**: 녹화, 라벨링, Workflow 생성, 수정이 장비와 무관",
                "**안전 규칙 공통**: 엔지니어 승인, 즉시 중단, 허용 조작 범위"],
            "확장의 조건과 검증", [
                "**공정 판단은 옮겨지지 않음**: 공정 Recipe 지식은 그 장비 엔지니어가 새로 쌓아야 함",
                "**안전 기준 상향 필요**: 계측 Recipe 오류는 측정 결과에, 공정 Recipe 오류는 __Wafer에 직접 영향__",
                "**첫 검증 (2027-12)**: 2차 장비군이 장비별 코드 없이 공통 플랫폼에 합류하는지 확인",
                "검증 뒤 공정 장비, SmartFactory 운영 확장 검토 (시점은 검증 결과로 정함)"],
            h=Inches(3.9), size=14, arrow=False)


# ---------- 팀장 피드백 대응표 (feedback-response-map.md 요약, 팀장에게 rev2 와 함께 가져감) ----------
def s_map():
    s = slide("팀장 피드백 대응표",
              "피드백 8건 가운데 **7건은 본 장표**에서, **1건은 구두와 팀원용 장표**로 답함")
    rows = [("#", "팀장 피드백", "본 장표 (10장)", "별첨 (2장)", "팀원용 장표, 기획안"),
            ("1", "BP사가 하면 안 되는가, 함께 할 수 없는가", "5장: BP사 위탁 행, RCS 업체 협업 검토", "없음", "\"BP사 협업 검토\" 장"),
            ("2", "왜 TF인가, 다른 구성 방법은 없는가", "5장: 추진 방식 5가지 비교", "없음", "기획안 5장"),
            ("3", "전문 지식이 없는 구성원도 참여할 수 있는가", "없음 (구두 설명)", "없음", "\"현업 엔지니어의 참여 방법\" 장"),
            ("4", "무엇이 문제이고 성공하면 무엇을 얻는가", "2장: 문제와 얻는 것. 7장: 시기별 목표", "별첨 2", "기획안 1장, 4장"),
            ("5", "과정별 업무 전환 효과, 정량 효과", "2장: 1명이 2~3명분 (추정), 효과 측정 방법", "별첨 2", "기획안 4장"),
            ("6", "연도별로 어느 장비를 하는가", "4장: 1차 장비군 목표. 7장: 2차 장비군 한 줄", "별첨 1", "기획안 4장"),
            ("7", "공정 장비로 확장할 수 있는 근거", "7장: 확장성 한 줄", "별첨 1 (공정 장비 행)", "\"확장성 근거\" 장"),
            ("8", "안 할 수 없는 Sales Point", "2장 전체", "없음", "기획안 1장")]
    table(s, rows, M, COL_Y, [Inches(0.5), Inches(3.6), Inches(3.7), Inches(1.8), Emu(W - 2 * M - Inches(9.6))], size=12,
          row_h=Inches(0.44), border=True, shade=lambda r: r[2].startswith("없음"))
    note_line(s, Inches(6.2), "rev2 순서", "1 표지, 2 왜 해야 하는가, 3 추진 현황, 4 TF 구성, 5 왜 TF인가, 6 개발 방향, 7 로드맵, 8 일정, 9 리스크, 10 결정 필요 사항", size=12)
    note_line(s, Inches(6.5), "별첨", "1 연도별 대상 장비, 2 단계별 목표와 업무 전환 효과", size=12)


def s_map_ask():
    s = slide("팀장님께 확인할 것",
              "**보고 형식**과 **임원 예상 질문**에 따라 본 장표와 별첨의 구분을 조정")
    rows = [("#", "질문", "답에 따라 바뀌는 것"),
            ("1", "임원 보고는 대면 발표인가, 장표 사전 회람인가", "회람이면 별첨도 그대로 읽힘. 별첨 2는 수치를 뺀 축소판으로 본 장표에 올림"),
            ("2", "팀장님이 보고 자리에 동석하는가", "동석하면 구두 답변(피드백 3번)을 나눠 맡을 수 있음"),
            ("3", "담당 임원이 8건 가운데 무엇을 물을 것 같은가", "물을 가능성이 높은 것은 별첨에서 본 장표로 올림"),
            ("4", "8건 가운데 임원 장표에 꼭 들어가야 하는 것은 무엇인가", "대응표의 \"본 장표\" 칸을 고침")]
    table(s, rows, M, COL_Y, [Inches(0.5), Inches(5.4), Emu(W - 2 * M - Inches(5.9))], size=13, row_h=Inches(0.46), border=True)
    ly = Inches(4.6)
    box(s, M, ly, W - 2 * M, Inches(1.95), fill=MIST)
    text(s, M + Inches(0.3), ly + Inches(0.12), W - 2 * M - Inches(0.6), Inches(1.7), [
        [("보고 전에 채울 것", {"bold": True, "color": BLUE})],
        runs_of("1. **수치 범위 정리**: \"하루 2~3건\"과 \"레시피당 30~60분\"이 재는 범위를 맞춤. 정리되면 2장과 별첨 2에 수치 반영"),
        runs_of("2. **2차 장비군명 확정**: 정해지면 별첨 1(연도별 대상 장비)을 본 장표로 올림"),
        runs_of("3. **R&D DMI 합의 상태**: 장비군 담당 6명 참여에 대한 합의 여부를 5장 아래 줄에 적음"),
        runs_of("4. **약식 확인**: 1차 장비군 엔지니어에게 하루 처리 건수를 물어 \"추정\"을 \"엔지니어 N명 확인 기준\"으로 바꿈")],
        size=13, gap=0.3)


MAP = [lambda: s_cover("팀장 피드백 대응표", "담당 임원용 장표 rev2 기준"), s_map, s_map_ask]
# rev2 (2026-11-23 카운슬): 본 장표 10장 + 별첨 2장. 연도별 대상 장비, 단계별 목표와 업무 전환 효과는 별첨으로 내림
BOSS = [s_cover, s_why, s_status, s_team, s_why_tf, lambda: s_platform(merged=True), lambda: s_exec_roadmap(second_line=True),
        s_kanban, s_safety, s_decision, lambda: s_equip(backup=True), lambda: s_effect(backup=True)]  # 업무 전략 1, 2(s_assets, s_where)는 team deck에만 둠
TEAM = [s_cover, s_why, s_status, s_team, s_why_tf, s_bp, s_engineer_role, s_assets, s_where,
        lambda: s_platform(merged=True), s_offline, s_online, s_gates, s_eval, s_stack,
        s_exec_roadmap, s_equip, s_effect, s_extend, s_kanban, s_safety, s_decision, s_why_not_llm]
EXEC = [lambda: s_cover("Auto Recipe Creation TF", None), s_status, lambda: s_platform(merged=True),
        s_team, s_exec_roadmap]
# arc-tf-proposal.pptx 는 팀장 피드백 전 보고본(8장)으로 보존하고 다시 만들지 않는다. boss 는 rev2 로 저장한다 (2026-11-23)
DECKS = {"boss": ("arc-tf-proposal-rev2.pptx", BOSS), "team": ("arc-tf-team-full.pptx", TEAM),
         "exec": ("arc-tf-exec-summary.pptx", EXEC), "map": ("arc-tf-feedback-map.pptx", MAP)}


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
        cover = sl is prs.slides[0]
        for sh in sl.shapes:
            if sh.top <= Inches(1.1):  # 제목과 핵심 메시지는 그대로
                continue
            if cover:  # 표지는 s_cover 의 좌표 그대로
                continue
            # 본문: SRC_TOP~SRC_BOTTOM 을 BODY_TOP~BODY_BOTTOM 으로 옮기며 같은 비율로 늘린다
            sh.top = round(BODY_TOP + (sh.top - SRC_TOP) * STRETCH)
            sh.height = round(sh.height * STRETCH)
            if sh.has_table:  # 표 높이는 행 높이가 정한다
                for r in sh.table.rows:
                    r.height = round(r.height * STRETCH)
    prs.save(HERE / out_name)
    print(f"wrote {out_name}, {len(prs.slides)} slides")


if __name__ == "__main__":
    for target in sys.argv[1:] or DECKS:
        build(target)
