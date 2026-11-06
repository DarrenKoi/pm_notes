# -*- coding: utf-8 -*-
"""Auto Recipe Creation TF 담당 보고 deck (.pptx) 생성.

    python my-task/arc-tf/deliverables/build_pptx.py

deck 은 01-tf-proposal.md 의 요약이다. 그 두 문서를 고치면
아래 SLIDES 내용도 같이 고친다 (자동 동기화 안 됨). 필요: python-pptx. 한글 폰트: 맑은 고딕.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "arc-tf-proposal.pptx"

FONT = "맑은 고딕"
INK = RGBColor(0x1E, 0x2B, 0x36)      # 주색: 짙은 청회색 (웨이퍼 장비 화면)
INK_SOFT = RGBColor(0x4A, 0x5A, 0x66)
MIST = RGBColor(0xEE, 0xF1, 0xF4)     # 카드 바탕
LINE = RGBColor(0xC9, 0xD1, 0xD8)
AMBER = RGBColor(0xD9, 0x82, 0x2B)    # 강조: 사람이 결정할 곳, 요청
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xB8, 0xC4, 0xCE)

W, H = Emu(12192000), Emu(6858000)  # exact 16:9 widescreen
M = Inches(0.6)
TITLE_Y, TITLE_H = Inches(0.45), Inches(0.9)
BODY_Y = Inches(1.6)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
prs.part._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldSz').attrib.pop('type', None)
BLANK = prs.slide_layouts[6]


# ---------- helpers ----------
def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.15, name=None):
    """runs: str, or list of paragraphs; each paragraph is str or list of (text, {opts})."""
    box = slide.shapes.add_textbox(x, y, w, h)
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if i:
            p.space_before = Pt(size * 0.5)
        for t, opt in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.color.rgb = opt.get("color", color)
    return box


def box(slide, x, y, w, h, fill=MIST, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
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


def badge(slide, x, y, label, fill=INK, color=WHITE, d=Inches(0.55), size=16):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), True, color
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def slide(title, notes, dark=False):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = INK if dark else WHITE
    if title:
        text(s, M, TITLE_Y, W - 2 * M, TITLE_H, title, size=30, bold=True,
             color=WHITE if dark else INK, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    s.notes_slide.notes_text_frame.text = notes
    return s


def footer(s, n=None):
    n = n or len(prs.slides)
    text(s, M, H - Inches(0.45), Inches(6), Inches(0.3), "Auto Recipe Creation TF 구성 기획안", size=10, color=INK_SOFT)
    text(s, W - M - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(n), size=10,
         color=INK_SOFT, align=PP_ALIGN.RIGHT)


TBD = AMBER  # ⟨확인⟩ 표시 색

LANE = [RGBColor(0xFB, 0xE3, 0xCC), RGBColor(0xDC, 0xE6, 0xF1), RGBColor(0xED, 0xED, 0xED)]


def table(s, rows, x, y, widths, size=14, head_fill=INK, zebra=True, row_h=None, bold_col0=False):
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, sum(widths, Inches(0)),
                           row_h * len(rows) if row_h else Inches(0.45) * len(rows)).table
    for j, wd in enumerate(widths):
        t.columns[j].width = wd
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = head_fill if i == 0 else (MIST if zebra and i % 2 == 0 else WHITE)
            c.margin_left = c.margin_right = Inches(0.12)
            tf = c.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = val
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0 or (bold_col0 and j == 0)
            r.font.color.rgb = WHITE if i == 0 else (TBD if "⟨" in val else INK)
    return t


# ---------- 1. 표지 ----------
s = slide(None, "Auto Recipe Creation TF 구성 기획안을 보고드리겠습니다.", dark=True)
text(s, M, Inches(2.75), W - 2 * M, Inches(1.2), "Auto Recipe Creation TF 구성 기획안", size=44, bold=True,
     color=WHITE, align=PP_ALIGN.CENTER)
ln = s.shapes.add_connector(1, W / 2 - Inches(0.8), Inches(4.15), W / 2 + Inches(0.8), Inches(4.15))
ln.line.color.rgb = AMBER
ln.line.width = Pt(3)
text(s, M, Inches(4.4), W - 2 * M, Inches(0.5), "2026-12", size=18, color=PALE, align=PP_ALIGN.CENTER)

# ---------- 2. 요청: 지정군 ----------
s = slide("R&D MI 현업 엔지니어들과 만드는 자동화 공통 플랫폼: TF 구성 요청",
          "요청드리는 것은 TF 구성과 지정군 엔지니어들의 협업입니다. 1차 지정군은 CD-SEM, Thickness, VeritySEM입니다. 이미 ARC 작업이 진척 중이고 엔지니어들도 데이터 흐름과 작업 방식을 알고 있어, 공통 플랫폼의 기초 틀을 만들기에 가장 적합합니다. 2차 지정군은 Inspection 장비군으로, 선택 사항입니다. 1차에서 얻은 노하우를 적용해 공통 플랫폼이 실제로 다른 장비군에도 통하는지 확인합니다. 개발은 2명이 각자 업무 시간의 50%를 투입해 실질 1명으로 운영합니다.")
box(s, M, BODY_Y, Inches(7.6), Inches(4.7), fill=MIST)
text(s, M + Inches(0.4), BODY_Y + Inches(0.3), Inches(4), Inches(0.4), "1차 지정군", size=15, bold=True, color=AMBER)
text(s, M + Inches(0.4), BODY_Y + Inches(0.7), Inches(7), Inches(0.5), "공통 플랫폼의 기초 틀", size=22, bold=True)
for i, eq in enumerate(["CD-SEM", "Thickness", "VeritySEM"]):
    x = M + Inches(0.4) + i * Inches(2.3)
    box(s, x, BODY_Y + Inches(1.45), Inches(2.1), Inches(0.9), fill=INK)
    text(s, x, BODY_Y + Inches(1.45), Inches(2.1), Inches(0.9), eq, size=18, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, M + Inches(0.4), BODY_Y + Inches(2.65), Inches(6.9), Inches(1.9),
     [[("지정 사유  ", {"bold": True}), ("이미 ARC 작업이 진척 중이며 엔지니어들도 기본적인 데이터 흐름과 작업 방식을 알고 있음", {})],
      [("요청  ", {"bold": True}), ("담당 엔지니어의 TF 조직도 협업 (장비군마다 1~2명, 주 5시간)", {})]],
     size=15, color=INK_SOFT)
x2 = M + Inches(7.9)
w2 = W - M - x2
box(s, x2, BODY_Y, w2, Inches(2.75), fill=WHITE, line=LINE)
text(s, x2 + Inches(0.35), BODY_Y + Inches(0.3), w2 - Inches(0.7), Inches(0.4), "2차 지정군 (선택)", size=15, bold=True, color=AMBER)
text(s, x2 + Inches(0.35), BODY_Y + Inches(0.7), w2 - Inches(0.7), Inches(0.5), "Inspection 장비군", size=22, bold=True)
text(s, x2 + Inches(0.35), BODY_Y + Inches(1.35), w2 - Inches(0.7), Inches(1.3),
     ["1차 지정군에서 얻은 작업 노하우를 적용", "공통 플랫폼을 실제로 적용하는 첫 대상"], size=14, color=INK_SOFT)
box(s, x2, BODY_Y + Inches(2.95), w2, Inches(1.75), fill=INK)
text(s, x2 + Inches(0.35), BODY_Y + Inches(3.15), w2 - Inches(0.7), Inches(0.4), "개발 인력", size=15, bold=True, color=AMBER)
text(s, x2 + Inches(0.35), BODY_Y + Inches(3.55), w2 - Inches(0.7), Inches(1.0),
     [[("2명 × 50%", {"size": 24, "bold": True}), ("  = 실질 1명", {"size": 16})],
      "개인 과제와 병행, 교육도 담당"], size=14, color=WHITE)
footer(s)

# ---------- 4. 문제 ----------
s = slide("숙련자 손에 의존하는 레시피 셋업과 Align Fail 장비 정지",
          "레시피 셋업은 레시피 한 건에 평균 30~60분이 듭니다. 현업 경험치입니다. 특히 측정 위치와 Align을 잡는 구간에서 측정, 실패, 재조정을 되풀이합니다. 장비에 조작 API가 없어 원격 화면에서 사람이 직접 해야 하고, 그 판단은 숙련자에게만 있습니다. 정확한 구간 시간은 온보딩 기간에 담당 엔지니어가 직접 재서 확정합니다.")
box(s, M, BODY_Y, Inches(4.3), Inches(4.9), fill=INK)
text(s, M + Inches(0.4), BODY_Y + Inches(0.6), Inches(3.6), Inches(1.4), "30~60분", size=60, bold=True, color=WHITE)
text(s, M + Inches(0.4), BODY_Y + Inches(2.1), Inches(3.6), Inches(1.2),
     ["레시피 1건 셋업 시간", "현업 경험치, 셋업 전체 기준"], size=16, color=PALE)
text(s, M + Inches(0.4), BODY_Y + Inches(3.55), Inches(3.6), Inches(1.0),
     "자동화 대상 구간 실측은 2027-01~02 온보딩 기간", size=13, color=PALE)
rows = [
    ("시행착오 반복", "측정 위치, Align을 잡을 때 측정 → 실패 → 재조정. 그동안 화면 앞을 떠날 수 없음"),
    ("노하우가 사람에게만", "판단 기준이 숙련자의 눈과 손에 있어 사람마다 품질이 다르고, 담당자가 바뀌면 사라짐"),
    ("장비가 늘면 사람도", "장비, 제품이 늘면 셋업 인력도 같은 비율로 늘어야 함. Align Fail 때는 장비도 멈춤"),
]
for i, (head, body) in enumerate(rows):
    y = BODY_Y + i * Inches(1.65)
    x = M + Inches(4.7)
    badge(s, x, y + Inches(0.1), str(i + 1))
    text(s, x + Inches(0.8), y, Inches(6.3), Inches(0.5), head, size=20, bold=True)
    text(s, x + Inches(0.8), y + Inches(0.5), Inches(6.3), Inches(1.0), body, size=15, color=INK_SOFT)
footer(s)

# ---------- 3. 핵심 개발 방향 ----------
s = slide("핵심 개발 방향: AI가 지식 자산으로 레시피를 셋업할 수 있는가",
          "가장 중요한 개발 방향입니다. 엔지니어의 마우스, 키보드 조작과 레시피 셋업 방법을 지식 자산으로 만들고, 공통 플랫폼을 통해 AI가 그것을 그대로 수행할 수 있느냐입니다. 2027년에는 두 가지를 타겟으로 합니다. 상반기에는 CD-SEM과 Thickness의 Align Fail 자동 대응을 완성하고, 연말까지 1차 지정군의 셋업 절차와 노하우를 지식 자산으로 쌓습니다. 셋업 시간 단축은 2028년 보조 단계, Agent가 셋업을 진행하는 자율 단계는 2029년 타겟입니다.")
flow = [("엔지니어 작업", "화면 녹화, 마우스, 키보드 조작"), ("지식 자산", "절차, 노하우, 결정 지점"),
        ("공통 플랫폼", "화면 인식, 입력 실행, 안전 규칙"), ("2029 자율 목표", "CD-SEM, Thickness 셋업 진행, 위험 시 엔지니어 개입")]
fw = Inches(2.75)
fg = (W - 2 * M - 4 * fw) / 3
for i, (h, b) in enumerate(flow):
    x = M + i * (fw + fg)
    last = i == len(flow) - 1
    box(s, x, BODY_Y, fw, Inches(1.7), fill=AMBER if last else (INK if i == 2 else MIST))
    fgc = WHITE if (last or i == 2) else INK
    text(s, x + Inches(0.2), BODY_Y + Inches(0.3), fw - Inches(0.4), Inches(0.5), h, size=20, bold=True,
         color=fgc, align=PP_ALIGN.CENTER)
    text(s, x + Inches(0.2), BODY_Y + Inches(0.9), fw - Inches(0.4), Inches(0.7), b, size=13,
         color=WHITE if (last or i == 2) else INK_SOFT, align=PP_ALIGN.CENTER)
    if not last:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + fw + Inches(0.05), BODY_Y + Inches(0.7), fg - Inches(0.1), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = LINE; a.line.fill.background()
text(s, M, Inches(3.65), Inches(4), Inches(0.4), "2027 타겟", size=16, bold=True, color=AMBER)
for i, (when, head, body) in enumerate([
    ("2027 상반기, 6월 판정", "Align Fail 자동 대응 완성", "CD-SEM, Thickness 실알람 무인 대응. 장비 가동 시간 회수"),
    ("2027 연말, 12월 판정", "지식 자산 축적", "1차 지정군 셋업 절차, 노하우를 축적, 막힌 지점에서 바로 찾아 씀")]):
    x = M + i * Inches(6.2)
    box(s, x, Inches(4.1), Inches(5.9), Inches(1.75), fill=MIST)
    text(s, x + Inches(0.4), Inches(4.3), Inches(5.1), Inches(0.35), when, size=13, color=INK_SOFT)
    text(s, x + Inches(0.4), Inches(4.65), Inches(5.1), Inches(0.5), head, size=21, bold=True)
    text(s, x + Inches(0.4), Inches(5.2), Inches(5.1), Inches(0.6), body, size=14, color=INK_SOFT)
text(s, M, Inches(6.1), W - 2 * M, Inches(0.4),
     "이후: 2028 보조 (엔지니어 앞에서 반복 입력 대행, 셋업 시간 단축)  →  2029 자율 (Agent가 셋업 진행)",
     size=14, color=INK_SOFT)
footer(s)

# ---------- 5. 효과 ----------
s = slide("타겟 시점별 기대 효과와 측정 지표",
          "효과는 언제 확인할 수 있는지로 나눴습니다. 2027 상반기에는 Align Fail로 멈추던 장비 시간을 회수합니다. 연말에는 노하우가 절차서로 남고 막힌 지점에서 바로 쓰입니다. 엔지니어 투입과 셋업 시간 감소는 2028년 보조 단계부터입니다. 현재 값은 아직 비어 있는 칸이 많습니다. 측정하기 전에는 목표를 숫자로 정하지 않고, 온보딩 기간에 기준선을 재서 채우겠습니다.")
metrics = [("효과", "측정 지표", "현재", "목표", "타겟 시점"),
           ("장비 가동 시간 회수", "Align Fail 장비 정지 시간 감소분 (월)", "⟨확인⟩", "⟨확인⟩", "2027 상반기"),
           ("지식 자산화", "절차서로 정리된 레시피 유형 수", "0", "장비군마다 ⟨N⟩개", "2027 연말"),
           ("막힘 해소", "돌파구 추천 채택률", "-", "⟨확인⟩%", "2027 연말"),
           ("엔지니어 투입 감소", "건당 직접 투입 시간, 예정 밖 개입", "⟨확인⟩", "⟨확인⟩", "2028"),
           ("셋업 시간 단축", "대상 구간 셋업 경과 시간", "대상 구간 실측 예정", "⟨확인⟩% 단축", "2028"),
           ("품질 균일화", "셋업 뒤 첫 측정 성공률", "⟨확인⟩", "⟨확인⟩", "2029")]
table(s, metrics, M, BODY_Y - Inches(0.05), [Inches(2.6), Inches(4.2), Inches(1.9), Inches(1.9), Inches(1.533)], size=14,
      row_h=Inches(0.56), bold_col0=True)
text(s, M, Inches(6.05), W - 2 * M, Inches(0.7),
     ["측정 기준: 측정 위치와 Align 셋업 구간, 실패와 재작업 포함, 승인 대기 별도 기록. 회사 효과 = 월 셋업 건수 × 건당 직접 투입 감소 시간",
      "확정 시점: 기준선 2027-02 / Align Fail 판정 기준 2027-01 말 / 지식화 목표는 시범 실측 후 확정"],
     size=12, color=INK_SOFT, spacing=1.0)
footer(s)

# ---------- 6. 이미 확보한 것 ----------
s = slide("0에서 시작하지 않음: PoC 기반과 현업 자동화 재사용",
          "1차 PoC의 확인 범위는 화면 조작과 실장비 반자동 운영입니다. 실알람 무인 대응 완성은 2027 상반기 목표입니다. API 없는 장비 화면을 사내 VLM으로 읽고 조작할 수 있다는 것을 확인했고, 최근 SecureVDI 경유 화면에도 입력할 수 있는 것을 확인했습니다. 그래서 VeritySEM과 Thickness도 같은 방식으로 다룰 수 있습니다. 또 Thickness 담당자들이 PyAutoGUI로 자동화해 둔 작업이 많습니다. 정해진 반복 입력은 이 스크립트를 재사용하고, 화면 인식과 판단이 필요한 구간은 VLM이 맡는 혼합 방식을 검토하겠습니다.")
cols = [("확인한 것 (PoC)", INK, WHITE, "✓  ", [
            "사내 VLM, OCR로 API 없는 장비 화면을 보고 조작",
            "VLM은 \"어디를 볼지\", CV는 \"정확히 어디인지\" 역할 분리",
            "기준 이미지로 Align Key 탐색 (오프라인 평가)",
            "반자동 운영: 위치 재조정은 자동, 확정은 엔지니어",
            "SecureVDI 경유 화면에 입력 가능 (2026-11 확인)"]),
        ("혼합 검토: Thickness PyAutoGUI", MIST, INK, "→  ", [
            "현업이 PyAutoGUI로 자동화해 둔 반복 작업이 많음",
            "정해진 순서의 반복 입력: 기존 스크립트 재사용",
            "위치 확인, 예외 화면, 막힘: VLM이 판단",
            "스크립트 앞뒤에 VLM 화면 확인, 다르면 멈춤",
            "온보딩 때 목록화, 2027-06 요소 목록에 반영"])]
for i, (head, fill, fg, mark, items) in enumerate(cols):
    x = M + i * Inches(6.2)
    box(s, x, BODY_Y, Inches(5.9), Inches(4.75), fill=fill)
    text(s, x + Inches(0.4), BODY_Y + Inches(0.35), Inches(5.1), Inches(0.5), head, size=21, bold=True, color=fg)
    text(s, x + Inches(0.4), BODY_Y + Inches(1.1), Inches(5.1), Inches(3.7),
         [[(mark, {"color": AMBER, "bold": True}), (t, {})] for t in items], size=15, color=fg, spacing=1.15)
text(s, M, Inches(6.6), W - 2 * M, Inches(0.35),
     [[("추가 검증  ", {"bold": True, "color": AMBER}), ("SecureVDI 경유 VLM 인식 정확도, PyAutoGUI 혼합 안정성 (2027 상반기 Thickness 운영)", {})]],
     size=13, color=INK_SOFT)
footer(s)

# ---------- 7. 범위 ----------
s = slide("시행착오가 가장 많은 측정 위치, Align 셋업부터 단계적 자동화",
          "레시피 셋업은 생성부터 저장까지 다섯 단계입니다. 자동화 대상은 가운데, 측정 위치와 Align 셋업 구간입니다. 2027년에는 이 구간을 관찰하고 지식화하고, 2028년부터 엔지니어 앞에서 Agent가 입력을 대행합니다. Agent는 엔지니어 본인 PC에서 돌고 장비 PC에는 아무것도 설치하지 않습니다. 측정 실행과 저장 같은 위험 조작은 반드시 엔지니어가 승인합니다. Align Fail 무인 대응만 예외로, 위치 재조정과 재측정 재시도만 허용합니다.")
flow = [("레시피 생성,\n기본 정보", False), ("측정 위치,\nAlign 셋업", True), ("파라미터\n설정", False), ("검증 측정\n판정", False), ("저장", False)]
fw = Inches(2.15)
for i, (t, agent) in enumerate(flow):
    x = M + i * (fw + Inches(0.36))
    box(s, x, BODY_Y + Inches(0.2), fw, Inches(1.5), fill=AMBER if agent else MIST)
    text(s, x, BODY_Y + Inches(0.2), fw, Inches(1.5), t.split("\n"), size=17, bold=True,
         color=WHITE if agent else INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1.0)
    text(s, x, BODY_Y + Inches(1.85), fw, Inches(0.4), "자동화 대상" if agent else "엔지니어", size=14,
         bold=agent, color=AMBER if agent else INK_SOFT, align=PP_ALIGN.CENTER)
text(s, M, BODY_Y + Inches(2.3), W - 2 * M, Inches(0.4),
     "자동화 대상 구간: 2027 관찰과 지식화  →  2028 엔지니어 앞 입력 대행  →  2029 자율 목표", size=15, bold=True,
     color=INK, align=PP_ALIGN.CENTER)
facts = [("반자동이 원칙", "위험 조작(측정 실행, 삭제)과\n레시피 저장 직전에 엔지니어 승인"),
         ("Align Fail은 승인된 예외", "무인 허용: 레시피 변경 없는 위치 재조정과 해당 지점 재측정만"),
         ("엔지니어 PC에서 실행", "CD-SEM은 RCS, VeritySEM, Thickness는 SecureVDI 경유. 장비 PC 설치 없음")]
for i, (h, b) in enumerate(facts):
    x = M + i * Inches(4.1)
    box(s, x, Inches(4.35), Inches(3.85), Inches(1.9), fill=WHITE, line=LINE)
    text(s, x + Inches(0.3), Inches(4.55), Inches(3.3), Inches(0.45), h, size=17, bold=True)
    text(s, x + Inches(0.3), Inches(5.05), Inches(3.3), Inches(1.1), b.split("\n"), size=14, color=INK_SOFT, spacing=1.0)
footer(s)

# ---------- 8. TF 구성 ----------
s = slide("엔진은 공통 플랫폼 팀, 절차는 장비를 아는 담당 엔지니어",
          "장비마다 자동화를 따로 만들면 화면 인식, 조작, 안전 장치를 장비 수만큼 다시 만들게 됩니다. 반대로 개발자가 모든 장비의 레시피 절차를 알 수는 없습니다. 그래서 공통 플랫폼 팀이 엔진을 만들고, 장비군 담당 엔지니어들이 절차와 판정 기준을 맡습니다. 2차 지정군 Inspection은 선택이며, 참여하면 1월에 함께 합류해 미리 학습하고 작업을 녹화해 둡니다.")
top_w = Inches(9.0)
tx = (W - top_w) / 2
box(s, tx, BODY_Y, top_w, Inches(1.45), fill=INK)
text(s, tx + Inches(0.4), BODY_Y + Inches(0.2), top_w - Inches(0.8), Inches(0.5),
     "TF 리드 + 공통 플랫폼 팀 (개발 2명, 각 50%)", size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
text(s, tx + Inches(0.4), BODY_Y + Inches(0.8), top_w - Inches(0.8), Inches(0.5),
     "화면 인식, 조작 엔진, 안전 규칙, 승인 화면, VLM 연동, 교육", size=15, color=PALE, align=PP_ALIGN.CENTER)
cells = [("1차 지정군", "CD-SEM", "TF 리드 직접 담당", True), ("1차 지정군", "Thickness", "MI 엔지니어, PyAutoGUI 공유", True),
         ("1차 지정군", "VeritySEM", "MI 엔지니어", True), ("2차 지정군 (선택)", "Inspection", "학습, 작업 녹화", False)]
cw = Inches(2.8)
gap = (W - 2 * M - 4 * cw) / 3
for x0, y0, x1, y1 in ((W / 2, BODY_Y + Inches(1.45), W / 2, Inches(3.35)),
                       (M + cw / 2, Inches(3.35), W - M - cw / 2, Inches(3.35))):
    ln = s.shapes.add_connector(1, x0, y0, x1, y1)
    ln.line.color.rgb = LINE
for i, (a, b, c, first) in enumerate(cells):
    x = M + i * (cw + gap)
    y = Inches(3.75)
    ln = s.shapes.add_connector(1, x + cw / 2, Inches(3.35), x + cw / 2, y)
    ln.line.color.rgb = LINE
    box(s, x, y, cw, Inches(1.9), fill=MIST if first else WHITE, line=None if first else LINE)
    text(s, x + Inches(0.3), y + Inches(0.25), cw - Inches(0.6), Inches(0.4), a, size=14, bold=True, color=AMBER)
    text(s, x + Inches(0.3), y + Inches(0.7), cw - Inches(0.6), Inches(0.5), b, size=19, bold=True)
    text(s, x + Inches(0.3), y + Inches(1.25), cw - Inches(0.6), Inches(0.5), c, size=13, color=INK_SOFT)
text(s, M, Inches(5.85), W - 2 * M, Inches(1.0),
     [[("상반기  ", {"bold": True, "color": AMBER}), ("TF 리드는 Align Fail (CD-SEM 겸임), 개발 1명은 녹화분 확보와 수집 기반", {})],
      [("하반기  ", {"bold": True, "color": AMBER}), ("TF 리드는 절차화와 지식 자산, 개발 1명은 상시 관찰 모드와 운영 유지", {})]],
     size=14, spacing=1.0)
footer(s)

# ---------- 9. 일정 칸반 ----------
s = slide("2027 일정: 상반기 Align Fail 완성, 연말 지식 자산 축적",
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
lw = Inches(1.55)
colw = (W - 2 * M - lw) / len(periods)
hy = BODY_Y - Inches(0.05)
for j, p in enumerate(periods):
    x = M + lw + j * colw
    box(s, x + Inches(0.04), hy, colw - Inches(0.08), Inches(0.42), fill=INK, shape=MSO_SHAPE.RECTANGLE)
    text(s, x, hy, colw, Inches(0.42), p, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
card_h, card_g = Inches(0.4), Inches(0.06)
lane_hs = [Inches(1.5), Inches(1.5), Inches(0.6)]
y = hy + Inches(0.52)
for li, ((name, cols_), lh) in enumerate(zip(lanes, lane_hs)):
    box(s, M, y, W - 2 * M, lh, fill=LANE[li], shape=MSO_SHAPE.RECTANGLE)
    text(s, M + Inches(0.12), y, lw - Inches(0.2), lh, name.split("\n"), size=13, bold=True, anchor=MSO_ANCHOR.MIDDLE, spacing=1.0)
    for j, items in enumerate(cols_):
        x = M + lw + j * colw + Inches(0.08)
        for k, it in enumerate(items):
            cy = y + Inches(0.12) + k * (card_h + card_g)
            gate = it.startswith("★")
            box(s, x, cy, colw - Inches(0.16), card_h, fill=AMBER if gate else WHITE, radius=0.15)
            text(s, x + Inches(0.1), cy, colw - Inches(0.3), card_h, it, size=13, bold=gate,
                 color=WHITE if gate else INK, anchor=MSO_ANCHOR.MIDDLE)
    y += lh + Inches(0.1)
text(s, M, y + Inches(0.02), W - 2 * M, Inches(0.95),
     [[("6월 관문  ", {"bold": True, "color": AMBER}), ("자동 해결률 기준 충족, 오측정 0건, 최소 평가 건수와 기간 충족", {})],
      [("12월 관문  ", {"bold": True, "color": AMBER}), ("장비군마다 절차서 ⟨N⟩개, 돌파구 추천 채택률, 엔지니어 PC 자원 한도 충족", {})],
      [("2028 진입  ", {"bold": True, "color": AMBER}), ("지식화 관문 통과와 입력 실행기 안전 요건 충족", {})]],
     size=13, color=INK_SOFT, spacing=1.0)
footer(s)

# ---------- 10. 확인 필요 사항 ----------
s = slide("녹화 DB 접근 가능 여부 12월 확정 요청과 스토리지 계획",
          "R&D MI H/W 담당자께 확인이 필요한 사항이 하나 있습니다. 장비 화면 녹화분 DB에 직접 접근할 수 있는지입니다. 가능하면 녹화분을 바로 받아 써서 개발 속도가 빨라집니다. 불가능하면 RCS에서 GUI 자동화로 녹화분을 내려받아야 합니다. 속도가 느리고, 다운로드 받는 자동화 작업이 별도로 필요합니다. 또 스크린샷과 녹화 영상을 매우 많이 쓰기 때문에 스토리지 소모가 큽니다. 2027년에는 할당 자원 안에서 보존 기한을 정해 운영하고, 사용량을 실측해 추후 스토리지 자산 확보를 요청드리겠습니다.")
box(s, M, BODY_Y + Inches(0.6), Inches(3.6), Inches(2.2), fill=INK)
text(s, M + Inches(0.3), BODY_Y + Inches(0.6), Inches(3.0), Inches(2.2),
     [[("확인 필요", {"size": 13, "color": AMBER, "bold": True})], "장비 화면 녹화분 DB 직접 접근", [("확인 담당: R&D MI H/W", {"size": 13, "color": PALE})], [("기한: 2026-12", {"size": 13, "color": PALE})]],
     size=18, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
for i, (h, b, fill, fg) in enumerate([
        ("가능하면", "녹화분을 바로 받아 사용, 개발 속도가 빨라짐", MIST, INK),
        ("불가능하면", "RCS에서 GUI 자동화로 다운로드, 속도 느림, 다운로드 자동화 작업 별도 필요", WHITE, INK)]):
    y = BODY_Y + i * Inches(1.75)
    x = M + Inches(4.4)
    ln = s.shapes.add_connector(1, M + Inches(3.6), BODY_Y + Inches(1.7), x, y + Inches(0.75))
    ln.line.color.rgb = LINE
    ln.line.width = Pt(2)
    box(s, x, y, W - M - x, Inches(1.5), fill=fill, line=None if i == 0 else LINE)
    text(s, x + Inches(0.4), y + Inches(0.2), Inches(6.5), Inches(0.45), h, size=19, bold=True, color=AMBER if i == 0 else INK)
    text(s, x + Inches(0.4), y + Inches(0.7), W - M - x - Inches(0.8), Inches(0.7), b, size=15, color=INK_SOFT)
box(s, M, Inches(5.25), W - 2 * M, Inches(1.2), fill=MIST)
text(s, M + Inches(0.4), Inches(5.25), W - 2 * M - Inches(0.8), Inches(1.2),
     [[("스토리지  ", {"bold": True, "color": AMBER}),
       ("스크린샷, 녹화 영상으로 소모량이 매우 큼. 2027 할당 자원과 보존 기한으로 운영, 사용량 실측 후 하반기 자산 확보 요청", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE)
footer(s)

# ---------- 11. 안전과 위험 ----------
s = slide("위험 조작은 엔지니어 승인, 무인 운영은 Align Fail 재조정만",
          "가장 걱정하실 부분은 Agent가 생산 레시피를 잘못 건드리는 것이라고 생각합니다. 측정 실행, 저장, 삭제 같은 위험 조작은 반드시 엔지니어가 승인하고, 엔지니어는 언제든 즉시 멈출 수 있습니다. Align Fail 무인 대응도 위치 재조정과 재측정 재시도만 허용합니다. 운영상 큰 위험은 겸직 시간과 개발 인력입니다. 개발 인력이 실질 1명이라 상반기는 Align Fail을 우선하고, 매월 진척을 보고해 늦어지면 범위 축소를 요청드리겠습니다.")
risks = [("오조작", "위험 조작, 저장 전 엔지니어 승인, 즉시 중단 수단. 무인 전환은 장비별 승인"),
         ("개발 인력 실질 1명", "상반기 Align Fail 우선, 관찰은 녹화 축적까지. 매월 진척 보고, 늦으면 범위 축소"),
         ("겸직 시간 부족", "주 5시간을 기획안에 명시, 월간 보고에 참여 시간 실적 보고"),
         ("장비군마다 다른 화면, 절차", "공통 엔진과 장비별 절차를 분리, 1차는 3개 장비군으로 제한")]
for i, (h, b) in enumerate(risks):
    col, row = i % 2, i // 2
    x = M + col * Inches(6.2)
    y = BODY_Y + row * Inches(2.0)
    box(s, x, y, Inches(5.9), Inches(1.75), fill=MIST)
    text(s, x + Inches(0.4), y + Inches(0.25), Inches(5.1), Inches(0.45), h, size=19, bold=True)
    text(s, x + Inches(0.4), y + Inches(0.8), Inches(5.1), Inches(0.9), b, size=15, color=INK_SOFT)
text(s, M, Inches(5.75), W - 2 * M, Inches(0.6),
     [[("가드레일  ", {"bold": True, "color": AMBER}),
       ("Align Fail 무인 운영에서 저장, 삭제, 모드 변경, 레시피 수정 금지. 불확실하면 자체 중단하고 엔지니어 호출. 오측정 0건", {})]], size=15)
footer(s)

# ---------- 13. 마무리 ----------
s = slide(None, "다시 한 번 요청드립니다. 2027년 1월부터 TF 운영과 지정군 엔지니어들의 협업, 개발 인력 배정을 승인해 주십시오. 승인해 주시면 1월에 온보딩을 시작하고, 기준선은 2월에 측정하며, 효과 목표는 지표별 확정 일정에 맞춰 보고드리겠습니다.", dark=True)
text(s, M, Inches(1.9), W - 2 * M, Inches(1.0), "결정을 요청드립니다", size=40, bold=True, color=WHITE)
for i, t_ in enumerate(["2027-01~12 ARC TF 운영 승인",
                        "CD-SEM, Thickness, VeritySEM 담당 지정: 장비군별 1~2명, 주 5시간",
                        "TF 리드 포함 개발 2명 배정: 각 업무 시간 50%, 교육 포함"]):
    y = Inches(3.2) + i * Inches(0.8)
    badge(s, M, y, str(i + 1), fill=AMBER, d=Inches(0.5), size=15)
    text(s, M + Inches(0.8), y, Inches(11), Inches(0.5), t_, size=20, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
text(s, M, Inches(5.65), W - 2 * M, Inches(0.4),
     [[("선택  ", {"bold": True, "color": AMBER}), ("2차 지정군 Inspection 장비군 참여 여부", {})]], size=16, color=WHITE)
text(s, M, Inches(6.25), W - 2 * M, Inches(0.6),
     "승인 후: 2027-01 온보딩 시작 → 2027-02 장비군별 기준선 측정 → 지표별 확정 일정에 맞춰 효과 목표 보고", size=16, color=PALE)

# ---------- 12. 요청 사항 ----------
s = slide("별첨: 요청 사항 상세",
          "요청 사항은 일곱 가지입니다. 12월에 결정을 부탁드리는 것은 위 세 가지, TF 구성과 지정군 엔지니어 협업, 개발 인력입니다. 인프라와 서버, 검증 장비는 1월 가동 전까지 협의하고, 스토리지는 사용량을 실측한 뒤 하반기에 요청드리겠습니다.")
asks = [("시점", "요청"),
        ("2026-12", "ARC TF 구성 승인"),
        ("2026-12", "1차 지정군(CD-SEM, Thickness, VeritySEM) 담당 엔지니어 협업 (장비군마다 1~2명, 주 5시간). 2차 지정군 Inspection은 선택"),
        ("2026-12", "공통 플랫폼 팀 개발 인력 2명 (TF 리드 포함, 1인당 50%, 실질 1명, 교육 포함)"),
        ("2027-01 전", "사내 추론 인프라 사용 보장, 추가 비용 승인 주체 지정"),
        ("2027-01 전", "사내 서버 자원 할당 (저장소, 검색, DB, 캐시, 새 서버 없음)"),
        ("2027-01 전", "비생산 검증용 장비 사용 인가 (1차 지정군마다 하루 1시간)"),
        ("2027 하반기", "녹화 스토리지 자산 확보 (사용량 실측 근거, 예상 용량 ⟨확인⟩)")]
t = s.shapes.add_table(len(asks), 2, M, BODY_Y, W - 2 * M, Inches(4.6)).table
t.columns[0].width, t.columns[1].width = Inches(2.1), W - 2 * M - Inches(2.1)
for i, (when, what) in enumerate(asks):
    today = when == "2026-12"
    for j, val in enumerate((when, what)):
        c = t.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = INK if i == 0 else (MIST if today else WHITE)
        tf = c.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].text = ""
        r = tf.paragraphs[0].add_run()
        r.text = val
        r.font.name, r.font.size = FONT, Pt(14)
        r.font.bold = i == 0 or today
        r.font.color.rgb = WHITE if i == 0 else (AMBER if (today and j == 0) or "⟨" in val else INK)
text(s, M, Inches(6.5), W - 2 * M, Inches(0.4), "음영 행: 12월에 결정을 요청하는 항목", size=12, color=INK_SOFT)
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
