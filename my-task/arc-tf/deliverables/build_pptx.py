# -*- coding: utf-8 -*-
"""ARC TF 임원 보고 deck (.pptx) 생성.

    python my-task/arc-tf/deliverables/build_pptx.py

deck 은 01-tf-proposal.md 와 00-problem-statement.md 의 요약이다. 그 두 문서를 고치면
아래 SLIDES 내용도 같이 고친다 (자동 동기화 안 됨). 필요: python-pptx. 한글 폰트: 맑은 고딕.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

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

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)
TITLE_Y, TITLE_H = Inches(0.45), Inches(0.9)
BODY_Y = Inches(1.6)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
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


def footer(s, n):
    text(s, M, H - Inches(0.45), Inches(6), Inches(0.3), "ARC TF 구성 기획안 · 초안", size=10, color=INK_SOFT)
    text(s, W - M - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(n), size=10,
         color=INK_SOFT, align=PP_ALIGN.RIGHT)


TBD = AMBER  # ⟨확인⟩ 표시 색


# ---------- 1. 표지 ----------
s = slide(None, "ARC TF 구성 기획안을 보고드립니다. 결론부터 말씀드리고, 왜 필요한지, 어떻게 운영할지, 무엇을 요청드리는지 순서로 설명하겠습니다.", dark=True)
text(s, M, Inches(2.2), Inches(10), Inches(1.2), "ARC TF 구성 기획안", size=48, bold=True, color=WHITE)
text(s, M, Inches(3.35), Inches(10), Inches(0.6), "레시피(Recipe) 셋업 자동화 태스크포스", size=22, color=PALE)
text(s, M, Inches(5.4), Inches(10), Inches(0.9),
     ["보고 대상: AI TF 담당 임원, 기반기술센터장", "2026-12 · TF 리드"], size=15, color=PALE)

# ---------- 2. 결론: 요청 ----------
s = slide("2027년 1월부터 ARC TF 운영을 승인해 주십시오",
          "오늘 결정을 부탁드리는 것은 두 가지입니다. TF 구성 승인과, 장비군별 MI 엔지니어의 겸직 참여입니다. 겸직은 근무일마다 1시간, 주 5시간입니다. 첫 성과는 2027년 6월, CD-SEM과 Thickness의 Align Fail 자동 대응 완성으로 보고드리겠습니다.")
for i, (head, body) in enumerate([
    ("TF 구성 승인", "공통 플랫폼 팀(개발 2명)과 장비군 담당으로 구성된 ARC TF를 2027-01부터 운영"),
    ("MI 엔지니어 겸직 참여", "1차 장비군(CD-SEM, VeritySEM, Thickness)마다 1~2명, 주 5시간 (근무일마다 1시간)"),
]):
    x = M + i * Inches(6.2)
    box(s, x, Inches(2.0), Inches(5.9), Inches(3.2), fill=MIST)
    badge(s, x + Inches(0.4), Inches(2.4), str(i + 1), fill=AMBER)
    text(s, x + Inches(0.4), Inches(3.15), Inches(5.1), Inches(0.6), head, size=24, bold=True)
    text(s, x + Inches(0.4), Inches(3.85), Inches(5.1), Inches(1.2), body, size=16, color=INK_SOFT)
text(s, M, Inches(5.65), W - 2 * M, Inches(0.8),
     "목표 (2027): 상반기 CD-SEM·Thickness Align Fail 자동 대응 완성, 연말까지 1차 3개 장비군 셋업 절차·노하우를 지식 자산으로 축적",
     size=16, color=INK)
footer(s, 2)

# ---------- 3. 문제 ----------
s = slide("숙련 엔지니어가 레시피 셋업에 손으로 매달려 있습니다",
          "레시피 셋업은 레시피 한 건에 평균 30~60분이 듭니다. 현업 경험치입니다. 특히 측정 위치와 Align을 잡는 구간에서 측정, 실패, 재조정을 되풀이합니다. 장비에 조작 API가 없어 원격 화면에서 사람이 직접 해야 하고, 그 판단은 숙련자에게만 있습니다. 정확한 구간 시간은 온보딩 기간에 담당 엔지니어가 직접 재서 확정합니다.")
box(s, M, BODY_Y, Inches(4.3), Inches(4.9), fill=INK)
text(s, M + Inches(0.4), BODY_Y + Inches(0.6), Inches(3.6), Inches(1.4), "30~60분", size=60, bold=True, color=WHITE)
text(s, M + Inches(0.4), BODY_Y + Inches(2.1), Inches(3.6), Inches(1.2),
     ["레시피 1건 셋업 시간", "현업 경험치, 셋업 전체 기준"], size=16, color=PALE)
text(s, M + Inches(0.4), BODY_Y + Inches(3.55), Inches(3.6), Inches(1.0),
     "자동화 대상 구간 실측은 2027-02 온보딩 8주차", size=13, color=PALE)
rows = [
    ("시행착오 반복", "측정 위치·Align을 잡을 때 측정 → 실패 → 재조정. 그동안 화면 앞을 떠날 수 없다"),
    ("노하우가 사람에게만", "판단 기준이 숙련자의 눈과 손에 있어, 사람마다 품질이 다르고 담당자가 바뀌면 사라진다"),
    ("장비가 늘면 사람도", "API가 없어 스크립트 자동화가 안 된다. 셋업 인력이 장비 수에 비례해 늘어야 한다"),
]
for i, (head, body) in enumerate(rows):
    y = BODY_Y + i * Inches(1.65)
    x = M + Inches(4.7)
    badge(s, x, y + Inches(0.1), str(i + 1))
    text(s, x + Inches(0.8), y, Inches(6.3), Inches(0.5), head, size=20, bold=True)
    text(s, x + Inches(0.8), y + Inches(0.5), Inches(6.3), Inches(1.0), body, size=15, color=INK_SOFT)
footer(s, 3)

# ---------- 4. 효과 ----------
s = slide("셋업 시간이 줄면 엔지니어 시간이 회사에 돌아옵니다",
          "효과는 이렇게 이어집니다. 건당 엔지니어 투입 시간이 줄면 그만큼 엔지니어 시간이 회수되고, 신규 제품 도입 때 셋업 대기가 줄고, 장비가 늘어도 셋업 인력을 같은 비율로 늘리지 않아도 됩니다. 현재 값은 대부분 아직 비어 있습니다. 측정하기 전에는 목표를 숫자로 약속하지 않고, 온보딩 기간에 기준선을 재서 다음 보고 때 채우겠습니다.")
steps = ["건당 셋업 시간 감소", "엔지니어 시간 회수\n월 ⟨확인⟩시간", "신규 제품·공정 셋업 대기 단축", "장비가 늘어도 인력 비례 증가 없음"]
sw = Inches(2.7)
for i, t in enumerate(steps):
    x = M + i * (sw + Inches(0.47))
    box(s, x, BODY_Y, sw, Inches(1.5), fill=INK if i == 0 else MIST)
    text(s, x + Inches(0.25), BODY_Y, sw - Inches(0.5), Inches(1.5), t.split("\n"), size=16, bold=True,
         color=WHITE if i == 0 else INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + sw + Inches(0.1), BODY_Y + Inches(0.6), Inches(0.27), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = LINE; a.line.fill.background()
metrics = [("측정 지표", "현재", "목표"),
           ("측정 위치·Align 구간 건당 소요 시간", "셋업 전체 30~60분 (경험치)", "⟨확인⟩% 단축"),
           ("건당 예정에 없던 엔지니어 개입", "⟨확인⟩", "⟨확인⟩"),
           ("셋업 뒤 첫 측정 성공률", "⟨확인⟩", "현재 이상"),
           ("절차서로 정리된 레시피 유형", "0", "장비군마다 1개 이상")]
tbl = s.shapes.add_table(len(metrics), 3, M, Inches(3.55), W - 2 * M, Inches(2.6)).table
for j, wd in enumerate((Inches(5.6), Inches(3.8), Inches(2.733))):
    tbl.columns[j].width = wd
for i, row in enumerate(metrics):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = INK if i == 0 else (WHITE if i % 2 else MIST)
        tf = c.text_frame
        tf.paragraphs[0].text = ""
        r = tf.paragraphs[0].add_run()
        r.text = val
        r.font.name, r.font.size = FONT, Pt(14)
        r.font.bold = i == 0
        r.font.color.rgb = WHITE if i == 0 else (TBD if "⟨확인⟩" in val else INK)
text(s, M, Inches(6.3), W - 2 * M, Inches(0.4),
     "⟨확인⟩은 2027-01~02 온보딩 기간에 담당 엔지니어가 직접 측정해 확정한다. 측정 전에는 숫자로 약속하지 않는다.",
     size=12, color=INK_SOFT)
footer(s, 4)

# ---------- 5. PoC ----------
s = slide("0에서 시작하지 않습니다: Align Fail PoC로 기반을 확인했습니다",
          "1차 PoC로 Align Fail 무인 대응을 했습니다. 여기서 API 없는 장비 화면을 사내 VLM으로 읽고 조작할 수 있다는 것, 오프라인 평가에서 Align Key를 찾을 수 있다는 것을 확인했습니다. 남은 과제는 오른쪽 네 가지입니다. 앞의 세 가지는 플랫폼 팀이, 마지막 장비별 절차는 현업 엔지니어가 풀어야 합니다. 그래서 TF가 필요합니다.")
cols = [("확인한 것", INK, WHITE, [
            "사내 VLM·OCR로 API 없는 장비 화면을 보고 조작",
            "VLM은 \"어디를 볼지\", CV는 \"정확히 어디인지\" 역할 분리",
            "기준 이미지로 Align Key 탐색 (오프라인 평가)",
            "반자동 운영 원칙: 위치 재조정은 자동, 확정은 엔지니어"]),
        ("아직 남은 것", MIST, INK, [
            "현장 신뢰성: 실알람 완주, 실장비 Align 검증 → 플랫폼",
            "실행 엔진 제품화 (현재 데모 단계) → 플랫폼",
            "신규 레시피의 위치 선정 (PoC는 기존 Key만) → 플랫폼 + 장비군 담당",
            "장비군마다 다른 화면·절차 → 장비군 담당 (현업 지식 필요)"])]
for i, (head, fill, fg, items) in enumerate(cols):
    x = M + i * Inches(6.2)
    box(s, x, BODY_Y, Inches(5.9), Inches(4.9), fill=fill)
    text(s, x + Inches(0.4), BODY_Y + Inches(0.35), Inches(5.1), Inches(0.5), head, size=22, bold=True, color=fg)
    text(s, x + Inches(0.4), BODY_Y + Inches(1.1), Inches(5.1), Inches(3.6),
         [[("✓  " if i == 0 else "→  ", {"color": AMBER, "bold": True}), (t, {})] for t in items],
         size=16, color=fg, spacing=1.2)
footer(s, 5)

# ---------- 6. 범위 ----------
s = slide("1차는 시행착오가 가장 많은 한 구간만 맡깁니다",
          "레시피 셋업은 생성부터 저장까지 다섯 단계입니다. 1차에서는 가운데, 측정 위치와 Align 셋업 구간만 Agent가 맡습니다. 나머지는 지금처럼 엔지니어가 합니다. Agent는 엔지니어 본인 PC에서 돌고 장비 PC에는 아무것도 설치하지 않습니다. 측정 실행과 저장 같은 위험 조작은 반드시 엔지니어가 승인합니다.")
flow = [("레시피 생성·\n기본 정보", False), ("측정 위치·\nAlign 셋업", True), ("파라미터\n설정", False), ("검증 측정\n판정", False), ("저장", False)]
fw = Inches(2.15)
for i, (t, agent) in enumerate(flow):
    x = M + i * (fw + Inches(0.36))
    box(s, x, BODY_Y + Inches(0.3), fw, Inches(1.6), fill=AMBER if agent else MIST)
    text(s, x, BODY_Y + Inches(0.3), fw, Inches(1.6), t.split("\n"), size=17, bold=True,
         color=WHITE if agent else INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1.0)
    text(s, x, BODY_Y + Inches(2.05), fw, Inches(0.4), "Agent" if agent else "엔지니어", size=14,
         bold=agent, color=AMBER if agent else INK_SOFT, align=PP_ALIGN.CENTER)
facts = [("반자동", "Agent가 조작하고, 측정 실행·저장 직전에는 엔지니어가 승인"),
         ("엔지니어 PC에서 실행", "설치형 프로그램. 장비 PC에는 아무것도 설치하지 않는다"),
         ("범위 밖", "장비 유지보수, 하드웨어 제어. 전 과정 자동화는 2028년 이후")]
for i, (h, b) in enumerate(facts):
    x = M + i * Inches(4.1)
    text(s, x, Inches(4.65), Inches(3.8), Inches(0.45), h, size=18, bold=True)
    text(s, x, Inches(5.15), Inches(3.8), Inches(1.1), b, size=15, color=INK_SOFT)
footer(s, 6)

# ---------- 7. TF 구성 ----------
s = slide("엔진은 한 곳에서, 절차는 장비를 아는 사람이 만듭니다",
          "장비마다 자동화를 따로 만들면 화면 인식, 조작, 안전 장치를 장비 수만큼 다시 만들게 됩니다. 반대로 개발자가 모든 장비의 레시피 절차를 알 수는 없습니다. 그래서 공통 플랫폼 팀이 엔진을 만들고, 장비군 담당의 MI 엔지니어가 절차와 판정 기준을 맡습니다. 2차 장비군 엔지니어도 1월에 함께 합류해 미리 학습하고 자기 장비 작업을 녹화해 둡니다.")
top_w = Inches(9.0)
tx = (W - top_w) / 2
box(s, tx, BODY_Y, top_w, Inches(1.45), fill=INK)
text(s, tx + Inches(0.4), BODY_Y + Inches(0.2), top_w - Inches(0.8), Inches(0.5),
     "TF 리드 + 공통 플랫폼 팀 (개발 2명)", size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
text(s, tx + Inches(0.4), BODY_Y + Inches(0.8), top_w - Inches(0.8), Inches(0.5),
     "화면 인식·조작 엔진 · 안전 규칙 · 승인 화면 · VLM 연동 · 교육", size=15, color=PALE, align=PP_ALIGN.CENTER)
cells = [("CD-SEM 담당", "CD-SEM", "리드 직접 담당", True), ("VeritySEM 담당", "VeritySEM", "MI 엔지니어", True),
         ("Thickness 담당", "Thickness", "MI 엔지니어", True), ("2차 장비군 담당", "나머지 장비군", "학습·작업 녹화", False)]
cw = Inches(2.8)
gap = (W - 2 * M - 4 * cw) / 3
for i, (a, b, c, first) in enumerate(cells):
    x = M + i * (cw + gap)
    y = Inches(3.75)
    ln = s.shapes.add_connector(1, x + cw / 2, BODY_Y + Inches(1.45), x + cw / 2, y)
    ln.line.color.rgb = LINE
    box(s, x, y, cw, Inches(1.9), fill=MIST if first else WHITE, line=None if first else LINE)
    text(s, x + Inches(0.3), y + Inches(0.25), cw - Inches(0.6), Inches(0.4), a, size=14, bold=True, color=AMBER)
    text(s, x + Inches(0.3), y + Inches(0.7), cw - Inches(0.6), Inches(0.5), b, size=19, bold=True)
    text(s, x + Inches(0.3), y + Inches(1.25), cw - Inches(0.6), Inches(0.4), c, size=14, color=INK_SOFT)
text(s, M, Inches(5.95), W - 2 * M, Inches(0.6),
     "담당 엔지니어가 맡는 일: 장비별 레시피 절차 정의, 성공 판정 기준, 기준선 측정, 현장 검증, 저장 최종 승인", size=15)
footer(s, 7)

# ---------- 8. 타임라인 ----------
s = slide("상반기에 Align Fail을 완성하고, 연말까지 지식 자산을 쌓습니다",
          "2027년은 두 갈래로 갑니다. 상반기에는 Align Fail 자동 대응을 CD-SEM과 Thickness에서 빠른 첫 통과로 완성하고, 6월에 판정받겠습니다. 이것을 버리는 시제품이 아닌 플랫폼 부품으로 만들어, 필요했던 요소를 장기 계획에 반영합니다. 같은 기간에 엔지니어 작업을 녹화해 쌓고, 하반기에 절차와 노하우를 지식 자산으로 만들어 12월에 판정받겠습니다.")
phases = [("2026-12", "승인", "TF 구성 승인, 보안 승인(관찰·입력), 입력 시험"),
          ("2027-01~03", "온보딩·반자동", "8주 학습·녹화 시작, CD-SEM Align Fail 반자동"),
          ("04~06", "Align Fail 완성", "CD-SEM·Thickness 무인 대응, 6월 판정, 요소 목록"),
          ("07~09", "절차화", "녹화에서 절차 초안, 노하우 수집, 의뢰서 연동"),
          ("10~12", "지식 자산", "지식 자산 DB, 돌파구 추천, 12월 판정")]
pw = Inches(2.25)
pg = (W - 2 * M - 5 * pw) / 4
ly = BODY_Y + Inches(0.95)
ln = s.shapes.add_connector(1, M + Inches(0.3), ly, W - M - Inches(0.3), ly)
ln.line.color.rgb = LINE
ln.line.width = Pt(2)
for i, (when, name, what) in enumerate(phases):
    x = M + i * (pw + pg)
    text(s, x, BODY_Y, pw, Inches(0.4), when, size=14, bold=True, color=INK_SOFT)
    badge(s, x, ly - Inches(0.25), str(i + 1), fill=AMBER if i == 0 else INK, d=Inches(0.5), size=14)
    text(s, x, ly + Inches(0.5), pw, Inches(0.5), name, size=20, bold=True)
    text(s, x, ly + Inches(1.05), pw, Inches(1.3), what, size=15, color=INK_SOFT)
box(s, M, Inches(5.2), W - 2 * M, Inches(1.25), fill=MIST)
text(s, M + Inches(0.4), Inches(5.2), W - 2 * M - Inches(0.8), Inches(1.25),
     [[("단계 통과 결정  ", {"bold": True}),
       ("리드가 통과 기준과 결과를 정리 → 담당 엔지니어 품질 확인 → 월간 임원 보고에서 승인. 미달이면 보완·축소·중단 중 하나를 정함", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 8)

# ---------- 9. 안전과 위험 ----------
s = slide("Agent는 생산 레시피를 혼자 확정하지 않습니다",
          "가장 걱정하실 부분은 Agent가 생산 레시피를 잘못 건드리는 것이라고 생각합니다. 측정 실행, 저장, 삭제 같은 위험 조작은 반드시 엔지니어가 승인하고, 엔지니어는 언제든 즉시 멈출 수 있습니다. 실장비 검증은 인가받은 비생산 시간, 하루 1시간에만 합니다. 운영상 가장 큰 위험은 겸직 시간 확보인데, 이번에 주 5시간을 명시해 승인받고 월간 보고에 실적을 보고하겠습니다.")
risks = [("오조작", "위험 조작·저장 전 엔지니어 승인, 즉시 중단 수단, 작업 전 원본 레시피 백업"),
         ("겸직 시간 부족", "주 5시간을 기획안에 명시, 월간 보고에 참여 시간 실적 보고"),
         ("현업의 AI 이해 부족", "2027-01~02 온보딩 8주, 실습 중심"),
         ("장비군마다 다른 화면·절차", "공통 엔진과 장비별 절차를 분리, 1차는 3개 장비군으로 제한")]
for i, (h, b) in enumerate(risks):
    col, row = i % 2, i // 2
    x = M + col * Inches(6.2)
    y = BODY_Y + row * Inches(2.0)
    box(s, x, y, Inches(5.9), Inches(1.75), fill=MIST)
    text(s, x + Inches(0.4), y + Inches(0.25), Inches(5.1), Inches(0.45), h, size=19, bold=True)
    text(s, x + Inches(0.4), y + Inches(0.8), Inches(5.1), Inches(0.9), b, size=15, color=INK_SOFT)
text(s, M, Inches(5.75), W - 2 * M, Inches(0.6),
     [[("가드레일  ", {"bold": True, "color": AMBER}),
       ("오조작으로 잘못 저장된 레시피 0건 · 승인 없는 위험 조작 0건 · 생산 측정 영향 0건", {})]], size=16)
footer(s, 9)

# ---------- 10. 요청 사항 ----------
s = slide("요청 사항",
          "요청 사항은 일곱 가지이고, 오늘 결정을 부탁드리는 것은 12월 항목 네 가지입니다. 보안 승인은 관찰과 입력을 나눠, 입력 심사가 늦어도 녹화가 막히지 않게 하겠습니다. 나머지는 1월 가동 전까지 협의하겠습니다.")
asks = [("시점", "요청"),
        ("2026-12", "ARC TF 구성 승인"),
        ("2026-12", "장비군별 MI 엔지니어 겸직 지정 (장비군마다 1~2명, 주 5시간)"),
        ("2026-12", "공통 플랫폼 팀 개발 인력 배정 (본인 포함 2명, 교육 포함)"),
        ("2026-12", "보안 승인 두 건: 관찰(설치형 프로그램, 화면 녹화)과 입력(Align Fail 한정)"),
        ("2027-01 전", "사내 추론 인프라 사용 보장, 추가 비용 승인 주체 지정"),
        ("2027-01 전", "사내 서버 자원 할당 (저장소·검색·DB·캐시, 새 서버 없음)"),
        ("2027-01 전", "비생산 검증용 장비 사용 인가 (1차 장비군마다 하루 1시간)")]
tbl = s.shapes.add_table(len(asks), 2, M, BODY_Y, W - 2 * M, Inches(4.6)).table
tbl.columns[0].width, tbl.columns[1].width = Inches(2.3), W - 2 * M - Inches(2.3)
for i, (when, what) in enumerate(asks):
    today = when == "2026-12"
    for j, val in enumerate((when, what)):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = INK if i == 0 else (MIST if today else WHITE)
        tf = c.text_frame
        tf.paragraphs[0].text = ""
        r = tf.paragraphs[0].add_run()
        r.text = val
        r.font.name, r.font.size = FONT, Pt(15)
        r.font.bold = i == 0 or today
        r.font.color.rgb = WHITE if i == 0 else (AMBER if today and j == 0 else INK)
text(s, M, Inches(6.4), W - 2 * M, Inches(0.4), "음영 행: 오늘 결정을 요청하는 항목", size=12, color=INK_SOFT)
footer(s, 10)

# ---------- 11. 마무리 ----------
s = slide(None, "다시 한 번 요청드립니다. 2027년 1월부터 ARC TF 운영과 MI 엔지니어 겸직 참여를 승인해 주십시오. 승인해 주시면 1월에 온보딩을 시작하고, 2월 말에 장비군별 기준선을 측정해 효과 목표를 숫자로 보고드리겠습니다.", dark=True)
text(s, M, Inches(1.9), W - 2 * M, Inches(1.0), "결정을 요청드립니다", size=40, bold=True, color=WHITE)
for i, t in enumerate(["ARC TF 구성 승인 (2027-01 가동)", "1차 장비군 MI 엔지니어 겸직 지정 (주 5시간)", "공통 플랫폼 팀 개발 2명 배정"]):
    y = Inches(3.2) + i * Inches(0.8)
    badge(s, M, y, str(i + 1), fill=AMBER, d=Inches(0.5), size=15)
    text(s, M + Inches(0.8), y, Inches(10), Inches(0.5), t, size=20, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
text(s, M, Inches(6.0), W - 2 * M, Inches(0.6),
     "승인 후: 2027-01 온보딩 시작 → 2027-02 장비군별 기준선 측정 → 효과 목표를 숫자로 보고", size=16, color=PALE)

prs.save(OUT)
print(f"wrote {OUT.name}, {len(prs.slides)} slides")
