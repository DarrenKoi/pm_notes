"""RE, REA 팀 입문 deck 의 Bento(HTML) 출력.

build_team_intro.py 가 부른다. 같은 SLIDES 를 받아 re-rea-team-intro.bento.html 의 #bento-doc JSON 을 바꿔 쓴다.
파일의 나머지(Bento 앱 런타임)는 건드리지 않는다. 형식: https://bento.page/agents.md
kind 이름 = 이 파일의 함수 이름. 좌표는 1280x720 px, 좌우 여백 96.
"""
import base64
import html
import json
import math
import mimetypes
import re
from pathlib import Path

HERE = Path(__file__).parent
BLOCK = re.compile(r'(<script type="application/bento\+json" id="bento-doc">)(.*?)(</script>)', re.S)

BG, FG, MUTE, ACC, ONACC = "#FFFFFF", "#1B2430", "#5F6B7A", "#D9480F", "#FFFFFF"
S1, S2, LINE = "#F5F6F8", "#E6E9EE", "rgba(27,36,48,0.16)"
KR = "'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', system-ui, sans-serif"
MONO = "Consolas, 'SF Mono', Menlo, monospace"
# SLIDES, extra json 에 16진수로 적힌 pptx 색을 이 deck 의 색으로 바꾼다
TINT = {"E8F0FA": "#DCE9FB", "FFF2CC": "#FFEBB5", "FCE4E4": "#FBD9D9", "D9D9D9": "#DDE1E6", "1F5AA6": "#B9D2F5"}
X, W = 96, 1088
MARCH = {"loop": {"type": "dash-march", "distance": 18, "duration": 1.4}}


def rich(s, mono=False):
    s = html.escape(s, quote=False)
    if mono:
        return s.replace(" ", "&nbsp;").replace("\n", "<br>")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r" {2,}", " &nbsp;&nbsp;", s).replace("\n", "<br>")


class Slide:
    """Element list of one slide. Ids are deterministic: <slide id>-<n>, or a shared id for morph."""

    def __init__(self, sid):
        self.sid, self.els, self.runs = sid, [], []

    def add(self, el, id=None, order=None, fx=None, **extra):
        el = {"id": id or f"{self.sid}-{len(self.els) + 1}", **el, "rotation": el.get("rotation", 0), "opacity": 1, **extra}
        for k in ("x", "y", "w", "h"):
            el[k] = round(el[k])
        if fx or order is not None:
            el["fx"] = fx or {"enter": "fade-up", "order": order}
        self.els.append(el)
        return el

    def text(self, x, y, w, h, s, size=16, color=FG, weight=400, align="left", valign="top", mono=False, lh=1.4, **kw):
        return self.add(dict(type="text", x=x, y=y, w=w, h=h, html=rich(s, mono), fontSize=size,
                             fontFamily=MONO if mono else KR, fontWeight=weight, color=color, align=align,
                             valign=valign, lineHeight=lh), **kw)

    def rect(self, x, y, w, h, fill=S1, stroke="none", sw=0, radius=10, **kw):
        return self.add(dict(type="shape", shape="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                             strokeWidth=sw, radius=radius), **kw)

    def line(self, x1, y1, x2, y2, color=MUTE, run=None, **kw):
        """Dashed marching arrow from (x1, y1) to (x2, y2). A Bento line is horizontal in its box, so rotate it.
        run: start delay in seconds of a dot that travels along the line again and again (shows what flows where)."""
        ln = math.hypot(x2 - x1, y2 - y1)
        if run is not None:
            self.runs.append(dict(type="shape", shape="ellipse", x=x1 - 6, y=y1 - 6, w=12, h=12, fill=color, stroke="none",
                                  strokeWidth=0, radius=0, fx={"loop": {"type": "motion-path", "duration": round(max(1.2, ln / 220), 1),
                                  "path": f"M0,0 L{round(x2 - x1)},{round(y2 - y1)}", "delay": run, "ease": "none"}}))
        return self.add(dict(type="shape", shape="line", x=(x1 + x2 - ln) / 2, y=(y1 + y2) / 2 - 2, w=ln, h=4, fill=color,
                             stroke=color, strokeWidth=2, radius=0, strokeStyle="dashed", lineEnd="arrow",
                             rotation=round(math.degrees(math.atan2(y2 - y1, x2 - x1)), 1)), fx=MARCH, **kw)

    def card(self, x, y, w, h, head, body="", sample=None, size=17, hh=48, hot=False, order=None, sample_size=13, hs=19):
        self.rect(x, y, w, h, S1, LINE, 1, 12, order=order)
        self.rect(x, y, w, hh, ACC if hot else S2, radius=12, order=order)
        self.text(x + 16, y, w - 32, hh, head, hs, ONACC if hot else FG, 700, valign="middle", lh=1.2, order=order)
        top = y + hh + 10
        if sample:
            ch = (sample.count("\n") + 1) * sample_size * 1.45 + 18
            self.rect(x + 10, top, w - 20, ch, BG, radius=8, order=order)
            self.text(x + 20, top, w - 40, ch, sample, sample_size, ACC, valign="middle", mono=True, lh=1.45, order=order)
            top += ch + 10
        if body:
            self.text(x + 16, top, w - 32, y + h - top - 8, body, size, order=order)

    def bullets(self, x, y, w, h, items, size=16, order=None):
        slot = min(h / len(items), size * 1.4 * 3.4)
        for i, it in enumerate(items):
            self.add(dict(type="shape", shape="ellipse", x=x, y=y + i * slot + size * 0.7 - 3, w=6, h=6, fill=ACC,
                          stroke="none", strokeWidth=0, radius=0), order=order)
            self.text(x + 18, y + i * slot, w - 18, slot - 4, it, size, order=order)

    def banner(self, y, s, h=46):
        self.rect(X, y, W, h, "transparent", ACC, 1.5, 10)
        self.text(X + 16, y, W - 32, h, s, 18, ACC, 700, "center", "middle")


# ---------- 슬라이드 종류 (kind 이름 = 함수 이름) ----------

def cards(b, y, h, s):
    n, gap = len(s["items"]), 24
    w = (W - gap * (n - 1)) / n
    for i, (head, body) in enumerate(s["items"]):
        b.card(X + i * (w + gap), y, w, h, head, body, size=18 if n < 4 else 17, order=i)


def two(b, y, h, s):
    w = (W - 32) / 2
    for i, (head, items) in enumerate(s["groups"]):
        x = X + i * (w + 32)
        b.card(x, y, w, h, head, hot=i == 1, order=i)
        b.bullets(x + 20, y + 68, w - 40, h - 78, items, 19, order=i)


def table(b, y, h, s):
    tint, n = s.get("tint", {}), len(s["rows"]) + 1
    small = h / n < 50
    rows = [{"cells": [{"html": rich(v)} for v in s["head"]]}]
    for row in s["rows"]:
        cells = []
        for j, v in enumerate(row):
            hit = next((c for k, c in tint.items() if k in v), None)
            cell = {"html": rich(v), "bold": j == 0 or bool(hit)}
            if hit:
                cell.update(bg=TINT.get(hit, S2), align="center")
            cells.append(cell)
        rows.append({"cells": cells})
    b.add(dict(type="table", x=X, y=y, w=W, h=min(h, n * 64), header=True, columns=[{"w": w} for w in s["widths"]], rows=rows,
               style=dict(headerBg=S2, headerColor=ACC, borderColor=LINE, borderWidth=1, cellPadX=14,
                          cellPadY=9 if small else 11, fontSize=15 if small else 17, color=FG, radius=10)), order=0)


def code(b, y, h, s):
    b.rect(X, y, 640, h, S1, LINE, 1, 12, order=0)
    b.rect(X, y, 4, h, ACC, radius=2, order=0)
    b.text(X + 24, y, 600, h, s["code"], 14, valign="middle", mono=True, lh=1.45, order=0)
    b.bullets(768, y + 6, 416, h - 6, s["side"], 16, order=1)


def image(b, y, h, s):
    """A screenshot fitted into the body area. Path is relative to this folder; the file is embedded."""
    b.add(dict(type="image", x=X, y=y, w=W, h=h, src="asset:" + s["image"], fit="contain", radius=8), order=0)


def draw_agenda(b, y, h, s):
    w = (W - 72) / 4
    for i, (head, body) in enumerate(s["items"]):
        b.card(X + i * (w + 24), y, w, 214, head, body, order=i, hs=18)
    b.text(X, y + 238, 300, 44, "시간 배분", 17, weight=700, valign="middle")
    b.text(X + 780, y + 232, 250, 50, "60", 44, ACC, 900, "right", "middle", mono=True, lh=1, fx={"countUp": True})
    b.text(X + 1040, y + 238, 48, 44, "분", 17, MUTE, 700, valign="middle")
    x = X
    for i, (label, mins) in enumerate(s["time"]):
        sw, big = W * mins / 60, mins >= 16
        b.rect(x, y + 290, sw - 4, 66, ACC if big else (S2 if i % 2 else S1), "none" if big else LINE, 0 if big else 1, 8, order=4)
        b.text(x, y + 290, sw - 4, 66, f"{label}\n{mins}분", 14, ONACC if big else FG, 700, "center", "middle", order=4)
        x += sw
    b.text(X, y + 372, W, 26, "한 장에 2분 안쪽으로 진행한다. 용어 표와 명령어 장은 훑고 지나간다.", 15, MUTE)


def draw_reverse(b, y, h, s):
    b.text(X, y, 470, 36, "보통의 개발 (Forward): 설계에서 제품으로", 18, MUTE, 700, valign="middle")
    b.line(X + 480, y + 18, X + W, y + 18, run=0)
    w, gap = 320, (W - 960) / 2
    for i, (head, body) in enumerate(s["nodes"]):
        b.card(X + i * (w + gap), y + 50, w, h - 100, head, body, order=i)
        if i < 2:
            b.text(X + i * (w + gap) + w, y + h / 2 - 20, gap, 40, "↔", 26, MUTE, align="center", valign="middle")
    b.line(X + 600, y + h - 18, X, y + h - 18, ACC, run=1.5)
    b.text(X + 610, y + h - 36, W - 610, 36, "리버스 엔지니어링 (Reverse): 제품에서 설계로", 18, ACC, 700, "right", "middle")


def draw_why(b, y, h, s):
    w = (W - 72) / 4
    qy = y + h - 64
    for i, (head, body) in enumerate(s["items"]):
        cx = X + i * (w + 24) + w / 2
        b.card(X + i * (w + 24), y, w, h - 130, head, body, size=16, order=i)
        b.line(cx, y + h - 124, 640 + (i - 1.5) * 120, qy - 6, run=i * 0.4, order=4)
    b.rect(240, qy, 800, 64, ACC, radius=12, order=5)
    b.text(240, qy, 800, 64, s["question"], 24, ONACC, 800, "center", "middle", order=5)


def draw_chain(b, y, h, s):
    nodes, labels = s["nodes"], s.get("labels")
    n = len(nodes)
    gap = 72 if labels else (56 if n <= 3 else 28)
    w = (W - gap * (n - 1)) / n
    ch = h - (56 if s.get("back") else 0)
    small = n > 3
    for i, (head, body, *sample) in enumerate(nodes):
        x = X + i * (w + gap)
        b.card(x, y, w, ch, head, body, sample[0] if sample else None, 15 if small else 17, order=i,
               sample_size=13 if small else 14, hs=15 if small else 19)
        if i < n - 1:
            b.line(x + w + 4, y + ch / 2, x + w + gap - 4, y + ch / 2, ACC, run=i * 1.2, order=i)
            if labels:
                b.text(x + w, y + ch / 2 - 30, gap, 22, labels[i], 13, MUTE, 700, "center", order=i)
    back, shape = s.get("back"), s.get("back_shape")
    if not back:
        return
    by = y + ch + 10
    if shape == "box":
        b.banner(by, back)
    elif shape == "both":
        left, right = re.split(r" {3,}", back)
        b.rect(X, by + 4, W, 4, ACC, radius=2, fillGradient={"angle": 90, "stops": [{"at": 0, "color": ACC}, {"at": 1, "color": S2}]})
        b.text(X, by + 14, W / 2, 30, left, 17, ACC, 700)
        b.text(X + W / 2, by + 14, W / 2, 30, right, 17, MUTE, 700, "right")
    else:
        b.line(X + W, by + 8, X, by + 8, ACC, run=2)
        b.text(X, by + 16, W, 30, back, 17, ACC, 700, "center")


def draw_loop(b, y, h, s):
    w, gap = 456, W - 912
    for i, (head, items) in enumerate(s["groups"]):
        x = X + i * (w + gap)
        b.card(x, y, w, h, head, hot=i == 1, order=i, hs=17)
        b.bullets(x + 20, y + 68, w - 40, h - 78, items, 17, order=i)
    mx = X + w
    b.text(mx, y + h / 2 - 74, gap, 24, "가설을 세운다", 15, ACC, 700, "center")
    b.line(mx + 10, y + h / 2 - 40, mx + gap - 10, y + h / 2 - 40, ACC, run=0)
    b.line(mx + gap - 10, y + h / 2 + 40, mx + 10, y + h / 2 + 40, run=1.2)
    b.text(mx, y + h / 2 + 52, gap, 44, "결과가 다르면\n다시 읽는다", 15, MUTE, 700, "center")


def draw_evidence(b, y, h, s):
    w, ch = (W - 120) / 4, h - 78
    for i, (head, body) in enumerate(s["nodes"]):
        x = X + i * (w + 40)
        b.card(x, y, w, ch, head, body, size=16, hot=i == 2, order=i, hs=17)
        if i < 3:
            b.line(x + w + 4, y + ch / 2, x + w + 36, y + ch / 2, ACC, run=i * 1.2, order=i)
    b.rect(X, y + h - 64, W, 64, "transparent", MUTE, 1.5, 10, strokeStyle="dashed", order=4)
    b.text(X + 16, y + h - 64, W - 32, 64, s["unknown"], 17, valign="middle", order=4)


def draw_zones(b, y, h, s):
    rows = sum(len(items) for _, items in s["groups"])
    rh = (h - 12 * (len(s["groups"]) - 1)) / rows
    for i, ((name, items), tint) in enumerate(zip(s["groups"], ("E8F0FA", "FFF2CC", "FCE4E4"))):
        gh = rh * len(items)
        b.rect(X, y, W, gh, S1, LINE, 1, 12, order=i)
        b.rect(X, y, 124, gh, TINT[tint], radius=12, order=i)
        b.text(X, y, 124, gh, name, 21, weight=800, align="center", valign="middle", order=i)
        for k, it in enumerate(items):
            b.text(X + 144, y + k * rh, W - 160, rh, it, 17, valign="middle", order=i)
        y += gh + 12


def draw_steps(b, y, h, s):
    n = len(s["nodes"])
    w = (W - 16 * (n - 1)) / n
    b.line(X, y + 140, X + W, y + 140, run=0, id="flow")
    for label, a, z in (("준비", 0, 3), ("분석", 3, 6), ("정리", 6, 7)):
        gx, gw = X + a * (w + 16), (z - a) * (w + 16) - 16
        b.text(gx, y, gw, 24, label, 15, MUTE, 700, "center")
        b.rect(gx, y + 26, gw, 3, MUTE, radius=2, opacity=0.5)
    for i, (head, body) in enumerate(s["nodes"]):
        x = X + i * (w + 16)
        b.rect(x, y + 40, w, 200, S1, LINE, 1, 12, id=f"st{i}", order=i)
        b.text(x + 12, y + 50, w - 24, 44, str(i + 1), 36, ACC, 900, mono=True, lh=1, order=i)
        b.text(x + 12, y + 100, w - 24, 26, head, 17, weight=700, order=i)
        b.text(x + 12, y + 132, w - 24, 104, body, 14, MUTE, order=i)
    x5, x6 = X + 4 * (w + 16) + w / 2, X + 5 * (w + 16) + w / 2
    b.add(dict(type="shape", shape="path", x=x5, y=y + 248, w=x6 - x5, h=40, fill="transparent", stroke=ACC, strokeWidth=2,
               radius=0, d=f"M {x6 - x5} 0 C {x6 - x5} 40, 0 40, 0 0", pathBox=[0, 0, x6 - x5, 40], strokeStyle="dashed"),
          id="loop", fx=MARCH)
    b.text(x5 - 200, y + 292, x6 - x5 + 400, 26, "6단계 결과가 가설과 다르면 5단계로 돌아간다", 16, ACC, 700, "center")


def draw_arch(b, y, h, s):
    my = y + 36 + (h - 36) / 2
    b.rect(648, y, 536, h, "transparent", MUTE, 1.5, 12, strokeStyle="dashed")
    b.text(664, y + 6, 300, 24, "내 PC에서 실행된다", 15, MUTE, 700)
    b.rect(X, my - 50, 150, 100, S1, LINE, 1, 50, order=0)
    b.text(X, my - 50, 150, 100, "**사람**\n질문을 고르고\n결과를 따진다", 16, align="center", valign="middle", order=0)
    b.text(246, my - 30, 80, 22, "질문", 15, MUTE, 700, "center")
    b.line(250, my, 322, my, run=0)
    b.rect(326, my - 50, 200, 100, S2, LINE, 1, 12, order=1)
    b.text(326, my - 50, 200, 100, "**AI 에이전트 (pi)**\n어떤 도구를 쓸지\n정한다", 16, align="center", valign="middle", order=1)
    b.rect(326, y, 200, 52, "transparent", ACC, 1.5, 12, order=1)
    b.text(326, y, 200, 52, "**사내 LLM** (AI 모델)", 16, align="center", valign="middle", order=1)
    b.line(426, my - 54, 426, y + 56, ACC, run=4.8)
    b.text(X, y + 64, 316, 24, "도구 결과가 모델로 전달된다", 15, ACC, 700, "right")
    b.text(530, my - 66, 134, 40, "MCP로\n도구 호출", 14, ACC, 700, "center", "bottom", lh=1.3)
    b.line(532, my - 18, 664, my - 18, ACC, run=1.2)
    b.line(664, my + 18, 532, my + 18, run=3.6)
    b.text(530, my + 26, 134, 40, "근거와 한계가\n붙은 결과", 14, MUTE, 700, "center", lh=1.3)
    b.rect(668, my - 85, 200, 170, ACC, radius=12, order=2)
    b.text(668, my - 85, 200, 170, "**REA**\n\n스킬: 작업 지침서\nMCP 서버:\n분석 도구 묶음", 17, ONACC, align="center", valign="middle", order=2)
    eh = (h - 36 - 30 - 10) / 4
    for i, t in enumerate(("**Ghidra**\nC, C++ 프로그램", "**JavaScript 분석기**\n웹 앱, Electron 앱", "**.NET, Python 분석기**",
                           "**Chrome**\n웹 페이지 관찰")):
        ey = y + 36 + i * (eh + 10)
        b.rect(968, ey, 200, eh, S2 if i else TINT["FFF2CC"], LINE, 1, 10, order=3)
        b.text(968, ey, 200, eh, t, 15, align="center", valign="middle", lh=1.3, order=3)
        b.line(872, my, 964, ey + eh / 2, run=2.4 if i == 0 else None, order=3)


def draw_prompt(b, y, h, s):
    lines = [c.count("\n") + 1 for *_, c in s["nodes"]]
    pad = (h - 12 * (len(lines) - 1) - sum(lines) * 21) / len(lines)
    for i, ((head, body, sample), n) in enumerate(zip(s["nodes"], lines)):
        rh = n * 21 + pad
        b.rect(X, y, 640, rh, S1, LINE, 1, 10, order=i)
        b.rect(X, y, 4, rh, ACC, radius=2, order=i)
        b.text(X + 22, y, 610, rh, sample, 14, valign="middle", mono=True, lh=1.5, order=i)
        b.line(X + 646, y + rh / 2, 764, y + rh / 2, ACC, order=i)
        b.text(772, y, 412, 26, f"{i + 1}  {head}", 19, ACC, 800, order=i)
        b.text(772, y + 28, 412, rh - 28, body, 15, order=i)
        y += rh + 12


def draw_tree(b, y, h, s):
    b.rect(360, y, 560, 44, S2, LINE, 1, 10, order=0)
    b.text(360, y, 560, 44, s["root"], 18, weight=700, align="center", valign="middle", order=0)
    b.line(640, y + 46, 640, y + 62)
    b.rect(360, y + 64, 560, 44, "transparent", ACC, 1.5, 10, order=1)
    b.text(360, y + 64, 560, 44, s["check"], 17, ACC, 700, "center", "middle", order=1)
    w, ly = (W - 72) / 4, y + 150
    for i, (head, tag, tool) in enumerate(s["nodes"]):
        x = X + i * (w + 24)
        b.line(640 + (i - 1.5) * 60, y + 110, x + w / 2, ly - 4, ACC if i == 0 else MUTE, run=0 if i == 0 else None, order=2)
        b.rect(x, ly, w, y + h - ly, S1, ACC if i == 0 else LINE, 1.5 if i == 0 else 1, 12, order=2 + i)
        b.text(x + 14, ly + 10, w - 28, 48, head, 16, weight=700, lh=1.35, order=2 + i)
        b.text(x + 14, ly + 64, w - 28, 26, tag, 16, ACC if i == 0 else MUTE, order=2 + i)
        b.text(x + 14, ly + 96, w - 28, y + h - ly - 104, tool, 13, MUTE, mono=True, order=2 + i)


def draw_native(b, y, h, s):
    n = len(s["nodes"])
    w = (W - 20 * (n - 1)) / n
    fw = 3 * w + 40
    b.add(dict(type="shape", shape="path", x=X, y=y, w=fw, h=36, fill=ACC, stroke="none", strokeWidth=0, radius=0,
               d=f"M 0 0 L {fw} 14 L {fw} 22 L 0 36 Z", pathBox=[0, 0, fw, 36], opacity=0.45,
               fillGradient={"angle": 90, "stops": [{"at": 0, "color": S2}, {"at": 1, "color": ACC}]}))
    b.text(X + 12, y, fw, 36, "넓게 훑은 뒤 한 곳만 깊게", 15, weight=700, valign="middle")
    sx = X + 4 * (w + 20)
    b.rect(sx, y + 14, 2 * w + 20, 8, ACC, radius=4)
    b.text(sx - 200, y, 190, 36, "닫기 전에 저장", 15, ACC, 700, "right", "middle")
    b.line(X, y + 76, X + W, y + 76, run=0)
    for i, (head, body) in enumerate(s["nodes"]):
        b.card(X + i * (w + 20), y + 52, w, h - 52, f"{i + 1}  {head}", body, size=15, hot=i == 4, order=i, hs=15)


def draw_result(b, y, h, s):
    b.text(X, y, 640, 26, s.get("head", "REA가 돌려주는 결과 (예시)"), 14, MUTE, 700)
    rows = s["rows"]
    rh = (h - 30 - 10 * (len(rows) - 1)) / len(rows)
    for i, (tag, text, tint) in enumerate(rows):
        ry = y + 30 + i * (rh + 10)
        b.rect(X, ry, 640, rh, S1, LINE, 1, 10, order=i)
        b.rect(X, ry, 96, rh, TINT.get(tint, S2), radius=10, order=i)
        b.text(X, ry, 96, rh, tag, 17, weight=800, align="center", valign="middle", order=i)
        b.text(X + 112, ry, 516, rh, text, 17, valign="middle", order=i)
    ih = (h - 20) / len(s["items"])
    for i, (head, body) in enumerate(s["items"]):
        iy = y + i * (ih + 10)
        b.rect(768, iy, 416, ih, "transparent", ACC, 1.5, 10, order=4 + i)
        b.text(784, iy + 10, 384, 26, head, 18, ACC, 800, order=4 + i)
        b.text(784, iy + 42, 384, ih - 48, body, 16, order=4 + i)


def draw_uses(b, y, h, s):
    w = (W - 72) / 4
    for i, (head, body) in enumerate(s["items"]):
        b.card(X + i * (w + 24), y, w, 150, head, body, size=16, order=i, hs=17)
    b.text(X, y + 168, W, 28, "계산 로직을 옮길 때는 검증까지 한다", 18, ACC, 700)
    by, bh = y + 206, h - 206
    b.rect(X, by, 300, bh, S1, LINE, 1, 12, order=4)
    b.text(X + 20, by, 260, bh, "**같은 입력**\n정상값\n경계값 (0, 음수, 아주 큰 값)\n빈 입력", 17, valign="middle", order=4)
    for k, t in enumerate(("원본 도구", "새로 만든 코드")):
        yy, hh = by + k * (bh / 2 + 6), bh / 2 - 6
        b.rect(490, yy, 300, hh, S2, LINE, 1, 12, order=5)
        b.text(490, yy, 300, hh, t, 18, weight=700, align="center", valign="middle", order=5)
        b.line(400, by + bh / 2, 486, yy + hh / 2, run=0, order=5)
        b.line(794, yy + hh / 2, 880, by + bh / 2, ACC, run=1.2, order=5)
    b.rect(884, by, 300, bh, ACC, radius=12, order=6)
    b.text(884, by, 300, bh, "**출력 비교**\n시험한 입력에서 같으면 통과\n다르면 다시 분석한다", 18, ONACC, align="center", valign="middle", order=6)


# ---------- 문서 ----------

def cover(b, s):
    hexdump = "8d 14 09 83 f9 65 b8 ff\nff ff ff 0f 4c c2 c3 cc\n48 89 5c 24 08 57 48 83\nec 20 48 8b d9 e8 3a 01\n" \
              "00 00 48 8b c8 ff 15 0e\n4c 8b c0 48 85 c0 74 2b\n41 b9 10 00 00 00 4c 8d\n05 9f 3e 01 00 48 8b d3"
    b.text(770, 150, 414, 420, hexdump, 25, MUTE, mono=True, lh=1.45, id="hex", opacity=0.22)
    b.add(dict(type="shape", shape="ellipse", x=900, y=120, w=220, h=220, fill="transparent", stroke=ACC, strokeWidth=2, radius=0),
          id="orbit", opacity=0.55, fx={"loop": {"type": "motion-path", "duration": 14, "delay": 0, "ease": "none",
                                                 "path": "M0,0 C40,-30 90,-10 110,30 C130,70 70,110 20,80 C-20,60 -30,20 0,0"}})
    b.text(X, 150, 700, 26, "REVERSE ENGINEERING / REA", 16, ACC, 700, id="kicker", letterSpacing=2)
    b.text(X, 186, 700, 214, s["title"], 62, weight=900, valign="bottom", lh=1.12, id="title", role="title")
    b.rect(X, 416, 320, 10, ACC, radius=5, id="accent-bar")
    b.text(X, 448, 760, 70, s["msg"], 24, weight=600, order=1, role="subtitle")
    b.text(X, 530, 760, 30, s["sub"], 18, MUTE, order=2)


def build_doc(slides, title):
    out, assets = [], {}
    for n, s in enumerate(slides, 1):
        b = Slide(f"s{n:02d}")
        if s["kind"] == "cover":
            cover(b, s)
        else:
            b.text(X, 40, 700, 26, s["part"], 16, ACC, 700, id="kicker")
            b.text(X, 68, W, 58, s["title"], 38, weight=800, valign="middle", lh=1.2, id="title", role="title")
            b.rect(X, 134, 48, 4, ACC, radius=2, id="accent-bar")
            b.rect(X, 152, W, 52, S1, radius=8, id="msg-bg")
            b.rect(X, 152, 4, 52, ACC, radius=2, id="msg-bar")
            b.text(X + 22, 152, W - 36, 52, s["msg"], 20, weight=700, valign="middle", lh=1.25, id="msg", role="subtitle")
            b.text(1024, 676, 160, 24, "{{page}} / {{pages}}", 14, MUTE, align="right", id="pg")
            y, bottom = 222, 652
            if s.get("foot"):
                bottom -= 76
                b.text(X, bottom + 10, W, 66, s["foot"], 15, MUTE, valign="middle")
            elif s.get("banner"):
                bottom -= 60
                b.banner(bottom + 14, s["banner"])
            globals()[s["kind"]](b, y, bottom - y, s)
            if "image" in s:
                f = HERE / s["image"]
                assets[s["image"]] = f"data:{mimetypes.guess_type(f.name)[0]};base64,{base64.b64encode(f.read_bytes()).decode()}"
        for i, el in enumerate(b.runs):  # dots go on top of everything, without an entrance (it fights the path)
            b.els.append({"id": f"{b.sid}-run{i}", **el, "rotation": 0, "opacity": 1})
        out.append(dict(id=b.sid, background=BG, transition="none" if n == 1 else "morph", notes=s["notes"], elements=b.els))
    doc = dict(format="bento/slides", version=1, title=title, size=dict(width=1280, height=720),
               theme=dict(background=BG, color=FG, accent=ACC, fontFamily=KR),
               meta=dict(subject="리버스 엔지니어링 입문과 REA 활용법", keywords="reverse engineering, REA, Ghidra"), slides=out)
    if assets:
        doc["assets"] = assets
    return doc


def build_bento(path, slides, shell=None):
    """Write the deck JSON into the Bento app file. The app runtime comes from `shell` (default: `path` itself)."""
    src = shell or path
    if not src.is_file():
        raise SystemExit(f"{src.name} 이 없다. Bento 앱 파일을 먼저 받는다:\n"
                         f"  curl -fsSL https://bento.page/releases/slides/Bento_Slides.bento.html -o {src}")
    page = src.read_text(encoding="utf-8")
    m = BLOCK.search(page)
    assert m, "bento-doc block not found"
    payload = json.dumps(build_doc(slides, slides[0]["title"]), ensure_ascii=False).replace("<", "\\u003c")
    path.write_text(page[:m.start(2)] + "\n" + payload + "\n" + page[m.end(2):], encoding="utf-8")
