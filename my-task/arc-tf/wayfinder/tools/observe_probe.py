# observe_probe.py - 장비 뷰어 창이 바뀔 때만 JPEG 1장 저장하는 시험 스크립트
# "엔지니어 PC 자원 실측과 Windows 버전 확인" 티켓용. 실제 프로그램이 아니다.
# pip install mss pillow numpy pywin32
# 실행: python observe_probe.py "창 제목 일부"     판정 로직 확인: python observe_probe.py --selftest
# 시험용이라 mss(BitBlt)를 쓴다: 대상 창이 맨 앞에 보여야 한다. 실제 프로그램은 WGC.
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

TILE, PIX_TH, TILE_TH, SETTLE = 16, 12, 3, 0.4   # 타일 크기, 밝기 차, 바뀐 타일 수, 안정 대기(초)
LIVE = []          # 라이브 영역 (왼쪽, 위, 오른쪽, 아래) 비율 좌표. 담당 엔지니어가 정함. 예: [(0.30, 0.10, 0.75, 0.70)]
LIVE_SCENE = 0.8   # 라이브 영역 타일이 이 비율 이상 한꺼번에 바뀌면 장면 전환으로 보고 신호로 씀
OUT = Path("frames")


def small_gray(img):  # 320x180 흑백
    return np.asarray(img.convert("L").resize((320, 180)), dtype=np.int16)


def tile_diff(a, b):  # 타일별 평균 밝기 차
    d = np.abs(a - b)
    h, w = d.shape
    return d[: h // TILE * TILE, : w // TILE * TILE].reshape(h // TILE, TILE, w // TILE, TILE).mean(axis=(1, 3))


def live_mask(shape):  # 타일 격자 위의 라이브 영역 표시
    th, tw = shape
    m = np.zeros(shape, bool)
    for l, t, r, b in LIVE:
        m[int(t * th):int(b * th), int(l * tw):int(r * tw)] = True
    return m


def is_change(a, b):
    t = tile_diff(a, b) > PIX_TH
    live = live_mask(t.shape)
    ui_changed = int((t & ~live).sum()) >= TILE_TH          # 조작 영역 변화
    scene = live.any() and (t & live).sum() >= LIVE_SCENE * live.sum()  # 라이브 영역 장면 전환
    return ui_changed or bool(scene)


def main():
    import mss, win32gui
    title = sys.argv[1] if len(sys.argv) > 1 else "RCS"
    hits = []
    win32gui.EnumWindows(lambda h, _: hits.append(h) if title in win32gui.GetWindowText(h) else None, None)
    if not hits:
        sys.exit(f"창 없음: {title}")
    hwnd = hits[0]
    OUT.mkdir(exist_ok=True)
    prev, dirty_since, n = None, None, 0
    with mss.mss() as sct:
        while True:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            shot = sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            cur = small_gray(img)
            if prev is not None and is_change(prev, cur):
                dirty_since = time.time()
            elif dirty_since and time.time() - dirty_since >= SETTLE:
                n += 1
                img.save(OUT / f"{int(time.time() * 1000)}.jpg", quality=80)  # 저장 사진은 라이브 영역 포함 원본
                print(f"saved #{n}")
                dirty_since = None
            prev = cur
            time.sleep(0.1)  # 10회/초 확인


def selftest():
    global LIVE
    z = np.zeros((180, 320), np.int16)
    ui = z.copy(); ui[:32, :32] = 100                 # 왼쪽 위 조작 영역 4타일 변화
    assert not is_change(z, z) and is_change(z, ui)
    LIVE = [(0.5, 0.0, 1.0, 1.0)]                      # 오른쪽 절반이 라이브
    noise = z.copy(); noise[:16, 160:176] = 100       # 라이브 영역 1타일 흔들림 -> 무시
    assert not is_change(z, noise)
    scene = z.copy(); scene[:, 160:] = 100            # 라이브 영역 통째로 바뀜 -> 장면 전환
    assert is_change(z, scene)
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
