#!/usr/bin/env python3
"""pi 세션 로그에서 분당 요청 수(RPM)와 토큰 수(TPM)를 집계한다.

pi 는 모델 응답 하나마다 assistant 메시지에 usage 를 남긴다. 자식 세션 로그도 같은
디렉터리 아래에 쌓이므로, 전부 합치면 게이트웨이가 보는 부하를 추정할 수 있다.
"요청"은 기록된 모델 응답·오류 수이고, "토큰"은 보고된 사용량이다. 게이트웨이가 센 값이 아니다.
빠지는 것: 같은 키를 쓰는 다른 harness, 다른 경로의 세션, watchdog 검토 호출(세션에 사용량을 남기지
않는다), 안쪽 HTTP 재시도(한 응답 안에서 일어난다). 한도 준수 판정은 게이트웨이 로그로 한다.

    python rate-usage.py                          # 최근 24시간, ~/.pi/agent/sessions
    python rate-usage.py --hours 2 --rpm 50 --tpm 500000
    python rate-usage.py --dir D:\\pi-sessions --provider my-local-provider
    python rate-usage.py --selftest

TPM 은 두 가지로 낸다. 게이트웨이가 캐시 적중 토큰을 세는지 모르기 때문이다.
  tpm_all     = input + output + cacheRead + cacheWrite
  tpm_nocache = input + output + cacheWrite
"""
import argparse, collections, glob, json, os, sys, time
from datetime import datetime, timezone


def _ms(d, m):
    ts = m.get("timestamp") if m else None
    if isinstance(ts, (int, float)):
        return ts
    try:  # 엔트리의 ISO 시각 (usage·compaction 엔트리)
        return datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).timestamp() * 1000
    except (KeyError, AttributeError, ValueError):
        return None


def load(paths, since_ms, provider=None):
    """모델 응답(assistant) + 대화 밖 사용량(usage, compaction, branch_summary) 을 모은다."""
    seen, rows = set(), []
    for p in paths:
        try:
            f = open(p, encoding="utf-8")
        except OSError:
            continue
        with f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict):
                    continue
                m = d.get("message")
                if d.get("type") == "message" and isinstance(m, dict) and m.get("role") == "assistant":
                    u, prov, model, ts = m.get("usage"), m.get("provider"), m.get("model"), _ms(d, m)
                elif d.get("type") in ("usage", "compaction", "branch_summary"):
                    u, prov, model, ts = d.get("usage"), d.get("provider"), d.get("model"), _ms(d, None)
                else:
                    continue
                # 압축 요약 엔트리에는 provider 가 없을 수 있다. 필터에서 버리지 않고 "미상"으로 남긴다.
                if not u or ts is None or ts < since_ms or (provider and prov not in (provider, None)):
                    continue
                g = lambda k: int(u.get(k) or 0)
                i, o, cr, cw = g("input"), g("output"), g("cacheRead"), g("cacheWrite")
                # 포크·복제된 세션은 과거 엔트리를 같은 id 로 복사해 둔다. 같은 호출을 두 번 세지 않는다.
                # id 가 다르면 시각과 사용량이 같아도 다른 호출이다 (병렬 자식).
                key = (d.get("id"), prov, model, int(ts), i, o, cr, cw)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((ts, f"{prov}/{model}" if prov else "(provider 미상: 압축·요약)", i, o, cr, cw))
    return rows


def rolling_peak(rows, col):
    """임의의 60초 구간에서 본 최댓값. col: 0=요청 수, 1=토큰(캐시 포함), 2=토큰(캐시 제외)."""
    ev = sorted((r[0], 1, r[2] + r[3] + r[4] + r[5], r[2] + r[3] + r[5]) for r in rows)
    win, acc, best = collections.deque(), [0, 0, 0], 0
    for e in ev:
        win.append(e); acc = [acc[0] + e[1], acc[1] + e[2], acc[2] + e[3]]
        while win[0][0] <= e[0] - 60000:
            x = win.popleft(); acc = [acc[0] - x[1], acc[1] - x[2], acc[2] - x[3]]
        best = max(best, acc[col])
    return best


def summarize(rows):
    per_min = collections.defaultdict(lambda: [0, 0, 0])  # req, tpm_all, tpm_nocache
    per_model = collections.defaultdict(lambda: [0, 0])
    for ts, model, i, o, cr, cw in rows:
        b = per_min[int(ts // 60000)]
        b[0] += 1; b[1] += i + o + cr + cw; b[2] += i + o + cw
        per_model[model][0] += 1; per_model[model][1] += i + o + cr + cw
    return per_min, per_model


def report(rows, rpm, tpm):
    per_min, per_model = summarize(rows)
    if not per_min:
        print("기간 안에 기록된 모델 응답이 없다.")
        return
    active = len(per_min)
    peak_r = max(v[0] for v in per_min.values())
    peak_t = max(v[1] for v in per_min.values())
    peak_tn = max(v[2] for v in per_min.values())
    reqs = sum(v[0] for v in per_min.values())
    toks = sum(v[1] for v in per_min.values())
    print(f"활동한 분: {active}   요청: {reqs}   토큰(캐시 포함): {toks:,}")
    print(f"응답당 평균 토큰: {toks // max(reqs, 1):,}")
    print(f"최대 RPM: {peak_r} / {rpm}   최대 TPM(캐시 포함): {peak_t:,} / {tpm:,}   (캐시 제외): {peak_tn:,}   [달력 분]")
    print(f"최대 RPM: {rolling_peak(rows, 0)} / {rpm}   최대 TPM(캐시 포함): {rolling_peak(rows, 1):,}   "
          f"(캐시 제외): {rolling_peak(rows, 2):,}   [임의의 60초 구간]")
    over_r = sum(1 for v in per_min.values() if v[0] > rpm)
    over_t = sum(1 for v in per_min.values() if v[1] > tpm)
    over_tn = sum(1 for v in per_min.values() if v[2] > tpm)
    print(f"한도 초과한 분: RPM {over_r}, TPM(캐시 포함) {over_t}, TPM(캐시 제외) {over_tn}")
    print(f"80% 넘긴 분: RPM {sum(1 for v in per_min.values() if v[0] > rpm * .8)}, "
          f"TPM(캐시 포함) {sum(1 for v in per_min.values() if v[1] > tpm * .8)}")
    print("\n모델별:")
    for model, (n, t) in sorted(per_model.items(), key=lambda x: -x[1][1]):
        print(f"  {model:45s} 요청 {n:6d}   토큰 {t:>13,}   평균 {t // max(n, 1):>9,}")
    print("\n가장 바빴던 5분 (분, 요청, 토큰 캐시 포함):")
    for minute, v in sorted(per_min.items(), key=lambda x: -x[1][1])[:5]:
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(minute * 60))}  {v[0]:4d}  {v[1]:>11,}")


def selftest():
    base = 1_000_000 * 60000
    rows = [(base, "p/a", 100, 10, 50, 0), (base + 1000, "p/a", 200, 20, 0, 5),
            (base + 70000, "p/b", 1, 1, 0, 0)]
    per_min, per_model = summarize(rows)
    assert per_min[base // 60000] == [2, 385, 335], per_min
    assert per_model["p/a"] == [2, 385] and per_model["p/b"] == [1, 2]
    # 59.5초 차이 두 요청은 달력 분이 달라도 같은 60초 구간에 든다
    edge = [(base + 59000, "p/a", 10, 0, 0, 0), (base + 118500, "p/a", 20, 0, 0, 0)]
    assert summarize(edge)[0][base // 60000][0] == 1 and rolling_peak(edge, 0) == 2
    # usage 엔트리, ISO 시각, 포크 복제 중복 제거
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        a = {"type": "message", "id": "e1", "timestamp": "x", "message": {"role": "assistant", "provider": "p", "model": "a",
             "timestamp": base, "usage": {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 0}}}
        b = dict(a, id="e2")  # 같은 시각·사용량의 다른 호출
        c = {"type": "compaction", "id": "c1", "timestamp": "2001-09-09T01:46:40Z", "usage": {"input": 7}}
        u = {"type": "usage", "id": "u1", "kind": "cache_warm", "provider": "p", "model": "a",
             "timestamp": datetime.fromtimestamp(base / 1000, timezone.utc).isoformat().replace("+00:00", "Z"), "usage": {"cacheRead": 50}}
        for n in ("s1.jsonl", "fork.jsonl"):
            with open(os.path.join(t, n), "w") as f:
                f.write("\n".join(json.dumps(x) for x in (a, b, c, u)) + "\nnot json\n")
        got = load(glob.glob(os.path.join(t, "*.jsonl")), 0, provider="p")
        assert len(got) == 4 and sum(r[4] for r in got) == 56, got
        assert sum(1 for r in got if r[1].startswith("(provider")) == 1
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions"))
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--provider")
    ap.add_argument("--rpm", type=int, default=50)
    ap.add_argument("--tpm", type=int, default=500_000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    paths = glob.glob(os.path.join(a.dir, "**", "*.jsonl"), recursive=True)
    since = (time.time() - a.hours * 3600) * 1000
    report(load(paths, since, a.provider), a.rpm, a.tpm)


if __name__ == "__main__":
    sys.exit(main())
