#!/usr/bin/env python3
"""
selftest.py - bre.py 회귀 검증

정답을 아는 합성 fixture(make_fixture.py)를 만들고, bre.py 의 각 subcommand 가
그 정답을 복원하는지 확인한다. Agent 는 실제 장비 파일에 toolkit 을 쓰기 전에
이 스크립트를 먼저 돌려 환경(numpy 버전 등)이 정상인지 확인해야 한다.

사용:
    python selftest.py
    echo $?     # 0 = 전부 통과, 1 = 실패

실패하면 어떤 단언이 깨졌는지 출력한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
BRE = HERE / "bre.py"
FIXTURE = HERE / "make_fixture.py"

# make_fixture.py 가 생성하는 파일의 ground truth
TRUTH = {
    "magic": "CDS1",
    "n_points_offset": 0x08,
    "timestamp_offset": 0x0C,
    "header_size": 0x30,
    "stride": 16,
    "cd_field_offset_in_record": 12,
    "cd_dtype": "<f4",
    "n_points": 256,
}

failures: list[str] = []


def run(*argv: str) -> dict:
    out = subprocess.run([sys.executable, str(BRE), *argv],
                         capture_output=True, text=True, check=True)
    if out.stderr.strip():
        failures.append(f"stderr 오염 ({argv[0]}): {out.stderr[:200]!r}")
    return json.loads(out.stdout)


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(f"{name}: {detail}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, str(FIXTURE), td], capture_output=True, check=True)
        fx = Path(td)
        base = str(fx / "base.dat")

        print("[1] triage - magic 과 recipe string 을 찾는가")
        t = run("triage", base)
        check("magic 'CDS1' 이 head_ascii 선두에 있다", t["head_ascii"].startswith(TRUTH["magic"]))
        check("recipe string 을 추출한다",
              any("RECIPE_A" in s["text"] for s in t["ascii_strings"]))
        check("압축으로 오판하지 않는다", t["entropy_verdict"] != "compressed_or_encrypted",
              t["entropy_verdict"])

        print("[2] diff (측정점 +1) - count field 와 stride 를 드러내는가")
        d = run("diff", base, str(fx / "one_more_point.dat"))
        check("size_delta == stride", d["size_delta"] == TRUTH["stride"], f"got {d['size_delta']}")
        first = d["field_candidates"][0]
        check("첫 변경 필드가 n_points offset", first["start"] == TRUTH["n_points_offset"],
              f"got {first['start_hex']}")
        check("n_points delta == 1", first["interpretations"]["<I"]["delta"] == 1)

        print("[3] diff (재저장) - timestamp 와 checksum 만 바뀌는가")
        d2 = run("diff", base, str(fx / "resaved.dat"))
        check("크기 동일", d2["size_delta"] == 0)
        offs = [f["start"] for f in d2["field_candidates"]]
        check("timestamp offset 이 변경 필드에 포함", TRUTH["timestamp_offset"] in offs, str(offs))
        check("변경 필드는 timestamp + checksum 둘뿐", len(d2["field_candidates"]) == 2, str(offs))

        print("[4] stride - 기본 주기 16 을 harmonic(32/48) 보다 우선하는가")
        s = run("stride", base, "--offset", str(TRUTH["header_size"]), "--max-stride", "512")
        check("best_stride == 16", s["best_stride"] == TRUTH["stride"], f"got {s['best_stride']}")

        print("[5] arrays --stride - interleaved CD 필드를 정확히 집어내는가")
        a = run("arrays", base, "--stride", str(TRUTH["stride"]),
                "--payload-offset", str(TRUTH["header_size"]),
                "--lo", "1", "--hi", "1000", "--top", "4")
        top = a["top_candidates"][0]
        check("1위 후보 = record 내 offset 12",
              top["field_offset_in_record"] == TRUTH["cd_field_offset_in_record"],
              f"got {top['field_offset_in_record']}")
        check("1위 후보 dtype == <f4", top["dtype"] == TRUTH["cd_dtype"], top["dtype"])
        check("원소 수 == n_points", top["element_count"] == TRUTH["n_points"],
              str(top["element_count"]))
        check("평균이 CD 중심값(45nm) 근처", 44.0 < top["mean"] < 46.0, str(top["mean"]))

        print("[6] variance - corpus 전체에서 magic 을 고정 구간으로 잡는가")
        v = run("variance", *[str(p) for p in sorted(fx.glob("corpus_*.dat"))])
        check("offset 0 이 고정 구간에 포함",
              any(r["start"] == 0 for r in v["constant_runs"]))
        check("약한 heuristic 임을 명시", "weak_boundary_caveat" in v)

        print("[7] stamps - 진짜 timestamp offset 을 후보에 포함하는가")
        st = run("stamps", base, "--max-bytes", "256")
        check("0x0c <I unix 후보가 존재",
              any(c["offset"] == TRUTH["timestamp_offset"] and c["struct_fmt"] == "<I"
                  for c in st["candidates"]))
        check("오탐 다수임을 hint 로 경고", "오탐" in st["hint"])

        # === recipe / 좌표 파일 detector ===
        recipe = str(fx / "recipe.dat")

        print("[8] tlv - 가변 TLV 체인을 정확한 config 로 전부 소비하는가")
        tv = run("tlv", recipe, "--start", "24")   # 0x18
        b = tv["best"]
        check("tag_size=2, len_size=2, le", b["tag_size"] == 2 and b["len_size"] == 2
              and b["endian"] == "le", str((b["tag_size"], b["len_size"], b["endian"])))
        check("coverage ~ 1.0 이며 EOF 안착", b["coverage"] >= 0.99 and b["lands_at_eof"],
              str(b["coverage"]))
        check("tag 1,2,3,4 순서로 파싱", [r["tag"] for r in b["first_records"]] == [1, 2, 3, 4])

        print("[9] offsets - offset 테이블(@0x08, 4 entry)을 잡는가")
        of = run("offsets", recipe)
        hit = [c for c in of["candidates"] if c["offset"] == 0x08 and c["fmt"] == "<I"]
        check("0x08 에 <I 4-entry 테이블", bool(hit) and hit[0]["count"] == 4,
              str([(c["offset_hex"], c["count"]) for c in of["candidates"]]))

        print("[10] strtab - 파라미터명 문자열을 형태별로 추출하는가")
        stx = run("strtab", recipe)
        check("null-term 'CD_TARGET' 추출",
              any(x["text"] == "CD_TARGET" for x in stx["null_terminated"]))
        check("u16 length-prefixed 'ETCH_STP' 추출",
              any(x["text"] == "ETCH_STP" and x["prefix_type"] == "u16le"
                  for x in stx["length_prefixed"]))

        print("[11] serial - 내부 XML wrapper 를 탐지하는가")
        se = run("serial", str(fx / "recipe_xml.dat"))
        check("xml 시그니처 탐지", any(h["format"] == "xml" for h in se["signature_hits"]))
        check("mostly_text 로 판정", se["mostly_text"])

        print("[12] arrays --stride - 좌표 x/y f8 필드를 top 후보에 올리는가")
        co = run("arrays", str(fx / "coords.dat"), "--stride", "20", "--payload-offset", "8",
                 "--lo", "1", "--hi", "200000", "--dtypes", "<f8,<f4,<i4", "--top", "6")
        ks = {(c["field_offset_in_record"], c["dtype"]) for c in co["top_candidates"]}
        check("x=f8@k4, y=f8@k12 둘 다 top6 에", (4, "<f8") in ks and (12, "<f8") in ks, str(ks))
        check("좌표 원소 수 == n_sites(200)",
              all(c["element_count"] == 200 for c in co["top_candidates"]))

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL PASS - toolkit 이 합성 fixture 의 ground truth 를 전부 복원함")
    return 0


if __name__ == "__main__":
    sys.exit(main())
