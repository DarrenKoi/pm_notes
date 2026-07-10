#!/usr/bin/env python3
"""
make_fixture.py - bre.py 검증용 합성 CD-SEM 유사 binary 생성기

실제 장비 파일을 얻기 전에 toolkit 이 정답을 찾아내는지 확인하기 위한 fixture.
"정답"을 알고 있는 파일을 만들어, bre.py 가 그 정답을 복원하는지 회귀 검증한다.

레이아웃 (little-endian):
    0x00  char[4]   magic "CDS1"
    0x04  u16       version = 3
    0x06  u16       reserved = 0
    0x08  u32       n_points
    0x0c  u32       timestamp (unix seconds)
    0x10  f8        x_start_nm
    0x18  f8        x_step_nm
    0x20  char[16]  recipe name (null-padded ASCII)
    0x30  ...       n_points x record{ u32 site_id; f32 x_nm; f32 y_nm; f32 cd_nm }  (stride 16)
    EOF-4 u32       trailing checksum (sum of all prior bytes)

사용:
    python make_fixture.py outdir/
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"CDS1"
HEADER_SIZE = 0x30
STRIDE = 16


def build(n_points: int, timestamp: int, recipe: str, seed: int, cd_center: float = 45.0) -> bytes:
    rng = np.random.default_rng(seed)

    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    struct.pack_into("<H", header, 0x04, 3)          # version
    struct.pack_into("<H", header, 0x06, 0)          # reserved
    struct.pack_into("<I", header, 0x08, n_points)
    struct.pack_into("<I", header, 0x0C, timestamp)
    struct.pack_into("<d", header, 0x10, 0.0)        # x_start_nm
    struct.pack_into("<d", header, 0x18, 90.0)       # x_step_nm
    name = recipe.encode("ascii")[:16]
    header[0x20 : 0x20 + len(name)] = name

    # 실제 계측 trace 처럼: 부드러운 변동 + 작은 noise (매끄러움이 arrays 점수의 근거)
    walk = np.cumsum(rng.normal(0, 0.12, n_points))
    walk -= walk.mean()
    cd = cd_center + walk + rng.normal(0, 0.05, n_points)

    body = bytearray()
    for i in range(n_points):
        body += struct.pack("<Ifff", i, float(i * 90.0), 0.0, float(cd[i]))

    blob = bytes(header) + bytes(body)
    checksum = sum(blob) & 0xFFFFFFFF
    return blob + struct.pack("<I", checksum)


def build_recipe() -> tuple[bytes, dict]:
    """가변 TLV + offset 테이블 + 문자열 테이블을 가진 recipe 유사 파일.

    레이아웃 (little-endian):
        0x00 char[4]  magic "RCP2"
        0x04 u16      version = 2
        0x06 u16      n_params
        0x08 u32[n]   offset table (각 TLV record 의 절대 offset)
        ...  TLV records: u16 tag, u16 length, value[length]  (length=value 바이트만)
             tag=1: null-term 문자열(파라미터명)
             tag=2: f8 setpoint
             tag=3: u16 length-prefixed 문자열(step 이름)
             tag=4: f8 setpoint
    """
    header = bytearray(b"RCP2")
    header += struct.pack("<H", 2)                     # version @0x04

    def tlv(tag: int, value: bytes) -> bytes:
        return struct.pack("<HH", tag, len(value)) + value

    recs = [
        tlv(1, b"CD_TARGET\x00"),                      # null-term name
        tlv(2, struct.pack("<d", 45.0)),               # setpoint
        tlv(3, struct.pack("<H", 8) + b"ETCH_STP"),    # u16 length-prefixed string
        tlv(4, struct.pack("<d", 90.0)),               # setpoint
    ]
    n_params = len(recs)
    header += struct.pack("<H", n_params)              # n_params @0x06
    table_off = len(header)                            # 0x08
    body_start = table_off + n_params * 4              # 0x18

    offs, pos = [], body_start
    for r in recs:
        offs.append(pos)
        pos += len(r)
    table = b"".join(struct.pack("<I", o) for o in offs)
    blob = bytes(header) + table + b"".join(recs)
    truth = {"magic": "RCP2", "n_params_offset": 0x06, "offset_table_offset": table_off,
             "offset_table_entries": offs, "tlv_start": body_start,
             "tlv_config": "tag_size=2,len_size=2,le,len=value-only"}
    return blob, truth


def build_coords(n_sites: int = 200, seed: int = 7) -> tuple[bytes, dict]:
    """웨이퍼 측정 좌표 파일. 고정 크기 record 라 arrays/stride/diff 로 커버된다.

    레이아웃 (little-endian):
        0x00 char[4]  magic "XY01"
        0x04 u32      n_sites
        0x08 ...      n_sites x record{ u32 site_id; f8 x_um; f8 y_um }  stride 20
    """
    rng = np.random.default_rng(seed)
    header = bytearray(b"XY01") + struct.pack("<I", n_sites)
    body = bytearray()
    cols = int(math.isqrt(n_sites)) or 1
    for i in range(n_sites):
        x = ((i % cols) - cols / 2) * 5000.0 + rng.normal(0, 3)   # um, 그리드
        y = ((i // cols) - cols / 2) * 5000.0 + rng.normal(0, 3)
        body += struct.pack("<Idd", i, float(x), float(y))
    blob = bytes(header) + bytes(body)
    truth = {"magic": "XY01", "n_sites": n_sites, "stride": 20, "payload_offset": 0x08,
             "x_field": "f8 @ record offset 4", "y_field": "f8 @ record offset 12"}
    return blob, truth


def build_recipe_xml() -> bytes:
    """recipe 가 사실은 내부 XML 인 경우 (serial 탐지 대상)."""
    xml = b"<?xml version='1.0'?><recipe name='CD_A'><step dwell='90'/></recipe>"
    return b"RCPX" + struct.pack("<I", len(xml)) + xml


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    outdir.mkdir(parents=True, exist_ok=True)

    # base: 256 points
    (outdir / "base.dat").write_bytes(build(256, 1_770_000_000, "RECIPE_A", seed=1))
    # 측정점 1개만 추가 -> count field 와 stride 를 differential 로 드러내기 위함
    (outdir / "one_more_point.dat").write_bytes(build(257, 1_770_000_000, "RECIPE_A", seed=1))
    # 같은 recipe 재저장 -> timestamp 만 달라짐
    (outdir / "resaved.dat").write_bytes(build(256, 1_770_003_600, "RECIPE_A", seed=1))
    # corpus 용: 동일 구조, 다른 데이터
    for k in range(4):
        (outdir / f"corpus_{k}.dat").write_bytes(
            build(256, 1_770_010_000 + k * 97, "RECIPE_A", seed=10 + k, cd_center=45.0 + k * 0.3)
        )

    # recipe / 좌표 파일 fixture
    recipe_blob, recipe_truth = build_recipe()
    (outdir / "recipe.dat").write_bytes(recipe_blob)
    coords_blob, coords_truth = build_coords()
    (outdir / "coords.dat").write_bytes(coords_blob)
    (outdir / "recipe_xml.dat").write_bytes(build_recipe_xml())

    truth = {
        "magic": "CDS1 @ 0x00",
        "n_points_field": "u32 @ 0x08",
        "timestamp_field": "u32 unix @ 0x0c",
        "header_size": hex(HEADER_SIZE),
        "record_stride": STRIDE,
        "record_struct": "u32 site_id; f32 x_nm; f32 y_nm; f32 cd_nm",
        "cd_array": "float32 LE, stride 16, first element @ 0x3c",
        "checksum": "u32 sum of prior bytes @ EOF-4",
        "recipe.dat": recipe_truth,
        "coords.dat": coords_truth,
        "recipe_xml.dat": "RCPX wrapper + 내부 <?xml> (serial 탐지 대상)",
    }
    print("Wrote fixtures to", outdir)
    print("GROUND TRUTH:")
    for k, v in truth.items():
        print(f"  {k:20s} = {v}")


if __name__ == "__main__":
    main()
