#!/usr/bin/env python3
"""
bre.py - Binary Reverse Engineering toolkit (agent-executable)

미지의 binary file 구조를 복원하기 위한 subcommand 모음.
모든 subcommand는 stdout으로 JSON을 출력한다. Agent가 결과를 그대로 파싱해
다음 phase의 입력으로 넘길 수 있도록 설계되었다.

의존성: Python 3.9+, numpy. (scipy 불필요 - 통계는 자체 구현)

사용 예:
    python bre.py triage   sample.dat
    python bre.py variance corpus/*.dat
    python bre.py diff     a.dat b.dat
    python bre.py arrays   sample.dat --max-offset 4096
    python bre.py stride   sample.dat --max-stride 4096
    python bre.py stamps   sample.dat

Exit code는 항상 0 (분석 실패도 JSON의 error 필드로 보고). Agent가 exit code로
분기하지 않고 JSON을 읽도록 하기 위함.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# 임의의 바이트를 정수/실수로 강제 해석하는 도구이므로 overflow/invalid 는 정상이다.
# stdout 의 JSON 을 agent 가 파싱하므로 stderr 노이즈를 억제한다.
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

# 압축/컨테이너 magic. (bytes prefix, 이름) - 파일 선두 검사용
MAGICS: list[tuple[bytes, str]] = [
    (b"\x1f\x8b\x08", "gzip"),
    (b"\x78\x01", "zlib(no/low compression)"),
    (b"\x78\x9c", "zlib(default)"),
    (b"\x78\xda", "zlib(best)"),
    (b"\x78\x5e", "zlib"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\x04\x22\x4d\x18", "lz4(frame)"),
    (b"\xfd7zXZ", "xz"),
    (b"BZh", "bzip2"),
    (b"PK\x03\x04", "zip"),
    (b"II\x2a\x00", "TIFF(little-endian)"),
    (b"MM\x00\x2a", "TIFF(big-endian)"),
    (b"II\x2b\x00", "BigTIFF(LE)"),
    (b"\x89HDF\r\n\x1a\n", "HDF5"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"BM", "BMP"),
    (b"CDF\x01", "NetCDF"),
    (b"\x93NUMPY", "npy"),
    (b"SQLite format 3\x00", "SQLite"),
    (b"\xd0\xcf\x11\xe0", "MS Compound File (OLE)"),
]

DTYPES = ["<f4", ">f4", "<f8", ">f8", "<i4", ">i4", "<i2", ">i2", "<u4", ">u4"]


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def _shannon_entropy(buf: np.ndarray) -> float:
    """바이트 배열의 Shannon entropy (bits/byte, 0.0~8.0)."""
    if buf.size == 0:
        return 0.0
    counts = np.bincount(buf, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / buf.size
    return float(-np.sum(probs * np.log2(probs)))


def _kurtosis(x: np.ndarray) -> float:
    """Fisher kurtosis (정규분포=0). scipy 없이 구현."""
    x = x.astype(np.float64)
    n = x.size
    if n < 4:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0 or not math.isfinite(s):
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _emit(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# triage : 파일 정체 파악 (magic / entropy / strings / 압축 여부)
# ---------------------------------------------------------------------------

def cmd_triage(args) -> dict:
    data = _read(args.path)
    buf = np.frombuffer(data, np.uint8)

    magic_hits = [name for sig, name in MAGICS if data.startswith(sig)]

    # 임베디드 signature (offset 0 이외 위치) - binwalk 의 최소 대체재
    embedded = []
    for sig, name in MAGICS:
        if len(sig) < 3:  # 2바이트 magic은 오탐이 너무 많음
            continue
        start = 1
        while True:
            i = data.find(sig, start)
            if i == -1 or len(embedded) >= 50:
                break
            embedded.append({"offset": i, "offset_hex": hex(i), "signature": name})
            start = i + 1

    # 블록 단위 entropy - 압축/암호화 구간 경계 탐지
    block = args.block
    blocks = []
    for off in range(0, len(data), block):
        chunk = buf[off : off + block]
        if chunk.size < 64:
            break
        blocks.append({"offset": off, "entropy": round(_shannon_entropy(chunk), 3)})

    ent_all = _shannon_entropy(buf)
    if ent_all >= 7.5:
        ent_verdict = "compressed_or_encrypted"
    elif ent_all >= 6.0:
        ent_verdict = "mixed_or_packed"
    elif ent_all >= 4.0:
        ent_verdict = "text_or_mixed"
    else:
        ent_verdict = "structured_records"

    # ASCII strings (offset 포함) - `strings -t x` 대체
    strings = []
    for m in re.finditer(rb"[\x20-\x7e]{%d,}" % args.min_str, data):
        strings.append({"offset": m.start(), "offset_hex": hex(m.start()),
                        "text": m.group().decode("ascii", "replace")})
        if len(strings) >= args.max_str:
            break

    # UTF-16LE strings (Windows 계열 장비 SW에서 흔함)
    utf16 = []
    for m in re.finditer(rb"(?:[\x20-\x7e]\x00){%d,}" % args.min_str, data):
        utf16.append({"offset": m.start(), "offset_hex": hex(m.start()),
                      "text": m.group().decode("utf-16-le", "replace")})
        if len(utf16) >= 40:
            break

    return {
        "command": "triage",
        "path": args.path,
        "size": len(data),
        "head_hex": data[:64].hex(),
        "head_ascii": "".join(chr(b) if 32 <= b < 127 else "." for b in data[:64]),
        "tail_hex": data[-32:].hex(),
        "magic_at_offset_0": magic_hits or None,
        "embedded_signatures": embedded[:50],
        "entropy_overall": round(ent_all, 3),
        "entropy_verdict": ent_verdict,
        "entropy_blocks": blocks[:200],
        "ascii_strings": strings,
        "utf16le_strings": utf16,
        "next_step": (
            "압축으로 판정되면 먼저 해제 후 재triage. 아니면 02-known-format-shortcuts 확인 → "
            "corpus 수집 후 `variance` 실행."
        ),
    }


# ---------------------------------------------------------------------------
# variance : corpus 전체의 offset별 분산 -> header/payload 경계, 고정 필드
# ---------------------------------------------------------------------------

def cmd_variance(args) -> dict:
    files = args.paths
    if len(files) < 2:
        return {"command": "variance", "error": "최소 2개 파일 필요"}

    blobs = [_read(f) for f in files]
    n = min(len(b) for b in blobs)
    m = np.array([np.frombuffer(b[:n], np.uint8) for b in blobs])

    var = m.var(axis=0)
    # 모든 파일에서 동일한 offset = 고정 필드 (magic/version/padding)
    constant = np.flatnonzero(var == 0)

    # 고정 구간을 연속 range로 묶기
    def _runs(idx: np.ndarray) -> list[dict]:
        out = []
        if idx.size == 0:
            return out
        start = prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i == prev + 1:
                prev = i
                continue
            out.append({"start": start, "end": prev, "length": prev - start + 1,
                        "start_hex": hex(start),
                        "bytes_hex": blobs[0][start : prev + 1][:32].hex()})
            start = prev = i
        out.append({"start": start, "end": prev, "length": prev - start + 1,
                    "start_hex": hex(start),
                    "bytes_hex": blobs[0][start : prev + 1][:32].hex()})
        return out

    const_runs = [r for r in _runs(constant) if r["length"] >= args.min_run]

    # header/payload 경계 추정: 분산이 지속적으로 높아지기 시작하는 지점
    win = 32
    if n > win * 4:
        smooth = np.convolve(var, np.ones(win) / win, mode="same")
        thresh = smooth.max() * 0.25 if smooth.max() > 0 else 0
        above = np.flatnonzero(smooth > thresh)
        boundary = int(above[0]) if above.size else None
    else:
        boundary = None

    return {
        "command": "variance",
        "files": files,
        "compared_bytes": int(n),
        "file_sizes": [len(b) for b in blobs],
        "constant_runs": const_runs[:60],
        "constant_byte_count": int(constant.size),
        # 주의: 이 값은 약한 heuristic 이다. 신뢰할 신호는 constant_runs.
        "weak_boundary_hint": boundary,
        "weak_boundary_hint_hex": hex(boundary) if boundary else None,
        "weak_boundary_caveat": (
            "smoothing 기반 추정이라 실제 header 크기와 수 바이트 어긋난다. "
            "확정은 constant_runs 와 `diff` 결과로 하라. 이 값만 믿고 parser 를 쓰지 말 것."
        ),
        "interpretation": {
            "variance==0": "모든 파일에서 동일 -> magic, version, 고정 struct 필드, padding",
            "low_variance": "enum / flag / counter 후보",
            "high_variance": "timestamp, checksum, 또는 payload 시작",
        },
        "next_step": "고정 구간 이후를 payload로 보고 `arrays` / `stride` 실행. 단일 변수만 바꾼 두 파일로 `diff` 실행.",
    }


# ---------------------------------------------------------------------------
# diff : 두 파일의 바이트 차이 (differential analysis)
# ---------------------------------------------------------------------------

def cmd_diff(args) -> dict:
    a, b = _read(args.a), _read(args.b)
    n = min(len(a), len(b))
    aa = np.frombuffer(a[:n], np.uint8)
    bb = np.frombuffer(b[:n], np.uint8)
    diff_idx = np.flatnonzero(aa != bb)

    changes = []
    for i in diff_idx[: args.max_changes]:
        i = int(i)
        changes.append({
            "offset": i, "offset_hex": hex(i),
            "a": int(aa[i]), "b": int(bb[i]),
            "a_hex": f"{int(aa[i]):02x}", "b_hex": f"{int(bb[i]):02x}",
        })

    # 인접한 변경 바이트를 필드 후보로 묶고, 정수/실수로 해석 시도
    fields = []
    if diff_idx.size:
        groups, start, prev = [], int(diff_idx[0]), int(diff_idx[0])
        for i in diff_idx[1:]:
            i = int(i)
            if i <= prev + args.gap:
                prev = i
                continue
            groups.append((start, prev))
            start = prev = i
        groups.append((start, prev))

        for s, e in groups[:40]:
            width = e - s + 1
            # 필드 경계는 정렬되어 있을 가능성이 높으므로 2/4/8 바이트로 확장 해석
            interp = {}
            for size, fmts in ((2, ["<H", ">H", "<h", ">h"]),
                               (4, ["<I", ">I", "<i", ">i", "<f", ">f"]),
                               (8, ["<Q", ">Q", "<d", ">d"])):
                if s + size <= n:
                    for f in fmts:
                        try:
                            va = struct.unpack_from(f, a, s)[0]
                            vb = struct.unpack_from(f, b, s)[0]
                            interp[f] = {"a": va, "b": vb,
                                         "delta": (vb - va) if isinstance(va, (int, float)) else None}
                        except Exception:
                            pass
            fields.append({
                "start": s, "start_hex": hex(s), "end": e,
                "changed_width": width,
                "interpretations": interp,
            })

    return {
        "command": "diff",
        "a": args.a, "b": args.b,
        "size_a": len(a), "size_b": len(b),
        "size_delta": len(b) - len(a),
        "changed_byte_count": int(diff_idx.size),
        "changed_bytes": changes,
        "field_candidates": fields,
        "hint": (
            "size_delta 가 record stride 의 배수이면 그 값이 stride. "
            "delta==1 인 정수 필드는 count field 후보. "
            "delta 가 불규칙한 말미 필드는 checksum 후보."
        ),
        "next_step": "count/length 필드를 확정한 뒤 `arrays` 로 payload dtype 확정.",
    }


# ---------------------------------------------------------------------------
# arrays : offset x dtype x endianness 격자 탐색으로 측정값 배열 찾기
# ---------------------------------------------------------------------------

def _plausibility(x: np.ndarray, lo: float, hi: float) -> float:
    """측정값 배열다움 점수. 높을수록 실제 데이터일 가능성 높음."""
    x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        return -1e9
    if np.ptp(x) == 0:                       # 전부 상수 = 의미 없음
        return -1e9
    absx = np.abs(x)
    nz = absx[absx > 0]
    if nz.size == 0:
        return -1e9
    # 값이 물리적으로 그럴듯한 범위에 있는가 (가장 강한 신호)
    in_range = float(np.mean((nz >= lo) & (nz <= hi)))
    # 대부분의 값이 범위를 벗어나면 측정 배열이 아니다. 강하게 배제.
    if in_range < 0.5:
        return -1e9
    # 실제 계측 trace는 국소적으로 매끄럽다 (인접 샘플 상관)
    d = np.abs(np.diff(x))
    scale = absx.mean() + 1e-12
    smooth = -math.log1p(float(d.mean()) / scale)
    # 뾰족하지 않은 분포
    kurt = -abs(_kurtosis(x)) * 0.05
    return smooth + kurt + 6.0 * in_range


def _strided_field(data: bytes, payload_off: int, stride: int, k: int, dt: str) -> np.ndarray | None:
    """record 배열에서 offset k 의 필드 컬럼만 뽑아낸다 (interleaved field 추출)."""
    itemsize = np.dtype(dt).itemsize
    if k + itemsize > stride:
        return None
    arr = np.frombuffer(data, np.uint8, offset=payload_off)
    n_rec = arr.size // stride
    if n_rec < 8:
        return None
    rows = arr[: n_rec * stride].reshape(n_rec, stride)
    col = np.ascontiguousarray(rows[:, k : k + itemsize])
    return col.view(np.dtype(dt)).ravel()


def _describe(x: np.ndarray, score: float, **extra) -> dict:
    return {
        "score": round(score, 3),
        "element_count": int(x.size),
        "preview": [round(float(v), 6) for v in x[:8]],
        "min": round(float(np.min(x)), 6),
        "max": round(float(np.max(x)), 6),
        "mean": round(float(np.mean(x)), 6),
        **extra,
    }


def cmd_arrays(args) -> dict:
    data = _read(args.path)
    dtypes = args.dtypes.split(",") if args.dtypes else DTYPES
    results = []

    if args.stride:
        # --- interleaved 모드: record 안의 필드 컬럼을 하나씩 검사 ---
        # `stride` 로 record 크기를 먼저 확정한 뒤 이 모드를 쓴다. CD-SEM 결과 파일의
        # 실제 측정값은 대개 record 내부 필드이지 연속 배열이 아니다.
        for dt in dtypes:
            for k in range(0, args.stride - np.dtype(dt).itemsize + 1):
                x = _strided_field(data, args.payload_offset, args.stride, k, dt)
                if x is None or x.size < args.min_count:
                    continue
                s = _plausibility(x[: args.sample], args.lo, args.hi)
                if s > -1e8:
                    results.append(_describe(
                        x, s, field_offset_in_record=k,
                        field_offset_hex=hex(args.payload_offset + k),
                        dtype=dt, mode="strided"))
        mode_desc = {"mode": "strided", "stride": args.stride,
                     "payload_offset": args.payload_offset}
    else:
        # --- 연속 배열 모드 ---
        max_off = min(args.max_offset, max(len(data) - 256, 1))
        for dt in dtypes:
            itemsize = np.dtype(dt).itemsize
            for off in range(0, max_off, args.step):
                avail = (len(data) - off) // itemsize
                if avail < args.min_count:
                    continue
                count = min(avail, args.sample)
                try:
                    x = np.frombuffer(data, dt, count=count, offset=off)
                except (ValueError, TypeError):
                    continue
                s = _plausibility(x, args.lo, args.hi)
                if s > -1e8:
                    results.append(_describe(
                        x, s, offset=off, offset_hex=hex(off), dtype=dt,
                        total_elements_from_offset=int(avail), mode="contiguous"))
        mode_desc = {"mode": "contiguous", "max_offset": max_off, "step": args.step}

    results.sort(key=lambda r: r["score"], reverse=True)
    return {
        "command": "arrays",
        "path": args.path,
        "size": len(data),
        "search": {**mode_desc, "plausible_range": [args.lo, args.hi], "dtypes": dtypes},
        "candidate_count": len(results),
        "top_candidates": results[: args.top],
        "hint": (
            "--lo/--hi 를 실제 계측 물리 범위로 좁히는 것이 정확도에 가장 큰 영향을 준다 "
            "(예: CD 값이 nm 단위면 --lo 1 --hi 1000). in_range<0.5 인 후보는 자동 배제된다. "
            "연속 배열 모드에서 결과가 전부 쓰레기면 record 구조일 가능성이 높다 -> "
            "`stride` 로 record 크기를 구한 뒤 --stride/--payload-offset 로 재실행하라. "
            "1위 후보의 element_count 가 tool UI 의 측정점 개수와 일치하면 확정."
        ),
        "next_step": "확정된 offset/dtype 을 05-formalize-parser 의 Kaitai/construct 스펙에 기록.",
    }


# ---------------------------------------------------------------------------
# stride : 자기상관으로 고정 크기 record 주기 탐지
# ---------------------------------------------------------------------------

def cmd_stride(args) -> dict:
    data = _read(args.path)[args.offset :]
    b = np.frombuffer(data, np.uint8).astype(np.float64)
    if b.size < 512:
        return {"command": "stride", "error": "데이터가 너무 짧음 (>=512 bytes 필요)"}
    b = b - b.mean()

    # FFT 기반 자기상관 (긴 파일에서도 빠름)
    nfft = 1 << (int(b.size * 2 - 1).bit_length())
    f = np.fft.rfft(b, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: b.size]
    if ac[0] != 0:
        ac = ac / ac[0]
    ac[0] = 0

    limit = min(args.max_stride, ac.size - 1)
    lags = np.arange(1, limit)
    vals = ac[1:limit]
    order = np.argsort(vals)[::-1][:40]
    peaks = sorted({int(lags[i]) for i in order if vals[i] > args.min_corr})

    # 기본 주기(fundamental) 추정: 상위 peak 들의 최대공약수적 성격
    def _is_fundamental(p: int, ps: list[int]) -> bool:
        return sum(1 for q in ps if q != p and q % p == 0) >= 1

    fundamentals = [p for p in peaks if _is_fundamental(p, peaks)] or peaks[:5]

    # 후보 stride 별 검증: stride 로 접었을 때 열 분산이 0인 컬럼(고정 필드)이 있는가
    verify = []
    for st in fundamentals[:8]:
        rows = b.size // st
        if rows < 4:
            continue
        raw = np.frombuffer(data, np.uint8)[: rows * st].reshape(rows, st)
        colvar = raw.var(axis=0)
        fixed = int(np.count_nonzero(colvar == 0))
        verify.append({
            "stride": st, "records": rows,
            "fixed_columns": fixed,
            "fixed_column_ratio": round(fixed / st, 3),
            "autocorr": round(float(ac[st]), 4),
        })
    # 32, 48 같은 harmonic 은 진짜 stride(16)와 fixed_column_ratio 가 같아진다.
    # 따라서 ratio 가 사실상 동률이면 가장 작은 stride(기본 주기)를 우선한다.
    verify.sort(key=lambda v: (-round(v["fixed_column_ratio"], 2), v["stride"]))
    best = verify[0]["stride"] if verify else None
    harmonics = [v["stride"] for v in verify[1:] if best and v["stride"] % best == 0]

    return {
        "command": "stride",
        "path": args.path,
        "search_offset": args.offset,
        "autocorr_peaks": peaks[:20],
        "fundamental_candidates": fundamentals[:8],
        "verified": verify,
        "best_stride": best,
        "harmonics_of_best": harmonics,
        "hint": (
            "fixed_column_ratio 가 높으면서 가장 작은 stride 가 진짜 record 크기다. "
            "그 배수(harmonics_of_best)는 같은 ratio 를 갖지만 기본 주기가 아니다. "
            "fixed column 위치가 record 내 고정 struct 필드다. "
            "반드시 `diff` 의 size_delta 로 교차 검증하라 - 측정점 1개 추가 시 커진 바이트 수가 곧 stride."
        ),
        "next_step": "확정 stride 로 record struct 를 05-formalize-parser 에서 기술.",
    }


# ---------------------------------------------------------------------------
# stamps : timestamp 후보 탐색 (Unix / FILETIME / OLE date)
# ---------------------------------------------------------------------------

EPOCH_UNIX = datetime(1970, 1, 1, tzinfo=timezone.utc)
EPOCH_FILETIME = datetime(1601, 1, 1, tzinfo=timezone.utc)
EPOCH_OLE = datetime(1899, 12, 30, tzinfo=timezone.utc)


def cmd_stamps(args) -> dict:
    data = _read(args.path)
    # timestamp 는 거의 항상 header 에 있다. 전체 스캔은 느리고 오탐만 늘린다.
    scan_end = min(len(data), args.max_bytes) if args.max_bytes > 0 else len(data)
    lo = datetime(args.year_lo, 1, 1, tzinfo=timezone.utc)
    hi = datetime(args.year_hi, 1, 1, tzinfo=timezone.utc)
    hits = []
    truncated = False

    def _add(off, kind, fmt, dt):
        hits.append({"offset": off, "offset_hex": hex(off), "kind": kind,
                     "struct_fmt": fmt, "decoded_utc": dt.isoformat()})

    for off in range(0, max(scan_end - 8, 0)):
        # u32 Unix epoch (seconds)
        for fmt in ("<I", ">I"):
            v = struct.unpack_from(fmt, data, off)[0]
            try:
                dt = EPOCH_UNIX + timedelta(seconds=v)
            except (OverflowError, OSError):
                continue
            if lo <= dt < hi:
                _add(off, "unix_seconds_u32", fmt, dt)

        # u64 Windows FILETIME (100ns ticks since 1601)
        for fmt in ("<Q", ">Q"):
            v = struct.unpack_from(fmt, data, off)[0]
            if 0 < v < 2**63:
                try:
                    dt = EPOCH_FILETIME + timedelta(microseconds=v / 10)
                except (OverflowError, OSError):
                    continue
                if lo <= dt < hi:
                    _add(off, "windows_filetime_u64", fmt, dt)

        # f8 OLE automation date (days since 1899-12-30)
        for fmt in ("<d", ">d"):
            v = struct.unpack_from(fmt, data, off)[0]
            if math.isfinite(v) and 30000 < v < 60000:
                try:
                    dt = EPOCH_OLE + timedelta(days=v)
                except (OverflowError, OSError):
                    continue
                if lo <= dt < hi:
                    _add(off, "ole_automation_date_f8", fmt, dt)

        if len(hits) >= args.max_hits:
            truncated = True
            break

    return {
        "command": "stamps",
        "path": args.path,
        "scanned_bytes": scan_end,
        "file_size": len(data),
        "window_years": [args.year_lo, args.year_hi],
        "candidate_count": len(hits),
        "truncated": truncated,
        "candidates": hits,
        "hint": (
            "오탐이 매우 많다 (임의의 4바이트가 우연히 유효 날짜로 해석되는 확률이 높음). "
            "단독으로 신뢰하지 말 것. 같은 recipe 를 두 번 저장한 두 파일의 `diff` 결과와 "
            "교차 검증하라 - 두 파일에서 값이 달라지는 offset 만 진짜 timestamp 후보다. "
            "truncated=true 면 max_hits 에 걸려 스캔이 중단된 것이므로 --max-bytes 를 좁혀라."
        ),
    }


# ===========================================================================
# recipe / 좌표 파일용 detector
#   측정값 배열과 달리 recipe 는 가변 길이 TLV, offset 테이블, 문자열 테이블,
#   직렬화 객체인 경우가 많다. 아래 4개가 그 구조를 공략한다.
# ===========================================================================

# ---------------------------------------------------------------------------
# strtab : 구조화된 문자열 테이블 추출 (recipe 파라미터명/값이 여기 산다)
# ---------------------------------------------------------------------------

def _printable(bs: bytes) -> bool:
    return all(0x20 <= c < 0x7f or c == 0x09 for c in bs)


def cmd_strtab(args) -> dict:
    data = _read(args.path)
    n = len(data)
    ml = args.min_len

    # 1) null-terminated C 문자열
    null_term = []
    for m in re.finditer(rb"[\x20-\x7e]{%d,}\x00" % ml, data):
        null_term.append({"offset": m.start(), "offset_hex": hex(m.start()),
                          "length": m.end() - m.start() - 1,
                          "text": m.group()[:-1].decode("ascii", "replace")})
        if len(null_term) >= args.max_hits:
            break

    # 2) length-prefixed (Pascal 스타일): u8/u16/u32 LE prefix 뒤 그 길이만큼 printable
    len_prefixed = []
    for psize, fmt in ((1, "<B"), (2, "<H"), (4, "<I")):
        off = 0
        while off + psize < n and len(len_prefixed) < args.max_hits:
            p = struct.unpack_from(fmt, data, off)[0]
            if ml <= p <= args.max_str and off + psize + p <= n:
                chunk = data[off + psize : off + psize + p]
                if _printable(chunk):
                    len_prefixed.append({
                        "offset": off, "offset_hex": hex(off),
                        "prefix_type": f"u{psize * 8}le", "declared_len": p,
                        "text": chunk.decode("ascii", "replace")})
                    off += psize + p           # 매칭 시 문자열 끝으로 점프 (겹침 방지)
                    continue
            off += 1

    # 3) UTF-16LE (Windows 계열 장비 SW)
    utf16 = []
    for m in re.finditer(rb"(?:[\x20-\x7e]\x00){%d,}" % ml, data):
        utf16.append({"offset": m.start(), "offset_hex": hex(m.start()),
                      "text": m.group().decode("utf-16-le", "replace")})
        if len(utf16) >= args.max_hits:
            break

    return {
        "command": "strtab",
        "path": args.path,
        "null_terminated": null_term,
        "length_prefixed": len_prefixed,
        "utf16le": utf16,
        "hint": (
            "recipe 파라미터명은 대개 length_prefixed 또는 null_terminated 로 저장된다. "
            "length_prefixed 는 오탐이 있으나 declared_len 이 실제 길이와 맞는 연속 항목이 "
            "촘촘히 나오면 진짜 문자열 테이블이다. 그 시작 offset 을 `offsets`/`tlv` 결과와 대조하라."
        ),
    }


# ---------------------------------------------------------------------------
# offsets : offset/pointer 테이블 탐지 (디렉토리 구조)
# ---------------------------------------------------------------------------

def cmd_offsets(args) -> dict:
    data = _read(args.path)
    n = len(data)
    found = []
    for fmt, size in (("<I", 4), (">I", 4), ("<Q", 8), (">Q", 8)):
        for base in range(0, min(args.max_base, max(n - size * args.min_run, 0)) + 1):
            run, pos = [], base
            while pos + size <= n and len(run) < args.max_entries:
                v = struct.unpack_from(fmt, data, pos)[0]
                # 각 값이 파일 안을 가리키고(0<v<=n) 단조 비감소이면 포인터 테이블 후보
                if 0 < v <= n and (not run or v >= run[-1]):
                    run.append(v)
                    pos += size
                else:
                    break
            if len(run) >= args.min_run:
                found.append({
                    "offset": base, "offset_hex": hex(base), "fmt": fmt,
                    "count": len(run),
                    "values": [int(x) for x in run],
                    "points_to_preview": [
                        {"offset": int(x), "bytes_hex": data[int(x):int(x) + 8].hex()}
                        for x in run[:6]],
                })
    # 겹치는 하위구간 제거: 같은 fmt 에서 더 긴 run 에 포함된 짧은 run 은 버림
    found.sort(key=lambda f: f["count"], reverse=True)
    kept, seen = [], []
    for f in found:
        span = (f["fmt"], f["offset"], f["offset"] + f["count"] * (4 if "I" in f["fmt"] else 8))
        if any(f["fmt"] == s[0] and s[1] <= f["offset"] < s[2] for s in seen):
            continue
        seen.append(span)
        kept.append(f)
        if len(kept) >= args.top:
            break

    return {
        "command": "offsets",
        "path": args.path,
        "size": n,
        "candidates": kept,
        "hint": (
            "count 가 크고 values 가 header 크기 이후를 가리키며 파일 끝까지 고르게 퍼져 있으면 "
            "진짜 디렉토리/포인터 테이블이다. 각 entry 가 가리키는 곳을 `tlv`/`strtab` 로 파싱하라. "
            "values 개수가 header 의 어떤 count 필드와 일치하는지 `diff` 로 교차 확인."
        ),
    }


# ---------------------------------------------------------------------------
# tlv : 가변 길이 TLV(tag-length-value) record 체인 탐지
# ---------------------------------------------------------------------------

_INT = {1: "B", 2: "H", 4: "I"}


def _walk_tlv(data: bytes, start: int, tag_sz: int, len_sz: int,
              endian: str, len_incl_hdr: bool, max_rec: int):
    n = len(data)
    e = "<" if endian == "le" else ">"
    tfmt, lfmt = e + _INT[tag_sz], e + _INT[len_sz]
    hdr = tag_sz + len_sz
    pos, recs = start, []
    while pos + hdr <= n and len(recs) < max_rec:
        tag = struct.unpack_from(tfmt, data, pos)[0]
        length = struct.unpack_from(lfmt, data, pos + tag_sz)[0]
        rec_len = length if len_incl_hdr else hdr + length
        if rec_len < hdr or pos + rec_len > n:
            break
        recs.append({"offset": pos, "tag": tag, "declared_len": length, "rec_len": rec_len})
        pos += rec_len
    return recs, pos


def cmd_tlv(args) -> dict:
    data = _read(args.path)
    n = len(data)
    trials = []
    # tail 여유: checksum/padding 이 붙을 수 있으므로 EOF 근처에서 끝나면 성공으로 본다
    for tag_sz in (1, 2, 4):
        for len_sz in (1, 2, 4):
            for endian in ("le", "be"):
                for incl in (False, True):
                    recs, end = _walk_tlv(data, args.start, tag_sz, len_sz,
                                          endian, incl, args.max_rec)
                    if len(recs) < args.min_rec:
                        continue
                    consumed = end - args.start
                    total = n - args.start
                    coverage = consumed / total if total else 0
                    # EOF(또는 EOF-tail) 에 정확히 안착하면 가산점
                    lands_clean = (n - end) <= args.tail
                    score = coverage + (0.5 if lands_clean else 0)
                    trials.append({
                        "tag_size": tag_sz, "len_size": len_sz, "endian": endian,
                        "len_includes_header": incl,
                        "records_parsed": len(recs), "bytes_consumed": consumed,
                        "coverage": round(coverage, 4),
                        "ends_at": end, "ends_at_hex": hex(end),
                        "lands_at_eof": lands_clean,
                        "score": round(score, 4),
                        "first_records": recs[:8],
                    })
    trials.sort(key=lambda t: t["score"], reverse=True)
    best = trials[0] if trials else None
    return {
        "command": "tlv",
        "path": args.path,
        "start": args.start,
        "size": n,
        "best": best,
        "other_candidates": trials[1:5],
        "hint": (
            "coverage≈1.0 이고 lands_at_eof=true 인 config 가 진짜 TLV 레이아웃이다. "
            "--start 는 보통 header 끝(고정 struct 이후) 또는 `offsets` 가 가리키는 첫 entry. "
            "tag 값들이 반복되면 (예: 1=name,2=value) 그 tag 사전을 만들어라. "
            "record 안이 또 TLV 이면(nested) 각 value 를 --start 로 재귀 실행."
        ),
    }


# ---------------------------------------------------------------------------
# serial : 내부 직렬화/컨테이너 포맷 시그니처 탐지 (recipe 가 사실은 XML/직렬화객체인 경우)
# ---------------------------------------------------------------------------

_SERIAL_SIGS = [
    (b"<?xml", "xml"),
    (b"<recipe", "xml-like"),
    (b"\x30\x82", "asn1-der-sequence"),
    (b"\x00\x01\x00\x00\x00\xff\xff\xff\xff", "dotnet-binaryformatter"),
    (b"PK\x03\x04", "zip/ooxml"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ms-compound-file(ole/mfc-doc)"),
    (b"\x1f\x8b\x08", "gzip"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"\x89HDF\r\n\x1a\n", "hdf5"),
    (b"\x08\x00\x00\x00", "possible-protobuf-or-len-prefixed"),
]


def cmd_serial(args) -> dict:
    data = _read(args.path)
    hits = []
    for sig, name in _SERIAL_SIGS:
        start = 0
        while True:
            i = data.find(sig, start)
            if i == -1 or len(hits) >= args.max_hits:
                break
            hits.append({"offset": i, "offset_hex": hex(i), "format": name,
                         "signature_hex": sig.hex()})
            start = i + 1

    # JSON 은 시그니처가 약하므로 '{' 뒤 '"' 밀도로 별도 추정
    json_like = [m.start() for m in re.finditer(rb'\{\s*"', data)][:20]

    # ASCII 비율 - 파일이 사실상 텍스트(INI/XML/CSV recipe)인지 판단
    printable_ratio = round(sum(1 for b in data if 0x20 <= b < 0x7f or b in (9, 10, 13)) / max(len(data), 1), 3)

    return {
        "command": "serial",
        "path": args.path,
        "size": len(data),
        "printable_ratio": printable_ratio,
        "mostly_text": printable_ratio > 0.85,
        "signature_hits": hits,
        "json_like_offsets": [hex(x) for x in json_like],
        "hint": (
            "printable_ratio>0.85 면 recipe 가 사실상 텍스트다(INI/XML/CSV) — 역공학 대신 텍스트 파서. "
            "xml/zip/ole/dotnet 시그니처가 잡히면 그 구간을 해당 표준 도구로 열어라 "
            "(zip→unzip, ole→olefile, xml→lxml). MFC CArchive 직렬화는 시그니처가 약하니 "
            "ms-compound-file 이 아니면 벤더 SW 가 MFC 인지 확인 후 스키마 역공학이 필요하다."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        prog="bre.py", description="Binary reverse engineering toolkit (JSON output)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="magic/entropy/strings 로 파일 정체 파악")
    t.add_argument("path")
    t.add_argument("--block", type=int, default=1024, help="entropy 블록 크기")
    t.add_argument("--min-str", type=int, default=6, help="최소 string 길이")
    t.add_argument("--max-str", type=int, default=120)
    t.set_defaults(func=cmd_triage)

    v = sub.add_parser("variance", help="corpus 의 offset별 분산 -> 고정 필드/경계")
    v.add_argument("paths", nargs="+")
    v.add_argument("--min-run", type=int, default=2, help="보고할 최소 고정 구간 길이")
    v.set_defaults(func=cmd_variance)

    d = sub.add_parser("diff", help="두 파일의 바이트 차이 (differential analysis)")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--gap", type=int, default=3, help="필드로 묶을 최대 간격")
    d.add_argument("--max-changes", type=int, default=200)
    d.set_defaults(func=cmd_diff)

    a = sub.add_parser("arrays", help="offset x dtype 격자 탐색으로 측정 배열 탐지")
    a.add_argument("path")
    a.add_argument("--max-offset", type=int, default=4096)
    a.add_argument("--step", type=int, default=1)
    a.add_argument("--min-count", type=int, default=64, help="배열로 인정할 최소 원소 수")
    a.add_argument("--sample", type=int, default=20000, help="점수 계산에 쓸 최대 원소 수")
    a.add_argument("--lo", type=float, default=1e-9, help="물리적으로 그럴듯한 최소 절대값")
    a.add_argument("--hi", type=float, default=1e9, help="물리적으로 그럴듯한 최대 절대값")
    a.add_argument("--top", type=int, default=10)
    a.add_argument("--dtypes", default="", help="쉼표 구분 dtype 목록 (기본: 전체)")
    a.add_argument("--stride", type=int, default=0,
                   help="record stride. 지정 시 interleaved 필드 모드로 동작")
    a.add_argument("--payload-offset", type=int, default=0,
                   help="--stride 모드에서 record 배열이 시작하는 offset")
    a.set_defaults(func=cmd_arrays)

    s = sub.add_parser("stride", help="자기상관으로 record 주기 탐지")
    s.add_argument("path")
    s.add_argument("--offset", type=int, default=0, help="payload 시작 offset")
    s.add_argument("--max-stride", type=int, default=4096)
    s.add_argument("--min-corr", type=float, default=0.05)
    s.set_defaults(func=cmd_stride)

    st = sub.add_parser("stamps", help="timestamp 후보 탐색")
    st.add_argument("path")
    st.add_argument("--year-lo", type=int, default=2000)
    st.add_argument("--year-hi", type=int, default=2035)
    st.add_argument("--max-hits", type=int, default=500)
    st.add_argument("--max-bytes", type=int, default=8192,
                    help="스캔할 선두 바이트 수 (timestamp는 header에 있다). 0=전체")
    st.set_defaults(func=cmd_stamps)

    # --- recipe / 좌표 파일용 ---
    st2 = sub.add_parser("strtab", help="구조화 문자열 테이블 추출 (recipe 파라미터명)")
    st2.add_argument("path")
    st2.add_argument("--min-len", type=int, default=4, help="최소 문자열 길이")
    st2.add_argument("--max-str", type=int, default=256, help="length-prefix 최대 선언 길이")
    st2.add_argument("--max-hits", type=int, default=200)
    st2.set_defaults(func=cmd_strtab)

    of = sub.add_parser("offsets", help="offset/pointer 테이블(디렉토리) 탐지")
    of.add_argument("path")
    of.add_argument("--min-run", type=int, default=4, help="테이블로 인정할 최소 entry 수")
    of.add_argument("--max-base", type=int, default=512, help="테이블 시작 offset 탐색 범위")
    of.add_argument("--max-entries", type=int, default=256)
    of.add_argument("--top", type=int, default=8)
    of.set_defaults(func=cmd_offsets)

    tv = sub.add_parser("tlv", help="가변 길이 TLV record 체인 탐지")
    tv.add_argument("path")
    tv.add_argument("--start", type=int, default=0, help="TLV 시작 offset (보통 header 끝)")
    tv.add_argument("--min-rec", type=int, default=3, help="유효로 볼 최소 record 수")
    tv.add_argument("--max-rec", type=int, default=100000)
    tv.add_argument("--tail", type=int, default=8, help="EOF 안착 허용 tail 바이트(checksum/padding)")
    tv.set_defaults(func=cmd_tlv)

    se = sub.add_parser("serial", help="내부 직렬화/컨테이너 포맷 시그니처 탐지")
    se.add_argument("path")
    se.add_argument("--max-hits", type=int, default=100)
    se.set_defaults(func=cmd_serial)

    args = p.parse_args()
    try:
        _emit(args.func(args))
    except Exception as e:  # agent 가 파싱할 수 있도록 에러도 JSON
        _emit({"command": args.cmd, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
