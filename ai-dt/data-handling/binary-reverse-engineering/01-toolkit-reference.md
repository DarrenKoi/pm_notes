---
tags: [binary, reverse-engineering, tooling, kaitai, hex-editor, entropy]
level: intermediate
last_updated: 2026-07-10
---

# 범용 Binary RE 도구·기법 총람

> 문서화되지 않은 binary 파일 구조를 복원하는 도구와 기법. triage → 해부 → 형식화 → 통계적 필드 탐지 순. 이 폴더의 `scripts/bre.py`가 여기 나오는 통계 기법(§4·§5)을 구현한다.

## 전체 흐름

1. **Triage**: 무엇인가? 컨테이너·압축·텍스트혼합·순수 record? (`file`, `binwalk`, entropy)
2. **Corpus 확보**: 파일 하나로는 알 수 없다. **변수 하나만 바꾼** 파일을 여러 개 확보 → 차분 분석(differential analysis)이 미지 포맷 복원의 최고 지렛대.
3. **대화식 해부**: 구조 인식 hex editor에서 header를 손으로 주석.
4. **형식화**: struct를 이해하면 Kaitai/construct/ImHex 패턴으로 적어 재현·검토 가능하게.
5. **미지 구간 공략**: 통계적 필드 탐지(float/timestamp/length/CRC) + numpy grid 스캔.

계측 파일은 거의: `[magic/version header][metadata block][측정값 배열][trailer/checksum]`. 할 일은 배열의 offset·dtype·endianness·stride를 찾는 것.

---

## 1. 파일 정체 파악 / triage

| 도구 | 하는 일 | 설치 | 예시 | 언제 |
|---|---|---|---|---|
| `file` / libmagic | magic 기반 분류, 30년 축적 5000+ 시그니처 | 기본 내장; `brew install libmagic` | `file -k -p f.dat` | 항상 제일 먼저 |
| `xxd` / `hexdump` | raw hex+ASCII 덤프 | 기본 내장 | `xxd -g 1 -l 256 f.dat` | header 눈으로 확인 |
| `strings` | ASCII/UTF 런 추출 | binutils | `strings -n 6 -t x f.dat` (`-t x`=hex offset) | version·장비 ID·단위·컬럼명 |
| **TrID** | 학습형 시그니처 분류(10000+), % 랭킹 | mark0.net (freeware) | `trid f.dat` | `file`이 "data"라 할 때 2차 소견 |
| **binwalk** | 임베디드 시그니처+엔트로피 선형 스캔 | `pip install binwalk` | `binwalk -E f.dat` | 임베디드 컨테이너/압축 1차 탐지 (오탐 많음) |
| **unblob** ⭐ | binwalk 후계. 포맷 스펙으로 파싱·carve, 오탐 적음, 재귀 | `pip install unblob` (+`--install-deps`) | `unblob -e out/ f.dat` | 추출은 binwalk보다 이걸 우선 |
| **PolyFile** | cleanroom libmagic + Kaitai 계측, 임의 offset 임베디드 탐지, HTML 뷰어 | `pip install polyfile` | `polyfile f.dat --html o.html` | polyglot/오프셋 임베디드 의심 시 |
| **ent** | 엔트로피·χ²·평균·상관 | `apt install ent` | `ent f.dat` | 구간 무작위성 정량화 |

`bre.py triage`가 위의 `file`+`strings`+entropy+임베디드 스캔을 JSON 한 방으로 대체한다 (외부 설치 불필요).

**엔트로피 읽기** (bits/byte): ~0=상수/padding · 1~4=구조적 record/정수 · 4~6=텍스트/혼합 · **7.5~8.0=압축 또는 암호화**(엔트로피만으로는 구분 불가). `binwalk -E`의 offset-엔트로피 그래프에서 급상승 계단 = 압축/암호 구간 경계.

**압축 magic** (선두 바이트, `xxd`로 확인):
- gzip `1f 8b 08` · zlib `78 01`/`78 9c`/`78 da` · **zstd `28 b5 2f fd`** · lz4 frame `04 22 4d 18` · xz `fd 37 7a 58 5a` · bzip2 `42 5a 68`(`BZh`) · zip `50 4b 03 04`.
- headerless zlib/deflate 시험: `python -c "import zlib,sys; print(zlib.decompress(open(sys.argv[1],'rb').read()))" f.dat`, raw deflate는 `zlib.decompressobj(-15)`. zstd는 `pip install zstandard` 후 `ZstdDecompressor().decompress(data)`.

---

## 2. Hex editor / 구조 탐색기

| 도구 | 강점 | 라이선스 | 스크립트/CLI | OS |
|---|---|---|---|---|
| **ImHex** ⭐ | RE 전용. C 계열 **Pattern Language**로 struct를 바이트 위에 색칠, 데이터 인스펙터·엔트로피·diff | **무료 GPLv2** | `.hexpat` 패턴, `plcli` 러너, 방대한 커뮤니티 패턴 | Win/mac/Linux |
| **010 Editor** ⭐ | **Binary Template**(`.bt`)의 사실상 표준, 300+ 내장 템플릿, 대용량 강함 | **유료 $49.99**(체험판) | 완전한 C 계열 스크립트 | Win/mac/Linux |
| **HxD** | 빠르고 견고한 무료 데일리 | 무료(closed) | **템플릿·스크립트 없음** | **Windows 전용** |
| **Hexinator/Synalyze It!** | "Grammar" 기반 파싱, insert/delete 감지 diff | freemium(스크립트 유료 ~$80) | Lua/Python(유료) | Hexinator: Win/Linux, Synalyze: **macOS** |
| **wxHexEditor** | 멀티 GB/raw device | 무료(GPL) | 제한적, **사실상 개발 중단** | Win/mac/Linux |

**추천:** **ImHex**로 시작(무료·최고의 패턴 언어·활발). 준비된 템플릿이 많이 필요하면 유료 **010 Editor**. wxHexEditor는 >2GB device 편집 아니면 피할 것.

**ImHex Pattern 예시** (계측 header):
```rust
#pragma endian little
struct Header {
    char magic[4]; u16 version; u32 n_points;
    double x_start, x_step; padding[8];
};
struct File { Header hdr; float data[hdr.n_points] @ 64; };
File file @ 0x00;
```

---

## 3. 포맷 기술 언어 / 파서 생성기 (§형식화, Runbook Phase 6)

struct를 이해하면 **형식 문법으로 적어** 재현·테스트·공유 가능하게 한다.

### Kaitai Struct ⭐ ("스펙 + 원하는 언어 파서"에 최적)
선언적 YAML `.ksy` → C++/C#/Go/Java/JS/Lua/Nim/Perl/PHP/Python/Ruby/Rust 파서로 컴파일. **Web IDE**(ide.kaitai.io)가 업로드한 파일 위에 struct를 실시간 시각화 → 미지 포맷 반복 탐색에 이상적.
```yaml
meta: { id: metro_file, endian: le }
seq:
  - { id: magic, contents: "MET1" }
  - { id: version, type: u2 }
  - { id: n_points, type: u4 }
  - { id: x_start, type: f8 }
  - { id: x_step, type: f8 }
  - { id: samples, type: f4, repeat: expr, repeat-expr: n_points }
```
`kaitai-struct-compiler -t python metro_file.ksy` → `MetroFile.from_file("f.dat").samples`. 설치: `brew install kaitai-struct-compiler`.

### Python `construct` (빠른 Python 전용 작업에 최적)
같은 선언으로 **파싱·빌드 둘 다**.
```python
from construct import Struct, Const, Int16ul, Int32ul, Float64l, Float32l, Array, this
metro = Struct(
    "magic" / Const(b"MET1"), "version" / Int16ul, "n_points" / Int32ul,
    "x_start" / Float64l, "x_step" / Float64l,
    "samples" / Array(this.n_points, Float32l),
)
obj = metro.parse_file("f.dat")   # obj.samples
```
`pip install construct`. 큰 배열은 numpy로, header만 construct로 하면 빠르다.

### 그 외
- **hachoir** (`pip install hachoir`; `hachoir-urwid f.dat`) — 미지 스트림을 비트 단위 트리로 **탐색**. construct가 *명세*라면 hachoir는 *탐험*.
- **Rust `binrw`** (`cargo add binrw`) — `#[derive]` 기반 고속 프로덕션 파서.
- **scapy** — 파일이 TLV/record 스트림이면 layer로 기술.
- **Apache Daffodil (DFDL)** — XML 스키마 기반, 벤더 중립·아카이벌·round-trip. float/NaN 비트 보존. 무거우니 아카이벌 스펙이 필요할 때만.

**실용 선택:** Kaitai Web IDE에서 반복 → Python export. 스크립트는 `construct`+numpy. 속도 필요 시 binrw.

---

## 4. 자동/알고리즘적 포맷 추론

두 갈래: **(A) 동적 분석**(파일을 읽는 프로그램을 계측해 바이트 소비를 관찰 — 가장 강력하나 실행파일 필요), **(B) corpus 추론**(파일만으로 통계 정렬).

### (A) 동적 taint (파서 실행파일 필요)
Polyglot/AutoFormat/Tupni의 공통 통찰: *프로그램이 어떻게 파싱하느냐가 곧 포맷이다*.
- **Tupni**(MS Research, CCS'08) — record 시퀀스·타입·제약 복원. 랜드마크 논문.
- **PolyTracker**(Trail of Bits) ⭐ — 실용 현대판. LLVM 기반 universal taint. 벤더 파서를 이 위에서 돌리면 어떤 입력 바이트가 어떤 함수에 닿는지 보이고, CFG 문법까지 추출.

### (B) corpus 추론 (파일만 필요) — 현재 쓸 만한 OSS
| 도구 | 방법 | 쓸만? | 비고 |
|---|---|---|---|
| **NETZOB** | 서열 정렬(Needleman-Wunsch)+클러스터링 | ✅ `pip install netzob` | corpus 필드 추론 범용 출발점 |
| **NEMESYS** | 바이트 델타/비트 congruence 세그먼트 | ✅ 연구코드 | 순수 binary 메시지에 강함 |
| **FieldHunter** | 통계로 필드 의미(length/counter/addr) 추론 | ✅ 재구현 | |
| **BinaryInferno** ⭐ | float/length/timestamp 탐지기 앙상블 투표 | ✅ Python | **원하는 것에 가장 근접** — 필드 자동 추측 |

**실용 한계 (중요):** 이들은 *네트워크 프로토콜*(짧은 메시지 다수)용이다. 계측 파일은 *긴 단일 메시지 + 큰 배열*이라 정렬 기반 도구는 배열 본체에서 약하다. → **header/metadata 영역(선두 수백 바이트, corpus)** 에만 BinaryInferno/Netzob를 쓰고, **배열 본체는 §6 numpy 스캔**으로. (`bre.py`가 후자를 구현.)

### 모든 도구를 이기는 기법: 차분 분석
변수 하나만 바꾼 파일을 만들고 `cmp -l a b` / `radiff2 -x a b` / `bre.py diff`.
- 측정점 +1 → **count field**(1 증가)와 **length/size field**(stride만큼 증가), 배열 끝 offset.
- setpoint 변경 → 그 스칼라 위치.
- 같은 데이터 재저장 → 달라진 바이트 = **timestamp/seq/checksum**.

---

## 5. 통계/휴리스틱 필드 탐지 (`bre.py`가 구현)

**IEEE-754 float 배열 (계측 payload).** offset+dtype+endianness마다 디코딩해 *그럴듯함* 점수:
- `NaN`/`Inf` 없음, 지수 정상(값 대략 `1e-9…1e9`).
- 장비 물리 범위(nm/µm/%/dB) 안.
- **낮은 kurtosis / 매끄러움**: 실제 trace는 국소적으로 매끄럽다(인접 샘플 상관). `mean(|diff(x)|)` 작고 kurtosis 정상.
- x86 장비는 float32-LE가 압도적. float64-LE, big-endian도 시험.

**Timestamp** (디코딩값이 "현재 근처"인 정수 필드 스캔):
- **Unix epoch**(u32 초): 현재 ~`1.7e9` → 2024–2026.
- **Windows FILETIME**(u64, 1601 기준 100ns): 현재 ~`1.3e17`.
- **OLE date**(f8, 1899-12-30 기준 일수): 현재 ~`45000`–`46000`. Windows/VB/Office 계열 SW에 흔함.

**Length-prefix / offset-table:**
- **length prefix**: 블록 직전 u16/u32 값 ≈ 블록 바이트수(또는 원소수×stride). 차분 분석으로 즉시 확인.
- **offset table**: 파일 앞의 단조증가 u32/u64 런, 각 값 `< filesize`, 델타가 record 크기 → 디렉토리/인덱스.

**Checksum / CRC:**
- **위치**: 보통 record/파일 말미 1/2/4바이트. 커버 바이트 하나 바꾸면 *불규칙하게* 바뀌는 필드(차분 분석으로 드러남).
- **파라미터 식별** ((메시지, checksum) 쌍 여러 개 필요):
  - **CRC RevEng**(`reveng`) — 113개 preset + `-s` 검색으로 poly/init/xorout/reflect 복원. `apt install reveng`.
  - **CRC Beagle** ⭐ — Python, 차분 기법으로 비표준 XOR-in/out도 복원, 재생성 코드까지 출력.
  - 실패 시 additive/XOR/Fletcher/Adler를 직접 시험.

**Corpus 열(offset) 분산** (같은 포맷 파일 다수일 때 킬러 기법): 파일들을 행으로 쌓아 열별 분산 계산.
- 분산 0 = 상수(magic/version/padding/고정 필드).
- 낮은 분산 = enum/flag/counter.
- 최대 엔트로피 = timestamp/checksum 또는 배열 시작.
```python
import numpy as np
rows = [np.frombuffer(open(f,'rb').read(), np.uint8) for f in files]
minlen = min(map(len, rows)); m = np.array([r[:minlen] for r in rows])
var = m.var(axis=0)   # var==0 → 고정, 급점프 → header/payload 경계
```
→ `bre.py variance`가 이걸 구현.

---

## 6. 실전 스캔 스크립트 (`bre.py`에 내장)

### (a) offset×dtype×endianness 격자로 측정 배열 찾기 — `bre.py arrays`
모든 합리적 시작 offset·자료형을 디코딩해 *그럴듯함* 점수. 진짜 배열이 노이즈보다 훨씬 높은 점수. `--lo/--hi`를 장비 물리 범위로 좁히면 정확도 급상승. record 안에 interleaved면 `--stride`/`--payload-offset` 사용.

### (b) 자기상관으로 record stride 찾기 — `bre.py stride`
고정 크기 record 스트림은 record 크기에서 *주기적*. 자기상관 피크가 곧 stride. `bre.py`는 FFT 자기상관 + **stride로 접었을 때 분산 0인 열(고정 필드) 비율**로 검증하고, harmonic(32/48…)보다 fundamental(16)을 우선한다.

### (c) endianness/폭 한눈에
```python
import struct
w = data[0x40:0x48]
for fmt in ['<I','>I','<Q','>Q','<f','<d']:
    try: print(fmt, struct.unpack_from(fmt, w))
    except: pass
```

---

## 계측 `.dat` 빠른 시작 체크리스트

1. `bre.py triage f.dat` — magic·version·장비 ID·단위.
2. 압축이면 `unblob -e out/ f.dat`로 해제 후 재triage.
3. 측정점 수만 다른 파일 ≥5개 → `bre.py variance` (header/payload 경계·고정 필드).
4. 측정점 +1 두 파일 → `bre.py diff` (count·stride).
5. `bre.py stride` + `bre.py arrays --stride` → 측정값 offset·dtype·endianness. `element_count == 측정점 수` 확인.
6. **Kaitai `.ksy`** 로 적고 corpus 전체 round-trip 검증.
7. 말미의 불규칙 변경 바이트 → **CRC RevEng/Beagle**로 checksum 확정(파일을 되쓸 경우).

## 참고 자료 (References)

- unblob.org · imhex.org · kaitai.io · construct.readthedocs.io · github.com/jam1garner/binrw
- github.com/binaryinferno/binaryinferno · github.com/netzob/netzob · github.com/vs-uulm/nemesys
- reveng.sourceforge.io · github.com/colinoflynn/crcbeagle · github.com/trailofbits/polyfile · github.com/trailofbits/polytracker
- Tupni (Microsoft Research, CCS'08) · Trail of Bits "Two new tools that tame the treachery of files"
- 큐레이션: github.com/techge/PRE-list · github.com/extremecoders-re/re-list
