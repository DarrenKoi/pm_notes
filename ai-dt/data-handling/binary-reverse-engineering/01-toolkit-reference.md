---
tags: [binary, reverse-engineering, tooling, kaitai, hex-editor, entropy]
level: intermediate
last_updated: 2026-07-10
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# 범용 Binary RE 도구·기법 총람

> 문서화되지 않은 binary 파일 구조를 복원하는 도구와 기법. triage → 해부 → 형식화 → 통계적 필드 탐지 순. 이 폴더의 `scripts/bre.py`가 여기 나오는 통계 기법(§4·§5)을 구현한다.

## 전체 흐름

1. **Triage**: 무엇인가? 컨테이너·압축·텍스트혼합·순수 record? (`file`, `binwalk`, entropy)
2. **Corpus 확보**: 파일 하나로는 알 수 없다. **변수 하나만 바꾼** 파일을 여러 개 확보 → 차분 분석(differential analysis)이 미지 포맷 복원의 최고 지렛대.
3. **대화식 해부**: 구조 인식 hex editor에서 header를 손으로 주석.
4. **형식화**: struct를 이해하면 Kaitai/construct/ImHex 패턴으로 적어 재현·검토 가능하게.
5. **미지 구간 공략**: 통계적 필드 탐지(float/timestamp/length/CRC) + numpy grid 스캔.

고정 record를 가정한 학습용 계측 구조 예: `[magic/version header][metadata block][측정값 배열][trailer/checksum]`. 실제 파일이 이 구조라는 보장은 없다. 배열이면 offset·dtype·endianness·stride·단위·개수와 UI/export 값 대응을 확인한다.

---

## 1. 파일 정체 파악 / triage

| 도구 | 하는 일 | 설치 | 예시 | 언제 |
|---|---|---|---|---|
| `file` / libmagic | magic 기반 형식 후보 분류; DB 범위/개수 미확인 | OS 설치 여부 확인; Homebrew 명령 미검증 | `file -k -p f.dat` | 항상 제일 먼저 |
| `xxd` / `hexdump` | raw hex+ASCII 덤프 | 기본 내장 | `xxd -g 1 -l 256 f.dat` | header 눈으로 확인 |
| `strings` | ASCII/UTF 런 추출 | binutils | `strings -n 6 -t x f.dat` (`-t x`=hex offset) | version·장비 ID·단위·컬럼명 |
| **TrID** | 시그니처 후보/% 랭킹; 현 DB 개수/방식 미확인 | mark0.net (freeware) | `trid f.dat` | `file`이 "data"라 할 때 2차 소견 |
| **binwalk** | 임베디드 시그니처+엔트로피 선형 스캔 | v3는 Rust 기반; 공식 Docker/Cargo/source 안내 확인 | `binwalk f.dat`; entropy 옵션은 설치 판본 help 확인 | 임베디드 컨테이너/압축 후보 탐지; 실제 형식 대조 |
| **unblob** ⭐ | handler/추출기로 재귀 분석; 비교 우월성 미확인 | 공식 설치 안내/외부 추출기 의존성 확인; 현 guide에 --install-deps 없음 | `unblob -e out/ f.dat` | 확인된 container의 별도 출력 폴더로 추출; 장비 지원 미확인 |
| **PolyFile** | cleanroom libmagic + Kaitai 계측, 임의 offset 임베디드 탐지, HTML 뷰어 | `pip install polyfile` | `polyfile f.dat --html o.html` | polyglot/오프셋 임베디드 의심 시 |
| **ent** | 엔트로피·χ²·평균·상관 | `apt install ent` | `ent f.dat` | 구간 무작위성 정량화 |

`bre.py triage`는 소스에 내장한 한정된 signature·ASCII 문자열·블록 entropy를 JSON으로 출력한다. libmagic/binwalk의 전체 기능을 대체하지 않는다. Python/NumPy가 필요하며 정상 분석/포착 예외와 help/인자/의존성 오류의 출력 계약은 [scripts README](./scripts/README.md)를 따른다.

표의 file/strings/TrID/PolyFile/ent 설치·옵션·OS 기능은 이 검토에서 개별 설치/일차 문서 대조를 완료하지 않은 참고 후보다. binwalkv3/unblob guide는 아래 공식 출처와 확인일을 따르며 실제 실행은 미확인이다.

**엔트로피 읽기** (bits/byte): 0에 가까운 구간은 같은 byte가 반복됨을 뜻한다. 높은 값은 분포가 고름을 뜻하며 압축/암호의 확정 증거가 아니다. 원래 1~4/4~6/7.5~8.0 분류는 휴리스틱으로만 읽는다. 균등한 비암호화 bytes도8이고 압축 파일도 낮은 값이 가능하다. 블록 크기·길이·혼합 구조에 영향을 받으며 급상승만으로 경계/방식을 확정하지 않는다.

**압축 magic** (선두 바이트, `xxd`로 확인):
- gzip `1f 8b 08` · zlib `78 01`/`78 9c`/`78 da` · **zstd `28 b5 2f fd`** · lz4 frame `04 22 4d 18` · xz `fd 37 7a 58 5a` · bzip2 `42 5a 68`(`BZh`) · zip `50 4b 03 04`.
- zlib header/trailer가 있는 스트림 시험: `python -c "import zlib,sys; print(zlib.decompress(open(sys.argv[1],'rb').read()))" f.dat`, header 없는 raw DEFLATE는 `zlib.decompress(data, wbits=-15)` 또는 `decompressobj(-15)`를 쓴다. 실패만으로 암호화를 확정하지 않으며 wrapper·offset·길이·사전·손상을 대조한다. zstd는 `pip install zstandard` 후 `ZstdDecompressor().decompress(data)`.

---

## 2. Hex editor / 구조 탐색기

| 도구 | 강점 | 라이선스 | 스크립트/CLI | OS |
|---|---|---|---|---|
| **ImHex** ⭐ | RE 전용. C 계열 **Pattern Language**로 struct를 바이트 위에 색칠, 데이터 인스펙터·엔트로피·diff | **무료 GPLv2** | `.hexpat` 패턴, `plcli` 러너, 방대한 커뮤니티 패턴 | Win/mac/Linux |
| **010 Editor** ⭐ | **Binary Template**(`.bt`)의 사실상 표준, 300+ 내장 템플릿, 대용량 강함 | 유료/체험 조건·가격 미확인(원래 $49.99는 현재 근거 없음) | 완전한 C 계열 스크립트 | Win/mac/Linux |
| **HxD** | 빠르고 견고한 무료 데일리 | 무료(closed) | **템플릿·스크립트 없음** | **Windows 전용** |
| **Hexinator/Synalyze It!** | "Grammar" 기반 파싱, insert/delete 감지 diff | 요금/라이선스 미확인(원래 ~$80는 현재 근거 없음) | Lua/Python(유료) | Hexinator: Win/Linux, Synalyze: **macOS** |
| **wxHexEditor** | 멀티 GB/raw device | 무료(GPL) | 스크립트/현재 개발 상태 미확인 | Win/mac/Linux |

**선택 기준:** 구조 패턴 표시가 필요하면 ImHex, 준비된 Binary Template가 필요하면010 Editor를 후보로 검토한다. 표의 현재 라이선스·OS·CLI·템플릿 개수·가격·개발 상태는 공식 자료/설치로 재확인하지 않았다. 최고/활발/피할 도구라는 순위를 근거 없이 확정하지 않는다. 원본은 읽기 전용으로 열고 수정 실습은 별도 사본에 한다.

**ImHex Pattern 초안** (가상 계측 header, Rust가 아님; compiler 미실행):
```text
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
선언적 YAML `.ksy`를 지원 target의 파서로 컴파일하는 참고 기법이다. Python 호출 예는 아래에 둔다. 원래 나열된 target(C++/C#/Go/Java/JS/Lua/Nim/Perl/PHP/Python/Ruby/Rust)의 현재 지원 단계는 설치 compiler 판본에서 확인해야 한다. Web IDE는 시각화 후보이며 실제 파일 업로드/설치/호환은 검증하지 않았다.
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
`pip install construct`는 설치 후보다. Construct2.10 공식 문서는 parse/build 양방향 계약을 설명하지만 전체 파일의 padding/unknown/checksum/NaN bytes 보존은 별도 확인한다. 큰 배열을 NumPy로 분리하는 방법은 최적화 후보이며 속도 향상은 측정하지 않았다. 위 MET1은 header26byte 뒤 float32 배열, ImHex 초안은 offset64, 합성 fixture의 CDS1은 별도 포맷이다. 세 예제를 같은 실제 형식으로 섞지 않는다.

### 그 외
- **hachoir** (`pip install hachoir`; `hachoir-urwid f.dat`) — 미지 스트림을 비트 단위 트리로 **탐색**. construct가 *명세*라면 hachoir는 *탐험*.
- **Rust `binrw`** (`cargo add binrw`) — `#[derive]` 기반 고속 프로덕션 파서.
- **scapy** — 파일이 TLV/record 스트림이면 layer로 기술.
- **Apache Daffodil (DFDL)** — XML 스키마 기반, 보존용 형식 명세 후보. round-trip/float·NaN bit 보존/현 지원 판본은 미확인이다.

**적용 선택지:** 명세/시각화는 Kaitai, Python parse/build는 Construct, Rust 통합은 binrw를 검토한다. hachoir/binrw/scapy/Daffodil과 앞선 ImHex compiler의 현재 옵션·설치·성능은 미확인이다. 일반 Kaitai compile은 reader 생성이며 byte 동일 writer를 자동 보증하지 않는다. [공식 serialization](https://doc.kaitai.io/serialization.html)의 read-write/runtime 조건과 미지 필드 보존을 확인해야 한다.

---

## 4. 자동/알고리즘적 포맷 추론

두 갈래: **(A) 동적 분석**(파일을 읽는 프로그램을 계측해 바이트 소비를 관찰 — 가장 강력하나 실행파일 필요), **(B) corpus 추론**(파일만으로 통계 정렬).

### (A) 동적 taint (파서 실행파일 필요)
Polyglot/AutoFormat/Tupni의 공통 통찰: *프로그램이 어떻게 파싱하느냐가 곧 포맷이다*.
- **Tupni**(MS Research, CCS'08) — record 시퀀스·타입·제약 복원. 랜드마크 논문.
- **PolyTracker**(Trail of Bits) ⭐ — 실용 현대판. LLVM 기반 universal taint. 계측 가능한 프로그램/빌드 조건에서 입력 byte의 전파/문법 추론을 검토하는 연구 후보다. 임의 closed vendor 실행파일을 그대로 분석할 수 있다는 보장은 없으며 현 설치/지원/논문 재현은 미확인이다. 파일 읽기만 하는 분석과 vendor 프로그램 실행을 구분하고 권한/판본을 먼저 확인한다.

### (B) corpus 추론 (파일만 필요) — 현재 쓸 만한 OSS
| 도구 | 방법 | 쓸만? | 비고 |
|---|---|---|---|
| **NETZOB** | 서열 정렬(Needleman-Wunsch)+클러스터링 | 참고 후보·현 설치 미확인 | corpus 필드 추론 범용 출발점 |
| **NEMESYS** | 바이트 델타/비트 congruence 세그먼트 | 연구 후보·재현 미확인 | 순수 binary 메시지에 강함 |
| **FieldHunter** | 통계로 필드 의미(length/counter/addr) 추론 | 연구 후보·재현 미확인 | |
| **BinaryInferno** ⭐ | float/length/timestamp 탐지기 앙상블 투표 | 참고 후보·재현 미확인 | **원하는 것에 가장 근접** — 필드 자동 추측 |

**실용 한계 (중요):** 이들은 *네트워크 프로토콜*(짧은 메시지 다수)용이다. 계측 파일은 *긴 단일 메시지 + 큰 배열*이라 정렬 기반 도구는 배열 본체에서 약하다. → **header/metadata 영역(선두 수백 바이트, corpus)** 에만 BinaryInferno/Netzob를 쓰고, **배열 본체는 §6 numpy 스캔**으로. (`bre.py`가 후자를 구현.)

### 모든 도구를 이기는 기법: 차분 분석
제목은 원래 학습 표현이며 모든 도구보다 우월하다는 측정 결과는 없다. 권한과 원본 보존 아래 변수 하나만 바꾼 파일을 비교한다. `cmp -l a b` / `radiff2 -x a b`는 미실행 호출 참고이며 `bre.py diff`의 실제 계약은 scripts README를 따른다.
- 측정점 +1 → count/length/배열 끝의 후보. 1 증가도 seq 등일 수 있고 파일 크기 차이가 stride가 되는 것은 고정 record·변하지 않는 다른 구간을 확인한 경우다.
- setpoint 변경 → 그 스칼라 위치.
- 같은 데이터 재저장 → timestamp/seq/checksum 후보. compression/padding/정렬/메타데이터 등도 달라질 수 있고 측정 timestamp는 유지될 수도 있다.

---

## 5. 통계/휴리스틱 필드 탐지 (`bre.py`가 구현)

**IEEE-754 float 배열 (계측 payload).** offset+dtype+endianness마다 디코딩해 *그럴듯함* 점수:
- `NaN`/`Inf` 없음, 지수 정상(값 대략 `1e-9…1e9`).
- 장비 물리 범위(nm/µm/%/dB) 안.
- **낮은 kurtosis / 매끄러움**: 현 점수의 가정이다. 불연속 trace/상수/0/극단값도 실제 데이터일 수 있다. 유한값·절댓값 범위·분산 조건 때문에 참값이 제외되거나 다른 bytes가 높은 점수를 받을 수 있다.
- x86 native byte order와 파일 encoding은 별개다. float32/float64·little/big endian은 후보이며 장비별 분포/우세는 미확인이다.

**Timestamp** (디코딩값이 "현재 근처"인 정수 필드 스캔):
- **Unix epoch**(u32 초): 기준일2026-10-04 UTC의 값은 아래 로컬 계산과 대조한다. 값의 epoch·초/ms·시간대·실제 저장 시점을 확인한다.
- **Windows FILETIME**(u64, 1601 기준 100ns): 1601 기준100ns를 가정한 후보. 실제 필드 판본/단위와 범위를 확인한다.
- **OLE date**(f8, 1899-12-30 기준 일수): 1899-12-30 기준 일수를 가정한 후보. 원래45000–46000 범위는 확인일2026-10-04를 포함하지 않으므로 고정 현재 범위로 쓰지 않는다. 실제 사용 빈도는 미확인이다.

**Length-prefix / offset-table:**
- **length prefix**: 블록 직전 u16/u32 값 ≈ 블록 바이트수(또는 원소수×stride). 차분/경계/단위·count 대조로 가설을 검증한다.
- **offset table**: 파일 앞의 단조증가 u32/u64 런, 각 값 `< filesize`, 델타가 record 크기 → 디렉토리/인덱스 후보. 상대 offset·길이·연결 대상도 확인한다.

**Checksum / CRC:**
- **위치**: 보통 record/파일 말미 1/2/4바이트. 변경 bytes는 후보일 뿐이다. additive checksum은 단순 증가/충돌도 가능하고 checksum이 끝에 있다는 보장도 없다. 알고리즘·적용 구간·byte order·여러 독립 샘플을 대조한다.
- **파라미터 식별** ((메시지, checksum) 쌍 여러 개 필요):
  - **CRC RevEng**(`reveng`) — CRC 모델 탐색 후보. 현재 preset 개수·-s/설치·지원 조건은 공식 문서/설치로 검증하지 않았다(원래113개 단정 철회).
  - **CRC Beagle** ⭐ — Python, 차분 기법으로 비표준 XOR-in/out도 복원, 재생성 코드까지 출력.
  - 실패 시 additive/XOR/Fletcher/Adler를 직접 시험.

**Corpus 열(offset) 분산** (같은 포맷 파일 다수일 때 킬러 기법): 파일들을 행으로 쌓아 열별 분산 계산.
- 분산0 = 이 corpus의 해당 offset에서 관측한 동일 byte. magic/version/padding 의미를 확정하지 않는다.
- 낮은 분산 = 제한된 변화; enum/flag/counter는 후보 의미다.
- 큰 분산/entropy 변화 = 조사 후보; 분산과 entropy는 다른 통계이며 timestamp/checksum/경계의 확정 증거가 아니다.
```python
import numpy as np
files = ["a.dat", "b.dat"]  # 같은 판본/정렬을 확인한 두 분석 사본
rows = [np.frombuffer(open(f,'rb').read(), np.uint8) for f in files]
minlen = min(map(len, rows)); m = np.array([r[:minlen] for r in rows])
var = m.var(axis=0)   # var==0 → 고정, 급점프 → 경계 후보
```
→ `bre.py variance`가 이걸 구현.

---

## 6. 실전 스캔 스크립트 (`bre.py`에 내장)

### (a) offset×dtype×endianness 격자로 측정 배열 찾기 — `bre.py arrays`
모든 합리적 시작 offset·자료형을 디코딩해 *그럴듯함* 점수. 점수는 후보 순위이며 정확도 향상/참값 우세는 미측정이다. `--lo/--hi`는0이외 절댓값 범위다. 음수와 양수가 같은 점수를 받을 수 있고 상수/0/NaN/Inf는 누락/제외될 수 있다. 실제 물리 범위를 알아야 하며 범위를 모르면 미확인으로 남긴다. record 안에 interleaved면 `--stride`/`--payload-offset` 사용.

### (b) 자기상관으로 record stride 찾기 — `bre.py stride`
고정 크기 record 스트림은 record 크기에서 *주기적*. 자기상관 피크는 stride/그 배수·약수 후보이지 구조 의미의 확정이 아니다. `bre.py`는 FFT 자기상관 + **stride로 접었을 때 분산 0인 열(고정 필드) 비율**로 검증하고, 작은 후보를 우선하는 휴리스틱을 쓴다. 실제 최소 record 주기를 보장하지 않으며 diff·count·경계·값/단위를 대조한다. stride 출력에는 coverage 필드가 없다.

### (c) endianness/폭 한눈에
```python
import struct
data = bytes(0x40) + struct.pack("<d", 1.0)  # 합성8byte; 실제 파일은 별도 읽기
w = data[0x40:0x48]
for fmt in ['<I','>I','<Q','>Q','<f','<d']:
    try: print(fmt, struct.unpack_from(fmt, w))
    except struct.error as exc: print(fmt, "decode error:", exc)
```

---

## 계측 `.dat` 빠른 시작 체크리스트

1. `bre.py triage f.dat` — magic·version·장비 ID·단위.
2. 압축이면 `unblob -e out/ f.dat`로 해제 후 재triage.
3. 측정점 수만 다른 파일 ≥5개 → `bre.py variance` (header/payload 경계·고정 필드).
4. 측정점 +1 두 파일 → `bre.py diff` (count·stride).
5. `bre.py stride` + `bre.py arrays --stride` → 측정값 offset·dtype·endianness. `element_count == 측정점 수` 확인.
6. **Kaitai `.ksy`** 로 적고 corpus 전체 파싱/의미를 대조하고 별도 writer가 있을 때 unknown/padding/checksum까지 byte 동일 round-trip을 검사한다.
7. 말미의 불규칙 변경 바이트 → **CRC RevEng/Beagle**로 checksum 후보를 검증한다. 불규칙 변화만으로 확정하지 않는다. 실제 파일 재작성은 별도 권한·보존 계약이 필요하다.

## 참고 자료 (References)

- unblob.org · imhex.org · kaitai.io · construct.readthedocs.io · github.com/jam1garner/binrw
- github.com/binaryinferno/binaryinferno · github.com/netzob/netzob · github.com/vs-uulm/nemesys
- reveng.sourceforge.io · github.com/colinoflynn/crcbeagle · github.com/trailofbits/polyfile · github.com/trailofbits/polytracker
- Tupni (Microsoft Research, CCS'08) · Trail of Bits "Two new tools that tame the treachery of files"
- 큐레이션: github.com/techge/PRE-list · github.com/extremecoders-re/re-list


## 확인 근거와 검토 결과

확인일 2026-10-04 00:00 UTC를 같은 epoch 정의로 로컬 계산하면 Unix 초1791072000, FILETIME100ns134355456000000000, OLE 일수46299다. 이는 실제 파일이 해당 epoch/시간대를 사용한다는 증거가 아니다.

2026-10-04 확인한 일차 자료: [binwalk v3](https://github.com/ReFirmLabs/binwalk), [unblob guide](https://unblob.org/guide/), [Python3.12 zlib](https://docs.python.org/3.12/library/zlib.html)·[struct](https://docs.python.org/3.12/library/struct.html), [Construct2.10](https://construct.readthedocs.io/en/latest/intro.html), [Kaitai serialization](https://doc.kaitai.io/serialization.html), [현 CLI 구현](./scripts/bre.py). unblob의 -e/--extract-dir는 공식 guide에서 유효하며 외부 추출기와 별도 출력 경로가 필요하다. version 없는 mutable 페이지는 확인일 상태이며 최신/설치 보증이 아니다.

원래 모든 절·도구/연구 후보·MET1/통계/struct 예제 맥락과 작성일을 보존했다. 성능/가격/개발 상태·연구 도구/CRC 모델 복원·IDE/compiler의 현 지원은 개별 미확인으로 남겼다. 로컬 Python3.12.12/NumPy1.26.4의 합성/반례 검사는 실제 장비 파일·외부 도구 설치/GUI·parser 생성의 증거가 아니다. HERDR_ENV=1/pane_not_found로 Claude 의견을 받지 못해 완전 중복 통합과 실제 장비 계약 선택은 보류했다. [정리 기록](../organization-log.md)에 검증 범위를 남긴다.
