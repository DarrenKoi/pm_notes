---
tags: [binary, reverse-engineering, recipe, coordinate, tlv, cd-sem]
level: intermediate
last_updated: 2026-07-10
---

# 좌표 파일 & Recipe 파일 역공학

> SEM 이미지·측정값 배열과 달리, **좌표 파일**과 **recipe 파일**은 구조가 다르다. 좌표는 측정값과 같은 고정 record 배열이라 기존 도구로 커버되지만, recipe는 가변 길이·계층·직렬화 구조라 전용 detector가 필요하다. 이 문서는 그 차이와 대응 도구를 정리한다.

## 왜 따로 다루나 (Why)

앞의 `arrays`/`stride`/`diff`는 **고정 크기 record의 연속 배열**을 가정한다. 측정값·좌표는 이 가정에 맞지만, recipe는 대개 다음 중 하나다.

- **가변 길이 TLV**(tag-length-value) — 스텝마다 파라미터 개수가 다름
- **offset/pointer 테이블**(디렉토리) — header가 파일 내 섹션 위치를 가리킴
- **문자열 테이블** — 파라미터명·스텝명이 길이 접두 또는 null-종단 문자열
- **직렬화된 객체** — 내부 XML/JSON, MFC CArchive, .NET BinaryFormatter, ASN.1

그래서 `bre.py`에 recipe 전용 detector 4종을 추가했다: `tlv`, `offsets`, `strtab`, `serial`.

## 핵심 개념 (What) — 두 파일 유형의 구조

### 좌표 파일 (측정값과 동형)
```
[magic][u32 n_sites][ n_sites × record{ site_id; x; y (; z) } ]
```
전형: 웨이퍼 측정 사이트, 다이 좌표, 정렬 마크. `(x,y)`가 f8 또는 f4, um/nm 단위. **고정 stride record라 기존 도구 그대로 적용**.

### Recipe 파일 (계층·가변)
```
[magic][version][n_sections]
[offset table: n_sections × u32]          <- offsets 로 탐지
[section: TLV | TLV | ...]                <- tlv 로 탐지
   각 TLV = [tag][length][value]
   value 안에 파라미터명(문자열), setpoint(f8), nested step 배열...
[string table]                            <- strtab 로 탐지
```
또는 통째로 내부 XML/직렬화 객체 — **serial**로 먼저 확인.

## 어떻게 하는가 (How)

### 좌표 파일 — 기존 파이프라인 그대로
```bash
# 1) 정체 + record 주기
python3 bre.py triage  coords.dat
python3 bre.py stride  coords.dat --offset <header_size>

# 2) x/y 필드 탐지 (물리 범위를 웨이퍼 좌표로: um면 대략 ±150000)
python3 bre.py arrays  coords.dat --stride <N> --payload-offset <header_size> \
        --lo 1 --hi 200000 --dtypes '<f8,<f4' --top 8
```
- 좌표는 **두 컬럼 모두 의미**가 있다. site_id 카운터가 최상위 점수를 받는 게 정상이고, x·y f8 필드는 그 아래 top 후보로 나온다.
- `element_count`가 `n_sites`와 일치하고, x·y의 mean/min/max가 웨이퍼 스케일(수 mm~수십 mm)이면 확정.
- **격자 검증**: x는 열 인덱스, y는 행 인덱스에 비례하므로 값이 일정 간격으로 반복된다 — 진짜 좌표의 강한 신호.

### Recipe 파일 — 전용 detector

**1) 먼저 텍스트/직렬화인지 확인 (`serial`)**
```bash
python3 bre.py serial recipe.dat
```
- `mostly_text: true`(printable_ratio>0.85) → 사실상 INI/XML/CSV. 역공학 말고 텍스트 파서.
- `signature_hits`에 `xml`/`zip`/`ms-compound-file`/`dotnet` → 그 구간을 표준 도구로(lxml/unzip/olefile). **여기서 끝날 수 있다.**

**2) 디렉토리 구조 확인 (`offsets`)**
```bash
python3 bre.py offsets recipe.dat
```
- `count`가 크고 `values`가 header 이후를 고르게 가리키면 포인터 테이블. entry 수가 header의 count 필드와 일치하는지 `diff`로 교차확인.

**3) TLV 체인 파싱 (`tlv`)**
```bash
python3 bre.py tlv recipe.dat --start <header_size 또는 offset-table 첫 entry>
```
- `best.coverage≈1.0` + `lands_at_eof:true` → 그 `(tag_size, len_size, endian, len_includes_header)`가 진짜 TLV 레이아웃.
- `first_records`의 `tag` 값이 반복되면 tag 사전을 만든다(예: 1=name, 2=setpoint, 3=step).
- record 안이 또 TLV(nested)면 각 value의 offset을 `--start`로 재귀 실행.

**4) 파라미터명 추출 (`strtab`)**
```bash
python3 bre.py strtab recipe.dat
```
- `null_terminated` / `length_prefixed` / `utf16le` 세 형태로 문자열 수집. recipe 파라미터명·스텝명이 여기 있다.
- `length_prefixed`는 오탐이 있으나 `declared_len`이 맞는 항목이 촘촘히 이어지면 진짜 문자열 테이블. 그 시작 offset을 `offsets`/`tlv` 결과와 대조.

### 권장 순서 (recipe)
```
serial ──text/직렬화?──► 표준 파서로 종료
   │ 아니오
   ▼
offsets ──디렉토리?──► 각 entry를 tlv --start 로
   │
   ▼
tlv ──TLV 체인 확정──► tag 사전 작성
   │
   ▼
strtab ──파라미터명 매핑──► FORMAT.md 에 필드+tag+문자열 정리
```

## 커버리지 요약

| 파일 유형 | 구조 | 도구 | 커버 |
|---|---|---|---|
| SEM 이미지 | TIFF + private tag | `tifffile`(역공학 불필요) | ✅ [02-cd-sem-formats](./02-cd-sem-formats.md) |
| 측정값 | 고정 record 배열 | `arrays`/`stride`/`diff` | ✅ |
| **좌표** | 고정 record `(x,y[,z])` 배열 | `arrays --stride`/`stride`/`diff` | ✅ |
| **recipe(TLV)** | 가변 tag-length-value | `tlv`+`offsets`+`strtab` | ✅ |
| **recipe(텍스트/XML)** | 내부 XML/INI/직렬화 | `serial`→표준 파서 | ✅ |
| recipe(MFC/독점 직렬화) | 클래스 스키마 직렬화 | `serial`로 식별 후 스키마 역공학 | ⚠️ 부분 — 벤더 SW가 MFC/.NET인지 확인 필요 |

**한계 명시:** MFC `CArchive`·독점 직렬화는 시그니처가 약해 자동 파싱이 어렵다. `serial`이 `ms-compound-file`을 잡으면 `olefile`로, 아니면 벤더 SW 스택(MFC/.NET/Qt)을 확인해 스키마를 역공학해야 한다. 이 경우 [03-legal-and-first-moves](./03-legal-and-first-moves.md)의 "벤더에 먼저 요청"이 특히 유효하다.

## 관련 문서

- [00-agent-runbook.md](./00-agent-runbook.md) — 전체 phase 파이프라인
- [01-toolkit-reference.md](./01-toolkit-reference.md) — 범용 RE 도구·기법
- [02-cd-sem-formats.md](./02-cd-sem-formats.md) — 벤더 포맷·기존 reader
