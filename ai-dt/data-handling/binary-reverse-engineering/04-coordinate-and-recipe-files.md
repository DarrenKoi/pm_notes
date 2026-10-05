---
tags: [binary, reverse-engineering, recipe, coordinate, tlv, cd-sem]
level: intermediate
last_updated: 2026-07-10
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "바이너리 역공학"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 좌표 파일 & Recipe 파일 역공학

> SEM 이미지·측정값 배열과 달리, **좌표 파일**과 **recipe 파일**은 구조가 다르다. 좌표가 고정 record 배열이면 기존 도구를 적용할 수 있고, recipe를 포함한 가변 길이·계층·직렬화 구조에는 다른 detector를 검토한다. 실제 형식은 장비/소프트웨어 판본과 샘플로 확인해야 한다. 이 문서는 그 차이와 대응 도구를 정리한다.

## 왜 따로 다루나 (Why)

앞의 `arrays`/`stride`/`diff`는 **고정 크기 record의 연속 배열**을 가정한다. 실제 측정값·좌표가 이 가정에 맞는지는 미확인이다. recipe를 조사할 때 검토할 구조 후보는 다음과 같다.

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
활용 맥락: 웨이퍼 측정 사이트, 다이 좌표, 정렬 마크. 위 도식은 합성 구조 예이며 f8/f4와 µm/nm는 후보일 뿐이다. 원점·축 방향·회전/반사·좌표계·단위·site 순서를 실제 export와 대조한다. **고정 stride와 경계를 확인한 파일에 기존 도구를 적용**한다.

### Recipe 파일 (계층·가변)
```
[magic][version][n_sections]
[offset table: n_sections × u32]          <- offsets 로 탐지
[section: TLV | TLV | ...]                <- tlv 로 탐지
   각 TLV = [tag][length][value]
   value 안에 파라미터명(문자열), setpoint(f8), nested step 배열...
[string table]                            <- strtab 로 탐지
```
또는 통째로 내부 XML/직렬화 객체 — **serial**로 먼저 후보를 찾는다. 결과는 실제 표준 파싱 성공과 다르다.

## 어떻게 하는가 (How)

### 좌표 파일 — 기존 파이프라인 그대로
```bash
# 1) 정체 + record 주기
python3 bre.py triage  coords.dat
python3 bre.py stride  coords.dat --offset "$HEADER_SIZE"

# 2) x/y 필드 탐지 (물리 범위를 웨이퍼 좌표로: um면 대략 ±150000)
python3 bre.py arrays  coords.dat --stride "$BEST_STRIDE" --payload-offset "$HEADER_SIZE" \
        --lo 1 --hi 200000 --dtypes '<f8,<f4' --top 8
```
`HEADER_SIZE`·`BEST_STRIDE`는 확인한 정수로 먼저 정의한다. 호출은 `scripts`가 현재 위치라고 가정한다. `--lo/--hi`는 0 이외 값의 절댓값 범위다. 0은 범위 계산에서 제외되고 음수는 양수와 같은 절댓값으로 평가되며, 유효한 상수 좌표도 점수에서 제외될 수 있다.

- 좌표는 **두 컬럼의 대응 관계**를 확인한다. site_id가 첫 후보이거나 x·y가 반드시 top8 안에 있다는 보장은 없다. 후보가 없으면 미확인이지 좌표 없음의 증거가 아니다.
- `element_count`·`n_sites`·범위 일치는 필요 대조 항목이며 단독 확정 조건이 아니다. 모든 대상 값과 단위·경계·판본·UI/export 대응을 확인한다. 위 ±150000µm는 가정한 스케일 예이지 공통 허용 범위가 아니다.
- **격자 검증**: 규칙적인 격자를 가정한 샘플에서는 열/행과 좌표 간격을 대조한다. 불규칙 측정 사이트도 가능하므로 주기 유무로 좌표의 진위를 확정하지 않는다.

### Recipe 파일 — 전용 detector

**1) 먼저 텍스트/직렬화인지 확인 (`serial`)**
```bash
python3 bre.py serial recipe.dat
```
- `mostly_text: true`는 반올림된 ASCII printable_ratio>0.85 조건이다. INI/XML/CSV 파싱 성공을 뜻하지 않고, UTF-16 등 텍스트도 false가 될 수 있다. 인코딩과 문법·필요 필드·단위를 별도로 검증한다.
- `signature_hits`의 `xml`/`zip/ooxml`/`ms-compound-file(ole/mfc-doc)`/`dotnet-binaryformatter`는 byte 패턴 후보다. lxml/unzip/olefile 같은 구조 파서로 경계를 확인하고 필요한 값까지 대조한 뒤 종료한다. .NET 후보에 `BinaryFormatter.Deserialize`를 실행하지 않는다. Microsoft 공식 안내는 이를 안전한 데이터 처리 수단으로 보지 않으며 .NET9부터 기본 구현은 사용 시 예외를 낸다.

**2) 디렉토리 구조 확인 (`offsets`)**
```bash
python3 bre.py offsets recipe.dat
```
- `count`가 크고 `values`가 header 이후를 고르게 가리키면 포인터 테이블 후보다. entry 수·기준(절대/상대 offset)·대상 구간의 길이와 의미·파일 경계를 교차확인한다. count 일치만으로 확정하지 않는다.

**3) TLV 체인 파싱 (`tlv`)**
```bash
python3 bre.py tlv recipe.dat --start "$SECTION_START"
```
`SECTION_START`는 경계가 확인된 정수다. `best=None`이면 미확인으로 남긴다.

- `best.coverage≈1.0` + `lands_at_eof:true`는 `(tag_size, len_size, endian, len_includes_header)` 후보 점수다. `lands_at_eof`는 `--tail` 이내 잔여 bytes도 허용한다. 실제 `ends_at`·`size`·`bytes_consumed`와 선언 길이·tag 의미·다른 샘플/판본을 대조한 뒤 확정한다.
- 반복 tag의 byte 위치와 변경된 값을 실제 UI/export로 대조해 tag 사전을 만든다. 1=name/2=setpoint/3=step은 합성 예이며 반복만으로 의미가 확정되지 않는다.
- nested TLV는 부모 value의 시작/끝을 먼저 검증한다. 현 CLI는 `--start` 이후 파일 EOF까지 탐색하고 부모 끝을 제한하는 옵션이 없다. 부모 value bytes만 저장소 밖 별도 분석 사본으로 추출해 start0으로 조사하고 절대 offset 변환을 기록한다. 전체 파일에서 start만 바꾸면 형제/후행 구간을 부모에 포함할 수 있다.

**4) 파라미터명 추출 (`strtab`)**
```bash
python3 bre.py strtab recipe.dat
```
- `null_terminated` / `length_prefixed` / `utf16le` 세 형태의 문자열 후보를 수집한다. recipe 파라미터명·스텝명인지는 실제 의미 대조가 필요하며 결과가 모든 문자열/인코딩을 포괄하지 않는다.
- `length_prefixed`는 오탐이 있으나 `declared_len`이 맞는 항목이 촘촘히 이어지면 문자열 테이블 후보다. 인코딩·선언 길이/종단·판본·문자열 의미도 확인하고 시작 offset을 `offsets`/`tlv` 결과와 대조.

### 권장 순서 (recipe)
```
serial ──text/직렬화 후보──► 표준 파싱 + 필요 값 대조
   │ 아니오
   ▼
offsets ──디렉토리 후보──► 각 구간 시작/끝 확인 후 TLV 조사
   │
   ▼
tlv ──TLV 후보 검증──► 실제 값에 근거한 tag 사전
   │
   ▼
strtab ──파라미터명 매핑──► FORMAT.md 에 필드+tag+문자열 정리
```

## 커버리지 요약

| 파일 유형 | 구조 | 도구 | 확인 범위 |
|---|---|---|---|
| SEM 이미지 | TIFF + private tag | `tifffile`(지원 판본/필드 대조 필요) | 일부 tag 구현, 실제 파일 미확인 [02-cd-sem-formats](./02-cd-sem-formats.md) |
| 측정값 | 고정 record 배열 | `arrays`/`stride`/`diff` | 합성 검사 대상, 실제 파일 미확인 |
| **좌표** | 고정 record `(x,y[,z])` 배열 | `arrays --stride`/`stride`/`diff` | 합성 검사 대상, 실제 파일 미확인 |
| **recipe(TLV)** | 가변 tag-length-value | `tlv`+`offsets`+`strtab` | 합성 검사 대상, 실제 파일 미확인 |
| **recipe(텍스트/XML)** | 내부 XML/INI/직렬화 | `serial`→표준 파서 | 합성 검사 대상, 실제 파일 미확인 |
| recipe(MFC/독점 직렬화) | 클래스 스키마 직렬화 | `serial`로 후보 탐지 후 경계/스키마 대조 | ⚠️ 부분 — 벤더 SW가 MFC/.NET인지 확인 필요 |

**한계 명시:** OLE compound-file signature는 MFC 사용의 증거가 아니며 signature가 없는 것도 MFC/.NET/Qt의 증거가 아니다. `olefile` 파싱은 컨테이너 구조 확인이며 내부 업무 스키마를 복원하지 않는다. Microsoft의 CArchive는 클래스별 Serialize와 schema 판본에 의존한다. 실제 벤더 stack·형식·권한은 미확인으로 남긴다. 이 경우 [03-legal-and-first-moves](./03-legal-and-first-moves.md)의 "벤더에 먼저 요청"이 특히 유효하다.

## 관련 문서

- [00-agent-runbook.md](./00-agent-runbook.md) — 전체 phase 파이프라인
- [01-toolkit-reference.md](./01-toolkit-reference.md) — 범용 RE 도구·기법
- [02-cd-sem-formats.md](./02-cd-sem-formats.md) — 벤더 포맷·기존 reader


## 근거와 검토 결과

2026-10-04 [CLI 구현](./scripts/bre.py)·[합성 검사](./scripts/selftest.py), [Microsoft CArchive/msvc-170](https://learn.microsoft.com/en-us/cpp/mfc/reference/carchive-class?view=msvc-170), [BinaryFormatter 지침](https://learn.microsoft.com/en-us/dotnet/standard/serialization/binaryformatter-security-guide)을 확인했다. CArchive 객체/schema 계약과 .NET9 기본 BinaryFormatter 제한을 실제 장비의 stack으로 추정하지 않는다. Python3.12.12/NumPy1.26.4의 합성 검사는 후보 산출/반례 확인이며 실제 좌표계·recipe·벤더 parser 지원 검증이 아니다. 원래 모든 절·도식·좌표/TLV/offset/string/직렬화 맥락을 보존하고 자동 확정/커버 보장과 경계 없는 재귀 설명을 정정했다. HERDR_ENV=1/pane_not_found로 Claude 의견이 없어 중복 통합·실제 포맷 계약은 보류했다. [정리 기록](../organization-log.md)에 검증 범위와 미확인을 남긴다.
