---
tags: [binary, reverse-engineering, runbook, agent, cd-sem]
level: intermediate
last_updated: 2026-07-10
---

# Agent Runbook — Binary 파일 구조 복원 파이프라인

> 다른 agent가 그대로 실행할 수 있는 phase 파이프라인. 각 phase는 **입력 · 명령 · 합격기준(acceptance) · 산출 JSON(handoff)** 을 명시한다. 한 phase의 산출 JSON이 다음 phase의 입력이 된다.

## 실행 원칙

- **모든 `bre.py` 명령은 stdout에 JSON을 낸다.** 그 JSON을 파싱해 다음 phase에 넘긴다. 사람이 hex를 다시 읽을 필요가 없도록 설계됐다.
- **각 phase는 합격기준을 통과해야 다음으로 넘어간다.** 통과 못 하면 해당 phase의 "실패 시" 지침을 따른다.
- **파괴적 동작 없음.** 이 파이프라인은 읽기 전용이다. 원본 파일을 수정하지 않는다.
- 시작 전 `python3 scripts/selftest.py`로 환경을 검증한다 (exit 0 = 정상).

## 산출물 규약 (Artifacts)

각 phase는 작업 디렉토리에 JSON을 저장한다. 권장 파일명:

```
work/
  01_triage.json       # phase 1
  04_variance.json     # phase 4a
  04_diff_points.json  # phase 4b (측정점 +1)
  04_diff_resave.json  # phase 4c (재저장)
  05_stride.json       # phase 5a
  05_arrays.json       # phase 5b
  05_stamps.json       # phase 5c
  FORMAT.md            # 최종 산출: 복원한 포맷 스펙 + Kaitai .ksy
```

---

## Phase 0 — 법적/실무 게이트 (사람 확인 필요)

**입력:** 대상 장비/파일 종류.
**행동:** [03-legal-and-first-moves.md](./03-legal-and-first-moves.md) 확인. 벤더 apps 엔지니어에게 (a) 포맷 스펙, (b) CSV/XML export, (c) SEMI EDA/Interface A 피드가 있는지 먼저 묻는다.
**합격기준:** "자사 데이터를 자사가 파싱"하는 범위이고 NDA/PO에 저촉 없음이 확인됨. 이미지 파일이면 Phase 2로 우회 가능.
**실패 시:** 역공학 대신 벤더 export/EDA 경로로 데이터를 얻는다. 여기서 종료.

---

## Phase 1 — Triage (정체 파악)

**입력:** 대상 파일 1개 (`FILE`).
**명령:**
```bash
python3 scripts/bre.py triage "$FILE" > work/01_triage.json
```
**읽을 필드:**
- `magic_at_offset_0`, `head_ascii` — 알려진 컨테이너인가?
- `entropy_verdict` — `compressed_or_encrypted`면 먼저 해제해야 한다.
- `embedded_signatures` — 파일 중간에 TIFF/zlib 등이 박혀 있나?
- `ascii_strings`, `utf16le_strings` — recipe명, 장비 ID, 단위, 컬럼명 노출.

**합격기준:**
- `entropy_verdict != compressed_or_encrypted` (구조 분석 가능), **또는** 압축이면 해제 후 이 phase 재실행.
- magic이 알려진 포맷(TIFF/HDF5/zip 등)이면 → **Phase 2로 분기** (기존 reader 사용).

**실패 시(압축):** `unblob -e out/ "$FILE"` 또는 Python `zlib`/`zstandard`로 해제 → 해제물로 Phase 1 재실행. 상세는 [01-toolkit-reference.md](./01-toolkit-reference.md) §1.

**Handoff:** `01_triage.json`. 특히 `entropy_verdict`, magic 여부, 눈에 띄는 string의 offset.

---

## Phase 2 — 알려진 포맷 우회 (하드코딩 전 필수 확인)

**입력:** Phase 1의 magic/확장자.
**행동:** 손으로 파서 짜기 전에 기존 reader로 이미 읽히는지 확인. [02-cd-sem-formats.md](./02-cd-sem-formats.md) 참조.
- **이미지(TIFF)면:**
  ```python
  import tifffile
  with tifffile.TiffFile(FILE) as t:
      print(t.is_fei, t.is_sem)          # 벤더 자동감지
      if t.is_fei: print(t.fei_metadata) # Thermo/FEI
      if t.is_sem: print(t.sem_metadata) # Zeiss CZ_SEM
      for p in t.pages:
          for tag in p.tags: print(tag.code, tag.name, repr(tag.value)[:80])
  ```
- **HDF5면** `h5py`, **SER/DM3/EMD면** `ncempy`, **STDF면** `pystdf`.

**합격기준:** 기존 라이브러리가 메타데이터·픽셀·측정값을 읽어주면 → **여기서 종료** (역공학 불필요). 못 읽으면 Phase 3로.

**Handoff:** 어떤 reader가 무엇까지 읽었는지 메모.

---

## Phase 3 — Corpus 확보 (차분 분석 준비)

**입력:** 장비 접근 권한 (파일을 새로 생성/저장할 수 있어야 함).
**행동:** **변수 하나만 바꾼** 파일 세트를 만든다. 이것이 이후 모든 정확도의 근거다.
- `base.dat` — 기준.
- `one_more_point.dat` — 측정점 **1개만 추가**. (count/stride를 드러냄)
- `resaved.dat` — **같은 recipe 재저장**. (timestamp/checksum만 바뀜)
- `corpus_*.dat` — 같은 구조, 다른 데이터 4개 이상. (offset별 분산용)

**합격기준:** 최소 2개(diff용), 권장 6개 이상(variance용). 각 변경이 "무엇을 바꿨는지" 기록됨.
**실패 시:** 장비에서 새 파일을 못 만들면, 기존 파일 중 구조가 같아 보이는 것들로 variance만 수행.

**Handoff:** 파일 목록 + 각 파일의 변경 내역.

---

## Phase 4 — 차분/분산 분석 (구조 골격 추출)

### 4a. Corpus 분산 → 고정 필드 + header/payload 경계
```bash
python3 scripts/bre.py variance work/corpus_*.dat > work/04_variance.json
```
- `constant_runs` = 모든 파일 동일 → magic/version/고정 struct 필드/padding.
- `weak_boundary_hint` = **약한 추정**. 이것만 믿지 말 것 (`weak_boundary_caveat` 참고). 확정은 `constant_runs` + diff로.

### 4b. 측정점 +1 diff → count·length·stride
```bash
python3 scripts/bre.py diff work/base.dat work/one_more_point.dat > work/04_diff_points.json
```
- `size_delta` = **record stride** (측정점 1개 늘어난 만큼 커진 바이트 수).
- `field_candidates` 중 `<I` delta==1 필드 = **count field**.

### 4c. 재저장 diff → timestamp·checksum 격리
```bash
python3 scripts/bre.py diff work/base.dat work/resaved.dat > work/04_diff_resave.json
```
- 여기서 바뀐 offset만이 timestamp/sequence/checksum. 보통 header의 시간 필드 + 말미 checksum 둘.

**합격기준:**
- stride(=`size_delta`) 확정.
- count field offset 확정 (delta==1).
- header 크기 추정 확정 (`constant_runs`의 마지막 + diff 교차검증).

**실패 시:** corpus가 부족하면 3~4개만으로도 variance는 유효. diff는 최소 2파일이면 됨.

**Handoff:** `{stride, count_offset, header_size, timestamp_offset, checksum_offset}`.

---

## Phase 5 — 배열/필드 탐지 (측정값 확정)

### 5a. Record stride 확정 (diff 교차검증)
```bash
python3 scripts/bre.py stride work/base.dat --offset <header_size> --max-stride 4096 > work/05_stride.json
```
- `best_stride` = 가장 작은 fundamental. `harmonics_of_best`는 그 배수(가짜).
- **Phase 4b의 `size_delta`와 반드시 일치해야 함.** 불일치 시 header_size 재검토.

### 5b. 측정값 필드 탐지 (interleaved)
```bash
python3 scripts/bre.py arrays work/base.dat \
    --stride <best_stride> --payload-offset <header_size> \
    --lo <물리최소> --hi <물리최대> --top 8 > work/05_arrays.json
```
- `--lo/--hi`를 **실제 계측 물리 범위로 좁히는 것이 정확도에 가장 큰 영향** (예: CD nm면 `--lo 1 --hi 1000`).
- 1위 후보의 `field_offset_in_record` + `dtype` = 측정값 필드.
- `element_count`가 tool UI의 측정점 수와 일치하면 확정.
- (측정값이 record가 아니라 연속 배열이면 `--stride` 없이 실행.)

### 5c. Timestamp 검증
```bash
python3 scripts/bre.py stamps work/base.dat --max-bytes 8192 > work/05_stamps.json
```
- **오탐 다수.** 4c diff에서 바뀐 offset과 교차하는 후보만 진짜.

**합격기준:**
- 측정값 필드의 (offset, dtype, endianness) 확정.
- `element_count == 측정점 수`.
- stride가 4b와 일치.

**Handoff:** `{payload_offset, stride, field_layout:[{offset,dtype,name}], value_field}`.

### 5d. 파일이 고정 record 배열이 아닐 때 (recipe/좌표)
Phase 5a~5c는 **고정 크기 record 배열**(측정값·좌표)을 가정한다. `stride`의 coverage가 낮거나 `arrays`가 전부 쓰레기면 **가변 구조(recipe)** 일 수 있다. → **좌표/recipe 전용 경로는 [04-coordinate-and-recipe-files.md](./04-coordinate-and-recipe-files.md)**.
```bash
python3 scripts/bre.py serial  work/target.dat            # 텍스트/XML/직렬화인지 먼저
python3 scripts/bre.py offsets work/target.dat            # 포인터 테이블(디렉토리)
python3 scripts/bre.py tlv     work/target.dat --start <header_size>   # 가변 TLV 체인
python3 scripts/bre.py strtab  work/target.dat            # 파라미터명 문자열
```
- 좌표 파일은 측정값과 동형이라 5a~5b 그대로 (단 `--lo/--hi`를 웨이퍼 좌표 스케일로).
- recipe는 `serial`→`offsets`→`tlv`→`strtab` 순. `serial`이 mostly_text/xml이면 표준 파서로 종료.

---

## Phase 6 — 파서 형식화 & 검증

**입력:** Phase 4·5의 확정 값.
**행동:** Kaitai `.ksy` 또는 Python `construct`로 스펙을 적고 **corpus 전체에 대해 round-trip 검증**. 문법·예시는 [01-toolkit-reference.md](./01-toolkit-reference.md) §형식화.

**합격기준:**
- 스펙이 corpus의 모든 파일을 에러 없이 파싱.
- 파싱한 측정값이 tool UI/CSV export 값과 일치 (grand-truth 대조).

**Handoff (최종):** `work/FORMAT.md` — 표로 정리한 필드 맵 + `.ksy` 스펙 + 검증 로그.

---

## 실패·중단 규칙 (rabbit-hole 방지)

- 한 phase에서 2~3회 시도해도 합격기준을 못 넘으면 **중단하고 사람에게 보고**한다. 무한 재시도 금지.
- 전체 파일이 고엔트로피(암호화 의심)면 역공학 불가 → 벤더 경로로 전환.
- `bre.py`가 `{"error": ...}` JSON을 내면 그 메시지를 그대로 보고한다.

## 참고 자료

- [01-toolkit-reference.md](./01-toolkit-reference.md), [02-cd-sem-formats.md](./02-cd-sem-formats.md)
- [agent-tasks.md](./agent-tasks.md) — 이 runbook의 각 phase를 subagent에 위임하는 프롬프트.
