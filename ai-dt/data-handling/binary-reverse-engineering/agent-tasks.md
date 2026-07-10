---
tags: [agent, subagent, task-spec, reverse-engineering]
level: intermediate
last_updated: 2026-07-10
---

# Agent Task 프롬프트 모음

> 각 블록을 복사해 subagent에 그대로 던진다. `<...>` 부분만 실제 값으로 치환. 각 task는 **읽기 전용**이고 **JSON/파일 산출물**로 끝나므로 결과를 다음 task 입력으로 넘길 수 있다. 파이프라인 전체 설명은 [00-agent-runbook.md](./00-agent-runbook.md).

## 사용 전 공통 전제

- toolkit 경로: `ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py`
- 시작 전 `python3 scripts/selftest.py` (exit 0 확인).
- 모든 `bre.py` 출력은 JSON. 작업 산출물은 `work/`에 저장.

---

## Task A — 단일 파일 정체 파악 (Phase 1)

```
목표: <FILE> 의 정체를 파악하고 다음 단계를 정하라.
제약: 읽기 전용. 파일을 수정하지 마라.
실행:
  python3 <REPO>/ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py triage "<FILE>" > work/01_triage.json
판정:
  - magic_at_offset_0 가 TIFF/HDF5/zip 등 알려진 포맷이면 "Task B(기존 reader)로 우회" 권고.
  - entropy_verdict == compressed_or_encrypted 이면 "먼저 압축 해제 필요" 보고(unblob 또는 zlib/zstd).
  - ascii_strings/utf16le_strings 에서 recipe명·장비ID·단위·컬럼명과 그 offset을 요약.
반환(JSON): { verdict, magic, entropy, notable_strings:[{offset,text}], recommended_next }
```

---

## Task B — 기존 reader로 우회 시도 (Phase 2)

```
목표: <FILE> 가 이미 오픈소스 reader로 읽히는지 확인. 읽히면 역공학 불필요.
참고: ai-dt/data-handling/binary-reverse-engineering/02-cd-sem-formats.md §2·§3
실행(이미지/TIFF 의심 시):
  python3 - <<'PY'
  import tifffile, json
  f = "<FILE>"
  try:
      with tifffile.TiffFile(f) as t:
          out = {"is_fei": t.is_fei, "is_sem": t.is_sem,
                 "fei": t.fei_metadata if t.is_fei else None,
                 "sem": {k:str(v) for k,v in (t.sem_metadata or {}).items()} if t.is_sem else None,
                 "tags": [(tg.code, tg.name, repr(tg.value)[:80]) for p in t.pages for tg in p.tags][:60]}
      print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
  except Exception as e:
      print(json.dumps({"error": str(e)}))
  PY
  # HDF5면 h5py, SER/DM3/EMD면 ncempy, STDF면 pystdf 로 유사 시도.
판정: reader가 메타/픽셀/측정값을 읽으면 "종료 — 역공학 불필요". 아니면 "Task C~E 진행".
반환(JSON): { reader_tried, readable:bool, extracted_summary, recommend }
```

---

## Task C — Corpus 확보 지시 (Phase 3, 사람에게)

```
목표: 차분 분석용 파일 세트를 만들도록 사용자에게 정확한 지시를 작성하라.
요구 파일:
  - base.dat        : 기준
  - one_more_point.dat : base 에서 측정점 1개만 추가
  - resaved.dat     : base 와 같은 recipe 재저장(데이터 동일, 저장만 다시)
  - corpus_0..3.dat : 같은 구조·다른 데이터 4개 이상
각 파일에 "무엇을 바꿨는지" 한 줄 기록 요청.
반환: 사용자에게 보낼 체크리스트 + 확보 후 Task D로 진행 안내.
```

---

## Task D — 차분/분산 분석 (Phase 4)

```
목표: corpus 로 파일의 구조 골격(고정 필드·count·stride·timestamp·checksum)을 추출.
실행:
  BRE=<REPO>/ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py
  python3 $BRE variance work/corpus_*.dat            > work/04_variance.json
  python3 $BRE diff work/base.dat work/one_more_point.dat > work/04_diff_points.json
  python3 $BRE diff work/base.dat work/resaved.dat   > work/04_diff_resave.json
판정:
  - 04_diff_points.json 의 size_delta = record stride.
  - field_candidates 중 <I delta==1 = count field offset.
  - 04_variance.json 의 constant_runs = 고정 필드(magic/version/padding). weak_boundary_hint 는 참고만.
  - 04_diff_resave.json 에서 바뀐 offset = timestamp + checksum.
합격기준: stride, count_offset, header_size(=constant_runs 끝, diff 교차검증), timestamp_offset 확정.
반환(JSON): { stride, count_offset, header_size, timestamp_offset, checksum_offset, fixed_field_runs }
```

---

## Task E — 배열/필드 탐지 (Phase 5)

```
목표: 측정값의 offset·dtype·endianness·stride 를 확정.
입력: Task D 의 { stride, header_size, timestamp_offset }.
물리범위: 측정 대상의 실제 물리 범위를 <LO>,<HI> 로. (예: CD nm → 1, 1000)
실행:
  BRE=<REPO>/ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py
  python3 $BRE stride work/base.dat --offset <header_size> --max-stride 4096 > work/05_stride.json
  python3 $BRE arrays work/base.dat --stride <stride> --payload-offset <header_size> \
          --lo <LO> --hi <HI> --top 8 > work/05_arrays.json
  python3 $BRE stamps work/base.dat --max-bytes 8192 > work/05_stamps.json
판정:
  - 05_stride.json best_stride 가 Task D 의 stride 와 일치해야 함(불일치 시 header_size 재검토).
  - 05_arrays.json top_candidates[0] 의 field_offset_in_record + dtype = 측정값 필드.
  - element_count 가 tool UI 측정점 수와 일치하면 확정.
  - 05_stamps.json 후보는 오탐 다수 → 04_diff_resave 에서 바뀐 offset 과 교차하는 것만 채택.
합격기준: (value_offset, dtype, endianness, stride) 확정, element_count == 측정점 수.
반환(JSON): { payload_offset, stride, value_field:{offset,dtype}, field_layout:[...], timestamp_field }
```

---

## Task E2 — 좌표/Recipe 파일 (고정 배열이 아닐 때)

```
목표: <FILE> 이 좌표 파일이거나 recipe 파일(가변 구조)일 때 구조를 복원.
판단: Task D/E 에서 stride coverage 가 낮거나 arrays 가 전부 쓰레기면 이 Task 로.
참고: ai-dt/data-handling/binary-reverse-engineering/04-coordinate-and-recipe-files.md
실행:
  BRE=<REPO>/ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py
  # 좌표 파일(고정 record)이면 Task E 를 --lo/--hi 웨이퍼 스케일(예: 1,200000)로 재실행.
  # recipe 파일이면:
  python3 $BRE serial  "<FILE>"                       > work/e2_serial.json
  python3 $BRE offsets "<FILE>"                       > work/e2_offsets.json
  python3 $BRE tlv     "<FILE>" --start <header_size> > work/e2_tlv.json
  python3 $BRE strtab  "<FILE>"                       > work/e2_strtab.json
판정:
  - e2_serial.json mostly_text=true 또는 xml/zip/ole/dotnet 시그니처 → 표준 파서로 종료(역공학 불필요).
  - e2_tlv.json best.coverage>=0.99 & lands_at_eof → 그 config 가 TLV 레이아웃. first_records 의 tag 사전화.
  - e2_offsets.json 의 포인터 테이블 entry 수가 header count 필드와 일치하는지 diff 로 확인.
  - e2_strtab.json 의 null_terminated/length_prefixed 로 파라미터명 매핑.
합격기준: recipe면 (tag 사전 + 파라미터명 매핑), 좌표면 (x/y 필드 offset·dtype 확정).
반환(JSON): { file_kind:"coords|recipe-tlv|recipe-text", tlv_config?, tag_dictionary?, string_table?, xy_fields? }
```

## Task F — 파서 형식화 & 검증 (Phase 6)

```
목표: 복원한 구조를 Kaitai .ksy 로 적고 corpus 전체 round-trip 검증.
입력: Task D·E 의 확정 값.
참고: ai-dt/data-handling/binary-reverse-engineering/01-toolkit-reference.md §3
행동:
  1. header + record struct 를 .ksy 로 작성(magic contents 로 검증 걸기).
  2. kaitai-struct-compiler -t python spec.ksy 로 파서 생성(또는 construct 로 동등 작성).
  3. corpus 의 모든 파일을 파싱해 에러 없는지, 측정값이 tool UI/CSV export 와 일치하는지 확인.
합격기준: 전 파일 파싱 성공 + ground-truth 값 일치.
반환: work/FORMAT.md (필드 맵 표 + .ksy + 검증 로그).
```

---

## 병렬화 힌트

- Task A(triage)와 Task B(기존 reader)는 **독립**이라 동시 실행 가능.
- corpus 파일이 많으면 Task D의 variance는 파일별 전처리를 병렬로 나눌 수 있으나, `bre.py variance`가 이미 벡터화돼 있어 보통 불필요.
- 여러 서로 다른 파일 종류(이미지 vs 결과 vs recipe)를 동시에 조사할 때 파일 종류별로 A~E 파이프라인을 각각 별도 subagent에 할당.
