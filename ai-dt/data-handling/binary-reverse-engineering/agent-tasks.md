---
tags: [agent, subagent, task-spec, reverse-engineering]
level: intermediate
last_updated: 2026-07-10
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: work_reference
---

# Agent Task 프롬프트 모음

> [!info] 검토 범위 — 2026-10-04
> 여기의 CDS1·offset48·stride16·recipe와 좌표는 합성 fixture의 학습 예다. 실제 CD-SEM/회사 데이터의 규격이 아니다. `bre.py`는 후보를 찾는 도구이며 출력의 hint가 “확정”이라고 표현해도 독립 근거를 검사한다. 정상 결과/포착한 분석 오류는 stdout JSON,인자 오류는 stderr/exit2일 수 있다. 구체 호출 계약은 [도구 README](./scripts/README.md)를 따른다. 원본 외의 전용 작업 경로에 결과를 저장하며 실제 장비 조사·벤더 연락·다른 agent 실행은 이 문서 정리에서 수행하지 않는다.


> 각 블록은 상황과 승인 범위를 검토해 위임할 때 쓰는 참고다. `<...>` 부분만 실제 값으로 치환. 각 task는 **원본 입력 읽기 전용**이며 별도 작업 디렉터리에 결과를 쓰고 **JSON/파일 산출물**로 끝나므로 결과를 다음 task 입력으로 넘길 수 있다. 파이프라인 전체 설명은 [00-agent-runbook.md](./00-agent-runbook.md).

## 사용 전 공통 전제

- toolkit 경로: `ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py`
- 시작 전 `python3 scripts/selftest.py` (exit 0 확인).
- 분석 결과는 exit/JSON/error/필수 구조를 함께 검사한다. help/인자 오류는 JSON이 아니다. 원본과 겹치지 않는 전용 `work/`를 먼저 만들고 산출물을 저장한다. 아래 task는 복사할 참고 template이지 지금 실행하는 요청이 아니다.

---

## Task A — 단일 파일 정체 파악 (Phase 1)

```
목표: <FILE> 의 정체를 파악하고 다음 단계를 정하라.
제약: 읽기 전용. 파일을 수정하지 마라.
실행:
  python3 <REPO>/ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py triage "<FILE>" > work/01_triage.json
판정:
  - magic_at_offset_0 가 TIFF/HDF5/zip 등 알려진 포맷이면 "Task B(기존 reader)로 우회" 권고.
  - entropy_verdict == compressed_or_encrypted 이면 "고entropy 후보; header/decoder로 압축 확인 필요" 보고. entropy만으로 암호화/압축을 확정하지 않는다.
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
판정: 필요한 메타/픽셀/측정값과 단위/개수/판본을 원천 대조한 경우만 해당 범위 종료. 위 코드는 메타 요약이며 픽셀/측정값을 실제 읽거나 대조하지 않는다. readable은 True/False/None(미확인)을 구분한다. 아니면 "Task C~E 진행".
반환(JSON): { reader_tried, readable:bool|null, extracted_summary, recommend }
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
  python3 "$BRE" variance work/corpus_*.dat            > work/04_variance.json
  python3 "$BRE" diff work/base.dat work/one_more_point.dat > work/04_diff_points.json
  python3 "$BRE" diff work/base.dat work/resaved.dat   > work/04_diff_resave.json
판정:
  - 04_diff_points.json 의 size_delta는 파일 크기 차이; 다른 블록/압축/정렬이 같고 record 하나만 추가된 조건에서 stride후보.
  - field_candidates 중 <I delta==1은 count 후보; counter와 구분해 여러 알려진 개수/endianness를 대조.
  - 04_variance.json 의 constant_runs = 고정 필드(magic/version/padding). weak_boundary_hint 는 참고만.
  - 04_diff_resave.json 에서 바뀐 offset은 timestamp/checksum/counter/metadata 등의 후보; 실제epoch/단위/저장시각으로 대조.
합격기준: stride, count_offset, header_size(실제 record 경계/소비 확인;constant_runs 끝과 같다고 가정하지 않음), timestamp_offset 확정.
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
  python3 "$BRE" stride work/base.dat --offset <header_size> --max-stride 4096 > work/05_stride.json
  python3 "$BRE" arrays work/base.dat --stride <stride> --payload-offset <header_size> \
          --lo <LO> --hi <HI> --top 8 > work/05_arrays.json
  python3 "$BRE" stamps work/base.dat --max-bytes 8192 > work/05_stamps.json
판정:
  - 05_stride.json best_stride 가 Task D 의 stride 와 일치해야 함(불일치 시 header_size 재검토).
  - 05_arrays.json top_candidates[0] 의 field_offset_in_record와 dtype은 후보. 첫 파일offset은 payload_offset+k; 의미/단위/순서를 원천 대조.
  - element_count 일치는 대조 조건 하나이며 필드 의미의 확정은 아니다. trailer/샘플상한을 별도 확인.
  - 05_stamps.json 후보는 오탐 다수 → 04_diff_resave 에서 바뀐 offset 과 교차하는 것을 추가검증 후보로 남기고 알려진 시각/epoch/단위/시간대로 검사.
합격기준: (value_offset, dtype, endianness, stride) 확정, element_count == 측정점 수.
반환(JSON): { payload_offset, stride, value_field:{offset,dtype}, field_layout:[...], timestamp_field }
```

---

## Task E2 — 좌표/Recipe 파일 (고정 배열이 아닐 때)

```
목표: <FILE> 이 좌표 파일이거나 recipe 파일(가변 구조)일 때 구조를 복원.
판단: Task D/E 에서 stride verified/autocorr가 약하거나 arrays 후보가 없으면 (stride에coverage필드는 없음) 이 Task 로.
참고: ai-dt/data-handling/binary-reverse-engineering/04-coordinate-and-recipe-files.md
실행:
  BRE=<REPO>/ai-dt/data-handling/binary-reverse-engineering/scripts/bre.py
  # 좌표 파일(고정 record)이면 Task E 를 --lo/--hi 웨이퍼 스케일(예: 1,200000)로 재실행.
  # recipe 파일이면:
  python3 "$BRE" serial  "<FILE>"                       > work/e2_serial.json
  python3 "$BRE" offsets "<FILE>"                       > work/e2_offsets.json
  python3 "$BRE" tlv     "<FILE>" --start <header_size> > work/e2_tlv.json
  python3 "$BRE" strtab  "<FILE>"                       > work/e2_strtab.json
판정:
  - e2_serial.json mostly_text=true 또는 xml/zip/ole/dotnet 시그니처 → 실제 표준parser/컨테이너 경계와 필요한 값이 검증돼야 종료. signature/mostly_text만으로 종료하지 않음. dotnet은 문자열탐지이며 일반 역직렬화를 실행하지 않음.
  - e2_tlv.json best.coverage>=0.99 & lands_at_eof → 그 config는 TLV후보. lands_at_eof는 실제EOF 또는 --tail여유까지 포함하므로 ends_at/size/bytes_consumed와 독립 필드값을 대조. first_records 의 tag 사전화.
  - e2_offsets.json 의 포인터 테이블 entry 수가 header count 필드와 일치하는지 diff 로 확인.
  - e2_strtab.json 의 null_terminated/length_prefixed 로 파라미터명 매핑.
합격기준: recipe면 (tag 사전 + 파라미터명 매핑), 좌표면 (x/y 필드 offset·dtype 확정).
반환(JSON): { file_kind:"coords|recipe-tlv|recipe-text", tlv_config?, tag_dictionary?, string_table?, xy_fields? }
```

## Task F — 파서 형식화 & 검증 (Phase 6)

```
목표: 복원한 구조를 Kaitai .ksy로 적고 corpus 전체 파싱/의미 검증. 바이트동일 round-trip은 별도의 writer/unknown/padding/checksum 보존 검사가 필요하며 일반 compiler 명령만으로 보장되지 않음.
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

`Task E`의 LO/HI는0이외 값의 절댓값 범위다. 음수/0좌표·상수setpoint는 도구 점수에서 누락/제외될 수 있다. `best=None`/빈 후보는 미확인으로 반환한다. task별 입력 SHA·판본·근거를 기록하고 한 findings 파일에 동시 작성하지 않는다.

## 검토 결과

2026-10-04 원래 절·phase/task·실습 호출의 목적을 보존하고 후보/확정, 입력/출력, JSON/exit, 실제 reader 값 대조와 미확인 반환을 구분했다. 근거는 [CLI 소스](./scripts/bre.py)와 [합성 검사](./scripts/selftest.py)다. Kaitai parse와 writer 조건은 [공식 serialization 문서](https://doc.kaitai.io/serialization.html)를 확인했으며 compiler/runtime는 실행하지 않았다. HDF5/SER/STDF 등의 외부 reader 후보와 unblob 명령의 설치 판본·옵션·실제 결과는 이 문서에서 확인하지 않았다. 실제 장비와 corpus, 작업 계약의 완성과 중복 통합은 미확인이다. HERDR_ENV=1/pane_not_found로 Claude 의견을 받지 못해 구조 통합은 보류했다. [정리 기록](../organization-log.md)에 남긴다.
