---
tags: [binary, reverse-engineering, cd-sem, metrology, file-format, data-handling]
level: intermediate
last_updated: 2026-07-10
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# Binary 파일 역공학 (Binary Reverse Engineering)

> CD-SEM 등 계측 장비가 뱉는 문서화되지 않은 binary 파일의 구조를 복원하기 위한 방법론 + 실행 가능한 toolkit. **다른 agent가 그대로 집어 실행할 수 있도록** runbook 형태로 정리했다.

## 왜 필요한가? (Why)

MI(Metrology & Inspection) 엔지니어로서 CD-SEM raw 데이터를 분석할 때, 이미지·recipe·측정결과 일부가 **binary 파일**이라 열어도 내용을 알 수 없다. 이 폴더는 그런 파일에서 다음을 복원하는 절차를 담는다.

- 파일이 무엇인지 (압축? 알려진 컨테이너? 순수 record 덩어리?)
- header 구조와 magic/version
- 측정값 배열의 위치·자료형(dtype)·endianness
- record stride와 필드 배치
- timestamp / count / length / checksum 필드

핵심 통찰 두 가지:

1. **먼저 기존 reader 지원을 확인한다.** `tifffile`에 FEI/Thermo·Zeiss 메타데이터 tag 구현이 있지만, SEM 전체의 TIFF 비율이나 특정 CD-SEM의 지원을 확인한 것은 아니다. 파일 판본과 필요한 픽셀·메타데이터·단위를 실제 UI/export와 대조한다 → [02-cd-sem-formats](./02-cd-sem-formats.md).
2. **기존 reader로 필요한 값을 확인할 수 없는 파일이 조사 대상이다.** 측정결과·recipe뿐 아니라 지원되지 않는 이미지/좌표도 해당한다. 구조와 권한을 확인한 뒤 [01-toolkit-reference](./01-toolkit-reference.md)의 기법을 쓴다.

## 핵심 개념 (What) — 파일 해부 순서

```
0. 법적/실무 확인   → 벤더에 스펙/export 먼저 요청 (03-legal-and-first-moves)
1. 정체 파악(triage) → magic / entropy / strings           [bre.py triage]
2. 알려진 포맷 우회  → 기존 reader로 이미 되는지 확인       (02-cd-sem-formats)
3. corpus 확보       → 변수 하나만 바꾼 파일 여러 개 수집
4. 차분 분석(diff)   → count/length/timestamp/checksum 위치 [bre.py diff, variance]
5. 배열/필드 탐지    → 측정값 offset·dtype·stride 확정      [bre.py arrays, stride, stamps]
6. 파서 형식화       → Kaitai/construct로 스펙 고정·검증    (05는 01 문서 §형식화)
```

고정 record를 가정한 학습용 구조 예:
`[magic/version header] [metadata block] [측정값 record 배열] [trailer/checksum]`

**파일 유형별 조사 후보** — 고정 record 배열인 측정값·좌표에는 `arrays`/`stride`/`diff`를, 가변 구조에는 `serial`/`offsets`/`tlv`/`strtab`를 사용한다. 파일 이름만으로 이 구조를 확정하지 않는다. 합성 fixture 검사는 실제 장비 포맷 지원과 다르다. 유형별 상세는 [04-coordinate-and-recipe-files](./04-coordinate-and-recipe-files.md).

## 어떻게 사용하는가? (How)

### Agent에게 통째로 맡기려면
→ **[00-agent-runbook.md](./00-agent-runbook.md)** 를 읽힌다. phase별 입력·명령·합격기준·산출 JSON이 명시돼 있어 subagent가 독립적으로 한 phase씩 실행할 수 있다.

### 직접 손으로 하려면
```bash
cd scripts
python3 selftest.py                       # 환경 점검 (numpy 정상? 12구간·28개 검사 통과?)
python3 bre.py triage  yourfile.dat       # 1단계
python3 bre.py variance corpus/*.dat      # 4단계 (파일 여러 개)
python3 bre.py diff    a.dat b.dat        # 4단계 (변수 1개만 다른 두 파일)
python3 bre.py stride  yourfile.dat --offset "$HEADER_SIZE"
python3 bre.py arrays  yourfile.dat --stride "$BEST_STRIDE" --payload-offset "$HEADER_SIZE" --lo 1 --hi 1000
python3 bre.py stamps  yourfile.dat --max-bytes 8192
```

`HEADER_SIZE`·`BEST_STRIDE`는 실제 파일에서 확인한 정수로 먼저 정의한다. 이 호출 예는 `scripts`를 현재 작업 위치로 가정하며 원본 입력과 출력 경로를 분리한다. selftest는 임시 합성 파일을 생성하고 일반 텍스트 결과를 낸다. `bre.py`의 정상 분석/포착 예외는 JSON이지만 help/인자 오류/의존성 오류는 다르다. exit code·JSON 파싱·`error` 필드·필수 값을 모두 검사한 뒤 다음 단계로 넘긴다.

## 문서 목록

| 문서 | 내용 |
|------|------|
| [reverse-engineering-concepts.md](./reverse-engineering-concepts.md) | **개념 입문** — 역공학이 무엇이고 왜 되는지, 용어(바이트·offset·dtype·endianness·stride)와 7단계 방법을 쉬운 말로 (사람이 처음 읽는 문서) |
| [00-agent-brief.md](./00-agent-brief.md) | **agent 작업 계약** — 미션·완료정의·게이트·`findings.json` 공유상태·가드레일 (먼저 읽음) |
| [00-agent-runbook.md](./00-agent-runbook.md) | **phase 파이프라인** — 각 phase의 입력/명령/합격기준/산출물 |
| [01-toolkit-reference.md](./01-toolkit-reference.md) | 범용 binary RE 도구·기법 총람 (triage·hex editor·Kaitai·통계 탐지) |
| [02-cd-sem-formats.md](./02-cd-sem-formats.md) | CD-SEM 벤더별 파일 포맷, TIFF private tag, 기존 오픈소스 reader, SEMI EDA |
| [04-coordinate-and-recipe-files.md](./04-coordinate-and-recipe-files.md) | **좌표·recipe 파일** — 가변 TLV/offset 테이블/문자열 테이블/직렬화 대응(`tlv`·`offsets`·`strtab`·`serial`) |
| [03-legal-and-first-moves.md](./03-legal-and-first-moves.md) | 역공학 적법성(DMCA 1201(f)/EU), NDA 주의, "벤더에 먼저 요청" |
| [agent-tasks.md](./agent-tasks.md) | 복사해서 subagent에 던지는 task 프롬프트 모음 |
| [scripts/](./scripts/) | `bre.py` toolkit, `make_fixture.py`, `selftest.py` (12구간·28개 합성 회귀검사) |

## 참고 자료 (References)

- 각 문서 하단 References 참고. 이 폴더는 standalone이며 다른 최상위 폴더와 링크하지 않는다.
- toolkit 검증: `scripts/selftest.py` — 정답을 아는 가상 계측/recipe fixture로 10개 subcommand와 28개 검사를 대조한다. 실제 CD-SEM 파일의 복원 정확도나 모든 오류 처리를 보증하지 않는다.


## 읽기 순서와 검토 결과

처음에는 개념 → 법률/첫 수순 → 벤더 형식 → 작업 계약 → runbook 순서로 읽는다. 도구 총람은 필요한 기법의 참고 자료, 좌표/recipe는 구조별 적용, task는 위임 template, scripts README는 실제 CLI 계약이다. task 본문은 이 문서 정리 작업의 실행 지시가 아니다. 모두 같은 주제 안의 학습/작업 참고이며 실제 장비 조사 기록과 구분한다.

2026-10-04 확인: [CLI 설명](./scripts/README.md)·[실제 구현](./scripts/bre.py)·[합성 검사](./scripts/selftest.py), [tifffile 공식 소스](https://github.com/cgohlke/tifffile) 표시2026.9.20. 로컬 Python3.12.12/NumPy1.26.4의 검사28개와 기존 오류 계약을 대조했다. 최신 버전/설치/실제 장비 지원은 보증하지 않는다. 원래 모든 절·문서 목록·실습 목적·작성일을 보존했고 무근거 포맷 비율/지원 일반화와 옛 검사 개수를 정정했다. HERDR_ENV=1/pane_not_found로 Claude 의견을 받지 못해 중복 통합과 업무별 계약 결정은 보류했다. [정리 기록](../organization-log.md)에 근거·검증·미확인을 남긴다.
