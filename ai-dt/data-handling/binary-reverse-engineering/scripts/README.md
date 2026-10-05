---
tags: [binary, reverse-engineering, toolkit, reference]
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: toolkit_reference
aliases: [바이너리 역공학 도구 사용법]
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "바이너리 역공학"
note_kind: "목차"
classified_on: "2026-10-05"
---

# scripts/ — bre.py toolkit

미지 binary 파일에서 구조 후보를 찾는 로컬 도구다. 결과를 원천 UI/export와 대조해 필드 의미를 확인할 때 사용한다. 입력에 접근할 권한과 원본 보존을 먼저 확인한다. 합성 파일의 성공은 실제 벤더 포맷 성공을 뜻하지 않는다.

소스는 Python3.9+와 NumPy를 의존성으로 설명한다. 2026-10-04 실제 확인 환경은 Python3.12.12·NumPy1.26.4이며 SciPy는 사용하지 않았다. Python3.9 실행 및 다른 NumPy 버전의 호환성은 미확인이다. 별도 가상환경에서 사용하는 것을 권한다.

## 파일

| 파일 | 역할 |
|------|------|
| [bre.py](./bre.py) | 메인 toolkit. 10개 subcommand. 정상 분석/포착한 분석 오류는 stdout JSON; help/인자 오류는 별도 |
| [make_fixture.py](./make_fixture.py) | 정답을 아는 합성 CD-SEM 유사 파일 생성 (검증·데모용) |
| [selftest.py](./selftest.py) | fixture로 bre.py를 회귀 검증 (12개 검사 구간·28개 check 호출, 정상 완료 exit0 = 해당 합성 검사 통과) |

## 빠른 시작

이 디렉터리에서 실행한다. 아래 `/tmp/fx`는 실습 산출 경로이며 `make_fixture.py`는 같은 이름 파일을 쓸 수 있으므로 비어 있는 전용 임시 경로로 바꾼다. 실제 장비 원본 디렉터리를 생성기의 출력 위치로 지정하지 않는다.

```bash
python3 selftest.py                          # 환경 점검 (제일 먼저)
python3 make_fixture.py /tmp/fx              # 합성 파일 생성 (실습용)
python3 bre.py triage   /tmp/fx/base.dat
python3 bre.py diff      /tmp/fx/base.dat /tmp/fx/one_more_point.dat
python3 bre.py variance  /tmp/fx/corpus_*.dat
python3 bre.py stride    /tmp/fx/base.dat --offset 48
python3 bre.py arrays    /tmp/fx/base.dat --stride 16 --payload-offset 48 --lo 1 --hi 1000
python3 bre.py stamps    /tmp/fx/base.dat --max-bytes 8192
python3 bre.py tlv       /tmp/fx/recipe.dat --start 24
python3 bre.py offsets   /tmp/fx/recipe.dat
python3 bre.py strtab    /tmp/fx/recipe.dat
python3 bre.py serial    /tmp/fx/recipe_xml.dat
```

## subcommand 요약

| 명령 | 하는 일 | 핵심 옵션 |
|------|---------|-----------|
| `triage` | magic·entropy·strings·임베디드 시그니처 | `--block`, `--min-str` |
| `variance` | corpus offset별 분산 → 고정 필드·경계 | (파일 2개 이상) |
| `diff` | 두 파일 바이트 차분 → count/length/stride/checksum | `--gap` |
| `arrays` | offset×dtype 격자 → 측정값 배열/필드 | `--stride`, `--payload-offset`, `--lo/--hi` |
| `stride` | 자기상관 → record 주기 | `--offset`, `--max-stride` |
| `stamps` | timestamp 후보(Unix/FILETIME/OLE) | `--max-bytes`, `--year-lo`, `--year-hi` |
| `tlv` | 가변 길이 TLV record 체인 탐지 (recipe) | `--start`, `--tail` |
| `offsets` | offset/pointer 테이블(디렉토리) 탐지 (recipe) | `--min-run`, `--max-base` |
| `strtab` | 구조화 문자열 테이블 추출 (recipe 파라미터명) | `--min-len`, `--max-str` |
| `serial` | 내부 직렬화/컨테이너(XML/zip/OLE/.NET) 탐지 | `--max-hits` |

앞 6개는 **고정 record 배열**(측정값·좌표)용, 뒤 4개는 **가변 구조**(recipe)용. 파일 유형별 사용법은 [좌표·recipe 문서](../04-coordinate-and-recipe-files.md).

## 설계 원칙

- **구조화된 분석 결과**: 정상 분석 결과는 stdout JSON이다. 결과 필드와 원시 바이트를 함께 검토한다. `--help`는 일반 텍스트라 JSON parser에 넘기지 않는다.
- **입력 읽기 전용**: `bre.py`는 입력을 읽고 stdout을 출력한다. 생성기는 지정 경로에 합성 파일을 쓰며 selftest는 임시 디렉터리를 만든다. stdout 리다이렉션 경로가 입력 원본과 같으면 shell이 원본을 비울 수 있으므로 결과는 다른 경로에 저장한다.
- **실패 판정**: `parse_args` 이후 포착한 분석 예외는 `{"error": ...}`와 exit0일 수 있다. 잘못된 인자는 stderr 일반 텍스트·exit2다. 의존성 import/프로세스 실패도 JSON 보장이 없다. 호출자는 exit status·stdout JSON 파싱·`error` 필드·필수 결과 구조를 함께 검사한다. exit0만으로 성공으로 분류하지 않는다.
- **정직한 신뢰도**: 약한 heuristic(variance의 boundary)은 caveat를 함께 낸다. 확정은 항상 diff 교차검증.

## 검증

`selftest.py`는 ground truth를 아는 fixture를 만들어 각 subcommand가 그 합성 파일에 넣은 조건을 검사한다:
triage(magic·string), diff(stride·count·timestamp·checksum 격리), stride(fundamental 우선), arrays(interleaved CD 필드 정확 추출), variance(고정 구간), stamps(진짜 timestamp offset). 추가로 recipe TLV·offset table·문자열·내부 XML과 좌표 배열을 검사한다. NumPy 버전이 바뀌면 다시 실행하되 실제 포맷·모든 오탐·새 버전 호환성을 이 검사만으로 보증하지 않는다.

## 검토 결과와 읽기 순서

2026-10-04 원래 절·명령·도구 역할을 보존하며 명령 수/검사 수와 오류 계약을 소스에 맞췄다. 자체 검사28개와10개 명령 등록을 확인했다. missing-file 분석 오류(JSON/exit0), 누락 인자(stderr/exit2), help(일반 텍스트/exit0), variance 파일 수 부족(JSON/exit0)을 로컬 확인했다. 실행 코드3개는 수정하지 않았다. 실제 장비·벤더 포맷·압축 해제·파서 명세의 완성은 미확인이다.

개념을 처음 읽으면 [역공학 개념](../reverse-engineering-concepts.md), 작업 범위는 [작업 계약](../00-agent-brief.md), 실행 순서는 [runbook](../00-agent-runbook.md)을 사용한다. 이 README는 현재 구현의 호출/오류 계약을 설명하며 위 작업 문서의 완전 통합은 Claude 연결 실패로 보류한다.

근거는 같은 모듈의 [CLI 구현](./bre.py), [fixture 생성기](./make_fixture.py), [자체 검사](./selftest.py)다. 실제 검증 범위와 협의 상태는 [정리 기록](../../organization-log.md)에 남긴다.
