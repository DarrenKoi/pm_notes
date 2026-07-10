# scripts/ — bre.py toolkit

미지 binary 파일 구조 복원용 실행 도구. 의존성: **Python 3.9+, numpy** (scipy 불필요).

## 파일

| 파일 | 역할 |
|------|------|
| `bre.py` | 메인 toolkit. 7개 subcommand, 전부 stdout에 JSON 출력 |
| `make_fixture.py` | 정답을 아는 합성 CD-SEM 유사 파일 생성 (검증·데모용) |
| `selftest.py` | fixture로 bre.py를 회귀 검증 (17개 단언, exit 0 = 통과) |

## 빠른 시작

```bash
python3 selftest.py                          # 환경 점검 (제일 먼저)
python3 make_fixture.py /tmp/fx              # 합성 파일 생성 (실습용)
python3 bre.py triage   /tmp/fx/base.dat
python3 bre.py diff      /tmp/fx/base.dat /tmp/fx/one_more_point.dat
python3 bre.py variance  /tmp/fx/corpus_*.dat
python3 bre.py stride    /tmp/fx/base.dat --offset 48
python3 bre.py arrays    /tmp/fx/base.dat --stride 16 --payload-offset 48 --lo 1 --hi 1000
python3 bre.py stamps    /tmp/fx/base.dat --max-bytes 8192
```

## subcommand 요약

| 명령 | 하는 일 | 핵심 옵션 |
|------|---------|-----------|
| `triage` | magic·entropy·strings·임베디드 시그니처 | `--block`, `--min-str` |
| `variance` | corpus offset별 분산 → 고정 필드·경계 | (파일 2개 이상) |
| `diff` | 두 파일 바이트 차분 → count/length/stride/checksum | `--gap` |
| `arrays` | offset×dtype 격자 → 측정값 배열/필드 | `--stride`, `--payload-offset`, `--lo/--hi` |
| `stride` | 자기상관 → record 주기 | `--offset`, `--max-stride` |
| `stamps` | timestamp 후보(Unix/FILETIME/OLE) | `--max-bytes`, `--year-lo/-hi` |
| `tlv` | 가변 길이 TLV record 체인 탐지 (recipe) | `--start`, `--tail` |
| `offsets` | offset/pointer 테이블(디렉토리) 탐지 (recipe) | `--min-run`, `--max-base` |
| `strtab` | 구조화 문자열 테이블 추출 (recipe 파라미터명) | `--min-len`, `--max-str` |
| `serial` | 내부 직렬화/컨테이너(XML/zip/OLE/.NET) 탐지 | `--max-hits` |

앞 6개는 **고정 record 배열**(측정값·좌표)용, 뒤 4개는 **가변 구조**(recipe)용. 파일 유형별 사용법은 상위 `04-coordinate-and-recipe-files.md`.

## 설계 원칙

- **JSON only**: 사람이 hex를 다시 읽지 않도록, 모든 결과가 기계 판독 가능. agent 파이프라인용.
- **읽기 전용**: 원본 파일을 절대 수정하지 않는다.
- **에러도 JSON**: exit code로 분기하지 않도록 실패도 `{"error": ...}`로 보고.
- **정직한 신뢰도**: 약한 heuristic(variance의 boundary)은 caveat를 함께 낸다. 확정은 항상 diff 교차검증.

## 검증

`selftest.py`는 ground truth를 아는 fixture를 만들어 각 subcommand가 그 정답을 복원하는지 확인한다:
triage(magic·string), diff(stride·count·timestamp·checksum 격리), stride(fundamental 우선), arrays(interleaved CD 필드 정확 추출), variance(고정 구간), stamps(진짜 timestamp offset). numpy 버전이 바뀌어도 이 스크립트로 회귀 확인.
