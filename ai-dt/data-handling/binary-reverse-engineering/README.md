---
tags: [binary, reverse-engineering, cd-sem, metrology, file-format, data-handling]
level: intermediate
last_updated: 2026-07-10
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

1. **이미지는 대개 이미 풀려 있다.** SEM 이미지는 거의 TIFF이고, `tifffile`이 FEI/Thermo·Zeiss의 private tag를 이미 파싱한다. 손으로 파서를 짜지 말 것 → [02-cd-sem-formats](./02-cd-sem-formats.md).
2. **진짜 역공학 대상은 측정결과·recipe binary다.** 여기에만 [01-toolkit-reference](./01-toolkit-reference.md)의 기법을 쓴다.

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

전형적 계측 파일 구조:
`[magic/version header] [metadata block] [측정값 record 배열] [trailer/checksum]`

**파일 유형별 커버리지** — 측정값·좌표는 고정 record 배열이라 `arrays`/`stride`/`diff`로, recipe는 가변 구조라 `serial`/`offsets`/`tlv`/`strtab`로 다룬다. 유형별 상세는 [04-coordinate-and-recipe-files](./04-coordinate-and-recipe-files.md).

## 어떻게 사용하는가? (How)

### Agent에게 통째로 맡기려면
→ **[00-agent-runbook.md](./00-agent-runbook.md)** 를 읽힌다. phase별 입력·명령·합격기준·산출 JSON이 명시돼 있어 subagent가 독립적으로 한 phase씩 실행할 수 있다.

### 직접 손으로 하려면
```bash
cd scripts
python3 selftest.py                       # 환경 점검 (numpy 정상? 17개 단언 통과?)
python3 bre.py triage  yourfile.dat       # 1단계
python3 bre.py variance corpus/*.dat      # 4단계 (파일 여러 개)
python3 bre.py diff    a.dat b.dat        # 4단계 (변수 1개만 다른 두 파일)
python3 bre.py stride  yourfile.dat --offset <header_size>
python3 bre.py arrays  yourfile.dat --stride <N> --payload-offset <H> --lo 1 --hi 1000
python3 bre.py stamps  yourfile.dat --max-bytes 8192
```

모든 명령은 **stdout에 JSON**을 낸다 → agent가 파싱해 다음 단계 입력으로 넘긴다.

## 문서 목록

| 문서 | 내용 |
|------|------|
| [00-agent-brief.md](./00-agent-brief.md) | **agent 작업 계약** — 미션·완료정의·게이트·`findings.json` 공유상태·가드레일 (먼저 읽음) |
| [00-agent-runbook.md](./00-agent-runbook.md) | **phase 파이프라인** — 각 phase의 입력/명령/합격기준/산출물 |
| [01-toolkit-reference.md](./01-toolkit-reference.md) | 범용 binary RE 도구·기법 총람 (triage·hex editor·Kaitai·통계 탐지) |
| [02-cd-sem-formats.md](./02-cd-sem-formats.md) | CD-SEM 벤더별 파일 포맷, TIFF private tag, 기존 오픈소스 reader, SEMI EDA |
| [04-coordinate-and-recipe-files.md](./04-coordinate-and-recipe-files.md) | **좌표·recipe 파일** — 가변 TLV/offset 테이블/문자열 테이블/직렬화 대응(`tlv`·`offsets`·`strtab`·`serial`) |
| [03-legal-and-first-moves.md](./03-legal-and-first-moves.md) | 역공학 적법성(DMCA 1201(f)/EU), NDA 주의, "벤더에 먼저 요청" |
| [agent-tasks.md](./agent-tasks.md) | 복사해서 subagent에 던지는 task 프롬프트 모음 |
| [scripts/](./scripts/) | `bre.py` toolkit, `make_fixture.py`, `selftest.py` (17 단언 회귀검증) |

## 참고 자료 (References)

- 각 문서 하단 References 참고. 이 폴더는 standalone이며 다른 최상위 폴더와 링크하지 않는다.
- toolkit 검증: `scripts/selftest.py` — 정답을 아는 합성 CD-SEM fixture로 7개 subcommand의 복원 정확도를 회귀 검증.
