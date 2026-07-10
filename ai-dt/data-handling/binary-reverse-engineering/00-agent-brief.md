---
tags: [binary, reverse-engineering, agent, cd-sem, metrology]
level: intermediate
last_updated: 2026-07-10
---

# Agent Brief — 미지 binary 포맷 복원 작업 계약

> 이 문서는 사람이 아니라 **agent 가 읽는 작업 계약서**다. 상세 명령은 [00-agent-runbook.md](./00-agent-runbook.md)에 있고, 이 브리프는 그 위의 미션·게이트·공유상태·가드레일을 정한다. 작업 시작 전 이것부터 읽는다.

## 미션

계측 장비(CD-SEM 등)가 뱉은 문서화되지 않은 binary 파일의 구조를 복원해, 재사용 가능한 parser 스펙으로 만든다.

**완료 정의(Definition of Done)**: [01-toolkit-reference.md](./01-toolkit-reference.md) §3 규격의 Kaitai `.ksy` 또는 `construct` 스펙이 존재하고, 보유한 모든 sample 파일에 대해 오류 없이 parsing 되며, 최소 한 개 필드의 값이 장비 UI/CSV export 값과 일치한다.

## 시작 전 게이트 (건너뛰지 말 것)

두 게이트를 통과하지 못하면 **작업을 시작하지 말고 사용자에게 보고**한다.

1. **법무/계약 게이트** — [03-legal-and-first-moves.md](./03-legal-and-first-moves.md)를 읽는다. 장비 구매계약·EULA·NDA에 reverse engineering 금지 조항이 있을 수 있다. 본인 데이터를 담은 본인 장비의 출력 파일을 파싱하는 것과, 벤더 소프트웨어를 역컴파일하는 것은 위험도가 전혀 다르다. **후자는 절대 자동으로 진행하지 않는다.**
2. **낭비 방지 게이트** — [02-cd-sem-formats.md](./02-cd-sem-formats.md)를 먼저 확인한다. 이미 열려 있는 포맷(TIFF private tag, HDF5, STDF 등)을 손으로 파싱하는 것은 순수한 시간 낭비다. **SEM 이미지의 경우 `tifffile`이 FEI/Zeiss 메타데이터를 이미 파싱한다.**

## 실행 순서 (phase pipeline)

각 phase 는 JSON artifact 를 남기고, 다음 phase 는 그것을 입력으로 받는다. 이 handoff 규약 덕분에 phase 별로 다른 agent 에게 위임할 수 있다. 각 phase 의 정확한 명령·합격기준은 [00-agent-runbook.md](./00-agent-runbook.md).

| Phase | 문서 | 산출 artifact | 통과 조건 |
|---|---|---|---|
| 0 | 이 브리프 + [03-legal](./03-legal-and-first-moves.md) | `findings.json` 초기화 | 게이트 2개 통과 |
| 1 | [runbook Phase 1](./00-agent-runbook.md) | `01_triage.json` | 압축/컨테이너 여부 판정 |
| 2 | [02-cd-sem-formats](./02-cd-sem-formats.md) | (기존 reader 시도 결과) | 기존 parser 적용 가능 여부 판정 |
| 3 | [runbook Phase 3~4](./00-agent-runbook.md) | `04_variance.json`, `04_diff_*.json` | count/length/timestamp 필드 확정 |
| 4 | [runbook Phase 5](./00-agent-runbook.md) | `05_stride.json`, `05_arrays.json` | payload dtype·offset·stride 확정 |
| 5 | [runbook Phase 6](./00-agent-runbook.md) + [01-toolkit §3](./01-toolkit-reference.md) | `FORMAT.md` (.ksy + validate) | 전 sample parsing 성공 |

Phase 2 에서 기존 parser 로 해결되면 **3~5는 실행하지 않는다.** 그게 성공이다.

## 환경 준비

```bash
cd ai-dt/data-handling/binary-reverse-engineering/scripts
python3 -m pip install numpy          # 유일한 필수 의존성
python3 selftest.py                   # 정답을 아는 합성 파일로 toolkit 회귀 검증
```

`selftest.py`가 `ALL PASS`가 아니면 toolkit 결과를 신뢰하지 말고 먼저 고친다. 이 스크립트는 `make_fixture.py`가 만든 합성 CD-SEM 유사 파일에서 magic·count 필드·timestamp·record stride·interleaved CD 배열을 전부 정확히 복원하는지 확인한다(17개 단언).

## 산출물 규약 — `findings.json`

Agent 는 작업 디렉터리에 `findings.json` 하나를 누적 갱신한다. 이것이 phase 간 유일한 공유 상태다. (runbook 의 개별 `NN_*.json` 은 각 phase 의 raw 출력이고, `findings.json` 은 그로부터 확정된 결론만 모은다.)

```json
{
  "target": "sample.dat",
  "corpus": ["base.dat", "one_more_point.dat", "resaved.dat"],
  "confirmed": {
    "endianness": "little",
    "magic": {"offset": 0, "bytes_hex": "43445331", "ascii": "CDS1"},
    "header_size": 48,
    "fields": [
      {"offset": 8,  "name": "n_points",  "dtype": "u4", "evidence": "diff: +1 point -> delta=1"},
      {"offset": 12, "name": "timestamp", "dtype": "u4", "evidence": "diff: resave -> delta=3600s"}
    ],
    "record": {"stride": 16, "count_field": "n_points",
               "fields": [{"k": 0, "name": "site_id", "dtype": "u4"},
                          {"k": 12, "name": "cd_nm", "dtype": "f4"}]}
  },
  "hypotheses": [
    {"claim": "EOF-4는 u32 checksum", "confidence": "medium", "how_to_test": "1바이트 변조 후 뷰어가 거부하는지"}
  ],
  "rejected": [
    {"claim": "stride=48", "why": "16의 harmonic. diff size_delta=16이 반증"}
  ]
}
```

**`confirmed` 에는 증거가 있는 것만 올린다.** 근거 없는 추측은 `hypotheses`로, 반증된 것은 `rejected`로 남긴다 — 다음 agent 가 같은 실수를 반복하지 않도록.

## 가드레일

- **원본 파일을 수정하지 않는다.** 모든 스크립트는 read-only다. 변조 실험이 필요하면 사본에 한다.
- **`bre.py`의 출력은 후보이지 정답이 아니다.** 특히 `stamps`는 오탐이 매우 많고, `variance`의 `weak_boundary_hint`는 수 바이트 어긋난다. 이 두 값은 단독 근거로 쓰지 않는다.
- **모든 확정에는 교차 검증 2개를 요구한다.** 예: stride 는 `stride` subcommand 의 autocorrelation **과** `diff`의 `size_delta` 가 일치해야 확정.
- **막히면 3회 시도 후 멈추고 보고한다.** 같은 방법을 반복하지 않는다. 특히 entropy 가 7.5 이상인데 압축을 못 푸는 경우는 암호화일 수 있으므로 즉시 사람에게 넘긴다.
- **벤더에게 먼저 물어보는 것이 거의 항상 더 빠르다.** Apps engineer 에게 포맷 스펙·SDK·CSV export·EDA(Interface A) feed 를 요청하는 선택지를 사용자에게 반드시 제시한다.

## 관련 문서

- [README](./README.md) — 시리즈 목차
- [00-agent-runbook.md](./00-agent-runbook.md) — phase별 정확한 명령·합격기준
- [agent-tasks.md](./agent-tasks.md) — 각 phase 를 subagent 에게 위임할 때 쓰는 프롬프트 원문
