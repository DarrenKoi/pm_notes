---
tags: [itc, aix, roadmap, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
---

# 로드맵 자료 정리 기록

## 범위와 결정

2026-10-04 원래 Markdown 12개/2,526행을 모두 읽고 개별 검토했다. 업무 기록·보고 초안·ADR·입력 텍스트·HTML을 구분했다. 기존 경로·파일명·절·수치 슬롯과 당시 결정은 유지하고 목차 및 공통 검토 안내를 추가했다. 기존 tags·version·last_updated·ADR Accepted 날짜를 유지했으며 검토일은 별도 속성으로 추가했다. 중복된 검토 주의사항은 대표 안내로 모았고 outline·speaker notes·카드·시간순 기록은 용도가 달라 합치지 않았다. 문서 간 이동은 없고 코드·HTML·CSS·bgk.txt는 수정하지 않았다.

| 원래 문서 | 검토 결과 |
|---|---|
| `CONTEXT.md` | 업무 용어와 가정; 원문 보존·주의사항 연결 |
| `ch1-context-asis-draft.md` | 외부 변화·현황 진단 초안; 원문 보존·주의사항 연결 |
| `ch2-vision-draft.md` | 비전·조직 차별 주장 초안; 원문 보존·주의사항 연결 |
| `ch3-tobe-draft.md` | 전환 단계·KPI 초안; 원문 보존·주의사항 연결 |
| `ch4-tracks-draft.md` | 능력·트랙 구성 초안; 원문 보존·주의사항 연결 |
| `ch6-gantt-draft.md` | 일정과 시각 표기 초안; 원문 보존·주의사항 연결 |
| `docs/adr/0001-aix-tf-ax-part-boundary.md` | 당시 Accepted 조직 경계 ADR; 원문 보존·주의사항 연결 |
| `grilling-session-2026-05-14.md` | 질문과 결정의 시간순 기록; 원문 보존·주의사항 연결 |
| `handoff-2026-05-15.md` | 당시 인계와 남은 작업; 원문 보존·주의사항 연결 |
| `itc-aix-roadmap-outline.md` | 통합 설계 v0.3; 원문 보존·주의사항 연결 |
| `quick-win-cards.md` | PoC 카드와 당시 졸업 조건; 원문 보존·주의사항 연결 |
| `team-brainstorm-prep.md` | v0.2 검토 회의 준비; 원문 보존·주의사항 연결 |

## Claude 협의와 보류

HERDR_ENV=1, `herdr pane current --current` → `pane_not_found`. 전용 Claude pane 연결이 없어 Claude 검토 의견을 받지 못했다. 다른 프로젝트 pane을 제어하지 않았다. 최종 일정·목표·fallback·조직 경계·원문 통합 판단은 협의 필요 결정으로 보류하고 선택지의 근거를 검토 안내에 기록했다. 과거 문구를 현재 업무 결정으로 고쳐 쓰지 않았다. 독립적으로 확인 가능한 수·경로·지표 정의와 학습 주의사항 정리를 진행했다.

## 근거·남은 미확인

문서 내부 대조 근거와 공식 자료 3건의 판본·범위·확인일은 검토 안내에 기록했다. 사내 실값·최종 승인·회사 모델/인프라·시장 규모·타사 적용·성과·배포와 원문의 미해결 질문은 미확인이다. 대회·회의·모델 훈련·실서비스를 실행하지 않았다.

## 세 단계 검증

1. **목록·고유 내용**: 원본 12개 모두 첫 제목 뒤 본문을 대조했다. 검토 callout과 의도한 앵커 복구 외에는 동일하다. 기존 frontmatter 값·고유 사례·수치 슬롯·당시 Accepted 결정·text fence 23개를 유지했다. 원본 텍스트·HTML·CSS 8개 SHA256 동일.
2. **정확성·로컬 확인**: 공식/일차 자료 3건을 대조했다. 실제 overview 유효 칸 14개와 간트 세부 행 19개를 파싱해 검산했다. 희귀 고장 1,000건 예제에서 accuracy 99%·고장 recall 0%·balanced accuracy 50%, TAT 감소 산술을 Python으로 확인했다. 실제 모델·API·사내 데이터·운영 증명은 아니다.
3. **참조·메타데이터·앱**: 기존 축약 QW 앵커 5개와 × 포함 절 앵커 1개를 복구했다. UI에서 GitHub식 slug가 문서 처음으로 이동하는 것을 발견해 원래 절 참조 14곳을 URL 인코딩한 정확한 제목 fragment로 바꿨다. Obsidian README → 검토 안내의 표·callout과 Ch.4 → outline Chapter 4 실제 절 이동을 재확인했다. Markdown 15개 고유 YAML 키·검토일·기존 related 경로와 CLI properties를 확인했다. 링크 검사 새 오류 0개; 기존 상위 규칙/CONTEXT-MAP 참조 3회는 원문대로 보존했다. `git diff --check` 통과. 일반 Markdown으로 표·내용은 읽을 수 있으나 모든 외부 렌더러의 heading fragment 호환성을 인증하지 않았다.

모든 원래 문서에 검토 결과가 있으나 남은 업무 가설·승인·Claude 협의 때문에 상태는 partial이다. 저장소 전체 완료를 뜻하지 않는다.
