---
type: review-log
tags: [ai-coding, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# AI 코딩 용어집 정리 기록

## 범위·분류와 보존

원래 Markdown 8개를 전체 읽고 같은 경로에 유지했다. 7개 섹션은 다른 개념군과 고유 대화를 다루므로 유지하고, 반복되는 출처·실제 적용 범위를 [검증된 적용 조건](./verified-conditions.md)으로 모았다. 모든 원래 last_updated 2026-05-05를 유지했다. 이동·삭제·실행 코드·첨부 변경과 신규 형제 주제 cross link는 없다. 기존 형제 주제 참조 10개는 유지했다.

| 원래 문서 | 개별 검토 결과 |
|---|---|
| [README](./README.md) | 7개 목차·읽기 순서 유지. 완전 번역본/현재 원문 60개 단정 대신 원래 일부 학습 노트라고 구분. 빈도 90% 미측정, 업계 평가를 저자 의견으로 구분 |
| [모델](./01-the-model.md) | 사용자 fine-tuning 가능·다양한 학습 목적/decoding 보완. 모델/실행 도구 구분. API 호환 조건·실제 요청 수·cache read/write·단가 범위 수정. 출력 비용 5배·변동 분포 단정 제거 |
| [세션·컨텍스트](./02-sessions-context-windows-turns.md) | 모델 weights와 API state 구분. history 전체 클라이언트 전송 필수 아님. 초기 지시/메모리와 압축 경계 보완. 모든 그래프 노드에 system prompt 필수·사내 역량 단정 수정 |
| [도구·환경](./03-tools-environment.md) | 도구 호출 수와 모델 요청 수 일대일 아님. MCP 2026-07-28 범위 명시. 모드는 제품별 계약, bypass와 sandbox 경계 분리. 격리만으로 모든 피해 방지한다는 보장 수정 |
| [실패 양상](./04-failure-modes.md) | 사실성/충실성은 관점이며 원인·처방 확정 아님. cutoff·사내 빈도·학습 데이터 개수 단정 수정. attention budget을 비유로 한정하고 query/head·mask·dense N² 조건 보완. 보편 100k 임계값 제거 |
| [인계](./05-handoffs.md) | clear 이후 초기 규칙이 남을 수 있음. handoff 반환 금지·compaction 항상 인메모리/새 세션이라는 일반화 수정. spec/ticket·의존성 도식 유지 |
| [메모리·조종](./06-memory-and-steering.md) | AGENTS.md 지원/로드 설정·본문 읽힌 후 비용 명시. subagent의 1단 강제 제약을 구현별 조건으로 수정. graph/API 이름과 context 격리 동치 아님. skill lazy load 비용 없음 표현 수정 |
| [작업 패턴](./07-patterns-of-work.md) | read-only 원본과 writable 리팩토링 복사본 구분. automated check가 판단 설계/다양한 결과를 가진다는 점 보완. 무인 실행의 보장 표현 수정. Brooks 책 본문의 정확한 근거는 미확인 표시 |

## 근거·협의·중요 결정

확인일은 2026-10-04다. 대표 조건 표에 원래 사전 main과 공식 제품/논문 18건을 연결했다. 사전 main에 추가 용어가 있지만 원래 목차를 최신 완전 번역으로 새로 만들지는 않았다. 저자의 용어 선택과 실무 비유를 학습 맥락으로 보존하고 실제 제품 계약은 공식 문서에 한정한다. 요금·모델 추천을 임의 최신 수치로 대체하지 않는다.

HERDR_ENV=1에서 `herdr pane current --current`가 pane_not_found였다. 전용 Claude 연결을 확보하지 못했고 다른 pane은 제어하지 않았다. 실제 Claude 의견은 없다. 7개 섹션 재합병, 저자 분류를 다른 체계로 바꾸기, 추가 용어 전면 번역과 정확한 Brooks 인용 확정은 협의/근거 대기로 보류했다. 공개 자료로 확인한 기술적 조건 수정은 진행했다.

## 세 차례 검증

1. **원문·목록 대조:** 원문 snapshot 8개 경로·절 제목과 현재 문서를 대조한다. 각 고유 용어·대화·인계/협업 도식은 유지했고 잘못된 진단/무인 예시는 조건을 수정했다. metadata/근거/기록을 포함한 현재 문서는 10개다.
2. **근거·예제:** 공식 자료의 API state·여러 tool calls·cache write/read·AGENTS.md·subagent 깊이를 대조했다. 실행 언어 예제는 없고 text 도식만 있다. attention softmax 합 1이 관련 key 가중치의 자동 균등 희석을 뜻하지 않음을 로컬 수치로 확인한다. 이는 실제 모델 성능이나 모든 원격 API의 재현성 검증이 아니다.
3. **참조·metadata·Obsidian:** 짧은 영문 fragment를 실제 한영 절의 anchor로 갱신했다. YAML·링크 검사와 CLI properties/읽기 화면 결과는 최종 확인에 남긴다.

## 남은 미확인

사내 사용 사례·빈도·역량·비용 절감률, 특정 계정의 모델/요금·실행 정책·메모리 로딩, 전체 원문 추가 용어 번역, Brooks 책 인용 절, 실제 Claude 협의는 미확인이다. 설명용 대화는 로컬 관측 자료로 취급하지 않는다.

## 최종 확인

원래 8개 문서의 절 제목 111개가 모두 남아 있고 text fence 7개도 유지했다. YAML 10개 중복 key·검토일, 원래 last_updated 보존이 통과했다. 로컬 softmax 수치의 두 경우 합은 1이고 관련 key 가중치는 동일하여 “추가 토큰마다 필연적으로 균등 희석”이라는 설명을 일반 정리로 쓸 수 없음을 확인했다. 실제 모델 품질 실험은 아니다. ai-dt 원래 non-Markdown 18개 SHA256이 초기 snapshot과 같다.

링크 검사 신규 문제는 0개다. 기존 형제 주제 참조 10개는 대상이 존재하지만 독립 주제 경계를 넘는 기존 참조로 기록했고 추가하지 않았다. Obsidian pm_notes에서 현재 10개 properties가 통과했다. CUA 첫 bundle 연결은 cgWindowNotFound였으나 실행 앱 목록을 확인하고 Obsidian 이름으로 재연결했다. 읽기 화면에서 README의 원래 날짜·검토 날짜·callout·7개 목차 표를 확인하고 적용 조건 링크를 눌러 올바른 ai-dt/ai-coding-dictionary 경로, 한국어 alias와 18개 출처 표가 노출됨을 확인했다. 모든 아래쪽 화면이나 모든 fragment 클릭을 검증한 것은 아니다.

최종 기록 작성 후 YAML·원문/절/첨부 보존·링크·diff 공백 검사를 다시 실행한다. 커밋·push는 하지 않았다.


## 추가 참조 재검증 — 2026-10-05

8개문서에추가했던GitHub slug형절참조292개를같은주제노트경로로복구하고표시용어/절이름을유지했다. 앞선Obsidian1.13.7검사에서slug절이동이실패했으므로파일존재검사만으로그참조를통과시키지않았다. renderer별앵커병기/HTML앵커보장대신노트+절이름참조를선택했다. Claude연결불가로추가설계협의는없으며새플러그인이나주제간링크는추가하지않았다. 원래절제목·text도식과고유예제/기술주장은이번복구에서동일하다. 일반Markdown/Obsidian에서는표시용어를본문에서찾아읽는다. 메타데이터/참조/CLI를재검사하며전체읽기/실제품실행은미완료다.
