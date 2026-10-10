---
tags: [my-task, ai-dt-curriculum, expert, benchmark]
category_major: "업무 기획·산출물"
category_middle: "AI/DT 교육 커리큘럼"
category_minor: "공개 커리큘럼 벤치마크"
note_kind: "조사 종합"
last_updated: 2026-10-09
---

# 공개 커리큘럼·인증 벤치마크 종합

> 세 조사자(Claude, Codex, OpenCode)가 서로의 결과를 보지 않고 같은 질문을 조사한 결과를 대조해, Expert 과정의 Builder·Guide 커리큘럼에 무엇을 반영할지 정리한 문서.

**읽는 법**: 출처 링크와 세부 표는 원본 세 문서에 있다. 이 문서는 대조 결과와 초안에 주는 의미만 담는다. "일치"는 둘 이상이 독립적으로 같은 결론을 낸 것이고, "단독"은 한 조사자만 찾아 교차 확인되지 않은 것이다. 모든 외부 자료는 2026-10-09에 확인했다.

| 원본 | 조사자 | 강점 |
|------|--------|------|
| [벤치마크 (Claude)](./public-curriculum-benchmark.md) | Claude 서브에이전트 | B1~B10 모듈 대조표, 시험 비중 비교, Kubernetes 승급 사다리 |
| [벤치마크 (Codex)](./public-curriculum-benchmark-codex.md) | Codex | 자산별 라이선스 판정, 사내 모델 이식 조건, 마켓 제출 묶음 |
| [벤치마크 (OpenCode)](./public-curriculum-benchmark-opencode.md) | OpenCode (GLM-5.3) | 2025~2026 신설 인증, 마켓 심사 사례, 보안 도구, 초안에 대한 반대 의견 |

## 1. 왜 조사했는가

Builder 96시간(B1~B10)과 Guide 24시간(G1~G4)의 모듈은 제안서 초안에서 온 값이고 근거가 없었다. 공개 커리큘럼과 비교해 빠진 것과 과한 것을 찾고, 교재를 처음부터 만들지 않아도 되는 부분을 가려내려는 것이다.

## 2. 세 조사가 일치한 것

| 결론 | 일치 | 초안에 주는 의미 |
|------|------|------------------|
| Agent 전용 Guide 인증은 없다. 확인한 인증은 모두 본인의 개발 역량만 본다 | 셋 모두 | Guide 과정은 유사 제도를 조합해 자체 설계한다 |
| 교재로 복사·각색할 수 있는 것은 Hugging Face Agents Course(Apache-2.0)가 가장 확실하다 | 셋 모두 | Builder 개념 교재의 1순위 |
| Microsoft AI Agents for Beginners(MIT), LangChain Academy 실습 저장소(MIT)도 각색 가능 | Claude, Codex | 메모리·컨텍스트·상태 실습의 원천 |
| 벤더 인증(AWS, Microsoft, Google, NVIDIA)은 출제 범위만 참고할 수 있다 | 셋 모두 | 사내 인증을 대신하지 못한다. 외부 배지 자동 인정 근거 없음 |
| 공개 과정의 축은 초안 B1→B8 순서와 같다 | 셋 모두 | 모듈을 크게 재편할 근거는 없다 |
| 메모리·컨텍스트 관리가 초안에 약하다 | Claude, Codex | B3·B4에 세션·장기 메모리, 컨텍스트 선택·압축 실습을 명시 |
| 관측(tracing)과 자동 평가를 명시해야 한다 | Codex, OpenCode | B6·B8의 수행 결과에 반영 |
| 다중 Agent는 선택 심화로 둔다 | 셋 모두 | 초안 유지 |
| B2(S/W 엔지니어링)는 공개 과정에 없고 입과 조건과 겹친다 | Claude, OpenCode | 시간을 줄일 후보 |
| 보안 비중을 높인다 | Claude, OpenCode | B7 시간 증가 또는 B4·B5에 권한 시험 포함 |
| 마켓에는 등록 후 변경 시 재심사가 필요하다 | 셋 모두 | 초안 관문은 등록 시점만 본다 |
| 마켓 등록과 안전 보증은 다르다 | Codex, OpenCode | Guide 1차 승인 + 기존 조직의 최종 심사, 2층 구조가 공개 설계와 같다 |
| Kubernetes의 Reviewer → Approver 승급이 Guide와 가장 닮았다 | Claude, OpenCode | 승인 권한 전에 실제 리뷰 이력을 요구하는 선례 |
| 초안 고유의 것(업무 문제 정의, 개인 실기, 운영 증거, 필수 탈락 기준, 인수인계)은 공개 과정에 없다 | 셋 모두 | 자체 제작 대상이자 차별점 |

## 3. 한 조사자만 찾은 것 (교차 확인 안 됨)

| 발견 | 조사자 | 쓸 곳 |
|------|--------|-------|
| Linux Foundation MCP Associate 인증. 보안·거버넌스가 출제의 24% | OpenCode | B5에서 MCP를 비교 대상이 아닌 기본 소재로 다룰 근거 |
| Microsoft AI-500(다중 Agent Expert 인증), AWS 실기형 Demonstrated 인증 | OpenCode | 실기형 검증이 확산 중이라는 근거 |
| PyRIT(MIT, 자동 레드팀 도구), Langfuse(오픈소스 관측 도구), OpenSSF Scorecard(Apache-2.0) | OpenCode | B7·B8 실습 도구와 마켓 자동 점검 후보. 사내 반입 가능 여부는 확인 필요 |
| 국내 공공 교육과 국내 대기업·제조 벤더의 공개 과정에는 Expert급 Agent 과정이 없다 | OpenCode | 국내 벤치마크 대상 없음 |
| Google Professional Agentic Architect는 필기 뒤 실습 랩으로 한 번 더 검증한다 | Claude, OpenCode | 초안의 개인 실기와 같은 구조 |
| Carpentries Instructor Trainer는 자격 보유와 활성 권한을 분리해 매년 갱신한다 | Codex | Guide 인증과 승인 권한의 활성 상태를 따로 관리할 근거 |
| Microsoft AI Champion 헌장은 보안·준수 승인을 챔피언 소관 밖으로 둔다 | Codex | 우리 Guide의 1차 승인은 별도 위임이 필요하다 |
| "OpenAI 호환"이어도 tool calling·구조화 출력·스트리밍이 모두 호환되는 것은 아니다 | Codex | 첫 실습에서 사내 모델의 호환 범위를 확인한다 |
| 공개 프레임워크는 추적 기록을 외부 서버로 보낼 수 있다 | Codex | 실습 환경에서 외부 전송을 끈다 |
| 도구마다 읽기 전용·파괴적 동작 여부를 선언하게 하는 심사 항목 | Claude, Codex | 마켓 제출 양식 |
| Carpentries는 가르치는 법에만 16시간을 쓴다 (초안 G2는 6시간) | Claude | G2 분량 검토 |

## 4. 엇갈린 것

| 항목 | 내용 | 처리 |
|------|------|------|
| DeepLearning.AI Agentic AI 과정 분량 | Claude·OpenCode는 9시간 55분, Codex는 7시간 45분 | 설계에 쓰지 않는 숫자라 미해결로 둔다 |
| NVIDIA 인증 출제 비중 | 인증 페이지와 연결된 PDF의 숫자가 다르다(Codex 확인). Claude가 읽은 값은 합이 98% | NVIDIA 비중은 비교 근거로 쓰지 않는다 |
| Kaggle·Google 2026년 과정 | Claude는 확인 못함, Codex는 후속 과정 페이지 확인 | 2025년 구성만 근거로 쓴다 |
| LangChain Academy 라이선스 | Claude·Codex는 실습 저장소 MIT 확인, OpenCode는 확인 못함 | MIT로 본다. 영상은 별도 |
| B2에서 줄인 시간을 어디에 쓸지 | OpenCode는 B5(MCP), Claude는 B7(보안) 쪽을 시사, Codex는 총량 유지하고 내용만 구체화 | 결정 필요 |

## 5. 조사자 의견이 갈린 설계 쟁점

**Guide의 리뷰·승인 역량을 교육 실습으로 확인하는 설계** (2026-10-09 확정 사항)

- OpenCode는 반대한다. 조사한 어느 제도도 실습만으로 승인 권한을 주지 않으며, 초기 심사단과 함께 실제 제출물을 채점하는 그림자 심사 3건과 첫 6개월의 2차 확인을 요구하라고 제안한다.
- Claude는 승인 권한 전 리뷰 이력으로 초안이 가볍다고 보고, 첫 승인 몇 건을 기존 승인자와 함께 하는 기간을 제안한다.
- Codex는 확정 사항을 받아들이되, 실습에서 정상 제출과 고의 결함 제출을 섞어 승인·보완·반려를 실제로 판정하게 하고, 채점 일치만으로 역량을 인정하지 말라고 한다.

세 의견의 공통점은 실습이 연습용 샘플이 아니라 실제 판정이어야 한다는 것이다.

**과정이 실패한다면 가장 유력한 이유** (OpenCode 의견)

커리큘럼 내용이 아니라 관문 운영이다. 운영 관찰 때문에 첫 기수에 "수료했지만 인증 유예"가 다수 나오고, 인증율을 본 조직이 통과 기준을 낮추라고 압박하는 경로다.

## 6. 재사용 판정 요약

| 구분 | 자료 | 조건 |
|------|------|------|
| 복사·각색 가능 | Hugging Face Agents Course, Microsoft AI Agents for Beginners, LangChain Academy 실습 저장소, Google 코드 리뷰 가이드(CC BY 3.0), Carpentries 강사 양성 교재(CC BY 4.0) | 라이선스·저작권 고지 유지, 변경 사실 표시. 영상·이미지·모델 가중치는 별도 확인 |
| 조건 확인 필요 | Anthropic courses 저장소(CC BY-NC 4.0, 2026-09-15 보관) | 사내 교육이 비상업 사용인지 확인 전에는 참조만 |
| 참조만 | Anthropic·OpenAI의 Agent 가이드, Kaggle 백서, DeepLearning.AI, Berkeley MOOC, 벤더 시험 가이드, ASQ·Kubernetes 제도 문서, 마켓 심사 정책 | 주제와 구조만 참고해 자체 문장으로 작성 |

라이선스 해석은 조사자의 판단이며 법무 검토를 거치지 않았다.

## 7. 한계

- 세 조사 모두 모듈 제목과 소개문을 근거로 대조했다. 강의 영상과 실습 노트북을 열어 깊이를 확인하지 않았다.
- 사내 모델에서 공개 실습을 실제로 돌려 보지 않았다.
- 시험 출제 비중과 교육 시간 비중은 같은 척도가 아니다. 방향만 참고한다.
- 3장의 단독 발견은 한 조사자의 확인에만 기대고 있다.

## 관련 문서

- [Expert 분과 브레인스토밍](./expert-curriculum-brainstorm.md)
- [Expert 양성 실행 제안](./expert-curriculum-proposal.md)
- [AI/DT 교육 커리큘럼 설계 목차](./README.md)
