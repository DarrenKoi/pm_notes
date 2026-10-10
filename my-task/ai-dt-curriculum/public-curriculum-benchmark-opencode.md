---
tags: [my-task, ai-dt-curriculum, expert, benchmark]
category_major: "업무 기획·산출물"
category_middle: "AI/DT 교육 커리큘럼"
category_minor: "Expert 분과"
note_kind: "공개 자료 벤치마크 조사"
last_updated: 2026-10-09
---

# 공개 커리큘럼·인증 독립 벤치마크 (opencode)

> Expert 분과의 Builder 교육(B1~B10, 12주 96시간), Guide 교육·승급(G1~G4, 24시간 + 실적), 사내 Agent·tool 마켓 1차 승인 설계를 공개 자료와 벤치마크한 독립 조사 결과와 의견.

**조사 방법·범위**

- 독립 조사다. 같은 폴더의 다른 벤치마크 문서(`public-curriculum-benchmark*.md`)는 읽지도 참고하지도 않았다.
- 1차 출처만 사용했다: 제공자 공식 페이지, 공식 저장소, 공식 시험 가이드, LICENSE. 블로그·언론·제3자 요약은 제외했다.
- 출처에 없는 숫자는 추정하지 않고 '명시 없음', 접근·확인 실패는 '확인 못함'으로 표기했다.
- 모든 외부 링크의 접속일은 2026-10-09다. 조사는 초안(제안서 6·7·8장, 브레인스토밍 1장)을 기준으로 했다.

## 1. 결론 먼저

1. **Agent 개발 Expert급 공개 인증이 2025~2026년에 집중 신설됐고, 그 축이 초안 B1~B10과 일치한다.** AWS AIP-C01(Generative AI Developer – Professional), Microsoft AI-500(Multi-Agent AI Solutions Expert), Google PAA(Professional Agentic Architect), Linux Foundation MCPA가 모두 최근 신설이며 공통 구조는 '설계·구현 → 평가 → 보안·거버넌스 → 배포·운영(+실습 검증)'이다. 모듈 대폭 재편의 근거는 못 된다.
2. **벤더 자격증은 실습·응시가 사내 환경에서 불가능하다.** Bedrock·Azure Foundry·Vertex에 종속된다. 다만 AWS는 대상 후보에 "open-source technologies" 경력을, MS AI-500은 MCP·LangGraph·RAG 이해를 명시하므로, 시험 영역 구조만 벤치마크할 자료다.
3. **사내 환경(오픈소스 모델 + OpenAI 호환 API)에서 직접 재사용 가능한 실무 교재는 소수다.** Hugging Face Agents Course(Apache-2.0), LlamaIndex local-models 튜토리얼, LangChain Academy(과정 무료), LF MCPA 대비 공식 문서(로컬 모델 Ollama 경험 명시). Guide용 리뷰 교재로는 Google eng-practices(CC BY 3.0).
4. **'Agent 전용 Guide 인증'은 공개적으로 확인되지 않는다.** 가장 근접한 원천은 Kubernetes의 reviewer→approver 승격(정량 리뷰 실적 + 스폰서), GDE/MVP의 연 1회 활동 리뷰, CNCF 졸업 심사의 '독립 adopter 3곳' 실사용 검증이다.
5. **마켓 1차 심사의 가장 가까운 공식 원천은 MCP 공식 레지스트리다.** 레지스트리는 private 서버를 지원하지 않고 "자체 private 레지스트리 구축"을 공식 권장한다 — 사내 마켓과 같은 위치다. 심사 항목은 MS의 AI 앱·에이전트 인증(보안 문서·시크릿 관리·최소 스코프), Slack 심사(최소 권한·설치 실측·변경 시 재심사), OpenSSF Scorecard(Apache-2.0)의 조합으로 설계할 수 있다.
6. **국내·제조업 공개 과정은 벤치마크 원천이 못 된다.** 국내 공공 AI 교육은 전 국민 리터러시 대상이고, 국내 대기업과 제조 벤더의 Expert급 공개 과정은 1차 출처로 확인하지 못했다(8장).
7. **의견(7장 요약)**: 96시간·24시간은 유지. B2에서 2시간을 B5(MCP)로 이동하고 B7에 자동화 레드팀 실습을 추가하라. Guide 승급에서 '리뷰·승인 역량 = 교육 실습 12시간'인 설계에는 반대하며 실제 심사 참여(그림자 채점) 실적을 요구해야 한다. 실패의 최 유력 원인은 커리큘럼 내용이 아니라 '수료↔인증 지연 → 통과 기준 하향 압박'이다.

## 2. 조사 결과 — Builder 벤치마크

### 2.1 2026-10-09 기준 신설·개편 흐름

| 자격증·과정 | 상태 | 공식 명시 사항 |
|---|---|---|
| AWS Certified Generative AI Developer – Professional (AIP-C01) | 시행 중 | 180분·75문항·300 USD, 시험 언어에 한국어 포함, 대상 "2년 이상 클라우드(AWS 또는 **open-source technologies**) + 생성형 AI 실무 1년" ([인증 페이지](https://aws.amazon.com/certification/certified-generative-ai-developer-professional/), [시험 가이드](https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01.html)) |
| AWS "Demonstrated" 실기형 크리덴셜 | 시행 중 | "timed, hands-on challenges"로 실무 준비도를 검증하는 형태. [AWS Agentic AI Demonstrated](https://skillbuilder.aws/learn/GTGKXBWUGU/aws-agentic-ai-demonstrated/SJK9ZKVCYU), [AWS Securing Agent Identities Demonstrated](https://skillbuilder.aws/learn/Z8M87YZWXB/aws-securing-agent-identities-demonstrated/9ZRC3U51YT) 등 — 시간제 실기 인증의 선례 (동일 페이지) |
| Microsoft Multi-Agent AI Solutions Expert (AI-500) | 시행 중 | 선행 자격 AI-103 필수. 평가 영역 4개(Architect / Develop in Azure / **Evaluate, optimize, and monitor** / **Secure, govern, and deploy**). "Microsoft Agent Framework, **MCP**, RAG, **LangGraph**" 이해 명시 ([자격증 페이지](https://learn.microsoft.com/en-us/credentials/certifications/multi-agent-ai-solutions-expert/), [시험 AI-500](https://learn.microsoft.com/en-us/credentials/certifications/exams/ai-500/)) |
| Azure AI Apps and Agents Developer Associate (AI-103) | AI-102 후속 | AI-102는 "This certification and the renewal assessment are retired"로 단종 표기 ([AI-102 페이지](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/)). AI-500이 AI-103을 선행으로 지정(위 링크) |
| Google Cloud Professional Agentic Architect (PAA) | beta 종료, GA 등록 11월 2일 개시 | beta 3시간·$120(정가 $200)·유효 1년·"3년 이상 클라우드 + 1년 이상 agentic(구글 클라우드)", 객관식 + 실습 랩(PAA labs) 이중 구조 ([인증 페이지](https://cloud.google.com/learn/certification/agentic-architect), [시험 가이드 PDF](https://services.google.com/fh/files/misc/professional_agentic_architect_exam_guide_english.pdf)) |
| Linux Foundation MCP Associate (MCPA) | 시행 중 | 90분·$250·유효 2년·초급. 도메인: Fundamentals 16% / Architecture 14% / Interactions & Execution 26% / **Security & Governance 24%** / Use Cases 20%. 권장 경험에 "local models via **Ollama**" 명시 ([인증 페이지](https://training.linuxfoundation.org/certification/model-context-protocol-associate-mcpa/)) |

의미: (a) 'agentic solutions 구현·평가·보안·운영'을 하나의 인증으로 묶는 방향이 업계 표준이 됐고 (b) 실습 랩·실기 챌린지형 검증(PAA labs, AWS Demonstrated)이 확산 중이며 (c) MCP가 벤더 중립 인증의 표준 소재로 등극했다.

### 2.2 Builder 비교표

| 자료 | 형태 | 시간·비용(공식 명시분) | 다루는 축 | 벤더 제거 후 사내 사용 |
|---|---|---|---|---|
| [HF Agents Course](https://huggingface.co/learn/agents-course) + [저장소](https://github.com/huggingface/agents-course) | 오픈소스 과정 | 무료. 총 학습 시간 명시 없음. **Apache-2.0** | 에이전트 기초 → 프레임워크(smolagents·LangGraph·LlamaIndex) → Agentic RAG → 최종 프로젝트(**자동 평가 + 리더보드**), 보너스: function-calling 파인튜닝, **관측·평가** | 높음 — 프레임워크가 OpenAI 호환 엔드포인트 지원. B1~B7·B10 실습 기반 |
| [LangChain Academy](https://academy.langchain.com/) | 오픈소스 계열 과정 | "Our course content is always free!"(공식 FAQ). 인증시험([LCAE](https://academy.langchain.com/pages/certifications-lcae))은 유료, 가격 명시 없음 | 카테고리가 **Build / Test / Deploy / Monitor** 4축 | 높음 — LangSmith(SaaS) 의존 과정은 개념 학습만, 사내 대체(오픈소스 관측 도구)로 치환 필요 |
| [LlamaIndex 공식 문서](https://developers.llamaindex.ai/python/framework/) | 공식 튜토리얼 | 무료 | "starter tutorial… **the local-models tutorial does the same without any hosted API**" — 호스팅 API 없는 로컬 모델 실습 경로가 공식 문서로 존재 | 높음 — B4·B5 실습 교재 |
| [LF MCPA](https://training.linuxfoundation.org/certification/model-context-protocol-associate-mcpa/) | 벤더 중립 인증 | 90분·$250·유효 2년 | MCP 전 도메인(보안·거버넌스 24%) | 높음(교재·영역 구조) — 시험 응시는 방화벽과 무관하나 실습은 로컬 모델로 가능 |
| [DeepLearning.AI](https://www.deeplearning.ai/courses/) | 단기 강좌 | 영상 무료, 수료증은 Pro(공식 FAQ 기준 $25~30/월). [Agentic AI 9h55m](https://www.deeplearning.ai/courses/agentic-ai), [Evaluating AI Agents 2h36m](https://www.deeplearning.ai/courses/evaluating-ai-agents), [MCP 1h58m](https://www.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic), [vLLM 1h38m](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm) | 개별 시간이 강좌 페이지에 명시 — 모듈당 분량 설계의 참고치 | 중간 — vLLM·평가 방법론 강좌는 사내 환경 직접 호환, OpenAI API 실습 강좌는 이식 필요 |
| [OpenAI Academy](https://academy.openai.com/) | 무료 아카데미 | 무료. 시간 명시 없음 | Codex·OpenAI API 중심 빌더 트랙 | 중간 — API 패턴이 사내 OpenAI 호환 스택과 동형 |
| [Anthropic Academy](https://academy.claude.com/) | 무료 아카데미 | 무료. AI Fluency 4시간, capabilities 3.5시간, human-agent teams(beta) 45분(과정 페이지 명시) | 4D 프레임워크, LLM 한계 이해, 사람-에이전트 협업 | 중간 — B3(LLM·컨텍스트 한계) 교재로 모델 불가지론적 |
| [AWS AIP-C01](https://aws.amazon.com/certification/certified-generative-ai-developer-professional/) | 자격증 | 180분·75문항·300 USD | 시험 가이드 5도메인: FM 통합·데이터·컴플라이언스 31% / 구현·통합 26% / 안전·보안·거버넌스 20% / 운영 효율 12% / 테스트·검증 11% ([시험 가이드](https://docs.aws.amazon.com/pdfs/aws-certification/latest/ai-professional-01/ai-professional-01.pdf)) | 낮음(응시) — **도메인 구조만 벤치마크**. B5~B8 영역 구성과 거의 1:1 |
| [AWS AIF-C01](https://aws.amazon.com/certification/certified-ai-practitioner/) | 입문 자격증 | 90분·65문항(채점 50)·100 USD·유효 3년 | "uses, but does not necessarily build" — 빌더 대상 아님 명시 | 낮음 — 입과 진단 게이팅 참고 |
| [MS AI-103](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-apps-and-agents-developer-associate/) | 자격증 | 가격·문항수 확인 못함 | generative AI·agentic 솔루션 구현 | 중간(개념) — Foundry 종속 |
| [MS AI-500](https://learn.microsoft.com/en-us/credentials/certifications/multi-agent-ai-solutions-expert/) | Expert 자격증 | 가격 "지역별 상이"(명시 수치 없음) | 멀티 에이전트 설계→구현→평가·관측→보안·거버넌스·배포. [스터디 가이드](https://aka.ms/AI500-StudyGuide) | 중간 — Expert급 평가·운영 영역 구성의 레퍼런스 |
| [MS AI-300](https://learn.microsoft.com/en-us/credentials/certifications/operationalizing-machine-learning-and-generative-ai-solutions/) | 자격증 | 확인 못함 | MLOps + **GenAIOps**(평가·관측·최적화) — [스터디 가이드](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-300)의 영역 구성이 B6·B8 참고 | 중간 |
| [Google PAA](https://cloud.google.com/learn/certification/agentic-architect) | Professional 자격증 | beta 3시간·$120 / GA $200·유효 1년 | low-code → 코딩 에이전트 → 커스텀 에이전트 → 평가·배포 → 보안·거버넌스 | 낮음 — '평가·배포·거버넌스까지 포함한 agentic 전문가 인증'이라는 선례만 |

### 2.3 B1~B10 대조 요약

- **일치**: 공개 인증들의 도메인(AIP-C01 5영역, AI-500 4영역, PAA 5능력)은 초안의 B1→B8 순서(정의→구현→연결→평가→보안→운영)와 같은 축이다. 캡스톤·실습 랩(PAA labs, AWS Demonstrated)도 B10·개인 실기와 같은 발상이다.
- **공개 인증이 다루는데 초안에 명시가 얇은 것**: MCP 표준(LF MCPA가 도메인의 74%를 MCP 자체에 배분, MS AI-500이 MCP 명시), 관측 가능성(observability·tracing — HF 보너스 유닛, LangChain Academy Monitor 축, AI-500 "monitor"), 자동 평가 파이프라인(HF Unit 4의 자동 평가 + 리더보드).
- **초안에 있는데 공개 과정이 거의 다루지 않는 것**(자체 제작 유지): 업무 문제 정의·Agent 필요성 판단(B1), 장애·복구·인수인계·운영 관찰(B8), AI 도구 사용 설명(B10·실기 정책) — 벤더 과정은 '동작하는 데모'까지만 다루는 경우가 많다.

## 3. 조사 결과 — Guide 벤치마크

### 3.1 Agent 전용 Guide 인증

'Agent 리뷰·승인을 전담하는 공개 인증'은 이번 조사 범위에서 확인하지 못했다. Guide 설계는 인접 제도의 조합으로 참고할 수밖에 없다.

### 3.2 인접 제도 비교표

| 제도 | 승격·선발 요건(공식 명시분) | 갱신·유지 | Guide 이식 포인트 |
|---|---|---|---|
| [Google Developer Experts](https://developers.google.com/community/experts) ([FAQ](https://developers.google.com/community/experts/faq)) | 전문성 + 커뮤니티 기여(멘토링 포함) + 커뮤니케이션. "ideally, at least 2 years" 활동. **현직 GDE 또는 Google 직원의 추천(nomination)** 필요 | **연 1회 활동 리뷰**, 활동 유지가 연장 조건. 탈락 후 재지원 6개월 대기 | 추천제 + 연간 활동 리뷰. 단 리뷰어·승인 권한이 없는 '영향력' 기반 제도 |
| [Microsoft MVP](https://mvp.microsoft.com/en-us/mvp/overview) ([FAQ](https://mvp.microsoft.com/en-us/faq?section=mvp)) | **최근 12개월** 기여만 심사. 4개 평가 영역(기술 전문성/커뮤니티 리더십/제품 피드백/커뮤니티 건전성). 기여 유형 11종에 **Mentorship/Coaching 포함**. 기여마다 **검증 가능 URL + 영향 지표** 필수 | 연 1회 단일 어워드 사이클([공식 아카이브](https://learn.microsoft.com/en-us/archive/blogs/mvpawardprogram/microsoft-mvp-award-evolution)). 미갱신 후 45일 대기 재추천 | '12개월 실적 + 검증 URL + 정량 지표'를 심사 양식에 내장하는 방식 |
| [Kubernetes 커뮤니티 멤버십](https://github.com/kubernetes/community/blob/master/community-membership.md) | member→**reviewer**: member 3개월 + 주 리뷰어로 5개 PR + substantial PR 20개 리뷰/병합 + 스폰서. reviewer→**approver**: 3개월 + PR 10개 + 리뷰/병합 30개 + owner 지명. approver 의무에 "**Mentor contributors and reviewers**" 명시 | 12개월 무기여 시 제거 후 재가입 | **정량 리뷰 실적 + 상급자 스폰서 + 이의 없음** 구조가 Guide 승급과 동일 패턴. [OWNERS 파일](https://github.com/kubernetes/community/blob/main/contributors/guide/owners.md)처럼 승인 범위를 영역별로 명시 |
| [Apache 성숙도 모델](https://community.apache.org/apache-way/apache-project-maturity-model.html) | 프로젝트 자기평가 8범주(Code/License/Release/Quality/**Community/Consensus**/…), 항목마다 고유 ID(CO40 "meritocratic", CO50 "권한 획득 방법의 문서화·일관 적용"). 단계형 부분 준수 수준은 정의하지 않음("complies with all the elements") | 해당 없음(프로젝트 단위). 개인 승급은 [contributor-ladder](https://community.apache.org/contributor-ladder.html) (초대제, 정량 기준 명시 없음) | 평가표 항목에 고유 ID를 부여하는 방식. '권한 획득 방법의 문서화' 원칙 |
| [Google eng-practices](https://github.com/google/eng-practices/) | 코드 리뷰 표준: 리뷰어가 볼 것 8종(Design/Functionality/Complexity/Tests/Naming/Comments/Style/Documentation). **"One business day is the maximum time it should take to respond"**([speed](https://google.github.io/eng-practices/review/reviewer/speed.html)). 페어 프로그래밍한 코드는 리뷰된 것으로 인정 | — | **CC BY 3.0**([LICENSE](https://github.com/google/eng-practices/blob/master/LICENSE)) — 사내 G1 리뷰 교재로 출처 표기 재사용 가능. 리뷰 SLA 규칙도 그대로 채택 가능 |
| [CNCF 졸업 기준](https://github.com/cncf/toc/blob/main/.github/ISSUE_TEMPLATE/template-graduation-application.md) | Sandbox→Incubating→Graduated 3단계([안내](https://www.cncf.io/projects/)). 졸업에 **독립 production adopter 3곳 + adopter 인터뷰 5~7건**, 유지보수자 **2개 이상 조직 출신**, contributor ladder 필수, **OpenSSF Best Practices passing 배지 필수** | — | '실사용 3건의 독립 검증'을 Guide 실적·마켓 등록 심사에 이식 가능 |

## 4. 조사 결과 — 사내 마켓 1차 심사 기준

### 4.1 MCP 공식 레지스트리 — 사내 마켓과 같은 위치의 공식 사례

[MCP Registry](https://modelcontextprotocol.io/registry/about)(preview 상태)의 공식 설계:

- **이름공간 인증**: 역방향 DNS 이름(`io.github.user/server`는 GitHub 계정 연계, `com.example/server`는 DNS(TXT)/HTTP 챌린지 검증)으로 소유권을 증명한다.
- **메타데이터 표준**: 표준화된 [`server.json`](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/draft/server.schema.json)(이름·패키지 위치·실행 방법·설명)으로 등록한다.
- **private 서버 미지원 → "자체 private 레지스트리를 구축해 등록하라"고 공식 권장.** 공식 코드베이스는 셀프호스팅용 설계가 아니라고 명시한다 — 사내 마켓은 이 안내가 전제하는 위치 그 자체다.
- **심사는 'unopinionated' 원칙**: 레지스트리는 메타데이터만 호스팅하고, 큐레이션·평점·심사는 downstream aggregators가 담당한다. 보안 스캔도 패키지 레지스트리·aggregator에 위임한다([Package Types](https://modelcontextprotocol.io/registry/package-types), [Moderation Policy](https://modelcontextprotocol.io/registry/moderation-policy)).

의미: '중앙 레지스트리는 소유권·메타데이터만 검증하고 품질·보안 심사는 별도 관문이 담당한다'는 2층 구조가 공식 표준 설계다. 초안의 'Guide 1차 승인 + 기존 조직 절차의 최종 심사' 2층 구조와 정확히 같은 골격이다.

### 4.2 상업 마켓플레이스 심사 항목

| 출처 | 심사·점검 항목(공식 명시분) | 사내 이식 |
|---|---|---|
| [MS AI 앱·에이전트 인증](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/artificial-intelligence-app-agent-certification) | **데이터 흐름 다이어그램·시크릿 관리 방식·통합 지점별 접근 범위 정의를 포함한 보안 문서 제출 필수**. 1~2페이지 보안 백서 권장. 출시 후 정책 준수의 지속 모니터링 요구 | 높음 — Agent 제품 심사의 직접 원천 |
| [MS 일반 인증 절차](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/review-publish-offer) | 3단 검증: 게시자 적격성 → 콘텐츠 검증 → 기술 검증(멀웨어 스캔·네트워크 호출 모니터링·패키지 분석·기능 전수 테스트). 실패 보고서 + 재제출 무제한 ([정책](https://learn.microsoft.com/en-us/legal/marketplace/certification-policies)) | 높음 — 심사 단계·상태 설계 |
| [AWS Marketplace 판매자·제품 심사](https://docs.aws.amazon.com/marketplace/latest/userguide/seller-eligibility.html) | 무료 제품도 "production-ready, 전체 기능", "정의된 고객 지원 프로세스", "취약점 없이 유지할 수단" 요구. 제출물은 정책·보안 컴플라이언스·취약점·사용성 관점에서 검토되고, 판매자가 최종 승인해야 게재되는 [상태 머신](https://docs.aws.amazon.com/marketplace/latest/userguide/product-submission.html) | 높음 — 제출→리뷰→승인 절차 골격 |
| [Slack Marketplace 심사](https://docs.slack.dev/slack-marketplace/slack-marketplace-app-guidelines-and-requirements) | **설치 후 기능 테스트**, TLS 1.2+, **요청하는 데이터 접근이 기능에 필요한 최소 범위인지 검토**, 개인정보 처리방침 필수, 실질 변경 시 재심사 + 상시 감사([절차](https://api.slack.com/start/distributing/directory)) | 높음 — '최소 스코프 + 실측 + 변경 시 재심사' |
| [GitHub Marketplace](https://docs.github.com/en/apps/github-marketplace/creating-apps-for-github-marketplace/requirements-for-listing-an-app) | 공개 사용 가능 상태, [퍼블리셔 검증](https://docs.github.com/en/apps/github-marketplace/github-marketplace-overview/applying-for-publisher-verification-for-your-organization)(조직 2FA·도메인 검증), 확정 보안 사고 24시간 내 통보([보안 모범사례](https://docs.github.com/en/apps/github-marketplace/creating-apps-for-github-marketplace/security-best-practices-for-apps-on-github-marketplace)) | 중간~높음 — 제출자 신원·소속 확인 |
| [VS Code Marketplace](https://code.visualstudio.com/api/working-with-extensions/publishing-extension) | 퍼블리셔 검증(6개월 이력·도메인 소유), 난독화 코드·시크릿 스캔 지적([공식 블로그](https://developer.microsoft.com/blog/security-and-trust-in-visual-studio-marketplace)) | 중간 — 난독화 금지·시크릿 스캔 |
| [npm Trusted Publishing](https://docs.npmjs.com/trusted-publishers/) / [PyPI](https://docs.pypi.org/trusted-publishers/) | CI/CD의 OIDC 단기 토큰으로만 배포, provenance(빌드 출처 증명 서명) | 높음 — '승인된 파이프라인에서 서명·출처 증명'을 심사 항목화 |
| [OpenSSF Scorecard](https://github.com/ossf/scorecard) ([checks](https://github.com/ossf/scorecard/blob/main/docs/checks.md)) | 자동 점검: Code-Review, Token-Permissions, Pinned-Dependencies, SAST, Signed-Releases, License 등 0~10점. **Apache-2.0**([LICENSE](https://raw.githubusercontent.com/ossf/scorecard/main/LICENSE)) | 높음 — 사내 Agent 저장소 자동 점검표. 일부 검사는 GitHub 전용이므로 사내 포지토리 관리체계 대응 확인 필요 |

### 4.3 리스크·보안 기준(심사 항목의 원천)

- [OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/)([공식 저장소](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10), CC BY-SA 4.0): 프롬프트 주입·민감 정보 노출 등 위험 목록 → B7·1차 심사 질문화.
- [NIST AI 600-1(생성형 AI 프로파일)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)([AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), 무료 PDF): GOVERN/MAP/MEASURE/MANAGE → 조직 수준 정렬용.
- [MITRE ATLAS](https://atlas.mitre.org/)([데이터 저장소](https://github.com/mitre-atlas/atlas-data)): AI 공격 시나리오 지식베이스 → 위협 모델링 참조.
- [ISO/IEC 42001](https://www.iso.org/standard/42001): 본문이 유료·비공개라 심사 항목의 직접 원천으로는 못 씀(공개 샘플만 열람).
- [EU AI Act GPAI 행동강령](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai): 모델 문서 양식 템플릿 중심 — 사내 '모델·도구 정보 카드' 양식 참고.

## 5. 놓치기 쉬운 자료 (국내·아시아·제조업·오픈소스 모델·평가·보안)

흔히 알려진 벤더 자격증 외에, 다른 조사자가 놓쳤을 만한 후보를 확인한 결과다.

- **국내 공공**: [AI디지털배움터](https://www.xn--2z1bw8k1pjz5ccumkb.kr/main.do)(과기정통부·[NIA 운영](https://www.nia.or.kr/site/nia_kor/ex/bbs/ListBusiness.do?businessMnCd=23000400))는 **전 국민 디지털 리터러시 대상 무료 교육**이다(AI 활용 교육 강화 명시). Expert급 개발자 과정이 아니다. [NIA STEP](https://nia.step.or.kr/)는 스마트 직업훈련 플랫폼이나 Agent 개발 Expert 과정은 확인 못함.
- **국내 기업(SDS·SK·LG CNS·현대차그룹 등)·제조 벤더(Siemens·Festo·Rockwell 등)**: 공개된 1차 출처(공식 교육 카탈로그)에서 Agent 개발 Expert급 과정을 **확인하지 못했다.** 사내 교육은 비공개로 추정되며, 벤치마크 원천으로 못 쓰는 것이 이번 조사의 결론이다.
- **오픈소스 모델 기반(사내 환경 최적)**: [meta-llama/llama-cookbook](https://github.com/meta-llama/llama-cookbook)(Meta 공식 학습 자료), [vLLM 강좌](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm)(오픈소스 서빙, 1h38m), [Langfuse](https://langfuse.com)(오픈소스 LLM 관측 도구, 셀프호스팅 문서 공개 — LangSmith 대체재 후보).
- **평가·보안 전문**: [PyRIT](https://github.com/microsoft/PyRIT)(**MIT**, 생성형 AI 레드팀 자동화 프레임워크 — 구 [Azure/PyRIT](https://github.com/Azure/PyRIT)는 2026-03-27 아카이브 확인, 저장소 이동 공지 포함), [HF Agents Course 보너스 유닛](https://huggingface.co/learn/agents-course/bonus-unit2/introduction)(관측·평가), [OpenAI Evals] 확인 못함(공식 저장소 이번 조사에서 미확인), [MS AI 앱·에이전트 인증 정책](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/artificial-intelligence-app-agent-certification)(Agent 심사 기준의 공개 실례 — 4.2표).
- **AWS Demonstrated 크리덴셜**(2.1표): 벤더 자격증 논의에서 잘 빠지는 '실기형(hands-on) 인증' 선례다.

## 6. 재사용 가능 여부 정리

판정: **직접 실습재**(사내 오픈소스 모델 환경에서 그대로 실습 가능) / **교재 준용**(출처 표기·변형 후 사용) / **구조 참고**(영역·절차 설계만 참고) / **참고만**.

| 자료 | 라이선스(확인분) | 사내 사용 판정 |
|---|---|---|
| HF Agents Course | Apache-2.0(저장소 명시) | 직접 실습재 (B1~B7, B10) |
| LlamaIndex local-models 튜토리얼 | 확인 못함(프레임워크는 오픈소스) | 직접 실습재 (B4·B5) |
| LangChain Academy | 확인 못함(과정 무료 공식 명시) | 직접 실습재(SaaS 과정은 개념 학습) |
| MCP 공식 문서·스펙(2026-07-28) | 확인 못함 | 직접 실습재 (B5) |
| Google eng-practices | **CC BY 3.0** | 교재 준용 (G1) |
| OpenSSF Scorecard | **Apache-2.0** | 교재 준용 → 자동 점검 도구 (심사) |
| OWASP LLM Top 10 | CC BY-SA 4.0(저장소 표기) | 교재 준용 (B7, 심사 질문) |
| NIST AI 600-1 | 공개 PDF(무료 배포) | 구조 참고 (조직 정렬) |
| Kubernetes community-membership 문서 | 확인 못함(공개 저장소) | 구조 참고 (Guide 승격) |
| CNCF 졸업 신청 템플릿 | 확인 못함(공개 저장소) | 구조 참고 (실적 검증) |
| MCP 레지스트리 설계(server.json·이름공간·2층 심사) | 확인 못함 | 구조 참고 (마켓 아키텍처) |
| AWS/MS/Google 자격증 영역 구성 | 해당 없음 | 구조 참고 (모듈·평가표) |
| DeepLearning.AI 강좌 | 영상 무료, 재배포 조항 확인 못함 | 참고만(내부 시청·분량 참조) |
| Anthropic/OpenAI Academy | 무료(재사용 조항 확인 못함) | 참고만 |
| ISO/IEC 42001 | 유료·비공개 | 참고만 |
| PyRIT | **MIT** | 직접 실습재 (B7 레드팀 실습) |
| 국내 공공 교육(AI디지털배움터 등) | 확인 못함 | 참고만(리터러시 분과용) |

## 7. 의견 — 조사 사실과 구분한 독립 판단

이 장은 내 판단이며 2~6장의 출처 사실과 구분된다.

### 7.1 B1~B10·G1~G4의 추가·삭제·시간 조정

- **추가 (1) MCP를 B5의 핵심 소재로 승격하라.** LF MCPA가 벤더 중립 인증 전체를 MCP에 배정하고, MS AI-500이 MCP·LangGraph를 명시하며, AWS 시험 가이드도 agentic solutions을 다룬다(2.1). 2026-10 기준 MCP는 '비교 대상'이 아니라 기본 소양이다. B5(8시간)에 server.json 계약·이름공간/소유권·신뢰 경계를 포함하고, 시간은 아래 조정으로 충당하라.
- **추가 (2) B6·B8에 '관측(트레이싱)과 자동 평가'를 명시하라.** HF 보너스 유닛, LangChain Academy의 Test/Monitor 축, MS AI-500의 "Evaluate, optimize, and monitor"가 같은 자리를 가리킨다. 초안 내용과 충돌하지 않지만 '평가 세트'만으로는 2026년 기준 얇다.
- **추가 (3) B7에 OWASP LLM Top 10 질문화 + 자동화 레드팀(PyRIT) 실습을 넣어라.** 현재 '권한 우회·기밀 노출·승인 누락 시험'은 방향이 맞으나, 수작업 시험만으로는 심사 재현성이 약하다. PyRIT(MIT)는 사내 오픈소스 환경에서 직접 돌릴 수 있다.
- **시간 조정: B2 12 → 10시간, B5 8 → 10시간.** 입과자가 이미 코딩하고 네트워크·클라우드를 이해한다는 확정 전제(브레인스토밍 1장)와 B2가 '기본기 재교육이 아니다'라는 초안의 원칙을 함께 놓으면 12시간은 과잉이고, MCP를 반영할 B5가 과소다. 파일럿 실측 후 되돌릴 수 있는 2시간 이동이다.
- **삭제: 없다.** 대신 **반대 하나** — '다중 Agent를 필수로 승격하자'는 유혹에 반대한다. MS AI-500은 다중 Agent 전문 인증이지만 연계 인증(AI-103)을 선행으로 요구하는 상위 Expert용이고, 초안이 다중 Agent를 선택 심화(B9)에 둔 판단이 옳다.
- **G1~G4: 시간(6×4)은 유지하되 실습 소재를 바꿔라.** G1은 eng-practices 8항목 + 리뷰 SLA(1영업일)를 교재로, G3은 '연습용 샘플'이 아니라 **초기 심사단의 과거 제출물로 그림자 채점**을 하고, G4에 OWNERS 파일식 '승인 범위의 문서화'를 포함하라.

### 7.2 96시간·24시간은 적절한가

- **96시간: 유지하되 근거는 내부 파일럿이어야 한다.** 조사한 공개 과정·자격증 어디에도 '총 교육 시간'이 명시돼 있지 않다(HF 명시 없음, 자격증은 시험 시간만 명시). 즉 96시간을 외부 벤치마크로 정당화하거나 반박할 수 없다. 다만 도메인 구성과 심사 중심(시험 90~180분 + 실습 랩) 설계가 초안과 같은 방향이므로, 96시간은 '업무 병행 12주·주 8시간'이라는 운영 제약에서 온 값으로 보고 유지하되, 파일럿에서 모듈별 소요를 실측해 다음 기수에 반영하는 지금 계획(제안서 13장)이 맞다.
- **24시간: 방향은 맞다.** 외부 제도는 교육 이수가 아니라 실적으로 승격한다(Kubernetes 정량 실적, GDE 2년 권장, MVP 12개월). '교육 24시간 + 실적' 이원 구조는 이 흐름과 일치한다. 문제는 총량이 아니라 실적 항목 구성이다(7.3).

### 7.3 Guide 승급 설계의 약점 — 여기는 분명히 반대한다

초안(2026-10-09 확정)은 Guide 승급을 '교육 이수 + 본인 Agent 1건 + 공동 개발 2건'로 하고, 리뷰·승인 역량의 공백은 "Guide 교육의 리뷰·심사 실습(G1+G3 12시간)으로 메운다. 실적 요건에 가이드 건을 따로 강제하지 않는다"고 정했다. **이 설계의 '12시간 실습으로 승인 역량 확인' 부분에 반대한다.**

- **근거 1**: 조사한 어느 승인·리뷰 제도도 교육 실습으로 승인 권한 역량을 인정하지 않는다. Kubernetes approver는 reviewer로서 substantial PR 30개의 **실제 리뷰 이력**을 요구하고, MVP는 12개월의 **검증 가능한 기여 이력**을 요구한다. Guide는 '타인의 산출물에 대한 1차 승인'이라는 판정 권한을 갖는데, 그 판정 역량을 실전 이력 없이 발급하는 공개 선례는 없었다(3.1).
- **근거 2**: 공동 개발 실적은 리뷰·판정 역량과 다른 역량이다. 같은 코드베이스에서 자기 결과물을 만든 이력이 남의 결과물의 결함·근거·우선순위를 판정하는 능력을 보증하지 않는다. Kubernetes가 contributor와 reviewer·approver를 구분해 승격하는 이유가 정확히 여기다.
- **근거 3 (구조적 닭-달걀)**: 첫 Guide 배출 전에는 실제 승인 경험이 생길 수 없다. 그래서 '실전 이력'을 요구하는 것이 불가능해 보이지만, 이미 확정된 '첫 Guide 배출 전 초기 심사단이 대행한다'는 장치가 해법이다 — 승급 요건에 **초기 심사단과 함께한 실제 심사 참여(그림자 채점) N회**를 넣으면, 대행 기간이 Guide의 실전 실습 기간이 된다. Kubernetes의 '상급자 스폰서'와 MVP의 '검증 가능한 이력'을 섞은 형태다.
- **구체 수정안**: (a) Guide 승급 심사에 실제 리뷰·심사 참여 증거 3건 이상(마켓 예비 제출물에 대한 그림자 채점 포함)을 요구. (b) G3 '평가 일치'는 승급 전 필수 통과로. (c) 첫 6개월은 Guide의 승인 건마다 심사단이 2차 확인하는 점진적 이양(승인 권한의 단계적 확대).
- **부분 동의**: '가이드 건을 실적 요건에 강제하지 않는다'는 대상 확대 유연성 측면은 이해한다. 그러나 승인 권한이 붙는 순간 최소한의 실전 심사 참여는 강제해야 하며, 이것이 없으면 Guide 인증의 신뢰가 초기 심사단의 명성에서 나오는 게 아니라 교육 수료증 수준으로 떨어진다.

### 7.4 이 과정이 실패한다면 가장 유력한 이유

**"수료↔인증 사이의 지연이 관문을 혼란스럽게 만들고, 조직이 인증율 지표 압박에 통과 기준을 낮추는 것"** 이 가장 유력하다고 본다.

- 초안의 인증은 교육 12주 + 운영 관찰(9주차 이후 시작, 4주 관찰) + 심사로 이어진다. 외부 사례가 보여주듯 실사용 검증은 원래 느리다 — CNCF 졸업은 독립 adopter 3곳과 인터뷰까지 요구한다. AWS조차 공식 페이지에서 "88% of agent pilots stall"이라고 표현한다([AWS Build & Deploy AI Agents](https://aws.amazon.com/ai/build-ai-agents/) — 벤더 마케팅 수치라 근거로는 약하지만, 파일럿 정체가 업계 공통 현상임을 시사한다).
- 그래서 첫 기수에서 '수료는 했으나 인증 유예'가 다수 나오는 것이 정상 시나리오다. 실패는 이 지표를 본 조직이 "기준이 너무 높다"며 통과선(80점·전 영역 3단계·필수 기준)을 낮추도록 압박할 때 온다. 초안의 "통과율을 높이기 위해 기준을 낮추지 않는다" 원칙이 무너지는 순간이 곧 실패다.
- 2차 후보: 초기 심사단 병목(첫 Guide 배출 전 모든 심사를 초기 심사단이 담당)과 교육시간 확보다. 다만 이 둘은 초안 9.2·12장이 이미 인지하고 대응안을 둔 항목이라, 인지되지 않은 위험을 꼽으면 위의 '기준 하향'이다.
- **반대 하나**: "커리큘럼 내용이 부족해서 실패할 것"이라는 가정에는 반대한다. 2~6장이 보이듯 실습 교재는 HF·LlamaIndex·MCP 공식 문서로 메울 수 있고, 오히려 초안이 자체 제작해야 하는 영역(B1 문제 정의, B8 운영·인수인계)이 공개 과정이 못 주는 차별점이다. 실패 위험은 내용이 아니라 관문 운영에 있다.

## 8. 확인하지 못한 것

- Microsoft AI-103·AI-500의 가격·문항 수·합격점(페이지가 지역별 상이 표기만 함), AI-500의 beta 여부 표기.
- AWS AIP-C01의 자격증 유효기간(인증 페이지에 명시 없음).
- LangChain Certified Agent Engineer의 시험 가격, HF Agents Course의 총 학습 시간.
- DeepLearning.AI 강의 소재의 재배포 가능 여부(이용약관상 재사용 조항 확인 못함).
- 국내 기업(SDS·SK·LG CNS·현대차그룹)과 제조 벤더(Siemens·Festo·Rockwell)의 공개 교육 카탈로그 — Agent 개발 Expert급 공개 과정을 1차 출처로 확인하지 못했다(5장).
- GDE·MVP·Kubernetes·CNCF 문서의 명시적 재사용 라이선스(일부 확인 못함, 6장 표).
- MCP 레지스트리 저장소의 최종 라이선스 전환 상태(MIT→Apache-2.0 여부 재확인 필요).
- OpenAI Evals 등 공개 평가 저장소의 이번 조사 확인(미확인).
- 'Agent 전용 Guide/챔피언 인증'의 존재(3.1 — 확인 못함이 결론).

## 9. 참고 자료 (접속일 2026-10-09)

본문의 모든 사실에 링크를 달았다. 아래는 그룹별 대표 링크 요약이다.

- **Builder — 벤더 자격증**: [AWS AIP-C01](https://aws.amazon.com/certification/certified-generative-ai-developer-professional/) · [시험 가이드](https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01.html) · [AWS AIF-C01](https://aws.amazon.com/certification/certified-ai-practitioner/) · [MS AI-500](https://learn.microsoft.com/en-us/credentials/certifications/multi-agent-ai-solutions-expert/) · [AI-500 스터디 가이드](https://aka.ms/AI500-StudyGuide) · [MS AI-103](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-apps-and-agents-developer-associate/) · [AI-102 단종](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/) · [MS AI-300](https://learn.microsoft.com/en-us/credentials/certifications/operationalizing-machine-learning-and-generative-ai-solutions/) · [Google PAA](https://cloud.google.com/learn/certification/agentic-architect) · [PAA 시험 가이드](https://services.google.com/fh/files/misc/professional_agentic_architect_exam_guide_english.pdf) · [LF MCPA](https://training.linuxfoundation.org/certification/model-context-protocol-associate-mcpa/)
- **Builder — 과정·교재**: [HF Agents Course](https://huggingface.co/learn/agents-course) · [HF 저장소(Apache-2.0)](https://github.com/huggingface/agents-course) · [LangChain Academy](https://academy.langchain.com/) · [LCAE 인증](https://academy.langchain.com/pages/certifications-lcae) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/) · [DeepLearning.AI 코스 목록](https://www.deeplearning.ai/courses/) · [OpenAI Academy](https://academy.openai.com/) · [Anthropic Academy](https://academy.claude.com/)
- **Guide — 제도**: [GDE](https://developers.google.com/community/experts) · [GDE FAQ](https://developers.google.com/community/experts/faq) · [MVP 개요](https://mvp.microsoft.com/en-us/mvp/overview) · [MVP FAQ](https://mvp.microsoft.com/en-us/faq?section=mvp) · [MVP 어워드 사이클 아카이브](https://learn.microsoft.com/en-us/archive/blogs/mvpawardprogram/microsoft-mvp-award-evolution) · [Kubernetes 멤버십](https://github.com/kubernetes/community/blob/master/community-membership.md) · [OWNERS](https://github.com/kubernetes/community/blob/main/contributors/guide/owners.md) · [Apache 성숙도 모델](https://community.apache.org/apache-way/apache-project-maturity-model.html) · [Apache contributor ladder](https://community.apache.org/contributor-ladder.html) · [eng-practices](https://github.com/google/eng-practices/) · [eng-practices LICENSE(CC BY 3.0)](https://github.com/google/eng-practices/blob/master/LICENSE) · [CNCF 졸업 신청 템플릿](https://github.com/cncf/toc/blob/main/.github/ISSUE_TEMPLATE/template-graduation-application.md) · [CNCF 프로젝트 단계](https://www.cncf.io/projects/)
- **마켓·심사·보안**: [MCP Registry](https://modelcontextprotocol.io/registry/about) · [MCP Package Types](https://modelcontextprotocol.io/registry/package-types) · [MCP Moderation Policy](https://modelcontextprotocol.io/registry/moderation-policy) · [MS AI 앱·에이전트 인증](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/artificial-intelligence-app-agent-certification) · [MS 인증 절차](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/review-publish-offer) · [AWS Marketplace 판매자 적격](https://docs.aws.amazon.com/marketplace/latest/userguide/seller-eligibility.html) · [AWS 제품 심사](https://docs.aws.amazon.com/marketplace/latest/userguide/product-submission.html) · [Slack 심사 가이드](https://docs.slack.dev/slack-marketplace/slack-marketplace-app-guidelines-and-requirements) · [Slack 심사 절차](https://api.slack.com/start/distributing/directory) · [GitHub Marketplace 요건](https://docs.github.com/en/apps/github-marketplace/creating-apps-for-github-marketplace/requirements-for-listing-an-app) · [VS Code 퍼블리셔 검증](https://code.visualstudio.com/api/working-with-extensions/publishing-extension) · [npm Trusted Publishing](https://docs.npmjs.com/trusted-publishers/) · [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/) · [OpenSSF Scorecard](https://github.com/ossf/scorecard) · [OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/) · [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) · [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) · [MITRE ATLAS](https://atlas.mitre.org/) · [ISO/IEC 42001](https://www.iso.org/standard/42001) · [EU GPAI 행동강령](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)
- **오픈소스 모델·평가·보안**: [PyRIT(MIT)](https://github.com/microsoft/PyRIT) · [llama-cookbook](https://github.com/meta-llama/llama-cookbook) · [vLLM 강좌](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm) · [Langfuse](https://langfuse.com) · [HF 관측·평가 보너스](https://huggingface.co/learn/agents-course/bonus-unit2/introduction)
- **국내**: [AI디지털배움터](https://www.xn--2z1bw8k1pjz5ccumkb.kr/main.do) · [NIA 디지털포용본부](https://www.nia.or.kr/site/nia_kor/ex/bbs/ListBusiness.do?businessMnCd=23000400) · [NIA STEP](https://nia.step.or.kr/)

## 관련 문서

- [Expert 분과 브레인스토밍](./expert-curriculum-brainstorm.md)
- [Expert 양성 실행 제안](./expert-curriculum-proposal.md)
- [AI/DT 교육 커리큘럼 설계(목차)](./README.md)
