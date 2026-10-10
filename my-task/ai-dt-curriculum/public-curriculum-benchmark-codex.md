---
tags: [my-task, ai-dt-curriculum, expert, curriculum, certification, benchmark]
category_major: "업무 기획·산출물"
category_middle: "AI/DT 교육 커리큘럼"
category_minor: "공개 커리큘럼·인증 벤치마크"
note_kind: "공개 자료 조사"
last_updated: 2026-10-09
status: research
---

# 사내 AI Agent Expert 과정: 공개 커리큘럼·인증 독립 벤치마크

## 결론 먼저

- Builder 교재의 우선 후보는 Hugging Face Agents Course와 Microsoft AI Agents for Beginners다. 각각 Apache-2.0·MIT로 배포되어 조건을 지키면 사내 교재로 복사·각색할 수 있다. [H-L][M-L]
- LangGraph Academy의 공개 노트북은 상태·메모리·사람 개입 실습에 적합하지만, 원본의 OpenAI·LangSmith·Tavily 연결과 배포 실습을 사내 구성으로 바꿔야 한다. [L-R][L-L]
- Google/Kaggle의 2025 과정과 2026 후속 과정은 교육 순서 참고에 유용하다. ADK가 다른 모델을 지원해도 Gemini·Google Search·Vertex 실습이 자동으로 이식되는 것은 아니다. [K25][K26][ADK-M]
- AWS AIP-C01·Microsoft AI-103·NVIDIA NCP-AAI는 공개 시험 범위의 참고 자료다. 이를 외부 제품 없이 수행하는 사내 개인 실기·운영 심사로 재작성해야 한다. [AWS-G][AZ-G][NV]
- 공개 모듈과 비교하면 B3·B4에 컨텍스트 선택·압축, 세션·장기 메모리, 상태 편집을 더 명확히 넣는 것이 우선이다. 총 교육시간 증설보다 기존 모듈의 수행 결과를 구체화하는 제안이다. [M-S][L][K25]
- 조사한 범위에서는 개발·리뷰·멘토링·마켓 승인까지 함께 인증하는 Agent 전용 Guide 제도를 확인하지 못했다. ASQ Master Black Belt의 포트폴리오, Carpentries의 교수 시연·활동 기반 권한 갱신을 조합할 수 있다. [ASQ-M][C-I][C-T]
- 마켓 등록과 안전 승인을 구분해야 한다. MCP Registry는 취약한 서버를 원칙적으로 삭제하지 않으며, OpenAI·Microsoft의 심사 사례는 실제 동작·권한·데이터 최소화·변경 후 재심사에 더 적합하다. [REG][O-MKT][MS-MKT]
- 무료 공개와 복사 허용은 별개다. Anthropic courses의 CC BY-NC 4.0은 사내 교육의 비상업 해당 여부를 확인해야 하므로 자동 재사용 후보에서 제외하고, 허락 전에는 주제·원리 참조로 제한한다. [A-L]

## 조사 범위와 판정 방법

접속 기준일은 **2026-10-09**다. 공식 제공자 페이지·공식 저장소·시험 가이드·LICENSE만 근거로 사용했다. 검색에 나온 제3자 수강기·시험 덤프·재배포 교재는 근거에서 제외했다. 일부 공식 사이트는 본문 수집이 실패해 공식 검색 색인에서 확인한 내용만 표시했다. 이런 항목의 접근 제한과 미확인 범위는 마지막 장에 모았다.

내부 대조 기준은 [제안서 6·7·8장](./expert-curriculum-proposal.md)과 [브레인스토밍 1장](./expert-curriculum-brainstorm.md)이다. B1~B10·G1~G4는 제안서 모듈명을 따른다. Guide 승급 실적은 브레인스토밍 1장의 최신 결정이 우선하며, 제안서 7.2의 예시 건수를 확정 기준으로 바꾸지 않았다. 회사명·사내 주소·실제 데이터·설비 정보는 수록하지 않았다.

표의 분량은 강의 영상, 권장 학습 속도, 시험 시간 등을 구분했다. 공식 출처에 없는 총 학습시간·합격률·교육비·재사용률은 추정하지 않고 **명시 없음**으로 표시한다. 읽은 출처에서 확인하지 못한 내용은 존재하지 않는다고 단정하지 않는다.

**벤더를 제거한 뒤 남는 내용**은 이 조사의 기술적 판단이다. 다음 세 범주로 적었다.

| 판정 | 의미 |
|---|---|
| 핵심 유지 | 학습 목표와 주요 알고리즘을 유지하고 모델 호출·외부 도구를 바꿀 수 있다. 원본 그대로 실행 가능하다는 뜻은 아니다 |
| 개념 유지·실습 재작성 | 아키텍처·평가·운영 원리는 남지만 관리형 서비스, SDK, 인증, 배포 실습의 재작성이 필요하다 |
| 참고만 | 공개 범위·접근·라이선스가 부족해 교재 채택이나 실습 이식 판단까지 하지 않는다 |

OpenAI 호환 API라는 이름만으로 Responses API, tool calling, JSON Schema, 스트리밍까지 호환된다고 가정하지 않는다. OpenAI Agents SDK도 비 OpenAI 제공자에 대해 API·구조화 출력 차이를 별도로 설명한다. [O-SDK]

## 1. Builder 벤치마크 비교표

라이선스 칸은 **확인한 자산**에만 적용한다. 영상·외부 이미지·데이터셋·모델 가중치·인증 로고까지 저장소 라이선스가 포괄한다고 보지 않는다.

| ID·제공자 / 과정·자료 | 대상·선수 조건 | 분량·형태 | 평가·인증 | 라이선스·벤더 제거 후 판단 |
|---|---|---|---|---|
| H / Hugging Face Agents Course | 입문~Agent 구축 학습자; Python·LLM 기초 | 본과정 Unit 1~4, 온보딩·보너스; 장당 1주·주 3~4시간 권장. 총시간 명시 없음 | Unit 1 기초 인증, 과제·최종 챌린지 수료 인증; 최종 증서 페이지는 점수 30% 초과 표기 | 저장소 Apache-2.0. 핵심 유지; Spaces·외부 검색·공개 제출을 내부 실행·채점으로 대체. [H][H-C][H-L] |
| M / Microsoft AI Agents for Beginners | Agent 입문자; 공식 영상은 Beginner. 구체적 선수 기준·필수 경력 명시 없음 | 현재 저장소 18 lessons; 총시간 명시 없음 | 자기 점검·실습·캡스톤 예시; 독립 개발자 자격시험 명시 없음 | 저장소 MIT. 핵심 유지 + Foundry 배포 부분 재작성. 현재 예제는 MAF·Azure OpenAI Responses 중심; Local 경로도 제공. [M][M-S][M-L] |
| K / Google·Kaggle AI Agents Intensive | 개발자; 2025 공식 발표는 시작 단계·심화 학습자 모두 언급. 필수 경력 명시 없음 | 2025년 11월 10~14일 5일 과정, 현재 자율학습 가이드. 2026 후속 Vibe Coding 과정도 공개; 총시간 명시 없음 | codelab·capstone; 2026 제출물은 영상·구현 이유·코드 링크. 감독형 개발·운영 인증 명시 없음 | 과정 전체 복사 라이선스 명시 없음; ADK 코드 Apache-2.0은 별개. 개념 유지·실습 재작성. [K-ANN][K25][K26][ADK-L] |
| L / LangChain Academy Introduction to LangGraph, Python | Python 개발 실습자; 설치 안내 Python 3.11~3.13. 지식 선수 수준 명시 없음 | 6개 학습 모듈·55 lessons·영상 6시간. 전체 실습시간 명시 없음 | 공개 과정표는 실습·피드백; 감독형 시험·실무 운영 증빙 명시 없음 | 공개 노트북 저장소 MIT. 핵심 유지; 외부 API·관측·배포 부분 재작성. 영상의 동일 라이선스 명시 없음. [L][L-R][L-L] |
| U / UC Berkeley RDI Agentic AI, Fall 2025 | 학부·대학원 학생, 공개 MOOC 학습자; 학내 과정은 ML·DL 기초 경험 권장 | 학기별 강의 일정; 총학습시간 명시 없음 | 학내 참여·퀴즈·프로젝트/기사 평가와 MOOC 등급별 수료증은 별도. MOOC 제출 마감 2026-01-31 | 강의·슬라이드 포괄 재사용 라이선스 명시 없음. 원리·평가 프로젝트 설계 참고; 원본 교재 복사는 보류. [U][U-M] |
| D / DeepLearning.AI Agentic AI, Andrew Ng | Intermediate; Python 구현을 사용. 경력·필수 선수 과목 명시 없음 | 7시간 45분 표기, 영상 31개·코드 예제 7개·채점 과제 8개. 실제 전체 소요시간 명시 없음 | 채점 과제·증서는 PRO 제공; 운영 증거·대면 개인 실기 명시 없음 | 공식 공개 페이지에 교재 재배포 라이선스 명시 없음. 패턴·평가 순서를 참조하고 자체 예제 작성. [D] |
| A / Anthropic 공식 courses 저장소·Engineering 자료 | Claude API 개발·프롬프트·tool 학습자; 통일된 선수 조건 명시 없음 | 저장소 5개 과정; 총시간 명시 없음. 2026-09-15 archive 표시 | 이 저장소 README에 감독형 시험·인증 기준 명시 없음. Academy 증서와 혼동하지 않음 | courses는 CC BY-NC 4.0. 기술 원리는 유지되지만 Claude SDK 호출 재작성·기업 용도 권리 확인 필요. Engineering 글의 동일 라이선스 명시 없음. [A-R][A-L][A-P] |
| O / OpenAI A practical guide to building agents + Agents SDK | 첫 Agent를 만드는 제품·엔지니어링 팀; 선수 수준 명시 없음 | PDF 34쪽; 강의·교육시간 명시 없음 | 가이드이며 자격증·수료 평가 명시 없음 | 가이드 재배포 허용 명시 없음. SDK 저장소는 MIT; 비 OpenAI 모델 경로 제공. 가이드는 참조, SDK 예제는 조건부 각색. [O-G][O-L][O-SDK] |
| E-AWS / AWS Certified Generative AI Developer – Professional, AIP-C01 | 권장: production 앱 개발 2년 이상, GenAI 구현 1년; 선행 자격증 의무 없음 | 시험 180분·75문항; 준비 학습 총시간 명시 없음 | 객관식·복수응답, 감독 시험. 운영 포트폴리오 심사와는 다름 | 시험 가이드에 교재화·문항 재사용 허용 명시 없음. 개념 유지·AWS 서비스 실습 재작성. [AWS][AWS-G] |
| E-AZ / Microsoft Azure AI Apps and Agents Developer Associate, AI-103 | Python 앱 개발·일반 AI·GenAI·Azure 이해; Foundry 기반 엔지니어 | 시험 범위 2026-04-16 기준. 조사한 페이지에서 전체 준비시간·문항 수·시험시간 명시 없음 | 인증 시험; 700 이상 합격, associate는 매년 무료 온라인 평가로 갱신 | 시험 자체 재사용 허용 명시 없음. 개념 유지·Foundry 서비스 실습 재작성. [AZ][AZ-G] |
| E-NV / NVIDIA Certified Professional Agentic AI, NCP-AAI | 페이지: AI/ML 1~2년·production Agent 경험; 연결 PDF는 권장 2~3년으로 차이 | 원격 감독 120분·60~70문항; 총 준비시간 명시 없음 | 유효 2년, 재시험 갱신. 페이지에 Coming soon 표시가 있어 현재 응시 가능성 미확인 | 가이드 교재 재사용 허용 명시 없음. 범용 주제는 유지, NVIDIA 플랫폼·추론 최적화는 별도 선택. [NV][NV-G] |

### 1.1 2026 신설 인증 중 추가 확인 후보

Anthropic Partner Certifications에는 Associate, Developer Foundations, Architect Foundations, Architect Professional이 공개되어 있다. 공식 페이지는 파트너 전용 자격임을 명시한다. Professional 준비 과정은 end-to-end Claude 설계자 대상이며, API·MCP·안전·위험 관련 학습을 포함한다. [A-CERT][A-PREP]

다만 공식 본문 접근이 403으로 실패해 연결된 **시험 가이드 원문**의 전체 영역·배점·문항 수·갱신 규칙을 검증하지 못했다. 이 값은 **명시 없음(이번 조사에서 원문 확인 불가)**으로 남긴다. 제3자들이 전재한 영역 비율로 채우지 않았다. 따라서 위의 시험 범위 검증 완료 후보와 분리하며, Builder·Guide 내부 인증 대체안으로 권고하지 않는다. [A-CERT][A-FAQ]

### 1.2 출처별 모듈·시험 영역

아래 목록은 공식 목차를 한국어로 요약했다. **세부 목차를 수록했다고 모든 강의를 수강·실행한 것은 아니다.**

| 출처 | 공개 모듈·영역 | 채택할 부분 / 걷어낼 부분 — 조사 판단 |
|---|---|---|
| H | 온보딩 → Agent 기본 → smolagents·LlamaIndex·LangGraph → Agentic RAG 사례 → 최종 구축·평가. 보너스: function calling 미세조정·관측/평가·게임 Agent. [H] | 기본 루프·tool·프레임워크 비교를 가져오고 게임은 선택. 공개 Spaces·leaderboard를 내부로 바꾼다 |
| M | 1 입문, 2 프레임워크, 3 설계 패턴, 4 tool, 5 RAG, 6 신뢰, 7 계획, 8 다중 Agent, 9 메타인지, 10 production, 11 프로토콜, 12 컨텍스트, 13 메모리, 14 MAF, 15 computer use, 16 확장 배포, 17 local Agent, 18 보안. [M-S] | 12·13·17과 보안/운영 주제를 선별. 제품별 14·16은 사내 실행 구조에 맞춰 재작성 |
| K25 | Day 1 Agent 소개, 2 tool·MCP 상호운용, 3 컨텍스트·세션·메모리, 4 품질, 5 prototype→production·A2A·배포. [K25] | 교육 순서는 재사용 가치가 높다. Gemini·검색·Vertex 배포를 제거하면 내부 모델·검색·배포 실습이 필요 |
| K26 | 공식 색인에서 확인: tool/API·code execution·Agent 통신, 장기 메모리·상태를 포함한 Agent Skills, 보안·평가. 전체 일자별 본문 확인 불가. [K26] | 2025 목차를 2026의 정확한 목차라고 표기하지 않음. 보안 실습 강화의 참고 후보 |
| L | 1 그래프·chain/router/Agent, 2 state·memory, 3 UX·human-in-the-loop, 4 병렬화·subgraph·research assistant, 5 장기 메모리, 6 배포. [L] | state reducer·요약/필터·중단·상태 편집을 실습으로 보완. LangSmith/외부 검색·배포 서비스는 대체 |
| U | LLM Agent 개관, 시스템 설계, 검증 가능한 Agent의 post-training, 평가, Agent 모델 학습, multi-agent, 평가 변동성, 과학 탐색, 실배포 경험, embodied Agent, 안전·보안. [U] | B6 평가 불확실성·평가자 구현을 참고. 모델 학습·로보틱스는 공통 필수보다 B9 후보 |
| D | 1 Agentic workflows 소개, 2 reflection, 3 tool use, 4 구축 실무·평가 팁, 5 고자율 Agent 패턴. 계획·multi-agent도 교육 주제. [D] | 프레임워크 전에 패턴을 직접 구현하는 순서와 오류 분석을 참고 |
| A | 저장소: API 기초, prompting tutorial, real-world prompting, prompt evaluations, tool use. 별도 글: chaining·routing·parallelization·orchestrator-workers·evaluator-optimizer·자율 Agent. [A-R][A-P] | 패턴 선택·프롬프트 평가 원리는 참조. SDK 코드·원문 교재는 권리·환경 대체 판단 후 |
| O | Agent 정의·필요성, model/tool/instructions, orchestration, guardrails·사람 개입. [O-G] | B1·B4 설계 리뷰 체크리스트 참고. OpenAI 모델 선택·호스팅 기능은 자체 예제로 대체 |
| E-AWS | ① 모델 통합·데이터·준수 31%, ② 구현·통합 26%, ③ 안전·보안·거버넌스 20%, ④ 운영 효율·최적화 12%, ⑤ 시험·검증·문제 해결 11%. [AWS-G] | 범용 실패·평가·운영 질문을 추출하고 Bedrock·IAM·CloudWatch의 제품 사용법은 제외 |
| E-AZ | Azure AI 계획·관리 25~30%, GenAI·Agent 30~35%, vision 10~15%, text analysis 10~15%, information extraction 10~15%. [AZ-G] | 앞 두 영역의 권한·상태·관측 주제 참고. 영상 생성 등은 직무 필요에 따라 선택 |
| E-NV | 설계·개발, 평가·튜닝, 배포·확장, 인지·계획·메모리, 지식·데이터, NVIDIA 플랫폼, 운영·유지, 안전·윤리·준수, 사람 상호작용·감독. [NV][NV-G] | 고급 설계·평가·운영 검토 범위 참고. 공개 배점 불일치가 있어 내부 시간 배분으로 환산하지 않음 |

**시험 영역 비율은 교육시간 배분이 아니다.** NVIDIA 페이지의 배포·확장 13%, 운영·유지 5%는 연결 PDF의 각각 5%, 7%와 다르다. 위 표의 NVIDIA 영역은 이름만 사용했다. 페이지와 PDF의 경력 권장값 차이도 원문 그대로 남겼다. 응시·가이드 버전은 제공자 확인이 필요하다. [NV][NV-G]

### 1.3 외부 LLM API가 없는 환경에서 실제로 남는 것

다음은 공식 모델 연결 문서에서 가능한 경로를 확인한 결과다. **사내 모델로 실행 검증한 결과는 아니다.**

| 후보 | 공식적으로 확인한 모델 대체 경로 | 남는 내용 | 이식 전에 확인할 것 |
|---|---|---|---|
| Hugging Face/smolagents | OpenAIModel의 api_base로 OpenAI 호환 서버 지정 가능. [H-M] | tool loop·평가·RAG 실습 | 모델 tool 표현·chat template, 생성 코드 실행 격리, 검색·파일 도구의 외부 접속 |
| Microsoft 입문 과정 | 공식 가이드에 Foundry Local·OpenAI 호환 경로. 기본 예제는 Responses API 중심. [M-S] | 설계·컨텍스트·메모리·local 도구 개념 | 임의 사내 서버에서 MAF 호출이 되는지 별도 확인; Local 제품 설치를 사내 모델 연동과 동일시하지 않음 |
| Google ADK | 공식 모델 목록에 OpenAI·Ollama·vLLM·LiteLLM 경로. [ADK-M] | runtime·tool·세션·평가·상호운용 개념 | Gemini 내장 도구·Google Search·Vertex 관리형 배포 대체 |
| LangGraph Academy | 원본 설정은 OpenAI·LangSmith·Tavily를 사용. [L-R] | graph/state/checkpoint·사람 개입·메모리 | 모델 adapter, 외부 tracing 끄기, 검색 대체, 배포 코드 변경 |
| OpenAI Agents SDK | custom base URL·Chat Completions model, 비 OpenAI tracing 처리 문서 제공. [O-SDK] | 실행·handoff·guardrail·세션 패턴 | Responses 기본값 변경, 구조화 출력 차이, OpenAI 서버로 trace가 나가지 않게 내부 processor 또는 비활성화 |
| AWS·Azure·NVIDIA 시험 가이드 | 시험 영역은 공개; 관리형 서비스 문제를 사내 모델로 실행하는 공식 대체 시험은 확인 못함. [AWS-G][AZ-G][NV] | 요구사항·평가·보안·운영 판단 문항의 주제 | 제품 지식을 제거한 독립 실기·장애 시나리오 새 작성 |

**권고:** 96시간 본과정의 공통 실습은 사내 모델·내부 검색·내부 배포로 한 경로만 만든다. 공개 프레임워크 세 개를 모두 익히는 과제로 확대하지 않는다. 첫 실습에서 tool calling·구조화 출력·중단/재개·trace 저장의 호환성을 확인하고, 지원하지 않는 기능은 교육 결과에 조건을 명시한다. 이는 제공자 인증의 규칙이 아니라 이번 조사에 따른 내부 설계 제안이다.

## 2. B1~B10 모듈 대조

대조는 **주제 대응**이며 동등한 숙련도·시간·시험 강도를 뜻하지 않는다. 내부 내용은 [제안서 6·8장](./expert-curriculum-proposal.md) 기준이다.

| 우리 모듈 | 직접 참고할 공개 주제 | 초안 판단·보완 제안 |
|---|---|---|
| B1 업무 문제·해법 선택 | Agent가 필요한 상황·단순 workflow 우선. [A-P][O-G] | 이미 대응. 업무 기준선·Agent 불필요 판단을 유지 |
| B2 S/W 엔지니어링 | 도구 입력 계약·단순 설계·테스트·문서에 대한 Google 리뷰 기준. [G-REV] | Agent 입문 강좌만으로 서비스 품질 전체를 대체하기 어려움. 재현·오류 처리·Git·테스트는 자체 훈련 유지 |
| B3 LLM·컨텍스트 | 메시지·토큰, 컨텍스트 선택/압축/격리, 짧은·긴 메모리. [H][M-S][L][A-CTX] | 컨텍스트 초과 처리 외에 검색 시점·요약 손실·다른 세션 정보 혼입 시험을 명시 |
| B4 도구·Agent 실행 | 패턴·state reducer·checkpoint·사람 개입/상태 편집. [M-S][L][D] | 실행 상태와 대화 메모리를 구분; tool 결과를 받은 뒤 승인 취소·상태 변경 시 재검증 실습 |
| B5 지식·시스템 연결 | Agentic RAG·MCP·A2A·지식 통합. [H][K25][NV-G] | API/MCP 비교는 유지. A2A는 선택 소개로 두고 필수 구축 요구로 늘리지 않음 |
| B6 평가·개선 | benchmark·prompt eval·오류 분석·평가 변동성·품질 게이트. [H-F][A-R][U][AWS-G] | 최종 답뿐 아니라 잘못된 tool 선택·검색·상태 전이도 판정. 모델 judge와 사람 판정의 불일치 확인 |
| B7 보안·통제 | 안전·권한·oversight·데이터 최소화·정확한 tool 메타데이터. [AZ-G][NV][O-MKT] | 기존 필수 기준에 공급망·tool 설명의 실제 효과 일치·메모리 접근통제 검토를 보완 |
| B8 배포·운영 | 관측·scaling·운영 최적화·배포 변경 심사. [K25][AWS-G][MS-MKT] | 자원·복구는 유지. release별 승인 범위·rollback·tool 변경 재심사 기록 추가 |
| B9 선택 심화 | multi-agent·모델 적응·게임/embodied Agent·computer use. [H][U][M-S] | 이미 대응. 공개 강좌가 다룬다는 이유만으로 공통 필수화하지 않음 |
| B10 통합 캡스톤 | benchmark 제출·capstone·평가자와 수행 Agent 구현. [H-F][K-ANN][U] | 자체 개인 실기·권한 실패·운영 증빙 유지. 외부 leaderboard 점수를 내부 통과선으로 옮기지 않음 |

### 2.1 공개 커리큘럼에는 있는데 우리 초안에 명시되지 않은 것

여기서 “없음”은 **제안서 6·7·8장의 명시적 세부 항목·수행 증거가 부족**하다는 뜻이다. B3·B4처럼 포괄 제목이 있어도 별도의 학습 결과가 없으면 보완 후보로 분류했다.

| 누락·약한 명시 | 공개 근거 | 제안 |
|---|---|---|
| 세션/장기 메모리 분리, 상태 스키마·reducer, 메모리 수정/삭제 | LangGraph의 state·memory 및 long-term memory 모듈. [L] | B3·B4의 필수 실습을 구체화; 민감한 내용을 기억하지 않게 하는 정책 포함 |
| 컨텍스트 선택·압축·격리와 just-in-time 검색 | Microsoft 컨텍스트 장, Anthropic context engineering. [M-S][A-CTX] | B3에 컨텍스트 예산을 넘긴 긴 대화·요약 손실 실습 추가 |
| streaming·상태 편집·time travel로 사람 개입을 설계 | LangGraph UX/HITL 모듈. [L] | B4에서 승인 요청 UI·중단 후 상태 변경을 시험. time travel은 선택 |
| reflection·evaluator-optimizer 패턴의 효용과 실패 | DeepLearning.AI 패턴과 Anthropic 패턴. [D][A-P] | B4·B6에서 기준선 대비 실제 개선 여부·추가 비용 확인 |
| 평가 결과의 변동성과 평가자 자체 구현 | Berkeley 평가·Predictable Noise 강의와 Green/White Agent 프로젝트. [U] | B6에 반복 평가·판정 일치 확인; 통계·모델 학습 전체는 B9 |
| A2A·Agent 간 상호운용 | Google production 단원. [K25] | B5 선택 소개. 단일 Agent 과제에 불필요한 multi-agent 구축을 요구하지 않음 |
| 직접적인 로보틱스·게임·Agent 모델 학습 | HF 보너스, Berkeley post-training·embodied Agent. [H][U] | 제조라는 이유만으로 설비 제어 실습에 연결하지 않음; 직무별 B9 검토 |

공급망·등록 후 재심사는 **커리큘럼 비교와 별개로 마켓 사례에서 도출한 보완**이다. 공개 입문 강좌의 공통 필수 모듈이라고 주장하지 않는다. [REG][MS-MKT]

### 2.2 우리 초안에만 있는 것

아래는 **이번에 확인한 공개 과정·시험 설명에서 동일한 묶음의 의무 요구를 찾지 못한 항목**이다. 전 세계 모든 과정에 없다는 뜻은 아니다. 내부 근거는 [제안서 6·8장](./expert-curriculum-proposal.md)이다.

- 12주·96시간 안에서 업무 기준선→구현→예외 시험→사용자 검증→운영 인수인계를 한 산출물에 누적하는 편성.
- 개인 변경·장애 진단 실기, 산출물, 운영 증거, 구조화 구술을 함께 요구하는 Builder 심사.
- 승인된 사용 관찰과 복구 시험을 요구하고, 운영 증거가 미완료이면 수료와 인증을 분리하는 규칙.
- 점수가 높아도 권한 위반·비밀 노출·승인 누락·중복 쓰기·통제 실패로 탈락시키는 공통 필수 기준.
- 읽기 전용 과제에서도 공통 쓰기·승인·중복 처리 역량을 별도로 확인하는 시험 설계.

**비교상 의미:** Hugging Face의 자동 채점, Berkeley의 수료증, 클라우드 객관식 시험은 각각 다른 증거를 요구한다. 내부 규칙을 삭제하고 외부 배지를 자동 인정할 근거는 없다. [H-F][U-M][AWS]

## 3. Guide 벤치마크

### 3.1 Agent 전용 제도의 확인 결과

조사한 Agent 개발 인증은 개발·설계 역량에 초점이 있다. **타인 Agent의 결함 리뷰·멘티 독립 수행·마켓 1차 승인 판단을 함께 심사하는 공식 자격**은 이번 조사에서 확인하지 못했다. Anthropic의 Architect·Microsoft의 Champions라는 명칭도 곧바로 이 역할의 인증을 뜻하지 않는다. [A-CERT][AZ][CHAMP]

### 3.2 유사 제도의 승급·실적·갱신

| 사례 | 승급·평가 요건 | 실적 증빙·갱신 | Guide에 가져올 것 / 한계 |
|---|---|---|---|
| ASQ Six Sigma Black Belt → Master Black Belt | BB: 관련 실무 3년+서명된 완료 프로젝트 1건 또는 서명 프로젝트 2건. MBB: 유효 ASQ BB+포트폴리오; BB/MBB 역할 5년 또는 BB 프로젝트 10건. [ASQ-B][ASQ-M] | MBB 포트폴리오는 Teaching/Coaching/Mentoring·직무 책임·기술 경험/혁신 각 영역 최소치 심사 후 시험. MBB·BB는 3년마다 18 RU 또는 시험 갱신. [ASQ-M][ASQ-R] | 선행 개발 역량+지도 실적의 영역별 심사. 제조 친화적이지만 Agent 품질 승인 자격은 아니며, 연차·프로젝트 수를 그대로 이식하지 않음 |
| Carpentries Instructor | Instructor Training 후 Welcome Session·Teaching Demonstration·Get Involved. 시연은 1시간 모임 안에서 5분 참여형 지도, Trainer 판정. [C-I] | 활동은 AMY 기록; 일반 checkout은 교육 후 90일, 연장 가능. 일반 Instructor 자격의 정기 갱신 주기는 확인한 checkout에 명시 없음. [C-I] | 교육 이수와 실제 교수 능력 구분. 멘티가 수행하도록 가르치는 모습을 관찰; Agent 개발·승인 심사는 별도 |
| Carpentries Instructor Trainer | 신청을 현직 Trainer 패널이 승인; 10주 교육. 교수 경험 또는 교육학 훈련 권장, 관찰·학습·행동강령·활동 약정 등 요구. [C-T] | 매년 Active 갱신. 활동 경로는 Instructor Training 1회·teaching demo 2회·모임 4회·분기 일정 설문 4회. 미갱신은 Alumni, 교수/시연 심사 권한 제한. 대화로 갱신하는 대체 경로도 있음. [C-T] | **자격 보유와 활성 승인 권한 분리**의 가장 직접적 사례. 숫자는 사내 규모에 맞춰 새로 정함 |
| Google Engineering Practices 코드 리뷰 기준 | 인증·승급 체계가 아니라 공개 리뷰 가이드. 설계·기능·복잡도·테스트·명명·주석·스타일·문서 검토. [G-REV] | 리뷰 범위를 명시하고 적절한 테스트·사용자 영향을 확인. 실적 최소건수·자격 갱신 명시 없음. [G-REV] | G1 기준·결함 사례·리뷰 기록 양식을 각색. Agent 특유 권한·평가·운영은 추가 |
| Microsoft AI Champion 역할 헌장 | AI 활용 실험·동료 지원·사례 공유·피드백·자원 연결 역할. 공인 자격·승급 시험 명시 없음. [CHAMP] | 역할 헌장에 참여 기대·지원 경로·성과 지표를 작성하도록 안내. 통일된 실적 건수·갱신 주기 명시 없음. [CHAMP] | 팀 챔피언 운영 참고. 공식 안내는 보안·준수 승인을 Champion 소관 밖으로 둔다. 우리 Guide의 1차 승인은 별도 위임·검증 필요 |

MBB의 객관식·상황 기반 평가가 Agent 리뷰 실기와 같지는 않다. Carpentries의 Instructor Trainer는 다른 강사의 교수 시연을 심사하는 역할이라는 점이 Guide와 유사하다. 일반 Instructor와 Trainer의 갱신을 섞지 않는 것이 중요하다. [ASQ-M][C-T]

### 3.3 G1~G4 대조

| 우리 모듈 | 벤치마크 | 초안 유지·보완 제안 |
|---|---|---|
| G1 설계·코드 리뷰 | Google 리뷰 기준; ASQ 기술 실적 포트폴리오. [G-REV][ASQ-M] | 결함·근거·우선순위는 유지. 정상 제출과 고의 결함 제출을 섞어 승인·보완·반려를 실제로 판정 |
| G2 멘토링·교육 설계 | Carpentries의 연습·피드백·인지부하·참여형 교수 시연. [C-I][C-RUB] | 설명의 유창함보다 학습자가 실행·설명·독립 변경하는지 관찰. 과제를 대신 수행한 활동은 지도 성과와 분리 |
| G3 평가·심사 일치 | ASQ 포트폴리오 영역별 판정, Carpentries 시연 rubric·Trainer 판정. [ASQ-M][C-RUB] | 동일 샘플 판정 비교는 유지. 오승인·과도한 반려·증거 부족·이해충돌 사례를 포함; 채점 일치만으로 승인 역량 인정하지 않음 |
| G4 재사용·운영 전파 | Carpentries 공동체 활동·Trainer 갱신; Champion 역할 헌장. [C-T][CHAMP] | 다른 사람이 쓰는 자료·인수인계는 유지. 등록 후 tool 변경·위험 발견·Guide 부재 때의 재심사/대행 절차를 연습 |

**승급 증거 제안:** 최신 내부 기준은 공동 개발과 가이드 실적을 동등하게 인정한다. 공동 개발은 본인 기여를, 가이드는 멘티의 독립 수행·피드백 전후 결과를 확인하고, 가이드 실적을 별도 의무로 추가하지 않는다. 공동 개발만으로 승급하는 후보자의 리뷰·승인 역량은 G1·G3 실습에서 직접 확인한다. 활동 건수는 합격 판정 자체가 아니다. ASQ의 영역별 포트폴리오와 Carpentries의 관찰형 평가를 내부 Agent 과제로 다시 설계하는 제안이다. [내부 기준](./expert-curriculum-brainstorm.md) [ASQ-M][C-I]

**갱신 제안:** Builder 역량 인증의 유효기간과 Guide 승인 권한의 활성 상태를 구분한다. 실제 지도·리뷰·심사 활동과 기준 변경 학습을 연간 확인하고, 휴면 시에는 승인 권한을 비활성화하는 방식이 타당하다. 공개 사례의 갱신 주기·건수를 그대로 사내 규칙으로 확정하지 않는다. [C-T][ASQ-R]

## 4. 사내 Agent·tool 마켓 승인 참고 사례

### 4.1 공개 사례

| 공식 사례 | 공개된 등록·심사 내용 | 사내에 남길 기준 — 제안 | 벤더 종속·주의 |
|---|---|---|---|
| OpenAI ChatGPT Directory Plugin guidelines | 실제 동작에 맞는 tool 이름·설명·schema, 명시적 readOnly/destructive/openWorld annotation, 최소 입력·데이터, 인증·권한, 예측 가능하고 감사 가능한 효과, 제출 전 시험·demo 계정. [O-MKT] | tool별 읽기/쓰기/삭제·전송·대상 범위 표기; 명세와 실제 효과를 같은 입력으로 재현; 권한 다른 계정 시험 | Apps SDK 구주소가 현재 Plugins 지침으로 이동. UI·디렉터리·플랫폼 계정 조건은 제외. annotation은 실제 권한 검사나 승인 장치를 대신하지 않음 |
| Microsoft MCP server certification, preview | 게시자 검증·endpoint 소유, package/schema 자동 검증, 기능·인증·보안·준수·telemetry 심사; 새 tool·주요 metadata/package 변경 시 재제출. [MS-MKT] | 운영 책임자·tool 소유자, 재현 환경·시험 증거, 승인된 버전 범위, 변경 시 재심사 | Partner Center·M365·Foundry·Key Vault 포맷은 제거. preview이며 품질/안전 심사와 단순 구조 검증을 구분 |
| 공식 MCP Registry moderation | 불법·악성·spam·비동작 서버 제거; 저품질·buggy·보안 취약 서버는 일반적으로 제거하지 않음. 이의제기 경로 제공. [REG] | 외부 registry 등록 여부 대신 사내 검증·허용 목록을 관리. 보류·회수·재심사·이의제기 상태를 기록 | 개방형 discovery 정책이다. 등록은 안전 검증·인증·제품 품질 보증이 아님 |

### 4.2 Guide가 1차 관문에서 확인할 제출 묶음

다음은 공식 심사 정책을 그대로 복사한 양식이 아니라 **사내 적용 제안**이다. 내부 기본 기준은 [제안서 8.5](./expert-curriculum-proposal.md)를 유지한다.

| 제출 항목 | Guide 확인 내용 | 참고 근거 |
|---|---|---|
| 기능·사용 범위·버전 | 허용 사용자·데이터·수행 작업·실패/중단 조건, tool 목록과 각각의 입력/출력·부작용 | tool 명세·behavior 일치. [O-MKT] |
| 권한·승인 시험 | 사용자가 달라지면 접근 범위가 실제로 달라지는지, 쓰기/삭제/전송 전에 필요한 승인이 있는지, 승인 취소 후 재검사 | 인증·최소 권한·명시적 효과·감사 가능성. [O-MKT][MS-MKT] |
| 기능·안전·회귀 증거 | 정상·오류·중복·오래된 문서·주입 입력, 모델/프롬프트/tool 버전과 재현 절차 | 대표 기능/안전 평가 증거·시험. [MS-MKT][O-MKT] |
| 데이터·메모리·로그 | 필요한 정보만 받는지, 입력/출력/trace의 저장 위치·보존·접근·삭제가 설명되는지 | 데이터 최소화·경계·관측 준비. [O-MKT][MS-MKT] |
| 운영 담당자·인수인계 | 담당 역할, 사용자 안내, 장애 연락·중단·rollback·인수인계 실행 가능 여부 | 내부 8.5 이후 운영 책임 기준 + 외부 지원·문서 요건. [MS-MKT] |
| 재사용 권리·의존성 | 교재/코드/model/tool 라이선스, package 버전·소유자, 외부 통신 요구·취약점 검토 결과 | 게시자 통제·보안 심사·registry 비보증 정책을 내부 공급망 기준으로 확장. [MS-MKT][REG] |
| 등록 후 변경·재심사 | 모델·tool 권한·동작·데이터 범위 변경 시 재평가, 위험 발견 시 승인 회수·사용 중단·재등록 근거 | Microsoft 변경 재제출·Registry 삭제/이의제기. [MS-MKT][REG] |

공개 정책에 없는 사내 통과점수·심사 SLA·보존 기간·재심사 주기는 **명시 없음**이며 이 문서에서 숫자를 만들지 않았다. Guide의 기술적 1차 승인과 최종 보안·출시 결정의 책임 범위는 별도로 정해야 한다. Champion 프로그램이 이런 승인 권한을 자동 부여하는 근거는 아니다. [CHAMP]

## 5. 재사용 가능 여부

### 5.1 판정 기준

- **복사 가능:** 확인한 라이선스가 해당 자산의 복제·배포를 허용한다. 고지·조건은 유지한다.
- **각색 가능:** 번역·수정·사내 API로 변경이 허용되는 범위다. 변경 사실·원저작자·라이선스를 적는다.
- **참조만:** 이번 조사에서 사내 복사·각색 권한을 확보하지 못한 자료에 대한 채택 방침이다. 법률상 모든 이용이 금지라는 뜻은 아니다.
- **조건 확인:** 허락 범위가 용도·자산에 따라 달라 아직 자동 승인할 수 없다.

사내 유료 교육 여부만으로 NonCommercial을 판정하지 않는다. CC BY-NC의 정의는 상업적 이익·금전 보상을 주된 목적으로 하는지에 관한 것이므로, 기업이라는 이유만으로 반드시 금지 또는 사내 무료 교육이라는 이유만으로 반드시 허용이라고 단정하지 않는다. 이번 목적에 대한 허락을 확인하기 전에는 해당 원문을 교재로 복제하지 않는 방침을 제안한다. [A-L]

### 5.2 자산별 재사용 표

| 자산 | 확인한 라이선스/권리 | 복사 | 각색 | 실제 가져올 방식·조건 |
|---|---|---|---|---|
| Hugging Face agents-course 저장소 자산 | Apache-2.0. [H-L] | 가능 | 가능 | 라이선스·기존 고지 유지, 변경 파일 표시, NOTICE가 있으면 반영. 외부 영상/이미지·GAIA·모델 가중치는 별도 확인 |
| Microsoft ai-agents-for-beginners 저장소 | MIT. [M-L] | 가능 | 가능 | 저작권·허가 고지 유지; 사내 API/배포 예제로 각색. 연결된 제품 화면·영상까지 일괄 MIT로 보지 않음 |
| LangChain Academy 공개 노트북·저장소 문서 | MIT. [L-L] | 가능 | 가능 | 고지 유지; 외부 서비스 연결 대체. Academy 영상·플랫폼 계정·배포 서비스의 권리는 별도 |
| Google ADK Python 코드 | Apache-2.0. [ADK-L] | 가능 | 가능 | framework/해당 코드 활용 가능; 이를 근거로 Kaggle whitepaper·영상 전체 복사를 허용하지 않음 |
| Google/Kaggle 과정 whitepaper·영상·개별 codelab | 과정 전체의 통일 재사용 라이선스 명시 없음. [K25][K26] | 참조만 | 참조만 | 목차와 학습 목표를 참고해 자체 실습 작성; notebook별 license 확인 후 해당 notebook만 채택 |
| OpenAI Agents Python SDK 저장소 | MIT. [O-L] | 가능 | 가능 | 고지 유지; 사내 model endpoint·내부 tracing 사용. 제품 이용약관·호스팅 tool과 별개 |
| OpenAI Agent PDF 가이드 | 해당 PDF의 교재 재배포·번역 허락 명시 없음. [O-G] | 참조만 | 참조만 | 설계 원리를 자체 표현·사례로 작성; PDF/도표 대량 복사·번역본 배포 권한은 별도 확인 |
| Anthropic courses 저장소 | CC BY-NC 4.0. [A-L] | 조건 확인 | 조건 확인 | 출처·라이선스·변경 표시에 더해 사내 용도의 NC 충족/별도 허락 필요. SDK 코드까지 MIT라고 가정하지 않음 |
| Anthropic Engineering·Academy·Partner 시험 자료 | courses 저장소와 동일 라이선스라는 명시 없음. [A-P][A-CERT] | 참조만 | 참조만 | 공개 주제·설계 판단만 참고. 접근 불가 시험 가이드·문항을 제3자 재배포본에서 복사하지 않음 |
| Berkeley 강의·슬라이드·MOOC | 확인한 과정 페이지에 포괄 복사/각색 라이선스 명시 없음. [U][U-M] | 참조만 | 참조만 | 평가자/수행 Agent의 역할과 과제 단계 참고. 슬라이드별 권리는 별도 확인 |
| DeepLearning.AI 교재·영상·과제 | 공식 과정 페이지에 오픈 재배포 허용 명시 없음. [D] | 참조만 | 참조만 | 모듈·패턴·평가 방식 참조, 자체 코드/시험 작성 |
| Google Engineering Practices | CC BY 3.0 Unported. [G-L] | 가능 | 가능 | 출처·라이선스·변경 고지; 리뷰 기준을 한국어로 각색. Google 인증/보증을 뜻하지 않음 |
| Carpentries Instructor Training 교육 자료 / 예제 코드 | 교육 CC BY 4.0 / 별도 표시 없으면 코드 MIT. [C-L] | 가능 | 가능 | 교육 내용은 출처·변경·라이선스 표시, 허용 권리를 막는 추가 제한 금지. 브랜드·공식 강사 인증 권한은 별개 |
| ASQ Body of Knowledge·포트폴리오·시험 제도 | 공개 열람과 교재 복사 허용은 별개; 확인한 페이지에 오픈 라이선스 명시 없음. [ASQ-M][ASQ-B] | 참조만 | 참조만 | 실적 제3자 확인·영역별 포트폴리오 구조 참고, 자체 rubric 작성 |
| AWS·Microsoft·NVIDIA 시험 가이드 | 공개 범위 확인; 시험 문항·증서·교육 재배포를 포괄 허용하는 근거는 확인 못함. [AWS-G][AZ-G][NV-G] | 참조만 | 참조만 | 공개 시험 영역으로 내부 출제 범위를 점검. 실제 시험 문항·배지·기관 이름을 사내 자격으로 재사용하지 않음 |
| Champion 헌장·마켓 정책 | 이번에 확인한 웹페이지의 교재 복사 허용 명시 없음. [CHAMP][O-MKT][MS-MKT][REG] | 참조만 | 참조만 | 역할 경계·심사 항목을 자체 문장과 사내 책임 구조로 작성 |

오픈 라이선스 교재를 사내망에서 제공할 때도 자산별 LICENSE·NOTICE·원저작자·변경 기록을 함께 두는 방식이 필요하다. 모델 가중치·데이터·의존 package의 라이선스와 보안은 교재 허가와 별도로 확인한다. Apache-2.0은 상표 사용을 자동 허용하지 않고, Carpentries도 로고·명칭의 상표를 구분한다. [H-L][C-L]

### 5.3 채택 조합 제안

| 용도 | 우선 채택 | 자체 작성할 것 |
|---|---|---|
| Builder 기본 교재 | H·M 저장소의 선별 자료, L의 상태/메모리 실습 | 사내 API adapter·내부 검색·배포·실패 복구 실습 |
| Builder 인증 범위 | E-AWS·E-AZ·E-NV의 범용 주제 참고 | 내부 업무 기준선·개인 변경 실기·운영 증거·필수 탈락 기준 |
| Guide 교육 | Google 리뷰 기준·Carpentries 교수 자료 | Agent 결함 리뷰·멘티 독립 수행·승인/반려 변형 과제 |
| Guide 승급·갱신 | ASQ의 영역별 증거 구조·Carpentries Active 역할 관리 참고 | 사내 실적 확인·이해충돌·대행·휴면·권한 복원 절차 |
| 마켓 | OpenAI·Microsoft 기능/안전 검토·변경 재심사 참고 | 책임자·승인 버전·실제 권한 시험·등록 후 중단 기준 |

이 조합은 외부 기관 자격을 취득하는 과정이 아니라 **공개 자료를 바탕으로 만드는 독립 사내 교육·인증**이다. 자료 재사용 허락은 외부 기관의 인증 발급·상표 사용 권한을 주지 않는다. [H-L][C-L]

## 6. 확인하지 못한 것

| 항목 | 확인 상태·이 보고서의 처리 |
|---|---|
| 실제 사내 모델·OpenAI 호환 서버에서 실습 실행 | 실행하지 않았다. tool calling·schema·스트리밍·Responses·checkpoint 호환, 외부 telemetry 차단은 구현 검증 필요 |
| 공개 교재의 사내 이식 비율·준비 인시 | 측정 자료 명시 없음. 강의 개수·시험 배점을 “재사용률”로 환산하지 않음 |
| Anthropic 최신 인증 시험 가이드 | 공식 Partner 페이지는 검색 색인으로 존재·역할을 확인했으나 본문 403. 연결 시험 PDF의 전체 범위·문항·갱신 원문 미확인. [A-CERT][A-FAQ] |
| Google 2026 과정 전체 목차·badge 세부 조건·자산별 라이선스 | 공식 Kaggle 본문이 동적으로 제공되어 텍스트 수집 결과가 비어 있음. 공식 색인의 확인 가능한 주제·제출 요구만 기록. 2025와 2026을 합치지 않음. [K25][K26] |
| DeepLearning.AI 모든 수업·실습 코드와 이식 가능성 | 공식 공개 과정 색인으로 분량·모듈·PRO 평가를 확인; 전체 본문/학습 환경 일부 접근 실패. API 의존과 notebook 권리는 완전 확인하지 못함. [D] |
| NVIDIA 시험의 현재 등록 가능성·확정 시험 배점 | Coming soon 표시, 페이지/PDF의 경험·배점 불일치. PDF 검색 결과 제목은 AI Operations로 보이지만 본문 표지는 Agentic AI임을 확인. 등록·배점 확정은 제공자 확인 필요. [NV][NV-G] |
| Berkeley 2026-10 현재 새 기수 수료증 | 확인한 것은 Fall 2025 자료와 2026-01 제출 마감. 기존 강의 열람과 현재 수료증 신청 가능성을 구분. [U-M] |
| 커리큘럼별 개설 연월 | 공식 연도가 확인된 Google·Berkeley·AI-103 범위만 표기. 다른 자료의 첫 신설 연월은 명시 없음 |
| 외부 인증의 제조 현장 성과·내부 운영 역량 예측력 | 공식 과정 소개·시험 가이드만으로 입증 불가. 합격률·업무 개선률을 만들지 않음 |
| Guide와 동일한 Agent 전용 공개 자격의 전 세계 부재 | 포괄적으로 증명하지 않았다. 이번 조사 범위에서 확인하지 못했다는 결론만 사용 |
| 회사 교육 용도의 NC 판정·권리자 허락 | 확보하지 않았다. 원문·슬라이드·과제·모델별 별도 조건 확인 필요. [A-L] |
| 저장소 고정 revision·모든 링크의 장기 유효성 | 접속일의 공개 main/master와 페이지를 확인했다. 실제 교재 채택 시 revision·LICENSE·NOTICE를 고정해야 함 |
| 인증기관의 비공개 문제·내부 심사 운영 | 접근하지 않았다. 공개 범위를 넘어 문항·승인 SLA·채점 규칙을 추정하지 않음 |

## 7. 참고 자료

**모든 자료 접속일: 2026-10-09.** 본문 표의 짧은 출처 표기는 아래 공식 자료로 연결된다. “검색 색인”은 제공자의 공식 URL에 대한 검색 결과로만 일부 확인한 항목이다.

| 표기 | 공식 자료·확인 목적 |
|---|---|
| H / H-F / H-C / H-L / H-M | [HF 과정·선수·학습 속도][H], [최종 채점·공개 제출][H-F], [증서 조건][H-C], [LICENSE][H-L], [smolagents 모델 연결][H-M] |
| M / M-S / M-L | [Microsoft 현재 과정 저장소][M], [공식 Study Guide][M-S], [MIT LICENSE][M-L] |
| K-ANN / K25 / K26 | [Google 2025 개설 발표][K-ANN], [2025 자율학습 가이드][K25], [2026 후속 과정][K26] — Kaggle 일부는 검색 색인 |
| ADK-M / ADK-L | [ADK 모델 선택][ADK-M], [ADK Python LICENSE][ADK-L] |
| L / L-R / L-L | [LangGraph 과정 목차·분량][L], [공개 notebook 설정][L-R], [LICENSE][L-L] |
| U / U-M | [Berkeley 학내 Fall 2025 syllabus·평가][U], [MOOC 수료 tier·마감][U-M] |
| D | [DeepLearning.AI Agentic AI][D] — 모듈·분량·PRO 평가, 일부 검색 색인 |
| A-R / A-L / A-P / A-CTX | [Anthropic courses][A-R], [CC BY-NC LICENSE][A-L], [Agent 패턴][A-P], [context engineering][A-CTX] |
| A-CERT / A-FAQ / A-PREP | [Partner Certifications][A-CERT], [공식 인증 FAQ][A-FAQ], [Architect Professional 준비 과정][A-PREP] — 검색 색인, 전체 본문 접근 제한 |
| O-G / O-SDK / O-L | [OpenAI Agent 가이드 PDF][O-G], [비 OpenAI 모델·tracing][O-SDK], [SDK MIT LICENSE][O-L] |
| AWS / AWS-G | [AWS 인증 안내][AWS], [AIP-C01 공식 시험 가이드][AWS-G] |
| AZ / AZ-G | [AI-103 인증 안내][AZ], [공식 study guide][AZ-G] |
| NV / NV-G | [NCP-AAI 인증·blueprint][NV], [연결된 공식 시험 PDF][NV-G] |
| ASQ-B / ASQ-M / ASQ-R | [Black Belt][ASQ-B], [Master Black Belt][ASQ-M], [갱신][ASQ-R] |
| C-I / C-T / C-RUB / C-L | [Carpentries Instructor checkout][C-I], [Instructor Trainer Handbook][C-T], [교수 시연 rubric][C-RUB], [교육·코드 LICENSE][C-L] |
| G-REV / G-L | [Google 코드 리뷰 기준][G-REV], [CC BY 3.0 LICENSE][G-L] |
| CHAMP | [Microsoft AI Champion 역할 헌장][CHAMP] |
| O-MKT / MS-MKT / REG | [OpenAI Plugin 심사 지침][O-MKT], [Microsoft MCP 인증 preview][MS-MKT], [MCP Registry moderation][REG] |

[H]: https://huggingface.co/learn/agents-course/en/unit0/introduction
[H-F]: https://huggingface.co/learn/agents-course/en/unit4/hands-on
[H-C]: https://huggingface.co/learn/agents-course/en/unit4/get-your-certificate
[H-L]: https://raw.githubusercontent.com/huggingface/agents-course/main/LICENSE
[H-M]: https://huggingface.co/docs/smolagents/reference/models
[M]: https://github.com/microsoft/ai-agents-for-beginners
[M-S]: https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/STUDY_GUIDE.md
[M-L]: https://github.com/microsoft/ai-agents-for-beginners/blob/main/LICENSE
[K-ANN]: https://blog.google/innovation-and-ai/technology/developers-tools/ai-agents-intensive/
[K25]: https://www.kaggle.com/learn-guide/5-day-agents?linkId=17674402
[K26]: https://www.kaggle.com/competitions/5-day-ai-agents-intensive-vibecoding-course-with-google
[ADK-M]: https://adk.dev/agents/models/
[ADK-L]: https://raw.githubusercontent.com/google/adk-python/main/LICENSE
[L]: https://academy.langchain.com/courses/intro-to-langgraph
[L-R]: https://raw.githubusercontent.com/langchain-ai/langchain-academy/main/README.md
[L-L]: https://raw.githubusercontent.com/langchain-ai/langchain-academy/main/LICENSE
[U]: https://rdi.berkeley.edu/agentic-ai/f25
[U-M]: https://agenticai-learning.org/f25
[D]: https://www.deeplearning.ai/courses/agentic-ai
[A-R]: https://github.com/anthropics/courses
[A-L]: https://raw.githubusercontent.com/anthropics/courses/master/LICENSE
[A-P]: https://www.anthropic.com/engineering/building-effective-agents
[A-CTX]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
[A-CERT]: https://anthropic-partners.skilljar.com/page/partner-certifications
[A-FAQ]: https://anthropic-partners.skilljar.com/page/faq-certifications
[A-PREP]: https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional
[O-G]: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
[O-SDK]: https://openai.github.io/openai-agents-python/models/
[O-L]: https://raw.githubusercontent.com/openai/openai-agents-python/main/LICENSE
[AWS]: https://aws.amazon.com/certification/certified-generative-ai-developer-professional/
[AWS-G]: https://docs.aws.amazon.com/pdfs/aws-certification/latest/ai-professional-01/ai-professional-01.pdf
[AZ]: https://learn.microsoft.com/en-us/credentials/certifications/exams/ai-103/
[AZ-G]: https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103
[NV]: https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/
[NV-G]: https://dam-cdn.nvd.orangelogic.com/AssetLink/64tei188l3tt132l265u1ipjoexdl1p5.pdf
[ASQ-B]: https://www.asq.org/cert/six-sigma-black-belt
[ASQ-M]: https://www.asq.org/cert/master-black-belt
[ASQ-R]: https://www.asq.org/cert/recertification
[C-I]: https://carpentries.github.io/instructor-training/checkout.html
[C-T]: https://docs.carpentries.org/handbooks/instructor_trainers.html
[C-RUB]: https://carpentries.github.io/instructor-training/demos_rubric.html
[C-L]: https://raw.githubusercontent.com/carpentries/instructor-training/main/LICENSE.md
[G-REV]: https://google.github.io/eng-practices/review/reviewer/looking-for.html
[G-L]: https://raw.githubusercontent.com/google/eng-practices/master/LICENSE
[CHAMP]: https://adoption.microsoft.com/en-us/copilot/define-your-ai-champion-role-and-charter/
[O-MKT]: https://developers.openai.com/plugins/plugin-guidelines
[MS-MKT]: https://learn.microsoft.com/en-us/microsoft-copilot-studio/mcp-certification
[REG]: https://modelcontextprotocol.io/registry/moderation-policy
