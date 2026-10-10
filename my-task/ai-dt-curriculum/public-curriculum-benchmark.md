---
tags: [my-task, ai-dt-curriculum, expert, benchmark]
category_major: "업무 기획·산출물"
category_middle: "AI/DT 교육 커리큘럼"
category_minor: "공개 커리큘럼 벤치마크"
note_kind: "조사"
last_updated: 2026-10-09
---

# 공개 커리큘럼·인증 벤치마크

> Expert 분과의 Builder·Guide 교육 모듈을 정하기 위해, 2026년 10월 현재 공개된 Agent 개발 커리큘럼·인증과 리뷰·멘토 자격 제도를 1차 출처로 확인하고 사내 환경에서 재사용할 수 있는지 판정한 조사.

**읽는 법**: "출처가 말하는 것"과 "이 문서의 판단"을 구분했다. 판단에는 `판단:`을 붙였다. 출처 페이지에 숫자가 없으면 "명시 없음"으로 적었다. 모든 외부 자료는 2026-10-09에 확인했다.

## 1. 결론 먼저

- **Builder 교재로 재사용할 수 있는 것**은 세 개다. Microsoft AI Agents for Beginners(MIT), Hugging Face Agents Course(Apache-2.0), LangChain Academy 실습 저장소(MIT). 이 중 Microsoft 과정만 로컬·OpenAI 호환 API 경로를 공식 문서에 적어 두었다.
- **Builder 모듈 구성·비중의 기준선으로 쓸 것**은 시험 개요(skills outline) 네 개다. NVIDIA NCP-AAI, Google Professional Agentic Architect, LangChain Certified Agent Engineer, AWS AIP-C01. 시험 자체는 외부 감독·벤더 제품 전제여서 사내 인증을 대신할 수 없다.
- **B1(Agent를 쓸지 판단)** 은 강의형 커리큘럼이 거의 다루지 않는다. Anthropic "Building effective agents"와 OpenAI "A practical guide to building agents" 두 문서가 가장 직접적인 근거다. 둘 다 공개 라이선스가 없어 인용·참조만 가능하다.
- **Guide에 해당하는 Agent 전용 인증은 찾지 못했다.** 확인한 Agent 인증은 모두 개인의 구축 역량을 필기·실습으로 본다. Guide는 인접 제도를 조합해야 한다: 리뷰 기준은 Google 코드 리뷰 가이드, 승급 사다리는 Kubernetes Reviewer/Approver, 교수법과 시연 평가는 The Carpentries 강사 양성, 프로젝트·멘토링 실적은 ASQ Six Sigma Black Belt/Master Black Belt.
- **초안에 없는데 공개 커리큘럼이 공통으로 가르치는 것**: 메모리·컨텍스트 엔지니어링(독립 단원), Agent 간 프로토콜(A2A), 계획·성찰 같은 설계 패턴, 코딩 Agent 활용·확장.
- **초안에만 있어 자체 제작해야 하는 것**: B2 S/W 엔지니어링, 중복 실행·중단 후 재개 시험, 권한별 검색 차이, 인수인계, 사내 모델·권한·마켓 절차, 그리고 Guide 과정 전체.
- **가장 큰 주의점**: 공개 과정의 실습은 대부분 해당 벤더의 클라우드나 계정을 전제로 한다. 사내에서 살아남는 것은 개념과 과제 설계이고, 실습 코드는 사내 OpenAI 호환 API에 맞춰 다시 써야 한다. 여기에 더해 이 분야 자료는 수명이 짧다. 이번 조사 중에도 Anthropic 강의 저장소와 Google 코드 리뷰 가이드 저장소가 보관(archived) 상태였고, Microsoft 과정은 GitHub Models 지원을 중단했다.

## 2. Builder 벤치마크 대상

### 2.1 비교표

"사내 환경에서 쓸 수 있는 정도" 열은 전부 이 문서의 판단이다. 기준은 외부 LLM API와 벤더 콘솔을 쓸 수 없고 사내 OpenAI 호환 API만 쓴다는 조건이다.

| 출처 | 대상·선수 조건 | 분량 | 평가 방식 | 라이선스·재사용 조건 | 사내 환경에서 쓸 수 있는 정도 (판단) |
|---|---|---|---|---|---|
| [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) | 공식 선수 조건 명시 없음. 생성형 AI 입문자는 별도 과정 권장 | 18개 레슨. 시간 명시 없음 | 없음 (인증·평가 언급 없음) | [MIT](https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/LICENSE) | 높음. 기본 경로는 Azure 구독과 Foundry지만 [설치 문서](https://github.com/microsoft/ai-agents-for-beginners/blob/main/00-course-setup/README.md)가 Foundry Local 등 OpenAI 호환 경로를 안내한다. 단 Foundry Local은 Chat Completions만 제공하고 Responses API 전체는 아니라고 적혀 있어 일부 예제는 고쳐야 한다 |
| [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | Python 기초, LLM 기초 | Unit 0~4와 보너스 3개. 장당 1주, 주 3~4시간 권장 | Unit 1 완료 시 기초 수료증. 과제 1개와 최종 과제까지 마치면 수료증. [최종 과제는 GAIA 벤치마크 일부에서 30% 이상](https://huggingface.co/learn/agents-course/unit4/introduction) | [Apache-2.0](https://raw.githubusercontent.com/huggingface/agents-course/main/LICENSE) | 중간. 교재 문서는 가져다 쓸 수 있다. 실습과 인증은 Hugging Face 계정과 인터넷을 전제로 한다. 자동 채점 벤치마크로 통과선을 두는 방식은 그대로 참고할 만하다 |
| [Kaggle·Google 5-Day AI Agents Intensive](https://www.kaggle.com/learn-guide/5-day-agents) | 선수 조건 명시 없음 | 5일. 2025년 11월 10~14일 라이브 후 자습 가이드로 공개. 시간 명시 없음 | 라이브 기수에 캡스톤 과제. 자습 가이드의 수료 기준은 명시 없음 | 명시 없음 | 낮음~중간. 코드랩이 ADK와 Gemini, AI Studio API 키를 전제로 한다. 백서와 5일 구성은 참고 가치가 크다 |
| [LangChain Academy: Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) | 선수 조건 명시 없음 | 55개 레슨, 영상 6시간 | 과정 자체는 무료, 수료 평가 명시 없음 | 실습 저장소 [MIT](https://github.com/langchain-ai/langchain-academy). 영상은 명시 없음 | 중간. 저장소 설정이 OpenAI, LangSmith, Tavily 키를 요구한다. LangGraph 자체는 오픈소스라 모델 호출부를 바꾸면 쓸 수 있으나 추적·배포 단원은 LangSmith에 묶여 있다 |
| [LangChain Certified Agent Engineer](https://academy.langchain.com/pages/certifications-lcae) | 무료 선행 과정 5개 권장 | 120분, 객관식 40문항(28개 이상 통과) | 온라인 감독 시험. [LangChain 자료만 참조 가능한 준오픈북, 일부 문항은 전용 LangSmith 조직에서 풀이](https://kb.langchain.com/articles/5297173898-what-is-the-langchain-certified-agent-engineer-exam). 99달러, 24개월 유효 | 시험. 재사용 대상 아님 | 낮음. 영역 구성(구축·테스트·배포·모니터링 각 25%)만 참고 |
| [NVIDIA NCP-AAI](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/) | AI/ML 직무 1~2년, 운영 수준 Agent 프로젝트 경험 | 120분, 60~70문항 | 온라인 원격 감독 시험. 200달러, 2년 유효, 재응시로 갱신 | 시험. 재사용 대상 아님 | 낮음(응시). 높음(영역 구성 참고). NVIDIA 제품 영역은 7%뿐이고 나머지는 벤더 중립적인 제목이다 |
| [Google Professional Agentic Architect](https://cloud.google.com/learn/certification/agentic-architect) | 클라우드 3년 이상, Google Cloud Agent 구축 1년 이상 권장 | 3시간, 객관식 약 80문항 + 합격 후 실습 랩 | 감독 시험 뒤 별도 hands-on 랩으로 실행·코딩 능력 검증. 유효 1년. 조회 시점 베타 등록 종료, 일반 등록은 11월 2일 개시로 표기 | 시험. 재사용 대상 아님 | 낮음(응시). 중간(구성 참고). 필기 뒤 실습으로 한 번 더 검증하는 2단 구조가 초안의 "개인 실기"와 닮았다 |
| [Microsoft AI-103](https://learn.microsoft.com/en-us/credentials/certifications/exams/ai-103/) | Python 개발 경험, Azure 서비스 이해 | 120분 | 감독 시험. [700점 이상 통과, 매년 무료 온라인 평가로 갱신](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103) | 시험. 재사용 대상 아님 | 낮음. 전 영역이 Microsoft Foundry 전제. 비전·음성·문서 추출이 35~45%를 차지해 Agent 전용 기준으로는 넓다 |
| [AWS AIP-C01](https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01.html) | 운영 수준 앱 2년 이상, 생성형 AI 구현 1년 | [180분, 75문항](https://aws.amazon.com/certification/certified-generative-ai-developer-professional/)(채점 65 + 비채점 10) | 감독 시험, 750점 이상. 300달러 | 시험. 재사용 대상 아님 | 낮음(응시). 중간(구성 참고). 과업 문장이 구체적이라 실습 주제를 뽑기 좋다 |
| [DeepLearning.AI: Agentic AI](https://www.deeplearning.ai/courses/agentic-ai/) | 중급 Python, LLM·API 기초 | 9시간 55분. 영상 31개, 코드 예제 7개, 채점 과제 8개 | 채점 과제. 수료증은 유료 멤버십 필요 | 명시 없음 | 낮음. 플랫폼 안에서 수강하는 형태다. 5개 모듈의 설계 패턴 순서만 참고 |
| [UC Berkeley Agentic AI MOOC (Fall 2025)](https://agenticai-learning.org/f25) | 공식 선수 조건 명시 없음. [학점 과정](https://rdi.berkeley.edu/agentic-ai/f25)은 ML·딥러닝 기초 권장 | 12회 강의 (2025-09-15 ~ 12-08) | 수료증 4단계. 기본 단계는 강의별 퀴즈 전부와 글쓰기 과제, 상위 단계는 대회 프로젝트 제출 | 명시 없음 | 낮음. 연구 동향 강연 중심이라 12주 실무 과정의 뼈대로는 맞지 않는다. 평가·보안 강의는 심화 참고 자료로 쓸 수 있다 |
| [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | Agent를 만드는 개발자 | 글 1편 (2024-12-19) | 없음 | 공개 라이선스 명시 없음 | 높음(개념). 벤더 제품 없이 읽히는 내용이다. 페이지 상단에 도구 환경이 2024년 12월 이후 많이 바뀌었다는 안내가 있다 |
| [OpenAI: A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) | 첫 Agent를 만드는 제품·엔지니어링 팀 | PDF 1편 (본문 쪽번호 32까지) | 없음 | 공개 라이선스 명시 없음 | 중간~높음(개념). 판단 기준과 가드레일 분류는 벤더 중립적이다. 코드 예제는 OpenAI Agents SDK 기준 |
| [Anthropic Courses 저장소](https://github.com/anthropics/courses) | Claude API 입문 개발자 | 과정 5개. 시간 명시 없음 | 없음 | [CC BY-NC 4.0](https://raw.githubusercontent.com/anthropics/courses/master/LICENSE) | 낮음. 저장소가 2026-09-15자로 보관되어 읽기 전용이다. Claude API 전제. 도구 사용·프롬프트 평가 단원의 구성만 참고 |

### 2.2 출처별 모듈 목록

**Microsoft AI Agents for Beginners.** [저장소 README](https://github.com/microsoft/ai-agents-for-beginners)의 레슨은 다음 18개다. 1 Intro to AI Agents and Agent Use Cases, 2 Exploring AI Agentic Frameworks, 3 Understanding AI Agentic Design Patterns, 4 Tool Use Design Pattern, 5 Agentic RAG, 6 Building Trustworthy AI Agents, 7 Planning Design Pattern, 8 Multi-Agent Design Pattern, 9 Metacognition Design Pattern, 10 AI Agents in Production, 11 Using Agentic Protocols (MCP, A2A and NLWeb), 12 Context Engineering for AI Agents, 13 Managing Agentic Memory, 14 Exploring Microsoft Agent Framework, 15 Building Computer Use Agents (CUA), 16 Deploying Scalable Agents, 17 Creating Local AI Agents, 18 Securing AI Agents. 레슨마다 독립적이어서 어디서든 시작할 수 있다고 적혀 있다. 예제는 Microsoft Agent Framework와 Foundry Agent Service 기준이다. [설치 문서](https://github.com/microsoft/ai-agents-for-beginners/blob/main/00-course-setup/README.md)에 따르면 Python 3.12 이상이 필요하고, GitHub Models는 더 이상 지원하지 않는다.

**Hugging Face Agents Course.** [저장소의 단원 표](https://github.com/huggingface/agents-course)는 Unit 0 과정 안내, Unit 1 Introduction to Agents(Agent 정의, LLM, 특수 토큰), Unit 1 보너스 함수 호출용 미세조정, Unit 2 프레임워크(2.1 smolagents, 2.2 LlamaIndex, 2.3 LangGraph), Unit 2 보너스 Observability and Evaluation, Unit 3 Agentic RAG 활용 사례, Unit 4 최종 프로젝트(자동 평가와 리더보드), Unit 3 보너스 게임 속 Agent로 구성된다. [과정 소개](https://huggingface.co/learn/agents-course/unit0/introduction)는 인증 과정이 무료이고 기한이 없다고 밝힌다.

**Kaggle·Google 5-Day AI Agents Intensive.** [학습 가이드](https://www.kaggle.com/learn-guide/5-day-agents)의 5일 구성은 Day 1 Introduction to Agents, Day 2 Agent Tools & Interoperability with Model Context Protocol (MCP), Day 3 Context Engineering: Sessions & Memory, Day 4 Agent Quality(관측성·로깅·추적과 평가 전략), Day 5 Prototype to Production(배포·확장, A2A 프로토콜)이다. 날마다 백서, 요약 팟캐스트, 코드랩이 짝을 이룬다. Day 2 코드랩에는 사람 승인을 기다리며 도구 호출을 멈췄다가 재개하는 장기 실행 작업이 들어 있다. [Google 블로그 회고](https://blog.google/technology/developers/ai-agents-intensive-recap/)는 2025년 기수에 약 150만 명이 참여했고 캡스톤 제출이 11,000건을 넘었다고 적었다.

**LangChain Academy.** [Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph)는 Module 1 핵심 개념, Module 2 State and memory, Module 3 UX와 human-in-the-loop, Module 4 어시스턴트 구축(병렬, 서브그래프, map-reduce), Module 5 장기 메모리, Module 6 배포로 구성된다. [인증 페이지](https://academy.langchain.com/pages/certifications-lcae)가 꼽는 무료 선행 과정은 Introduction to LangChain - Python, Introduction to Deep Agents, Building Reliable Agents with LangSmith, Monitoring Production Agents, Introduction to LangSmith Deployment다.

**NVIDIA NCP-AAI.** [인증 페이지](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/)의 출제 비중은 Agent Architecture and Design 15%, Agent Development 15%, Evaluation and Tuning 13%, Deployment and Scaling 13%, Cognition, Planning, and Memory 10%, Knowledge Integration and Data Handling 10%, NVIDIA Platform Implementation 7%, Run, Monitor, and Maintain 5%, Safety, Ethics, and Compliance 5%, Human-AI Interaction and Oversight 5%다. 조회한 값의 합은 98%였다. 페이지 표기 문제인지 추출 오류인지는 확인하지 못했다.

**Google Professional Agentic Architect.** [시험 가이드](https://services.google.com/fh/files/misc/professional_agentic_architect_exam_guide_english.pdf)는 5개 영역이다. Section 1 로우코드 도구로 Agent 구축(약 13%), Section 2 코딩 Agent를 활용한 애플리케이션 개발(약 17%), Section 3 커스텀 Agent 개발(약 33%), Section 4 평가와 배포(약 22%), Section 5 보안과 거버넌스(약 15%). Section 3에는 모델 선택(LLM과 SLM, 자체 호스팅과 SaaS, 오픈소스와 상용), 세션·메모리, RAG, Agent 권한, MCP·A2A 오케스트레이션이 들어 있다. Section 4에는 골든 데이터·엣지 케이스로 평가 세트 만들기, 지속 평가 파이프라인, 드리프트·추론 루프 같은 장애 진단이 들어 있다.

**Microsoft AI-103.** [학습 가이드](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103)(2026-04-16 기준 skills measured)의 영역은 Azure AI 솔루션 계획·관리 25~30%, 생성형 AI와 Agent 솔루션 구현 30~35%, 컴퓨터 비전 10~15%, 텍스트 분석 10~15%, 정보 추출 10~15%다. 계획·관리 영역 안에 추적 로그·출처 메타데이터·승인 워크플로를 통한 감사, 감독 모드·제약·도구 접근 통제로 Agent 행동을 관리하는 항목이 있다.

**AWS AIP-C01.** [시험 가이드](https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01.html)의 영역은 Domain 1 파운데이션 모델 통합·데이터 관리·컴플라이언스 31%, Domain 2 구현과 통합 26%, Domain 3 AI 안전·보안·거버넌스 20%, Domain 4 운영 효율과 최적화 12%, Domain 5 테스트·검증·문제 해결 11%다. [Domain 2의 Task 2.1](https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01-domain2.html)이 Agent 과업으로, 메모리·상태 관리, 종료 조건·타임아웃·서킷 브레이커, 사람 검토·승인 절차, 도구 오류 처리와 파라미터 검증, MCP 서버 구현을 나열한다.

**DeepLearning.AI Agentic AI.** [과정 페이지](https://www.deeplearning.ai/courses/agentic-ai/)의 모듈은 1 Introduction to Agentic Workflows, 2 Reflection Design Pattern, 3 Tool use, 4 Practical Tips for Building Agentic AI, 5 Patterns for Highly Autonomous Agents다. 실습에 쓰는 프레임워크와 모델 제공자는 페이지에 명시 없음.

**UC Berkeley Agentic AI MOOC.** [Fall 2025 페이지](https://agenticai-learning.org/f25)의 강의 주제는 LLM Agent 개요, AI 엔지니어 관점의 시스템 설계 변화, 학습 이후 검증 가능한 Agent, Agent 평가, Agent 모델 학습의 교훈, 멀티 Agent, LLM의 예측 가능한 잡음, 과학 발견 자동화, 실제 배포에서 얻은 교훈, LLM 시대의 멀티 Agent 시스템, 체화·상호작용·학습, Agent 안전과 보안이다. 수료증은 Trailblazer(퀴즈 전부 + 글쓰기 과제), Mastery(+ 대회 프로젝트 제출), Legendary(대회 입상·결선), Honorary로 나뉜다. [RDI 교육 페이지](https://rdi.berkeley.edu/education)에는 조회 시점에 2026년 기수가 올라와 있지 않았다.

**Anthropic Building effective agents.** [원문](https://www.anthropic.com/engineering/building-effective-agents)의 목차는 What are agents?, When (and when not) to use agents, When and how to use frameworks, Building blocks·workflows·agents, 패턴의 조합, 요약, 부록 2개다. workflow 패턴으로 prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer를 든다. 부록 2는 도구 정의를 다듬는 방법을 다룬다.

**OpenAI A practical guide to building agents.** [PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)의 목차는 What is an agent?, When should you build an agent?, Agent design foundations, Guardrails, Conclusion이다. Agent가 맞는 업무로 복잡한 판단, 유지하기 어려운 규칙, 비정형 데이터 의존 세 가지를 들고, 이 기준에 분명히 들지 않으면 결정론적 해법으로 충분할 수 있다고 적는다. 가드레일은 relevance classifier, safety classifier, PII filter, moderation, tool safeguards, rules-based protections, output validation으로 나눈다. tool safeguards는 읽기·쓰기, 되돌릴 수 있는지, 필요한 권한, 금전 영향에 따라 도구마다 낮음·중간·높음 위험 등급을 매기라는 내용이다. 사람 개입이 필요한 계기로는 실패 한도 초과와 고위험 동작 두 가지를 든다.

**Anthropic Academy와 Claude 인증.** [Skilljar 과정 목록](https://anthropic.skilljar.com/)에는 Building with the Claude API, Introduction to Model Context Protocol, Model Context Protocol: Advanced Topics, Introduction to agent skills, Introduction to subagents, Claude Code in Action, Teaching AI Fluency 등 23개 항목이 있다. 과정별 시간·수료 조건은 목록 페이지에 명시 없음. 인증은 [2026-03-12 발표](https://www.anthropic.com/news/claude-partner-network)로 Claude Certified Architect, Foundations가 파트너 대상으로 나왔고, [2026-07-23 발표](https://www.claude.com/blog/four-role-based-claude-certifications)로 Associate: Foundations, Developer: Foundations, Architect: Professional이 추가되어 4종이 되었다. 모두 감독 시험이고 Claude Partner Network 회원 대상이다. 가격과 영역 비중은 두 발표문에 명시 없음.

**대학 강의.** CMU 11-768 AI Agents는 [과목 페이지](https://cmu-agents.com/)의 설명문에서 2026년 가을 대학원 과목이며 도구 사용, 계획, 메모리, 학습, 안전, 상호작용을 다룬다는 것까지만 확인했다. 주차별 일정은 읽지 못했다.

## 3. 모듈 대조표

### 3.1 초안 B1~B10과 공개 출처

칸의 내용은 해당 출처의 모듈·영역 제목을 근거로 한다. 제목만으로 판단한 것이어서 실제 강의 깊이는 확인하지 않았다. "없음"은 모듈·영역 제목에 드러나지 않는다는 뜻이다. 출처 링크는 2장과 같다.

| 초안 모듈 | Microsoft (18레슨) | Hugging Face | Kaggle·Google 5-Day | LangChain Academy·LCAE | NVIDIA NCP-AAI | Google PAA |
|---|---|---|---|---|---|---|
| B1 업무 문제·해법 선택 (6h) | 부분. 1 Agent 활용 사례 | 없음 | 부분. Day 1에서 기존 LLM 앱과의 차이 | 없음 | 부분. Agent Architecture and Design 15% | 없음 |
| B2 S/W 엔지니어링 (12h) | 없음 | 없음 (Python 기초를 선수 조건으로 둠) | 없음 | 없음 | 없음 | 다른 방향. Section 2가 코딩 Agent 활용 17% |
| B3 LLM·컨텍스트 (8h) | 12 Context Engineering, 13 Agentic Memory | Unit 1 LLM·특수 토큰 | Day 3 Context Engineering: Sessions & Memory | Module 2 State and memory, Module 5 장기 메모리 | Cognition, Planning, and Memory 10% | 3.1 모델 선택, 세션·메모리 |
| B4 도구·Agent 실행 (12h) | 3 설계 패턴, 4 Tool Use, 7 Planning | Unit 1 Tools·Thoughts·Actions·Observations, Unit 2 프레임워크 | Day 2 도구 설계, 장기 실행 작업 | Module 1, Module 3 human-in-the-loop | Agent Development 15% | 3.1 커스텀 Agent, 3.3 오케스트레이션 |
| B5 지식·시스템 연결 (8h) | 5 Agentic RAG, 11 프로토콜(MCP, A2A) | Unit 3 Agentic RAG | Day 2 MCP | 모듈 제목에 없음 | Knowledge Integration and Data Handling 10% | 3.2 RAG, Agent 권한, MCP 서버 |
| B6 평가·개선 (10h) | 제목에 없음 (10 Production에 포함 여부 미확인) | Unit 2 보너스 Observability and Evaluation, Unit 4 벤치마크 | Day 4 Agent Quality | LCAE Testing Agents 25%, Building Reliable Agents with LangSmith | Evaluation and Tuning 13% | 4.1 평가 세트, 지속 평가 |
| B7 보안·통제 (8h) | 6 Trustworthy Agents, 18 Securing AI Agents | 없음 | 부분. Day 1 백서의 신원·정책, Day 2 MCP 위험과 사람 승인 | 부분. Module 3 human-in-the-loop | Safety, Ethics, and Compliance 5% + Human-AI Interaction and Oversight 5% | Section 5 보안·거버넌스 15% |
| B8 배포·운영 (10h) | 10 AI Agents in Production, 16 Deploying Scalable Agents | 부분. Unit 2 보너스 관측성 | Day 4 관측성·로깅·추적, Day 5 Prototype to Production | Module 6 배포, LCAE Deploying 25% + Monitoring 25% | Deployment and Scaling 13% + Run, Monitor, and Maintain 5% | 4.2 배포·장애 진단·모니터링 |
| B9 선택 심화 (6h) | 8 Multi-Agent, 15 Computer Use, 17 Local Agents | Unit 1 보너스 미세조정, Unit 3 보너스 | Day 5 A2A 멀티 Agent | Module 4, Introduction to Deep Agents | 해당 없음 | Section 1 로우코드 13% |
| B10 통합 캡스톤 (16h) | 없음 | Unit 4 최종 프로젝트 (GAIA 30% 이상) | 라이브 기수 캡스톤 | LCAE 실습 환경 문항 | 감독 객관식 시험 | 감독 시험 + 합격 후 hands-on 랩 |

B1은 강의형 과정보다 두 문서가 직접 다룬다. [Anthropic 글](https://www.anthropic.com/engineering/building-effective-agents)의 "When (and when not) to use agents"와 [OpenAI 가이드](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)의 "When should you build an agent?"다.

### 3.2 비중 비교

초안의 시간 비중은 96시간 기준으로 B4 12.5%, B2 12.5%, B6 10.4%, B8 10.4%, B7 8.3%다. 공개 시험 개요와 나란히 놓으면 다음과 같다.

| 주제 | 초안 | NVIDIA NCP-AAI | Google PAA | LCAE | AWS AIP-C01 |
|---|---|---|---|---|---|
| 평가 | B6 10.4% | 13% | 평가+배포 약 22% | 25% | 11% |
| 배포·운영 | B8 10.4% | 13% + 5% | (위에 포함) | 25% + 25% | 12% |
| 보안·통제 | B7 8.3% | 5% + 5% | 약 15% | 영역 제목에 없음 | 20% |

판단: 초안의 평가·운영 비중은 공개 시험의 범위 안에 있다. 보안·통제는 Google PAA(약 15%)와 AWS(20%)보다 낮다. Guide의 승인 기준이 제안서 8.5의 필수 통과 기준이고 그 대부분이 권한·승인·중복 실행이므로, B7을 늘리거나 B4·B5 실습에 권한 시험을 끼워 넣는 쪽을 검토할 만하다. 시험 문항 비중과 교육 시간 비중은 같은 척도가 아니므로 방향만 참고한다.

### 3.3 공개 커리큘럼은 가르치는데 초안에 없는 것

- **메모리와 컨텍스트 엔지니어링을 독립 단원으로.** Kaggle Day 3, Microsoft 12·13, LangGraph Module 2·5, NVIDIA 10% 영역이 모두 따로 다룬다. 초안 B3은 토큰·컨텍스트·구조화 출력이고 장기 메모리와 세션 관리는 드러나 있지 않다.
- **Agent 간 프로토콜(A2A).** Microsoft 11, Kaggle Day 5, Google PAA 3.3이 MCP와 나란히 다룬다. 초안 B5는 API·MCP 비교까지다.
- **계획·성찰 설계 패턴.** Microsoft 7·9, DeepLearning.AI Module 2. 초안 B4는 실행 루프·상태·재시도 중심이다.
- **프레임워크 비교.** Hugging Face Unit 2가 세 프레임워크를 나란히 가르치고 Microsoft 2도 다룬다. 초안은 특정 도구보다 원리를 가르친다는 방향이라 의도적으로 뺀 것으로 읽힌다.
- **코딩 Agent 활용과 확장.** Google PAA Section 2(약 17%)가 코딩 Agent에 MCP 서버·스킬·서브에이전트를 붙이고 샌드박스에서 쓰는 것을 출제한다. Anthropic 과정 목록에도 agent skills, subagents 과정이 따로 있다. 초안은 실기에서 AI 코딩 도구를 허용한다고만 적었다. 마켓에 올라가는 대상에 tool이 포함되므로 관련이 있다.
- **모델 선택과 비용.** Google PAA 3.1(LLM과 SLM, 자체 호스팅), AWS Domain 4(12%). 초안 B8에 자원이 한 단어로만 있다.
- **도구별 위험 등급.** OpenAI 가이드의 tool safeguards. 초안 B7의 최소 권한·승인과 이어지지만 등급 매기기 자체는 없다.
- **멀티 Agent를 필수로.** Microsoft 8, Kaggle Day 1·5, Berkeley 강의 2회. 초안은 선택 심화에 두었다. 판단: 초안의 위치가 Anthropic 글의 "필요한 복잡도부터"와 맞으므로 그대로 둘 만하다.

### 3.4 초안에는 있는데 공개 커리큘럼이 다루지 않는 것 (자체 제작 대상)

- **B2 S/W 엔지니어링 전체.** 확인한 과정은 모두 Python과 개발 기초를 선수 조건으로 두고 가르치지 않는다. 초안도 입과 대상이 이미 코드를 짤 수 있다고 전제한다. 판단: 12시간을 그대로 둘지, 입과 진단으로 넘기고 Agent 서비스에 특화한 테스트·재현성만 남길지 결정이 필요하다.
- **B1의 과제 정의서.** 사용자·기준선·성공 기준을 적고 일반 자동화와 비교하는 산출물은 공개 과정에 없다. 근거 문서 두 편은 있으나 양식과 사례는 직접 만들어야 한다.
- **중복 요청, 중단 후 재개, 재시도가 업무를 두 번 처리하지 않는지에 대한 시험.** AWS Task 2.1이 종료 조건·타임아웃·서킷 브레이커를 언급하고 Kaggle Day 2가 승인 대기 후 재개를 다루지만, 중복 처리 방지를 시험 항목으로 둔 과정은 찾지 못했다.
- **오래된 문서와 권한별 검색 결과 차이.** RAG 단원은 어디에나 있으나 문서 갱신과 사용자별 접근권한을 실패 사례로 다루는지는 제목에서 확인되지 않는다.
- **변경 전후 회귀 비교와 기준선.** 평가 세트 구축은 공통으로 있으나 기존 수작업·자동화를 기준선으로 삼는 설계는 사내 과제에 맞춰야 한다.
- **인수인계와 운영 담당 지정.** 공개 과정에 없다. Guide의 승인 기준이기도 하다.
- **사내 환경 고유 내용.** 사내 모델의 OpenAI 호환 API, 사내 권한 체계, 배포 공간, 마켓 등록 절차.
- **개인 실기·구술·4주 운영 관찰이라는 평가 방식.** 공개 인증은 객관식 감독 시험이 주류다. 실습을 붙인 것은 Google PAA의 사후 랩과 LCAE의 실습 환경 문항, Hugging Face의 벤치마크 점수 정도다.

## 4. Guide 벤치마크 대상

### 4.1 Agent 전용 Guide 인증은 찾지 못했다

2장에서 확인한 Agent 관련 인증은 모두 응시자 본인의 설계·구축 역량을 본다. 타인의 Agent를 리뷰하거나 지도한 실적을 요구하는 것은 없었다. 가장 가까운 것은 Anthropic 과정 목록의 [Teaching AI Fluency](https://anthropic.skilljar.com/)로, 강사가 이끄는 환경에서 AI 활용 역량을 가르치고 평가하는 과정이라고 소개되어 있다. 다만 대상이 AI 활용 교육이고 Agent 설계 리뷰가 아니며, 과정 내용은 목록의 한 줄 설명까지만 확인했다. 이 결론은 이번에 조사한 범위 안에서의 것이다.

### 4.2 인접 제도 비교표

| 출처 | 승급 요건 | 리뷰·멘토링 품질을 어떻게 증빙하는가 | 갱신·유지 | 재사용 조건 |
|---|---|---|---|---|
| [Google 코드 리뷰 가이드](https://google.github.io/eng-practices/review/reviewer/looking-for.html) | 자격 제도 아님. 리뷰어가 볼 항목을 정의 | 해당 없음 | 해당 없음. [저장소는 보관 상태](https://github.com/google/eng-practices) | CC-BY 3.0 |
| [Kubernetes 커뮤니티 멤버십](https://github.com/kubernetes/community/blob/master/community-membership.md) | Reviewer: 멤버 3개월 이상, 주 리뷰어로 PR 5건 이상, 실질적인 PR 20건 이상 리뷰 또는 병합. Approver: Reviewer 3개월 이상, 주 리뷰어로 실질적인 PR 10건 이상, PR 30건 이상 리뷰 또는 병합 | 상위 역할자가 추천하고 다른 소유자의 이의가 없어야 한다. 권한은 OWNERS 파일에 이름을 올리는 공개 변경으로 부여 | 12개월간 기여가 없으면 조직에서 제외, 재신청 필요 | 페이지에서 라이선스 확인 못함 |
| [The Carpentries 강사 양성](https://carpentries.org/instructor-training/) | 16시간 교육(이틀 종일 또는 반일 4회) 후 [체크아웃 3단계](https://carpentries.github.io/instructor-training/checkout.html): 환영 세션 1시간, 교수 시연, 커뮤니티 참여 활동 1건. 교육 종료 후 90일 이내 | 교수 시연에서 5분간 참여형 수업을 하고 Instructor Trainer가 평가한다. 수정 사항을 지정해 다시 시키기도 한다. 다른 교육생도 피드백한다 | 확인한 페이지에 명시 없음 | [교재 CC-BY 4.0](https://carpentries.github.io/instructor-training/) |
| [ASQ Six Sigma Black Belt](https://www.asq.org/cert/six-sigma-black-belt) | 서명된 확인서(affidavit)가 있는 프로젝트 2건, 또는 프로젝트 1건 + 관련 경력 3년. 필기 165문항(채점 150), 4시간 18분, 오픈북 | 프로젝트 확인서. 멘토링 실적 요구는 이 단계에 없음 | [3년마다 갱신. 18 RU 또는 재시험](https://www.asq.org/cert/recertification) | 자격 제도. 구조만 참고 |
| [ASQ Master Black Belt](https://www.asq.org/cert/master-black-belt) | 유효한 Black Belt 보유 + 포트폴리오 심사 통과 + (Black Belt 또는 MBB 역할 5년 이상 또는 Black Belt 프로젝트 10건) + 시험 | 포트폴리오를 MBB 전문가 패널이 심사한다. Teaching·Coaching·Mentoring, Occupational Experience & Responsibility, Technical Experience/Innovation 세 지표 각각에서 최저점을 넘어야 한다. 미달이면 사유를 받는다 | [3년마다 갱신](https://www.asq.org/cert/recertification). [Green Belt는 평생 유효](https://www.asq.org/cert/recertification) | 자격 제도. 구조만 참고 |

Google 가이드가 리뷰어에게 보라고 하는 항목은 Design, Functionality, Complexity, Tests, Naming, Comments, Style, Consistency, Documentation, Every Line, Context, Good Things다. Carpentries 강사 양성 교재의 단원은 Building Skill With Practice, Expertise and Instruction, Memory and Cognitive Load, Building Skill With Feedback, Motivation and Demotivation, Equity·Inclusion·Accessibility, Teaching is a Skill, Live Coding is a Skill, Preparing to Teach, Working With Your Team 등 25개다.

### 4.3 G1~G4 대조

| 초안 모듈 | 가장 가까운 공개 근거 | 가져올 것 (판단) |
|---|---|---|
| G1 설계·코드 리뷰 (6h) | Google 코드 리뷰 가이드, Kubernetes Reviewer/Approver 정의 | Google의 항목은 일반 코드용이다. Agent 고유 항목(도구 계약, 권한, 종료 조건, 평가 세트)은 제안서 8.5와 OpenAI 가이드의 도구 위험 등급에서 뽑아 덧붙여야 한다. Kubernetes가 Reviewer(품질·정확성)와 Approver(전체 수용 판단)를 나눈 것은 Guide의 리뷰와 마켓 승인을 구분하는 데 쓸 수 있다 |
| G2 멘토링·교육 설계 (6h) | Carpentries 강사 양성 교재 | 교수법만 16시간이다. 초안은 6시간이다. 교재가 CC-BY 4.0이라 연습·피드백·인지 부하 단원을 출처 표기 후 고쳐 쓸 수 있다. 내용이 LLM과 무관해 사내 제약을 받지 않는다 |
| G3 평가·심사 일치 (6h) | Carpentries 교수 시연(Trainer 평가), ASQ MBB 포트폴리오 패널 | 심사자 기준 정렬(calibration)을 직접 규정한 공개 표준은 확인하지 못했다. MBB가 세 지표 각각에 최저점을 두는 방식은 초안 9.1의 "모든 영역 3단계 이상"과 같은 구조다 |
| G4 재사용·운영 전파 (6h) | Kubernetes Approver 책임(기여자·리뷰어 멘토링), Carpentries 체크아웃의 참여 활동 | 직접 대응하는 공개 과정은 없다. 자체 제작 |

### 4.4 승급 요건 대조

초안의 Guide 승급 요건은 교육 이수 + 본인 Agent 1건 + 타인과 함께한 Agent 2건이다. 공동 개발 이력과 가이드 이력을 동등하게 인정하며, 공동 개발 건은 본인 기여를, 가이드 건은 멘티의 독립 수행을 확인한다.

- **프로젝트 건수.** ASQ Black Belt의 "서명된 확인서가 있는 프로젝트 2건"과 규모가 같다. 제조 회사 구성원에게 익숙한 틀이다. 판단: 멘티나 과제 책임자가 서명하는 확인서 형식을 빌리면 실적 증빙이 단순해진다.
- **리뷰 건수.** 초안 7.2는 실질 리뷰 3건 이상이다. Kubernetes는 Reviewer에 주 리뷰 5건과 리뷰·병합 20건, Approver에 주 리뷰 10건과 30건을 요구한다. 판단: 대상과 규모가 다른 오픈소스 프로젝트의 숫자이므로 그대로 옮길 수는 없다. 다만 승인 권한을 주기 전에 쌓게 하는 리뷰 이력으로는 3건이 적은 편이다. 첫 승인 몇 건을 기존 승인자와 함께 하는 기간을 두는 방법이 있다.
- **멘티의 독립 수행 확인.** 확인한 제도 중 지도받은 사람의 독립 수행을 직접 요건으로 둔 것은 없다. ASQ MBB가 교육·코칭·멘토링을 포트폴리오 지표로 심사하는 것이 가장 가깝다. 이 요건은 초안 고유의 것이고 근거 사례 없이 직접 설계해야 한다.
- **추천과 이의 없음.** Kubernetes는 상위 역할자의 추천과 다른 소유자의 무이의를 요구한다. 초안은 심사단 채점만 있다. 판단: 같은 팀 구성원이나 기존 Guide의 추천을 보조 증거로 넣을 수 있다.
- **갱신 주기.** 초안은 2년 유효, 매년 활동 확인, Guide 실적이 1년간 없으면 Builder로 유지다. 공개 사례는 Kubernetes 12개월 무활동 시 제외, ASQ 3년, NVIDIA 2년, LCAE 24개월, Google PAA 1년, Microsoft 매년이다. 판단: 초안 값은 이 범위 안에 있다.

## 5. 마켓 승인 관문 참고 사례

사내 Agent 마켓의 심사 체크리스트를 공개한 조직은 찾지 못했다. 공개된 것은 외부 마켓의 심사 지침과, 조직 내부 배포를 다루는 제품 문서다.

| 출처 | 공개된 심사 내용 | 초안 관문에 주는 시사점 (판단) |
|---|---|---|
| [OpenAI 앱·플러그인 제출 지침](https://developers.openai.com/apps-sdk/app-submission-guidelines) | 도구 설명이 스키마와 실제 동작에 일치할 것. 도구마다 readOnlyHint, destructiveHint, openWorldHint를 명시적 불리언으로 표기. 필요한 최소 입력만 요청. 권한 요청은 필요한 범위로 제한. 체험판·데모는 거절. 지원 연락처를 정확히 유지. 위반이 드러나면 승인 후에도 제거 | 도구마다 읽기 전용인지 파괴적인지 선언하게 하는 항목은 제안서 8.5의 "승인 전 쓰기 금지"를 점검 가능한 형태로 바꿔 준다 |
| [Slack Marketplace 심사 가이드](https://docs.slack.dev/slack-marketplace/slack-marketplace-review-guide) | 가장 덜 허용적인 권한 범위를 최소 개수로 요청. 앞으로 만들 기능의 권한은 승인하지 않음. 미완성·베타는 반려. 유지보수와 사용자 지원 준비. 게시된 앱의 변경은 재제출이 필요하고 기능·권한 추가에는 시연 영상이나 스테이징 앱 요구 | 초안의 승인 기준에는 등록 시점 기준만 있다. 등록 후 변경 시 재승인 규칙이 없다. 권한이나 도구가 늘어날 때만 재승인하는 식의 구분을 참고할 수 있다 |
| [Microsoft 365 Copilot 조직 내 게시](https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/publish-plugin-organization) | 시험을 마친 정확한 버전을 제출. 게시자·지원·개인정보·이용 약관 정보 제공. 관리자에게 넘길 때 시험 증거, 알려진 한계, 지원 정보, 업데이트 담당자를 전달. [게시 기록](https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/publish)에 지원·업데이트·보안 대응·제거 책임자를 남김. 게시가 승인·배포·활성화를 자동으로 끝내지 않는다고 명시 | 초안의 "운영 담당자·인수인계 문서 유무"와 같은 방향이다. 제거 책임자와 알려진 한계까지 기록하게 한 점을 더할 수 있다 |
| [Claude Code 조직용 플러그인 관리](https://code.claude.com/docs/en/plugins/org) | 체크리스트가 아니라 통제 수단을 제공한다. 관리자가 허용할 마켓 출처를 allowlist로 제한하고(strictKnownMarketplaces), 차단 목록을 두고, 설치·로드 이벤트를 감사한다. 허용된 마켓 안의 개별 항목은 allowlist로 걸러지지 않는다고 적혀 있다 | 마켓 단위로 신뢰를 주면 그 안의 항목 품질은 마켓 운영자가 책임져야 한다. Guide의 1차 승인이 그 자리에 해당한다 |

## 6. 재사용 가능 여부 정리

라이선스 열은 출처가 밝힌 내용이다. 나머지 열은 그 라이선스에 대한 이 문서의 해석이며 법무 검토를 거친 것이 아니다.

| 출처 | 라이선스 | 복사 | 수정·각색 | 참조만 | 출처 표기 |
|---|---|---|---|---|---|
| Microsoft AI Agents for Beginners | [MIT](https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/LICENSE) | 가능 | 가능 | | 저작권·라이선스 고지 유지 |
| Hugging Face Agents Course | [Apache-2.0](https://raw.githubusercontent.com/huggingface/agents-course/main/LICENSE) | 가능 | 가능 | | 라이선스 고지 유지, 변경 사실 표시 |
| LangChain Academy 실습 저장소 | [MIT](https://github.com/langchain-ai/langchain-academy) | 가능 (저장소 코드) | 가능 | 영상 강의 | 저작권·라이선스 고지 유지 |
| The Carpentries 강사 양성 교재 | [CC-BY 4.0](https://carpentries.github.io/instructor-training/) | 가능 | 가능 | | 필요 |
| Google 코드 리뷰 가이드 | [CC-BY 3.0](https://github.com/google/eng-practices) | 가능 | 가능 | | 필요 |
| Anthropic Courses 저장소 | [CC BY-NC 4.0](https://raw.githubusercontent.com/anthropics/courses/master/LICENSE) | 비상업적 목적에 한함 | 비상업적 목적에 한함 | | 필요. 사내 교육이 비상업적 사용에 해당하는지는 확인 필요 |
| OWASP Top 10 for Agentic Applications | [사이트 콘텐츠 CC BY-SA 4.0](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) | 가능 | 가능. 파생물도 같은 라이선스로 공개해야 함 | | 필요 |
| Kaggle·Google 5-Day 백서·코드랩 | 명시 없음 | | | 링크·인용 | 인용 시 |
| Anthropic Building effective agents | 명시 없음 | | | 링크·인용 | 인용 시 |
| OpenAI A practical guide to building agents | 명시 없음 | | | 링크·인용 | 인용 시 |
| DeepLearning.AI, Berkeley MOOC | 명시 없음 | | | 링크·수강 안내 | 인용 시 |
| 벤더 인증 시험 개요 (NVIDIA, Google, Microsoft, AWS, LangChain) | 공개 라이선스 아님 | | | 영역 구성·비중 참고 | 인용 시 |
| Kubernetes 멤버십 문서, ASQ 자격 요건 | 확인 못함 / 자격 제도 | | | 제도 구조 참고 | 인용 시 |

OWASP 문서는 B7의 위협 목록 근거로 쓸 수 있어 표에 넣었다. 발행일(2025-12-09)과 사이트 라이선스만 확인했고 10개 항목 본문은 읽지 못했다.

## 7. 확인하지 못한 것

- **Kaggle 5-Day 2026년 기수.** 검색 결과에 2026년 6월 15~19일 기수가 있다고 나왔으나 Kaggle·Google의 1차 페이지에서 확인하지 못했다. 본문은 2025년 11월 기수의 자습 가이드만 근거로 했다.
- **Claude 인증의 시험 형식.** 문항 수, 시간, 합격선, 가격, 영역 비중은 제3자 글에만 있었고 Anthropic 페이지에서 확인하지 못해 본문에 넣지 않았다. academy.claude.com 첫 화면에는 인증 안내가 없었다.
- **Microsoft AB-100(Agentic AI Business Solutions Architect).** 검색 결과로 존재와 학습 가이드 주소만 확인했고 페이지를 읽지 않았다. 본문에서 제외했다.
- **OpenAI 인증.** Agent 개발자 대상 인증은 검색에서 찾지 못했다. OpenAI 사이트에서 직접 확인하지는 않았다.
- **NVIDIA 출제 비중 합계.** 조회한 값의 합이 98%였다. 원인을 확인하지 못했다.
- **LangChain 인증의 재응시 규정.** 공식 인증 페이지에 없었다.
- **Berkeley MOOC.** 요청에 있던 llmagents-learning.org/f25는 404였다. 같은 과정이 agenticai-learning.org/f25에 있었다. 2026년 기수와 수강료 여부는 페이지에 명시 없음.
- **대학 강의.** CMU 11-768은 설명문만 읽었고 일정·과제는 확인하지 못했다. Stanford CS329A는 검색 결과로만 확인했다.
- **GitLab 메인테이너 승급 절차.** 핸드북 페이지가 길어 본문을 읽지 못했다. 본문에서 제외했다.
- **심사자 기준 정렬 표준.** ISO/IEC 17024 같은 인력 인증 표준은 조사하지 않았다. 4장의 "확인하지 못했다"는 조사하지 않았다는 뜻이다.
- **The Carpentries.** 강사 자격의 갱신·활동 유지 규칙과 교육비는 확인한 페이지에 없었다. carpentries.org/become-instructor와 docs.carpentries.org의 체크아웃 페이지는 열리지 않았다.
- **ASQ.** asq.org/cert/six-sigma-black-belt는 403이었고 www.asq.org 주소로 읽었다. 프로젝트 확인서에 누가 서명하는지는 1차 페이지에서 확인하지 못했다.
- **Microsoft 과정의 평가 단원.** 레슨 10 "AI Agents in Production"이 평가를 다루는지는 제목만으로 알 수 없어 대조표에 미확인으로 적었다.
- **모듈 대조표 전반.** 모듈·영역 제목과 소개 문단을 근거로 했다. 강의 영상과 실습 노트북을 열어 깊이를 확인하지는 않았다.
- **Kaggle 가이드 본문.** 일반 조회로는 제목만 반환되어 브라우저로 렌더링한 화면에서 읽었다. Day 3 이후의 과제 상세는 앞부분 요약까지만 확인했다.

## 8. 참고 자료

모두 2026-10-09에 확인했다.

**Builder: 커리큘럼**

- https://github.com/microsoft/ai-agents-for-beginners
- https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/LICENSE
- https://github.com/microsoft/ai-agents-for-beginners/blob/main/00-course-setup/README.md
- https://huggingface.co/learn/agents-course/unit0/introduction
- https://huggingface.co/learn/agents-course/unit4/introduction
- https://github.com/huggingface/agents-course
- https://raw.githubusercontent.com/huggingface/agents-course/main/LICENSE
- https://www.kaggle.com/learn-guide/5-day-agents
- https://blog.google/technology/developers/ai-agents-intensive/
- https://blog.google/technology/developers/ai-agents-intensive-recap/
- https://academy.langchain.com/
- https://academy.langchain.com/collections
- https://academy.langchain.com/courses/intro-to-langgraph
- https://github.com/langchain-ai/langchain-academy
- https://www.deeplearning.ai/courses/agentic-ai/
- https://agenticai-learning.org/f25
- https://rdi.berkeley.edu/agentic-ai/f25
- https://rdi.berkeley.edu/education
- https://cmu-agents.com/
- https://www.anthropic.com/engineering/building-effective-agents
- https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- https://github.com/anthropics/courses
- https://raw.githubusercontent.com/anthropics/courses/master/LICENSE
- https://anthropic.skilljar.com/
- https://academy.claude.com/
- https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/

**Builder: 인증**

- https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/
- https://cloud.google.com/learn/certification/agentic-architect
- https://services.google.com/fh/files/misc/professional_agentic_architect_exam_guide_english.pdf
- https://learn.microsoft.com/en-us/credentials/certifications/exams/ai-103/
- https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103
- https://aws.amazon.com/certification/certified-generative-ai-developer-professional/
- https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01.html
- https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01-domain2.html
- https://academy.langchain.com/pages/certifications-lcae
- https://kb.langchain.com/articles/5297173898-what-is-the-langchain-certified-agent-engineer-exam
- https://www.anthropic.com/news/claude-partner-network
- https://www.claude.com/blog/four-role-based-claude-certifications

**Guide**

- https://google.github.io/eng-practices/review/reviewer/looking-for.html
- https://github.com/google/eng-practices
- https://github.com/kubernetes/community/blob/master/community-membership.md
- https://carpentries.org/instructor-training/
- https://carpentries.github.io/instructor-training/
- https://carpentries.github.io/instructor-training/checkout.html
- https://www.asq.org/cert/six-sigma-black-belt
- https://www.asq.org/cert/master-black-belt
- https://www.asq.org/cert/recertification

**마켓 승인 관문**

- https://developers.openai.com/apps-sdk/app-submission-guidelines
- https://docs.slack.dev/slack-marketplace/slack-marketplace-review-guide
- https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/publish
- https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/publish-plugin-organization
- https://code.claude.com/docs/en/plugins/org
- https://code.claude.com/docs/en/plugin-marketplaces

**열리지 않은 주소**

- https://llmagents-learning.org/f25 (404)
- https://academy.langchain.com/pages/certifications (404)
- https://docs.carpentries.org/resources/instructors/checkout.html (404)
- https://carpentries.org/become-instructor/ (내용 없음)
- https://asq.org/cert/six-sigma-black-belt (403, www 주소로 대체)
- https://handbook.gitlab.com/handbook/engineering/workflow/code-review/ (본문 미확인)

## 관련 문서

- [Expert 분과 브레인스토밍](./expert-curriculum-brainstorm.md)
- [Expert 양성 실행 제안](./expert-curriculum-proposal.md)
- [AI/DT 교육 커리큘럼 설계 목차](./README.md)
