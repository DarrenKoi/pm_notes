# SK하이닉스 맞춤 AX Transformation Roadmap

이 로드맵은 SK하이닉스와 교육 업체가 공동 워크숍에서 도출했다. 로드맵과 과정별 학습 내용은 첨부 사진을 바탕으로 정리했다. 2027년에는 기반기술센터의 업무와 교육 대상자의 역량에 맞는 커리큘럼을 업체에 요청할 때 이 자료를 참고할 수 있다.

## 1. 트랙별 로드맵

### AX 비개발자 활용 트랙

업무 생산성 향상을 목표로, 로드맵은 다음 순서로 과정을 배치한다.

| 과정 | 로드맵 표기 | 주요 내용 |
|---|---|---|
| 생성형 AI 활용 업무 혁신 | 기초 · 14시간 | 생성형 AI와 프롬프트 엔지니어링, 문서 작성과 고도화, 반도체 시장 동향 분석 자동화, 경쟁사 데이터 취합 자동화, 사내 기술 문서 기반 지식베이스 구축 |
| 바이브코딩을 활용한 데이터분석/업무자동화 | 중급 · 21시간 | 수율과 가동률 데이터의 패턴을 분석하고 글로벌 반도체 사이클과 원자재 가격 변동 원인을 살펴본다. |
| AI 활용 자동화 실습 | 중급 · 28시간 | 노코드 기반 Agent 구현, 업무 자동화 워크플로우 설계, Dify와 n8n을 연계한 구현 |

### AX 리더십 과정

**AI Governance & AX 리더십 과정**은 AI 활용 기준을 수립하고 AX 추진 전략을 설계하는 리더를 대상으로 한다. 생성형 AI 트렌드와 업무 활용, 기업 AI 활용 원칙과 거버넌스 체계, 생성형 AI의 리스크·보안·윤리 기준, 부서별 AX 과제 발굴과 우선순위 도출, AI Agent 도입 전략과 조직 운영, 활용 성과지표와 ROI 평가체계 설계를 다룬다.

### AI 개발자 생성형 AI 트랙

로드맵의 상단은 **AI 에이전트 프레임워크 구현 → LLM 활용과 고급화 → 실전 AI 에이전트 Harness / 오케스트레이션** 순서다. 평가·관측성 과정은 하단에 배치되어 있다.

| 과정 | 로드맵 표기 | 주요 내용 |
|---|---|---|
| AI 에이전트 프레임워크 구현 | 중급 · 28시간 | LangChain·LangGraph로 End-to-End 실습 프로젝트를 진행하고, 실무에 적용할 수 있는 에이전트를 설계한다. |
| LLM 활용과 고급화 | 고급 · 35시간 | Advanced RAG, MCP, 소형 언어 모델 파인튜닝, 답변 품질 개선, 도메인 특화 에이전트 구현 |
| 실전 AI 에이전트 Harness / 오케스트레이션 | 고급 · 28시간 | 하네스 엔지니어링과 에이전트 행동 제어, 보안 샌드박스와 MCP 연동, 멀티 에이전트 오케스트레이션, AI 가드레일 |
| Agent Evaluation & Observability | 고급 · 14시간 | 에이전트 성능과 환각 평가, 비용·속도·품질 모니터링, 로그 추적, 평가 리포트와 RAG 품질 개선 대시보드 |

### AI 개발자 심화 트랙

반도체 데이터를 분석하고 모델을 학습·배포하는 과정이다.

| 과정 | 로드맵 표기 | 주요 내용 |
|---|---|---|
| MLOps 및 Edge AI 배포 | 고급 · 35시간 | 반도체 데이터 분석과 보안을 위한 로컬 구축, 데이터 드리프트 감지와 모델 재학습 자동화, 모델 경량화(양자화·프루닝·지식 증류), Edge AI 배포 전략 |
| 시계열 데이터와 딥러닝 | 고급 · 35시간 | 설비·공정 센서의 시계열 데이터 분석과 전처리, LSTM·GRU·Transformer 기반 모델링, Autoencoder·Isolation Forest를 활용한 이상 탐지 |
| 이미지 데이터와 딥러닝 | 고급 · 35시간 | 공정·웨이퍼 이미지의 특성 분석과 전처리, CNN·Vision Transformer 기반 이미지 분류, Detection·Segmentation 기반 결함 영역 탐지 |

## 2. 과정별 상세 커리큘럼

교육 시간은 각 과정의 일차별 표에 기재된 시간을 합산했다. 바이브코딩으로 완성하는 풀스택 서비스는 첨부 자료의 상세 과정 개요에 나온다.

### 생성형 AI 활용 업무 혁신 (2일 · 14시간)

- 대상은 SK하이닉스 생산직 구성원 중 생성형 AI 입문자다. 기본적인 PC 활용과 문서 작성 능력을 전제로 한다.
- 프롬프트를 설계해 반복 업무용 AI 비서(Skill, GPTs/Gems)를 만든다. 보고서 작성, 반도체 공정과 시장 데이터 분석에 적용한다.
- 도구는 ChatGPT, GPTs, Gemini/Gems, Claude, Excel, CSV다. 원문은 실습 비중을 70%로 제시하며, 반도체 현장 문서와 사내 지식베이스 구축 사례를 다룬다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | ASK · AI에게 제대로 일 시키는 법 | AI 답변 원리와 도구별 강점 비교, 사내 보안 기준과 AI 답변 검증, 역할·맥락·형식·조건을 지정하는 프롬프트, Few-shot과 단계적 사고, 반복 업무를 위한 Skill과 GPTs. 에러 메시지를 해석하고 사내 작업표준서를 참고하는 챗봇을 만든다. |
| 2일차 · 7시간 | SOLVE · 데이터로 답을 만든다 | 업무 자동화 포인트 진단, 배경·현상·원인·대책·기대효과를 담은 보고서, 수율·불량률·가동률의 이상 신호 탐색, 시장·경쟁사 동향 비교와 출처 검증. 보고서 자동 완성, 상위 5개 불량 유형 시각화, 개인 업무 적용 계획을 실습한다. |

### AI 활용 자동화 실습 (3일 · 21시간)

- 대상은 생성형 AI 기본 활용 경험자다. 프롬프트 작성 경험과 문서·데이터 기반 업무 프로세스에 대한 이해가 필요하다.
- 바이브코딩과 Dify로 AI 앱·챗봇을 만들고 n8n과 연계해 문서 처리와 반복 업무를 자동화한다. 사내 규정 Q&A, 반도체 기술 용어 챗봇 등 현업 사례를 다룬다.
- 도구는 Dify, n8n, 바이브코딩 도구, OpenAI API, Excel, CSV, PDF다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | START · 말로 만드는 AI의 시작 | 바이브코딩의 원리와 한계, 요구사항과 프롬프트 작성, Dify 앱·워크플로우·Agent, 사내 문서 RAG 연결과 검색 튜닝. 설비 점검 체크리스트 앱, 사내 규정·매뉴얼 Q&A, 반도체 기술 용어 챗봇을 실습한다. |
| 2일차 · 7시간 | CONNECT · 흩어진 업무를 하나로 잇다 | 자동화 대상 선정, n8n 트리거와 처리 흐름, 조건 분기·반복·예외 처리, Excel·CSV·PDF 처리와 메일·메신저 연동. 생산 실적 파일 취합과 요약 메일 발송, 불량률 기준 초과 알림을 실습한다. |
| 3일차 · 7시간 | AUTOMATE · 사람 손을 떠난 업무 | Dify와 n8n의 API 연동, 수집·요약·정리·배포, 웹 자료 수집과 출처 검증, 오류 알림·재시도·로그·보안·권한 점검. 반도체 시장·경쟁사 동향 요약, 사내 기술 문서 자동 응답, 팀 업무 자동화 프로젝트를 진행한다. |

### AI Agent 프레임워크 구현 (4일 · 28시간)

- 대상은 AI Agent 구현 입문자다. Python 기초와 Jupyter/Colab 사용 경험이 필요하다.
- LangChain·LangGraph 기반 Tool, Memory, RAG를 연결하고 Excel·SQL·멀티 에이전트까지 활용한다. 직무 주제의 합성 데이터를 사용한다.
- 도구는 Python, OpenAI API, LangChain, LangGraph, Chroma, pandas, SQLite다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | LangChain & Naive RAG | OpenAI API, ChatModel·Messages, Tool·Memory·Middleware, 문서 로딩·분할·임베딩·벡터 검색, 기본 RAG 흐름. 문서 Q&A와 Excel 데이터 분석 Agent를 구현한다. |
| 2일차 · 7시간 | LangGraph 워크플로우 | 상태 머신과 노드·엣지, 조건 분기, 상태 관리, Tool 실행 흐름을 다룬다. 업무 시나리오를 바탕으로 LangGraph Agent를 설계·구현한다. |
| 3일차 · 7시간 | Agentic RAG & 멀티 Agent | 쿼리 분해, 재순위화, Self-RAG, ChromaDB·Retriever 구성, Supervisor·Handoff, 역할 분담과 메시지 전달을 다룬다. |
| 4일차 · 7시간 | 통합 Mini Project | 주제 선정과 요구사항 정리, 시스템 설계, LangGraph 대화 흐름 구현, RAG 연동과 테스트·디버깅, 시연·발표, 동료 피드백을 진행한다. |

### LLM 활용과 고급화 (5일 · 35시간)

- 대상은 RAG·Agent 구현 경험이 있는 SK하이닉스 개발·공정·품질 데이터 실무자다. Python, OpenAI API, LangChain/RAG 기본 파이프라인 이해가 필요하다.
- LLM 구조와 활용 전략을 정리하고, RAG·Agent 품질 개선과 파인튜닝 데이터 설계를 연결한다.
- 도구와 기법은 LangChain, LangGraph, MCP, Deep Agents/Harness, Ragas, LLM-as-a-Judge, Unsloth/TRL이다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | LLM 이론 & 프롬프트 | Token·Context·Transformer·Attention, Prompt·Few-shot·CoT·ReAct를 다루고 프롬프트 품질을 비교한다. |
| 2일차 · 7시간 | RAG 품질 진단과 검색 기반 | Semantic Search·Chunking·Embedding·VectorStore를 살펴보고 기존 RAG의 검색 기준선을 진단한다. |
| 3일차 · 7시간 | Advanced RAG & 평가 | Multi-Query, HyDE, Query Decomposition, BM25, Hybrid Search, RRF, Reranking을 적용해 개선 전후를 평가한다. |
| 4일차 · 7시간 | MCP & Deep Agent/Harness | Tool Calling·MCP·Deep Agent/Harness, LLM-as-a-Judge·Ragas·Self-CRAG를 다룬다. 도구 연동 Agent를 평가한다. |
| 5일차 · 7시간 | Fine-Tuning Mini Project | Full Fine-Tuning과 PEFT, LoRA·QLoRA·양자화, Instruction/Embedding 학습 데이터를 다룬다. 반도체 Q&A와 검색 데이터를 주제로 튜닝 프로젝트를 진행한다. |

### 실전 AI 에이전트 Harness / 오케스트레이션 (4일 · 28시간)

- 대상은 고급 AI 개발자다. AI Agent의 구조와 활용을 이해해야 한다.
- 문제에 맞는 오케스트레이션 패턴을 선택하고 하네스, 샌드박스, MCP와 가드레일을 통합한 멀티 에이전트를 구현한다.
- 도구는 Claude, Claude Code(유료), Claude Agent SDK, OpenAI API, LangGraph다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | 멀티 에이전트 오케스트레이션 | Fan-out, Pipeline, Debate, Supervisor, Swarm을 비교한다. Planner·Executor·Critic·Reviewer·Reporter의 역할을 설계하고 수율 분석 리포트 자동화에 적용한다. |
| 2일차 · 7시간 | 에이전트 행동 제어 및 실행 하네스 | 승인·중단·재시도·재개, Trace 기반 실행 이력, Checkpoint 복구, Agent 지침 파일과 Skill 명세를 다룬다. |
| 3일차 · 7시간 | Tool Ecosystem 및 MCP 통합 | Structured Output, 도구 선택과 오류 처리, MCP Client·Server·Tool·Resource·Prompt, 외부 시스템 연동, 도구 권한을 다룬다. |
| 4일차 · 7시간 | AI Guardrails 및 Secure Agent | 프롬프트 인젝션·탈옥 사례, 입력·출력 검증, JSON Schema Validation, 개인정보 마스킹, 도구 권한, 사람 승인, 샌드박스를 다룬다. 공정 이상 분석·리포트 생성 Agent MVP를 구현한다. |

### 바이브코딩으로 완성하는 풀스택 서비스 (4일 · 28시간)

- 대상은 AI 협업 개발이 필요한 개발자다. 풀스택 개발 기초가 필요하다.
- 풀스택 웹 애플리케이션을 기획부터 배포까지 구현하고, AI를 활용한 개발 방법과 실무 아키텍처를 검증 가능한 MVP로 적용한다. 원문은 기존 대비 3~4배 빠른 개발 속도를 학습 목표에 제시한다.
- 기본 도구는 Claude Code, GitHub다. 세부 실습에는 Supabase, Smithery, Vercel, Swagger, React, Vite, GitHub Actions가 나온다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | Vibe Coding 개요, 환경 설정과 프롬프트 엔지니어링 | Claude Code 설치, 계정과 개발 환경 설정, 프롬프트 패턴과 작성 전략, 도메인별 예시를 다룬다. |
| 2일차 · 7시간 | Vibe Coding 도구와 MVP 기획 | CLI, MCP, 슬래시 명령어, 서브 에이전트, Hook, Skill을 다룬다. 요구사항, PRD, 사용자 시나리오, 와이어프레임, ERD와 데이터 모델을 작성한다. |
| 3일차 · 7시간 | Swagger와 MVP 백엔드 | 시나리오와 데이터 모델로 API 명세와 백엔드 목업을 구성한다. DB 연동, SOLID·Clean Architecture, 단위 테스트와 보안 점검을 진행한다. |
| 4일차 · 7시간 | MVP 프론트엔드, 연동 및 배포 | HTML·JSX와 React/Vite 프로젝트를 구성한다. 프론트엔드·백엔드를 연동하고 단위·E2E 테스트, 디버깅, GitHub Actions 기반 CI/CD를 진행한다. |

### Agent Evaluation & Observability (2일 · 14시간)

- 대상은 RAG나 Agent 애플리케이션 개발 경험이 있는 AI 개발자다. Python, LLM API, LangChain/LangGraph 기본 활용이 필요하다.
- 평가 데이터셋을 구축하고 Ragas·DeepEval로 품질을 평가한다. Langfuse로 실행을 추적해 품질·비용·응답 시간을 분석하고 프롬프트를 개선한다.
- 도구는 Langfuse, Ragas, DeepEval, LangChain, LangGraph, Deep Agents다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | RAG·Agent 평가 지표 및 데이터셋 구축 | RAG·Agent 구조를 복습하고 문서 기반 합성 평가셋을 만든다. Faithfulness·Answer Relevancy·Context Precision·Context Recall과 Task Completion·Tool Correctness·Argument Correctness·Step Efficiency를 측정한다. |
| 2일차 · 7시간 | Langfuse 기반 관측 및 Agent 품질 개선 | 검색·모델·도구 호출과 상태를 추적하고 실행 시간·토큰·비용을 확인한다. 실패 사례를 골라 평가 데이터셋으로 전환하고, 프롬프트 버전과 배포 라벨을 관리하며 변경 전후 품질을 비교한다. |

### MLOps 및 Edge AI 배포 (5일 · 35시간)

- 대상은 AI 모델 운영·배포 담당자다. 딥러닝 모델 학습과 Python 경험이 필요하다.
- MLflow로 실험 추적과 모델 등록을 수행한다. 데이터 수집, 학습, 서빙, 운영과 재학습을 자동화하고, 모델을 경량화해 폐쇄망 Edge에 배포한다.
- 기본 도구는 Python, MLflow, Docker, ONNX/TensorRT다. 세부 실습에는 Prefect, FastAPI, Evidently가 포함된다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | MLOps 개요 및 로컬 환경 구축 | MLflow 실험 추적, 온프레미스 모델 레지스트리 구성. 반도체 공정 데이터로 실험을 추적하고 모델을 등록한다. |
| 2일차 · 7시간 | 학습 파이프라인 자동화 | 수집·전처리·학습·평가 단계 분리와 재사용 설계, Prefect 기반 실행 흐름을 다룬다. 데이터 수집부터 모델 등록까지 자동화한다. |
| 3일차 · 7시간 | 모델 서빙 및 배포 | Batch/Online Inference, REST API, FastAPI 모델 서비스와 Docker 배포를 다룬다. 수율 예측 모델 API를 구축한다. |
| 4일차 · 7시간 | 모델 모니터링 및 운영 자동화 | 추론 로그와 성능 지표, Evidently 기반 데이터 품질 분석과 Drift 리포트를 다룬다. 드리프트 감지와 재학습 조건을 설정한다. |
| 5일차 · 7시간 | Edge AI 배포 | ONNX/TensorRT 런타임, 폐쇄망 디바이스 보안 배포와 배포 후 모니터링을 다룬다. 공정 이상 탐지 모델의 MLOps 파이프라인과 Edge 배포를 구현한다. |

### 시계열 데이터와 딥러닝 (5일 · 35시간)

- 대상은 설비·공정 엔지니어다. Python과 기초 통계 지식이 필요하다.
- 설비 센서 시계열 데이터를 전처리하고 모델링해 가동률을 예측한다. 딥러닝 기반 이상 탐지를 적용해 설비 예지보전 체계를 구축한다.
- 도구는 Python, PyTorch, statsmodels, scikit-learn이다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | 시계열 기초·전처리 | 센서 시계열 데이터의 특성, 추세·계절성 분해, 정상성 검정, 리샘플링과 결측치·이상치 처리를 다룬다. 설비 센서 데이터를 전처리한다. |
| 2일차 · 7시간 | 통계 분석·특성 공학 | ARIMA·SARIMA와 Random Forest·XGBoost를 다룬다. 설비 진동 신호의 특성을 추출하고 예측에 적용한다. |
| 3일차 · 7시간 | 딥러닝 모델링 (RNN/LSTM/GRU) | 시퀀스 모델과 다변량 시계열, LSTM·GRU 구조와 학습을 다룬다. LSTM으로 설비 가동률 예측 모델을 만든다. |
| 4일차 · 7시간 | Transformer 시계열 예측 | Transformer와 Self-Attention 구조, 다변량 장기 시퀀스 처리를 다룬다. Transformer로 공정의 다변량 시계열을 예측한다. |
| 5일차 · 7시간 | 이상 탐지·예지보전 | Autoencoder의 재구성 오차와 Isolation Forest의 임계값 설정을 다룬다. 최종 프로젝트에서 센서 기반 설비 이상 탐지와 예지보전 알림 시스템을 구축한다. |

### 이미지 데이터와 딥러닝 (5일 · 35시간)

- 대상은 품질·공정 엔지니어다. Python과 딥러닝 기초 지식이 필요하다.
- 웨이퍼·공정 이미지를 전처리하고 결함을 분류한다. 객체 검출과 영역 분할로 결함 위치를 찾고, 정상 패턴 이미지만으로 이상 탐지 모델을 학습한다.
- 도구는 Python, PyTorch, OpenCV, Ultralytics(YOLO)다.

| 일차 | 모듈 | 학습 내용 |
|---|---|---|
| 1일차 · 7시간 | 이미지 프로세싱·전처리 | 이미지 데이터의 특성, 정규화와 데이터 증강, 클래스 불균형 처리를 다룬다. 이미지 전처리·증강 파이프라인을 구성한다. |
| 2일차 · 7시간 | CNN 기반 이미지 분류 | CNN 구조와 전이학습, 최신 CNN 모델을 다룬다. 전이학습으로 불량 이미지를 분류한다. |
| 3일차 · 7시간 | Vision Transformer | ViT 구조와 패치 임베딩, Attention을 다룬다. ViT를 사용해 미세 패턴의 결함을 분류한다. |
| 4일차 · 7시간 | Detection (결함 검출) | YOLO와 Faster R-CNN을 비교하고 IoU·mAP로 객체 검출 성능을 평가한다. YOLO로 공정 결함을 검출한다. |
| 5일차 · 7시간 | 영역 분할·이상 탐지 | U-Net·Mask R-CNN을 활용한 결함 영역 분석과 Autoencoder의 재구성 오차를 다룬다. U-Net으로 결함 영역을 분할하고 합성곱 Autoencoder로 이상을 탐지한다. |

## 3. 2026년 과정 개요와 교육 시간 대조

| 과정 | 이번 로드맵 | 이번 상세 자료 | 2026년 과정 개요의 차수당 시간 |
|---|---|---|---|
| AI 활용 자동화 실습 / AI자동화 활용 실습 과정 | 28시간 | 3일 · 21시간 | 4일 · 28시간 |
| MLOps 및 Edge AI 배포 | 35시간 | 5일 · 35시간 | 4일 · 28시간 |

교육 시간은 각 자료의 표기를 따랐다. [2026년 SK하이닉스 AI 교육 과정 개요](./2026-vendor-training-overview.md)

## 출처

SK하이닉스와 교육 업체의 공동 워크숍에서 도출한 「SK 하이닉스 맞춤 AX Transformation Roadmap」과 과정별 교육 개요. 첨부 사진 `21112.jpg`부터 `21117.jpg`까지 총 6장에서 판독 가능한 본문을 정리했다.
