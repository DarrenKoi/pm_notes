---
tags: [ml, data-processing, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# ML/DL 정리 진행 기록

2026-10-04 원래 Markdown 19개/9,572행의 분포를 확인했다. 현재 루트 README·데이터 처리 4개·클래식 ML 6개·딥러닝 5개·배포 3개 모두 개별 검토했다. 아래 순차 기록의 당시 미검토 상태와 현재 상태를 구분한다. 외부/화면 검증·Claude 협의 미완료는 유지한다.

루트 README는 교육용 레시피임을 밝히고 데이터 처리 목차와 검토 범위를 추가했다. 실제 존재하지 않는 FastAPI 외부 경로 링크는 제거했다. 기존 상위/RAG 참조는 유지했고 폴더 간 신규 통합·이동·링크는 만들지 않았다.

데이터 처리 폴더의 README·verified-conditions·organization-log에 네 문서의 역할·변경·공식 근거·검증·Claude 미연결과 남은 미확인을 남겼다. 각 폴더의 개별 검토 후 아래 전체 대조를 수행했다.

데이터 처리의 세 단계 검증: 원래 루트/데이터 문서 5개 제목 97개·fence 72개 보존, Python AST 70개·실행 fixture 18종 통과, 새 상대 참조 오류 0개, 현재 9개 CLI 속성 및 README → 적용 조건 읽기 화면 탐색 통과. 비 Markdown 자료는 보존했다. 외부 다운로드·실제 API·plot 한글 글리프·Claude 협의는 미확인이다. 나머지 문서의 개별 검토는 계속 진행 중이다.

클래식 ML 워크플로우는 R²·weighted F1·분할·Pipeline·seed 단정을 수정하고 원래 절 27개/fence 15개를 보존했다. 공식 자료 7건, AST 13개/로컬 실행 7종, 현재 3개 상대 참조/CLI 및 실제 목차 탐색을 확인했다. 다른 클래식 ML 5개는 미검토다.

모델 평가 문서 추가 검토: 공식 API 7건·AST 15개·실제 전체 블록 순차 실행/plot 8개/검증 8종, 제목 20개/fence16개 보존, 현재 4개 참조/속성 및 실제 목차 탐색 확인. ML/DL 원문 7개 검토, 12개 대기.

회귀 문서 추가 검토: 제목18/fence10 보존, AST10·scikit-learn 흐름6종·PNG6, 새 참조 오류0/CLI5 및 실제 표 복구 재검증. XGBoost 3.4.1은 libomp 부재로 import 실패해 실행 미확인. ML/DL 원문8개 검토, 11개 대기.

분류 추가 검토: 원래16절/7fence·AST6개·LR/RF 검증9종/PNG1 확인, 새 참조0/CLI6 성공. native 두 패키지는 libomp 부재 import 실패, 분류 실제 읽기 화면은 CUA timeout으로 미완료. ML/DL 원문9개 검토, 10개 대기.

군집 추가 검토: 원래15절/15fence 보존·AST14·검증9종/PNG8·새 참조0/CLI7 확인. 실제 군집 읽기 화면은 CUA timeout으로 미완료. ML/DL 원문10개 검토,9개 대기.

튜닝 추가 검토와 클래식 전체 대조: 원래6개120절/75fence·AST69·새 참조0·CLI8 확인. 튜닝 실제 탐색은 축소 예산이며 별도 pruning/plot 구조를 검증했다. 전체 예산/native 실행과 분류·군집·튜닝 실제 읽기는 미완료다. ML/DL 원문11개 검토,8개 대기. 개별 근거와 세 단계 결과는 해당 폴더 정리 기록에 남겼다.

PyTorch 기초 추가 검토: 원래14절/9fence·AST8 보존, 실제 CPU20epoch·checkpoint 복원 포함8종 검증, 새 참조0·uniqueYAML3·CLI3 확인. GPU/worker/변환·실업무 성능·읽기 화면은 미확인. ML/DL 원문12개 검토,7개 대기.

학습 루프 추가 검토: 원래14절/8Python fence 보존·AST8, 작은 CPU fixture9종(표본 손실 집계/early stopping/저장·재개·best복원/실제SimpleCNN 포함)·PNG2, 새 참조0·uniqueYAML4·CLI4 확인. 원래 CIFAR/30epoch/GPU/worker/RNG replay·읽기 화면은 미확인. ML/DL 원문13개 검토,6개 대기.

CNN 추가 검토: 원래22절/13fence·AST11·실제 torchvision/CPU fixture7종·PNG1·새 참조0·uniqueYAML5·CLI5 확인. 반복된 모드 함수를 통합한 뒤 실제 학습 분기와 전체 fixture를 재검증했다. CIFAR/가중치 다운로드·50epoch/GPU/worker/읽기 화면은 미확인. ML/DL 원문14개 검토,5개 대기.

시퀀스 추가 검토: 원래19절/10fence·AST7·CPU 검증7종·PNG1 한글 표시 재검증·새 참조0·uniqueYAML6·CLI6 확인. 원래2,000점의 시간 분할과 packing 순서 복원을 확인했다. 50epoch/실제 센서/GPU/Obsidian 읽기·Claude 협의는 미확인. ML/DL 원문15개 검토,4개 대기.

전이학습 추가 검토와딥러닝 전체 대조: 원래5개85절/48fence·AST40·전이학습CPU7종/공식HF sourceAST2·새 참조0·uniqueYAML7·CLI7 확인. weights/CIFAR/BERT/전체10epoch/GPU/worker/읽기·Claude 협의는 미확인. ML/DL 원문16개 검토,배포3개 대기.

저장/로딩 추가 검토: 원래29절/16fence·AST14·CPU9종(실제 ONNX18/ORT 수치 비교 포함)·새 참조0·uniqueYAML3·CLI3 확인. GPU/판본변경/장애복구/DVC/MLflow 서비스/읽기·Claude 협의는 미확인. ML/DL 원문17개 검토,배포2개 대기.

FastAPI 서빙 추가 검토: 원래17절/17fence·AST11·CPU TestClient9종·bash2/YAML1·새 참조0·uniqueYAML4·CLI4 확인. 실제 socket/Docker/GPU/부하/CDN·읽기·Claude 협의는 미확인. ML/DL 원문18개 검토,실험 추적1개 대기.


## 현재 전체 대조 결과

실험 추적 원문 추가 검토: 원래18절/fence13개·AST10·실제 SQL/CPU 검증6종·bash2 syntax 확인. MLflow3.16.1/skops 신뢰 조건과 registry2versions/alias·전체 세 모델/CV5/Pipeline·validation 선택/test 단일 평가·CSV/JSONL 경계 조건을 기록했다. 상세 출처/판본·협의 보류는 deployment 정리 기록에 있다.

원래19개/9,572행의 문서가 모두 원래 경로에 있고 원래 code fence 수가 유지된다. 현재29개 문서에 검토일·개별 결과/목차/정리 기록이 있다. Python AST214개 오류0·unique YAML29개·링크/앵커 새 오류0·pm_notes Obsidian CLI properties29개 검토일 확인·ai-dt 비 Markdown18개 초기 해시 동일을 확인했다. 기존 root 상위/다른 주제 링크는 새로 만들지 않았다. 기존 제목 변경/고유 예제 보존과 실행 결과는 하위 정리 기록에 남겼다.

실제 Obsidian 읽기 화면은 데이터/워크플로우/평가/회귀에서 확인한 범위만 성공으로 남기고 분류 이후 문서는 반복 CUA timeout으로 미완료다. 실제 native XGBoost/LightGBM·외부 모델/데이터 다운로드·GPU/worker·전체 학습/튜닝 예산·운영 socket/Docker/인증/원격 registry·업무 승인·Claude 협의도 해당 문서에서 미확인으로 구분한다. 개별 문서 검토19/19와 모든 외부 검증 완료는 다르다.
