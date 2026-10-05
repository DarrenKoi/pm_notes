---
tags: [data-handling, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# 데이터 처리 문서 정리 기록

## 범위와 보존 원칙

기준일 2026-10-04. 원래 Markdown 32개와 비Markdown 자료 4개(실행 코드 3개·설정 1개)를 목록화했다. 원문 스냅샷과 SHA를 임시 검증 공간에 보존했다. 문서만 수정하며 실행 코드·첨부·기존 사용자 변경·원래 작성일을 보호한다. 이동·삭제·커밋·푸시는 없다. 다른 주제와 통합하거나 새 교차 링크를 만들지 않는다.

## 분류와 협의 상태

목차에 기존 네 주제 항목을 모두 보존하고 누락된 의존성·이벤트 참고 문서를 추가했다. 커리큘럼/패턴/시작 조건/저장소 연결의 차이를 설명해 같은 문법을 독립적인 목적에 맞게 읽도록 했다. 원래 문서의 고유 예제는 삭제하지 않았다.

Claude 협의를 시도하기 전에 HERDR_ENV=1을 확인했고 `herdr pane current --current`는 pane_not_found였다. 작업 전용 pane을 만들거나 다른 pane을 제어하지 않았다. Claude 의견은 없다. 완전 중복 통합·파일 분할/이동·웹훅 내구성 및 재전송 정책 결정은 협의가 필요해 보류한다. 공식 근거로 설명 오류를 바로잡는 작업은 진행했다.

## 개별 검토 결과

| 원래 문서 | 검토 결과 | 남은 검증 |
|---|---|---|
| [README](./README.md) | 네 주제와 읽기 순서 보존; Task/이벤트 목차 추가, 유사 문서 역할 구분 | 남은4개(역공학) 개별 검토 후 목차 재대조 |
| [Task 의존성](./task-dependencies.md) | 2.10.5 기준; trigger rule/skip/leaf 판정/슬롯/다른run·DAG 할당/exit·ds·공유파일·논리날짜/retry jitter 정정; 모든 기존 절 보존 | 실제 Airflow DAG parsing/scheduler·사내 executor·설치 패키지 |
| [이벤트 기반 실행](./event-driven-execution.md) | 2.x와3.x API/인증 구분; exactly-once/99%/즉시 표현 철회; Dataset 성공·조건/평문·수신자 미완성, 불필요·잘못된 import 정정; HTTP 오류 검사 | 실제 provider/MinIO 알림/API 인증·중복 제거/배포 설계 |

## 확인 근거와 판본

모두 2026-10-04 확인. Airflow 2.10.5를 예제 대조 판본으로 고정했으며 최신이나 사내 설치 버전이라는 뜻은 아니다.

- [DAG와 의존성](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/dags.html), [BashOperator](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/operator/bash.html), [템플릿 날짜](https://airflow.apache.org/docs/apache-airflow/2.10.5/templates-ref.html).
- [Best Practices](https://airflow.apache.org/docs/apache-airflow/2.10.5/best-practices.html), [Python 격리 Operator](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/operator/python.html).
- [TriggerDagRunOperator tag 소스](https://github.com/apache/airflow/blob/2.10.5/airflow/operators/trigger_dagrun.py), [ExternalTaskSensor](https://github.com/apache/airflow/blob/2.10.5/airflow/sensors/external_task.py), [TaskInstance retry 계산](https://github.com/apache/airflow/blob/2.10.5/airflow/models/taskinstance.py).
- [Dataset](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/datasets.html), [deferral](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/deferring.html), [Sensors](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/sensors.html).
- [Airflow 3 public interface](https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html), [3.3.2 API 인증](https://airflow.apache.org/docs/apache-airflow/3.3.2/security/api.html): 확인 당시 문서 표시3.3.2, 최신 보증 아님.
- [S3 일관성](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html), [HTTPX 상태 검사](https://www.python-httpx.org/quickstart/).

MinIO 문서·Amazon provider 일부 페이지와 legacy generated API 페이지는 조회 오류였다. 확인하지 못한 MinIO/설치 provider 기능은 미확인으로 남겼다. core 일부 API는 Apache 공식 2.10.5 tag 소스로 교차 확인했다.

## 참조 정리

의존성·이벤트 문서의 기존 상위 AI/DT 링크를 제거해 이 주제 안의 참조로 유지했다. 고유 본문 내용은 삭제하지 않았다. 현재 남은4개 역공학 문서의 기존 참조는 개별 검토 때 처리한다. 일반 Markdown의 상대 링크와 기본 callout/frontmatter만 사용하며 추가 Obsidian plugin은 요구하지 않는다.

## 나머지 원문 목록 — 개별 검토 대기

아래 항목은 발견한 원문 목록이며 검토 완료 기록이 아니다.

- [binary-reverse-engineering/01-toolkit-reference.md](./binary-reverse-engineering/01-toolkit-reference.md) — 검토 대기
- [binary-reverse-engineering/02-cd-sem-formats.md](./binary-reverse-engineering/02-cd-sem-formats.md) — 검토 대기
- [binary-reverse-engineering/04-coordinate-and-recipe-files.md](./binary-reverse-engineering/04-coordinate-and-recipe-files.md) — 검토 대기
- [binary-reverse-engineering/README.md](./binary-reverse-engineering/README.md) — 검토 대기

## 검증 결과

원래 32개 전체의 검토는 진행 중이며 세 단계 결과를 아래에 차례로 남긴다. 1차 당시 Native Obsidian 창은 cgWindowNotFound로 연결되지 않아 읽기 화면 검증이 미완료였다. 이후 연결이 복구돼 현재 수정/신규25개는 실제 읽기 탐색 결과를 추가했다. CLI 성공과 화면 렌더링 증거를 구분한다.


### 1차 수정 후 세 단계 검증

1. 원래32개 Markdown/비Markdown4개 경로 모두 보존했다. 개별 검토한 원문3개 외의29개는 SHA가 동일하다. 비Markdown4개(실행 코드3·설정1개) SHA도 동일하다. 의존성/이벤트 원래 모든 절 제목과 예제 fence 수가 유지됐고 2026-05-02 작성일을 보존했다. README의 기존 네 주제를 모두 포함했다.
2. Python 예제20개 AST 통과. 순수 추출→변환 함수의 생성 입력 처리1건, 실제 HTTPX0.28.1 MockTransport의 200/401/409/500응답4건, bash 종료/pipeline3건을 검증했다. 수신자가 HTTP 실패를 성공 처리하지 않는 것을 확인했다. Airflow를 설치하거나 실제 DAG parsing/scheduler·FastAPI 서버/인증·MinIO/장비·사내 endpoint를 실행하지 않았다. mock이 Airflow 실행 보증은 아니다. Dataset AND/OR는 2.9.3 공식 문서에서도 교차 확인했다.
3. 현재 수정3개+새 정리 기록1개, 총4개의 상대 링크/앵커/첨부 오류0, unique YAML와 검토일/status/type/tags 검사 및 exact pm_notes CLI properties4 통과. git diff --check -- ai-dt 통과. Native 창은 cgWindowNotFound로 연결되지 않아 4개 읽기 화면 렌더링 검증은 미완료다. CLI properties 성공을 화면 탐색 성공으로 간주하지 않았다.

추가로 cleanup/알림 leaf가 성공하면 중간 작업 실패에도 DAG run 성공이 될 수 있다는 [2.10.5 공식 판정 설명](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/dag-run.html)을 경고로 추가했다. 실행 순서 시연을 완전한 운영 실패 감지 설계로 소개하지 않는다. 원문3/32 검토,29개 대기이며 다음은 Airflow 커리큘럼·MinIO 문서를 문서별로 처리한다.


## Airflow 커리큘럼 첫 검토 — 2026-10-04

목차·01·02·08의 4개를 추가 검토했다. 원래 8단계 목차, 모든 기존 절, 작성일을 보존했다. 08의 H1 이후 원문 전체는 “2026-05-02 원문 시나리오” 아래에 정확히 보존하고 현재 지침을 앞에 추가했다. 최초 대조에서 구분 제목 뒤의 불필요한 줄바꿈 1개를 발견해 제거한 뒤 원문 비교를 통과했다. 고유 예제와 당시 가정을 현재 회사 사실로 덮어쓰지 않았다.

Connection 관리 화면에 접근하지 못해도 환경변수나 secret backend에서 제공한 값을 Task가 사용할 수 있으며, 그 값은 UI에 나타나지 않을 수 있다. Git Sync는 source 배포와 secret 공급, 패키지 설치를 구분해 설명했다. 회사 다수가 2.x라는 미측정 주장과 현재 권한 단정을 철회하고, 2.10.5 예제 판본과 3.x public API의 차이를 명시했다.

01의 빈 DAG는 골격이다. 02의 FTPClient와 URI replace는 설명 예제다. 08의 날짜용·시간용 main은 서로 다른 예제이며, CLI 진입점 누락·시간 문자열 slice·주석 처리된 업로드는 미완성 구현이다. 실제 정책, 구현 계약 통합, 파일 분할과 중복 통합은 Claude 협의 대기다. HERDR_ENV=1 확인 후 현재 pane 조회는 pane_not_found였으며 다른 pane을 제어하지 않았다.

2026-10-04 확인 근거: [Connection 2.10.5](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/connection.html), [secret backend](https://airflow.apache.org/docs/apache-airflow/2.10.5/security/secrets/secrets-backend/index.html), [module path](https://airflow.apache.org/docs/apache-airflow/2.10.5/administration-and-deployment/modules_management.html), [Helm DAG 배포](https://airflow.apache.org/docs/helm-chart/stable/manage-dag-files.html), [2.10.5 public interface](https://airflow.apache.org/docs/apache-airflow/2.10.5/public-airflow-interface.html), [3.x public interface](https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html), [MinIO API](https://docs.min.io/aistor/developers/sdk/python/api/). MinIO 페이지는 AIStor로 redirect됐고 사내 OSS 버전·권한·CA는 미확인이다. 기존 secure=False는 역사 시나리오이며 현재 권장값으로 사용하지 않는다.

### 문서별 추가 결과

| 원래 문서 | 검토 결과 | 남은 확인 |
|---|---|---|
| [커리큘럼 목차](./airflow-basic-to-advanced/README.md) | 8단계와 판본·사내 가정·검토 상태 구분 | 03~07 검토를 후속 기록으로 보완; 사내 조건은 미확인 |
| [01 기초](./airflow-basic-to-advanced/01-basic-concepts.md) | 파싱과 worker 실행, 빈 DAG·logical date·secret 공급 정정 | 실제 executor/scheduler·사내 권한 |
| [02 첫 DAG](./airflow-basic-to-advanced/02-first-dag-python-file.md) | schedule=None·bash 조건문 예외·module 이름 충돌·설명용 코드 구분 | 실제 FTP·서버 배포 |
| [08 Git Sync/Secret](./airflow-basic-to-advanced/08-bitbucket-git-sync-and-code-secrets.md) | 원문 정확 보존; env/backend 기능·누락 시 실패하는 loader·미완성 예제 구분 | secret 정책·실제 CA/MinIO·CLI 계약 협의 |

### 로컬 실행 범위

저장소 밖의 임시 Python 3.12.12 환경에 Airflow 2.10.5와 공식 constraints-3.12를 사용해 137개 패키지를 설치했다. uv pip를 사용했으며 [공식 설치 문서](https://airflow.apache.org/docs/apache-airflow/2.10.5/installation/installing-from-pypi.html)는 pip만 공식 지원한다고 명시한다. 따라서 uv 설치를 공식 지원 방식으로 소개하지 않는다. 당시 constraints의 재현성이 현재 보안 갱신이나 사내 적합성을 보장하지 않는다. 이 환경은 운영 서비스로 사용하지 않으며 원래 저장소 환경을 수정하지 않았다.

Python 예제 38개가 AST 검사를 통과했다. 실제 DagBag에서 8개 DAG(01 골격 1·02 완성 정의 5·08 역사 정의 2)를 발견해 Task ID와 순서 edge를 대조했고 import 오류는 없었다. source를 임시 파일로 추출하고 문서의 hello/FTP helper와 config fixture를 사용했다. 실제 PythonOperator의 외부 접속 없는 함수 1건을 실행했다. 환경변수 loader는 필수 값 4가지 누락 시 실패했고 오류 메시지에 비밀 값을 넣지 않았으며 TLS 설정 True를 반환했다.

실제 scheduler와 실행 worker, FTP·MinIO·API 접속, 인증·CA·TLS 연결, 업무 파일과 실제 secret은 사용하지 않았다. 08의 역사 DAG가 파싱된다고 원래 job 구현이 작동한다는 뜻은 아니다.

원래 32개 중 7개를 검토했다. 미수정 25개와 비Markdown 4개의 SHA가 동일하다. 모든 원래 경로, 고유 절, 08 본문, 작성일 보존 검사를 통과했다. 현재 수정/신규 8개의 상대 링크·앵커·첨부 오류는 없고 unique YAML·검토일/status/type/tags 및 exact pm_notes CLI properties 8개, git diff --check -- ai-dt 검사를 통과했다. CLI 출력의 오래된 4개 label은 저장된 8개 결과와 대조해 수정했으며 실제 loop는 8개를 처리했다.

Native 창 연결은 재시도에서 복구됐다. 이전 연결 불가 기록은 당시 상태이며, 아래 후속 화면 결과로 보완한다. Claude 협의는 미완료다.


복구된 pm_notes vault/Obsidian 1.13.7에서 현재 8개 문서를 빠른 전환기의 정확한 기존 경로로 열고 breadcrumb·읽기 모드·검토일을 확인했다. PageDown으로 끝까지 이동하며 native 접근성 상태의 본문 제목을 수집했다. frontmatter를 제외한 원문 고유 절 수와 모두 일치했다. 정리 기록에서 본문 밖 링크 미리보기의 제목 4개를 제외했다. 대표 실제 screenshot은 커리큘럼 목차·02·의존성·정리 기록 하단을 확인했고, 화면에서 발견한 검증 요약의 문장 간격을 개선해 다시 확인했다. 모든 행의 모든 픽셀을 검수한 것은 아니며 실제 모델·업무 서버 품질의 증거는 아니다.

| 문서 | 탐색 상태 수 | 원문 고유 절 = 관측 고유 절 |
|---|---:|---:|
| [README.md](./README.md) | 3 | 3 |
| [airflow-basic-to-advanced/README.md](./airflow-basic-to-advanced/README.md) | 7 | 8 |
| [airflow-basic-to-advanced/01-basic-concepts.md](./airflow-basic-to-advanced/01-basic-concepts.md) | 14 | 16 |
| [airflow-basic-to-advanced/02-first-dag-python-file.md](./airflow-basic-to-advanced/02-first-dag-python-file.md) | 23 | 14 |
| [airflow-basic-to-advanced/08-bitbucket-git-sync-and-code-secrets.md](./airflow-basic-to-advanced/08-bitbucket-git-sync-and-code-secrets.md) | 20 | 20 |
| [task-dependencies.md](./task-dependencies.md) | 22 | 24 |
| [event-driven-execution.md](./event-driven-execution.md) | 17 | 25 |
| [organization-log.md](./organization-log.md) | 8 | 12 |

이 추가 검증으로 현재 8개 문서의 이전 “읽기 화면 미완료” 기록을 보완했다. 미검토 25개와 다른 주제의 남은 읽기 화면 검증은 계속 대기다.


## 커리큘럼 03~07 추가 검토 — 2026-10-04

원문 5개를 추가 검토했다. 모든 원래 절 제목·예제 fence 수·작성일과 경로를 보존했다. 동일 주제의 기존 설명을 근거에 따라 정정했으며 파일 이동·삭제·회사 정책 통합은 하지 않았다.

| 원래 문서 | 수정 내용과 이유 | 남은 확인 |
|---|---|---|
| [03 스케줄·재시도](./airflow-basic-to-advanced/03-dependencies-scheduling-retry.md) | logical date/처리 구간/표시 timezone 분리; catchup 실행 순서 보장 철회; per-attempt timeout·all_done leaf·외부 trigger·골격 구분 | 실제 scheduler·backfill·외부 job 취소 |
| [04 데이터·상태](./airflow-basic-to-advanced/04-data-and-state.md) | 정의 시 XComArg와 runtime 값 구분; delete/insert transaction 필요; S3 copy/delete와 atomic rename 구분; 비밀 Git 저장 필수 추론 철회 | DB 격리·동시 writer·batch 게시/reader 계약·실제 MinIO |
| [05 실행 환경](./airflow-basic-to-advanced/05-packages-and-environments.md) | Bash PATH 선택과 PythonOperator 환경 구분; 전체 freeze 로그 제거; virtualenv 조건·캐시/함수 격리·CA 검증·constraints/core pin 보완 | 사내 index/CA·실제 venv·컨테이너/provider 실행 |
| [06 로컬 테스트](./airflow-basic-to-advanced/06-local-development-and-testing.md) | Parquet 내용까지 검사; DagBag의 예상 DAG/Task/edge 검사; tasks test 실제 부작용·미구현 query/storage stub 구분 | 실제 storage/client/CI·회사 배포 |
| [07 운영](./airflow-basic-to-advanced/07-advanced-operations.md) | Dataset 식별/성공 event·runtime mapping·pool slot/active run·Celery queue와 Pod 배치·callback 조건 구분 | scheduler 병렬성·클러스터·외부 알림·SLA 운영 |

확인 근거는 각 장에 판본·2026-10-04 확인일과 함께 남겼다. Airflow 2.10.5 문서와 Apache 공식 tag 소스, AWS S3 copy/move, PostgreSQL transaction(확인 페이지18), pip HTTPS/freeze(확인 페이지26.2.1), Python Packaging direct URL 표준을 대조했다. direct URL metadata는 인증정보 제거를 요구하지만 사내 경로 전체의 공개 적합성을 보장하지 않는다. 이를 일반적인 비밀번호 노출 단정으로 쓰지 않고 환경 조사 범위를 선택 패키지로 제한했다. Kubernetes의 core 문서 일부는 조회 오류였으며 2.10.5 공식 tag의 `execute_async`/`PodGenerator.from_obj`와 Celery `apply_async(queue=queue)`를 읽어 잘못된 queue 설명을 정정했다. 독립 provider release의 설치 성공이나 현재 cluster 검증으로 주장하지 않는다.

Claude 협의 연결은 HERDR_ENV=1 확인 후 `pane_not_found`였다. 의견을 받은 것으로 기록하지 않는다. DB/manifest 구현 계약·secret 정책·중복 예제의 완전 통합은 보류했고 공식 근거가 분명한 설명·예제 검사는 진행했다.

### 세 단계 추가 검증

1. 원래 Markdown 32개와 비Markdown 4개가 모두 존재한다. 새로 검토한 5개에서 모든 원래 절과 fence 수·2026-05-02 작성일을 보존했다. 08 원문 역사 본문은 정확히 동일하다. 미수정 20개와 실행 코드 3개·설정 1개의 SHA도 동일하다. 현재 원문 12/32개와 새 기록 1개를 검토했다.
2. 추가 Python fence 64개 AST 통과. Python 3.12.12/Airflow 2.10.5 환경에 공식 constraints-3.12의 virtualenv20.29.1·pandas2.1.4·numpy1.26.4·pyarrow19.0.0·pytest8.3.4를 임시 설치했다. 실제 DagBag에서 11개 DAG의 ID/Task 집합을 확인하고 순차·TaskFlow·mapping edge를 대조했으며 import 오류가 없었다. URI 반환만 하는 Task callable 1건, Airflow 실제 KST template의 UTC 날짜 경계 1건과 bash 구문, 실제 CSV→Parquet 내용 보존 pytest 1건이 통과했다. 보강한 문서 DAG test는 정상 graph를 통과하고 빈 graph·DAG 누락의 음성 fixture 2건을 실패로 검출했다. pytest의 이미 import된 plugin assert-rewrite 경고 1건은 실행 성공과 구분해 기록한다. 설치 도구 uv는 공식 지원 pip 설치와 구분한다. 실제 virtualenv payload 설치/실행·Kubernetes provider import·scheduler·외부 서비스·secret·업무 데이터는 사용하지 않았다.
3. Obsidian 1.13.7의 정확한 pm_notes 문서 경로와 breadcrumb·읽기 모드·검토일을 확인한 뒤 03~07을 끝까지 PageDown으로 읽었다. 본문 밖 미리보기 제목을 제외한 고유 절 수를 source와 대조한다. 대표 07 하단 screenshot에서 출처 링크·code span·callout 이후 검증 범위가 렌더링됐다. 현재 13개 문서의 상대 링크/앵커/첨부 오류 0, metadata 기본 검사·exact vault CLI properties 13개·git diff --check -- ai-dt가 통과했다. 추가 5개의 unique YAML 검사를 통과했으며 기존 8개는 앞선 검증 결과를 유지한다. 모든 행의 모든 픽셀·외부 렌더러 호환성 검사는 아니다.


| 추가 문서 | 탐색 상태 수 | source 고유 절 = 관측 고유 절 |
|---|---:|---:|
| 03 스케줄·재시도 | 20 | 17 |
| 04 데이터·상태 | 17 | 17 |
| 05 실행 환경 | 18 | 17 |
| 06 로컬 테스트 | 17 | 16 |
| 07 운영 | 20 | 21 |

5개가 모두 일치했다. 05의 일반 KubernetesPodOperator 안내 페이지는 provider 10.23.0으로 표시됐고 2.10.5와의 설치 호환으로 추론하지 않았다. 마지막 표·문장 보완 후 두 목차와 정리 기록을 처음부터 다시 읽고 참조를 재확인했다. 목차의 고유 절 3/8개와 기록의 14개가 source와 일치했고 새 검증 표의 실제 하단 screenshot도 확인했다. 현재 남은 원문 20개는 MinIO 1·정규화 9·역공학 10개다.


## MinIO 연결 튜토리얼 추가 검토 — 2026-10-04

원문 [Airflow + MinIO](./airflow-minio-tutorial.md)의 모든 기존 절·28개 fence·작성일과 TaskFlow·다운로드 두 변형·preprocess·upload·Windows/WSL/Compose/Astro·관찰성 이벤트/Handler 예제를 보존했다. 이 문서는 MinIO 연결 사례로 유지하고 커리큘럼/의존성/데이터 handoff의 역할을 설명했다. 기존 상위 AI와 RAG 교차 링크는 같은 데이터 처리 주제의 참조로 바꿨다. 이동·삭제·실행 코드 수정은 없다.

공식 2.10.5와 판본별 일차 자료를 대조해 분산 worker의 `/tmp` handoff 제한, logical date·Task 상태와 업무 검증의 차이, catchup/retry/active run의 보장 한계, TaskFlow XComArg edge, TLS·CA·UI/CLI/env/backend Connection, Windows POSIX/AST와 실제 import 차이를 정정했다. 근거 없는 90%·가장 흔한 사내 환경·보편적인 ship/캐시 원인·“tasks test 부작용 없음”을 철회했다. WSL/Compose 예제를 2.10.5 기준으로 맞추고 standalone은 여러 구성요소 기동임을 설명했다. Astro 설치 안내는 CLI 1.44·1.32 이후 Podman 기본 옵션/Windows WSL2를 확인했으며 고정 컨테이너 수를 철회했다. 회사 topology·설치 정책·현대 배포 판본이라는 보증은 없다.

같은 basename의 서로 다른 object key를 hash로 구분해 조용한 덮어쓰기를 막고, directory marker만 있는 prefix는 실패시켰다. 분석 임시 CSV는 실행별 directory로 분리하고 fixture 후 삭제되는 것을 확인했다. 관찰성은 부분 context의 누락을 None으로 보존하고 예약 필드를 payload가 덮어쓰지 못하게 했다. 전송 실패는 선택적 best-effort로 분리하며 원래 업무 예외를 유지했다. 인증서/hostname 검증을 켜고 CA 조건을 명시했고 raw 오류 문자열을 이벤트/전송 실패 로그에 복사하지 않는다. 원문의 logging Handler는 실제 실패 시 buffer 유실을 재현했으며 미완성 비교 예제로 명시했다. 자동 root handler attach/외부 client 생성은 parse-time에 실행되지 않는 주석 예시로 바꿨다.

Claude 협의는 pane_not_found로 불가하다. 분산 저장소 계약·raw schema/quality·입력 snapshot·event 내구성/감사 정책·Handler 완성·중복 통합은 결정하지 않았다. 부분 local 예제를 운영 완성본으로 사용하지 않도록 callout과 남은 조건을 기록했다.

### 추가 세 단계 검증

1. 모든 기존 절 제목·28개 fence·작성일·경로가 보존됐다. 원래 Markdown 32개 전체가 존재하고 미수정 19개·비Markdown 4개 SHA는 동일하다. 08 역사 본문은 이전 정확 보존 결과를 유지한다. 현재 원문 13개와 새 정리 기록 1개를 검토했다.
2. Python AST15 통과. S3Hook 대역으로 다운로드 두 변형에서 같은 basename의 두 서로 다른 CSV를 모두 보존했고 directory-only prefix 실패 2건을 확인했다. 실제 pandas2.1.4/pyarrow19.0.0 CSV→Parquet 값 보존과 임시 summary 업로드 대역·정리 1건이 통과했다. 실제 Airflow2.10.5 DagBag은 S3Hook import 대역/문서 helper를 사용해 DAG2개와 pipeline Task/edge를 발견했다. Amazon provider 설치/Connection/S3/MinIO 통신 증거가 아니다. 부분 context4건·예약 필드 보호·TLS kwargs/HTTP 거부·전송 실패와 업무 예외 보존을 mock으로 검증했다. logging Handler의 실패 시 buffer가 지워지는 결함은 음성 fixture로 재현했고 해결된 운영 Handler로 주장하지 않는다. PyArrow가 sandbox에서 CPU sysctl 접근 경고 4건을 출력했지만 실제 파일 값 검사와 명령 exit 0을 확인했다. 실제 회사 endpoint·secret·장비·scheduler·Windows/WSL/Docker/Astro는 실행하지 않았다. CLI는 tasks test --help를 읽어 dependency/state 기록 한계를 확인했다.
3. 정확한 pm_notes vault의 읽기 화면에서 40개 상태로 끝까지 이동했다. 기존/추가 source 고유 절 56개와 native 본문 제목 56개가 일치하고 breadcrumb·검토일·읽기 모드를 확인했다. 대표 하단 screenshot에서 출처·판본·남은 확인·로컬 범위를 확인했다. 마지막 문장과 기록/목차 갱신 후 MinIO/정리 기록을 다시 처음부터 읽었고 동일한 MinIO 56절·기록 16절을 확인했다. 현재 14개 참조 오류0·unique YAML·exact vault CLI properties14와 diff 검사가 통과했다. 모든 행의 모든 픽셀 검사는 아니다.

근거와 확인 판본은 원문 문서에 남겼다. Amazon provider 확인 페이지 9.37.0·Astro CLI 1.44·MinIO AIStor redirect는 회사 판본/설치 성공/최신 보증이 아니다. OpenSearch kwargs는 mock이며 실제 SDK·TLS handshake는 미확인이다.


## 정규화 입문 세 문서 추가 검토 — 2026-10-04

| 문서 | 수정과 역할 | 남은 확인 |
|---|---|---|
| [목차](./normalization/README.md) | 정규형/값 표준화/검색 점수 변환을 구분하고 01~02 → 저장소 → 의미/RAG → 통합 순서 제공 | 03~08 제품별 검토 |
| [01 핵심 개념](./normalization/01-normalization-core.md) | BCNF 결정자는 후보키가 아닌 슈퍼키; 모든 후보키의 2NF와 prime 예외의 3NF; FD/원자성 조건·NULL·이메일·현재 주소와 거래 스냅샷 구분 | 실제 업무 키·이력·동기화·권한 |
| [02 설계 절차](./normalization/02-modeling-process-checklist.md) | 반복 참여/주문항목 키 가정·기간 계약·별도 제약/보안·5분 예시와 SLA 구분; 정확한 정의는 01 참조 | 업무 유일성·기간·projection 구현 |

저자 제공 Database System Concepts 5판(2006) 7장과 Itzik Ben-Gan T-SQL Fundamentals 3판(2016) 공개 본문, RPI Sibel Adali Fall 2026 강의를 대조했다. 2NF는 모든 후보키, 3NF/BCNF는 비자명 FD·슈퍼키/prime 조건으로 설명했다. 원자성은 도메인 사용 조건을 함께 읽는다. 이론 판본을 최신 제품 버전으로 주장하지 않는다. RFC 5321(2008) §2.4는 local-part 대소문자 보존과 도메인 구분을 확인했다. SQLite 공식 FK 문서는 연결별 PRAGMA와 NOT NULL의 별도 필요성을 확인했다. PostgreSQL 18 제약 안내는 실제 설치 검증과 구분했다. 각 문서에 출처·확인일·적용 조건을 남겼다.

WikiDocs 글은 저자의 모델링 관점으로 보존하고 형식적 정규형의 근거와 구분했다. Ontario Tech의 검색 결과 페이지는 AI 생성 요약 표시를 확인해 정의 근거에서 제외했다. Claude 협의는 HERDR_ENV=1 상태에서 pane_not_found였다. 의견 수신·협의 완료로 기록하지 않으며 완전 통합·업무 모델 선택은 보류했다. 01은 정의, 02는 적용 절차로 구분해 고유 예제를 보존했다.

### 세 단계 검증

1. 원래 세 문서의 모든 절 제목·text 예제·2026-05-02 작성일·경로를 보존했다. 원래32개 Markdown/비Markdown4개가 모두 존재한다. 남은16개와 코드3·설정1의 SHA가 동일하다. 파일 이동·삭제·실행 코드 변경은 없다. 원문16/32개, 기록 포함 현재17개를 검토했다.
2. SQLite 3.50.4의 새 :memory: 연결에서 문서 SQL을 실행해 현재 주소 부산/과거 주문 주소 서울을 확인했다. 잘못된 FK·NULL 고객·NULL 스냅샷 3건을 IntegrityError로 검출했다. FD closure의 모든 부분집합을 검사해 후보키 AB/AC, 3NF 충족·BCNF 위반, 무손실 분해와 종속성 비보존을 확인하고 component 제약을 지키면서 join에서 AB→C를 위반하는 반례를 검사했다. 실제 업무 DB·동기화·권한·제품별 서비스는 미실행이다.
3. 현재17개 상대 링크·앵커·첨부 오류0, unique YAML/검토 메타데이터와 exact pm_notes CLI properties17개가 통과했다. 새 세 문서 전체를 읽기 화면에서 순차 탐색해 source 제목과 관측 제목이 정확히 일치했다(목차5절/5상태, 01 15절/11상태, 02 16절/11상태). 대표02 하단 screenshot에서 출처·상대 링크·code span이 렌더링됐다. 모든 행의 픽셀 또는 모든 외부 Markdown 렌더러 검사는 아니다. 정리 기록과 상위 목차는 마지막 수정 후 다시 읽는다.


## OpenSearch 정규화 적용 추가 검토 — 2026-10-04

[03 OpenSearch](./normalization/03-opensearch-normalization.md)의 모든 원래 절·8개 JSON 본문·작성일·경로를 보존했다. HTTP 요청 첫 줄을 JSON과 구분해 fence를 http로 바꿨다. 이메일 식별값과 검색 변환·trim 앞뒤 공백·원천 source와 별도 exact 필드·object/nested 의미·join routing·snapshot/version·chunk ID 변경·벡터 mapping 조건을 명시했다. 원래 pipeline 설정에 검색 요청 연결 단계를 추가했고 2차원 숫자는 실제 임베딩 모델 결과로 주장하지 않았다. 순서는 입문 → 제품 적용을 유지했다.

3.4 URL의 공식 normalizers/normalizer/nested query/join/synonym_graph, 2.15 object 배열/nested·normalization processor·hybrid search/k-NN index 문서를 2026-10-04 대조했다. 확인된 normalizers3.4/pipeline2.15 페이지는 유지보수 종료로 표시됐다. 최신·회사 설치 판본·서로 다른 판본의 무조건 호환을 주장하지 않는다. normalizer2.15와 trim3.4 단독 URL은 조회 오류였고 정상3.4 compatible 목록/예제로 대조했다. 01은 일반 정의, 03은 OpenSearch 적용이라 고유 예제를 유지한다. HERDR_ENV=1에서 Claude 연결은 pane_not_found로 불가했고 완전 통합·업무 모델/권한/버전 계약의 선택은 보류했다.

### 추가 검증 결과

1. 원래32개 Markdown·비Markdown4개가 모두 존재한다. 이 장의 모든 원래 절·8개 본문·작성일을 보존하고 새 요청2개를 추가했다. 남은15개·실행 코드3·설정1의 SHA가 동일하다. 이동·삭제·실행 코드 수정은 없다. 현재 원문17/32개, 기록 포함18개를 검토했다.
2. HTTP 행/JSON 본문10개 분리 파싱 통과. 가상의 배열에서 필드별 AND는 참·같은 원소의 AND는 거짓·OR는 참임을 로컬 fixture로 확인했다. 원문의 nested path와 두 must 조건을 대조했고 query2개/weights2개/합계1/범위 조건과 음성4건·벡터2차원 유한값을 확인했다. 이는 실제 Lucene 분석기·서버 mapping·색인·권한·검색 품질 검증이 아니다.
3. 현재18개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·exact pm_notes CLI properties18개를 통과했다. 03을 Obsidian1.13.7 읽기 화면에서13상태로 끝까지 탐색했고 본문11절이 source와 정확히 일치했다. 대표 하단 screenshot에서 판본/조회 오류/출처·상대 링크·code span의 렌더링을 확인했다. 모든 행의 픽셀·외부 Markdown renderer·실제 OpenSearch 검색 검증은 아니다. 마지막 기록 수정 후 두 목차와 기록도 다시 읽는다.


## MongoDB와 Redis 정규화 적용 추가 검토 — 2026-10-04

| 문서 | 수정과 이유 | 남은 확인 |
|---|---|---|
| [04 MongoDB](./normalization/04-mongodb-normalization.md) | 단일 문서/다중 문서 원자성·16MiB/배열 성장·manual reference의 애플리케이션 해석·validator 범위·unique 누락/null/sharding 조건·메일 case·별도 glossary schema·Vector Search 배포/index/filter 조건 구분 | 실제 mongosh/BSON/서버 제약·업무 partial/권한·Atlas/self-managed 검색 |
| [05 Redis](./normalization/05-redis-normalization.md) | shell 호출·메일 hash/key 일치·chunk/version·placeholder/hash tag·TTL/원천 동기화/동시성·8.0 JSON/Search·vector bytes/index 조건 보완 | 실제 서버 명령/TTL/Cluster/Search·업무 alias/key/권한 계약 |

모든 원래 절·작성일·경로와 예제 역할을 보존했다. MongoDB는7개 원래 JSON 중 이메일 도메인 처리값 하나를 바로잡고 나머지 본문을 보존했다. JSON fence의 컬렉션 설명을 본문으로 옮겼고 strict/error와 최소 유효 JSON을 추가했다. canonical_label의 glossary는 별도 구조이며 terms validator와 동일하지 않다고 설명해 고유 예제를 보존했다. Redis는10개 원래 명령을 shell 호출로 보존하면서 메일 key/value와 chunk/version을 보완했다. 실제 실행 코드·자료 파일은 수정하지 않았다.

공식 MongoDB8.0의 JSON Schema draft4/BSON 차이·validation level·unique index·원자성/manual reference, 확인 당시 manual9.0의 embedding/reference, 별도 Vector Search 개요를 2026-10-04 대조했다. 이전 Atlas 링크는 새 vector-search 개요로 갱신했다. 추측 stage/type URL과 개요의 ANN/ENN 링크는 조회 오류여서 실행/전문 확인으로 주장하지 않는다. 해당 개요의 self-managed 안내를 임의 Community 설치에서 즉시 기능이 된다는 보증으로 쓰지 않는다. Redis 공식 자료형/명령·transaction/Cluster·Open Source8.0 릴리스를 대조했다. HEXPIRE since7.4.0과8.0 JSON/Search 통합을 구분했고 latest URL을 최신/회사 설치 판본으로 단정하지 않았다. 처음 추측한 Redis 릴리스 URL 오류 후 실제 redisos-8.0-release-notes 페이지를 확인했다.

Claude 협의는 HERDR_ENV=1 확인 후 pane_not_found였다. 의견은 없으며 완전 중복 통합·업무별 key/unique/기간/alias/보안/배포 선택은 보류한다. 01은 이론,03은 검색 projection,04는 문서 저장,05는 캐시 접근/무효화로 역할을 구분했다. 원천 버전/권한/freshness 미확인은 일치로 기본 처리하지 않는다.

### MongoDB/Redis 세 단계 검증

1. 원래32개 Markdown·비Markdown4개 경로를 보존했다. 두 문서의 원래 절·작성일·예제 맥락/명령을 모두 대조했다. 남은13개·실행 코드3·설정1의 SHA가 동일하다. 원문19/32개와 기록 포함20개를 검토했다. 이동·삭제·실행 코드 변경은 없다.
2. MongoDB JSON8개 파싱, Node24.13.0 VM의 JavaScript fence2개/DB 대역호출3개를 검사했다. jsonschema4.23.0 Draft4에서 문서의 object/array/string bsonType만 type으로 변환한 제한된 schema로 정상1개·필수/타입 음성5개를 검사했다. 빈값/추가필드 허용과 glossary 부적합도 확인했다. 일반 Draft4 검사가 실제 BSON/mongosh 서버 증거는 아니다. ASCII 이메일 원형 보존·현재 grade 변경 후 snapshot 보존 fixture를 확인했다. Redis bash fence7개 syntax/대역 redis-cli 호출10개를 실행해 인자·JSON.SET 본문·HSET 쌍·메일 hash/key·chunk/version·alias lookup·Unix 초 timestamp를 대조했다. 실제 Redis 클라이언트 연결·서버·TTL·transaction·Cluster·벡터 검색은 실행하지 않았다.
3. 현재20개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·exact pm_notes CLI properties20개를 통과했다. 04/05를 Obsidian1.13.7 읽기 화면에서13/12상태로 끝까지 탐색했고 본문12/12절이 source와 정확히 일치했다. 대표Redis 하단 screenshot에서 출처·상대 링크·code span 렌더링을 확인했다. 모든 행의 픽셀·다른 renderer·서버 실행 검증은 아니다. 마지막 기록 수정 후 두 목차와 기록을 다시 읽는다.


## 정규화 RAG·온톨로지·통합 추가 검토 — 2026-10-04

| 원래 문서 | 수정과 역할 | 남은 확인 |
|---|---|---|
| [06 RAG](./normalization/06-llm-rag-normalization.md) | 논리 문서/판본/chunking revision·반열린 날짜/시간대·인증 주체와 후보 query·모호성·ID/label·벡터/score 조건과 평가 구분 | 실제 retrieval/모델/권한/승인 manifest |
| [07 온톨로지](./normalization/07-ontology-normalization.md) | 학습 계층과 관계형 정규형·glossary/taxonomy·JSON/OWL 추론·RDF/SHACL 검증의 차이; NULL/효과 보장 정정 | 실제 RDF/OWL/SHACL 실행·업무 의미 승인 |
| [08 통합](./normalization/08-cross-layer-cheatsheet.md) | 연속 JSON을 배열로 묶고 모델별 알람 복합키·장비 모델/instance 구분·매뉴얼 판본 resolver·BGE-M3 preview1024·routing/ID/label·cache 문맥/미확인 거부 보완 | 실제 DB/색인/모델/무효화/권한·장비 조치 |

세 문서의 모든 원래 절·작성일·경로와 원래 예제의 객체·값·흐름 맥락을 보존했다. 잘못된04월 종료 경계·논리 문서 _id·2차원 dense BGE 표기와 cache query-only 가정을 명시적으로 정정했으며 상세 변경은 각 문서에 남겼다. 06의 다른 AI/RAG 주제 링크는 같은 정규화03 참조로 바꿨다. 폴더 이동/삭제/실행 코드 변경은 없다. 이 시리즈9개 모두 정의 → 설계 절차 → 저장소 → 의미/RAG → 통합으로 읽고 같은 이름 canonical_terms의 ID/label 차이를 명세로 구분하게 했다. 새로운 공통 스키마로 업무 데이터/코드를 이동하지 않았다.

2026-10-04 확인한 일차 근거는 Lewis et al. RAG원논문2020, W3C SKOS2009·OWL2 Primer2판2012·RDF1.1 Concepts2014·SHACL2017, BAAI공식 BGE-M3 모델 카드 dense1024다. CVD 용어 배경은 NIST2019 기체 전구체/박막 증착 연구와 대조했다. 가상 CVD-2000/A-17 설명은 실제 장비 지침이 아니다. 제품별 기능은 이미 확인한03~05의 판본·미확인 범위를 따른다. 모델 카드는 mutable하며 확인일 상태이지 최신/설치보증이 아니다. glossary/ontology는 모든 RAG의 필수/정답 보장이 아니라 설계 선택지다. RDF triple이나 class/JSON을 RDB정규형/추론/필수값 검증으로 혼동하지 않는다.

HERDR_ENV=1 확인 후 Claude 연결은 pane_not_found였다. 의견 수신/협의 완료로 기록하지 않으며 완전 중복 통합·업무별 ID/기간/alias/권한/cache/원천 갱신 계약은 보류했다. 정의·적용·통합의 역할을 구분해 고유 예제를 유지했다.

### RAG/온톨로지/통합 세 단계 검증

1. 원래32개 Markdown·비Markdown4개 경로가 모두 존재한다. 세 장의 원래 절·작성일·예제 객체/값/맥락을 대조했다. 바뀐 날짜·판본 _id·embedding_preview와 연속 JSON 배열화는 의도한 정정으로 따로 검사했다. 남은10개·실행 코드3·설정1의 SHA가 동일하다. 현재 원문22/32개와 기록 포함23개, 정규화9개 전체를 검토했다.
2. JSON11개 파싱, Python fence1개 AST/실제 순수 함수 실행 통과. cache 함수는 같은 문맥/인자 순서 변경에 동일 key,7개 항목 변경에 다른 key,누락/잘못된 타입/미확인 문맥33건을 ValueError로 거부했다. 날짜 구간은4월30일23:59:59 포함/5월1일 제외를 확인했다. 두 매뉴얼 판본 동시 보존과 정확한 source_doc_id/version resolver,SQLite메모리 fixture에서두 모델 A-17 허용/누락 복합FK 거부를 확인했다. cosine scale 정상과0벡터/차원/nonfinite 음성3건은 순수 수학 fixture로 확인했다. 실제 모델·DB업무 배포·RedisTTL/cache·retrieval·ACL·RDFreasoner/SHACL engine 실행은 아니다.
3. 현재23개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·exact pm_notes CLI properties23개를 통과했다. 06/07/08을 Obsidian1.13.7 읽기 화면에서13/12/16상태로 끝까지 탐색했고 본문14/14/18절이 source와 정확히 일치했다. 대표08 하단 screenshot에서 출처·상대 링크·code span 렌더링을 확인했다. 모든 행의 픽셀·다른 renderer·실제 DB/모델 실행 검증은 아니다. 마지막 기록 수정 후 두 목차와 기록을 다시 읽는다.

## 역공학 실행 도구 README 개별 검토 — 2026-10-04

[scripts README](./binary-reverse-engineering/scripts/README.md)의 기존 모든 절·10개 명령 목록·원래 실습 호출을 보존했다. 실제 구현과 다른7개 명령/17개 단언 표기를10개 명령·12구간28개 검사로 정정했다. 정상 분석/포착 예외의 JSON과 argparse/의존성/프로세스 오류를 구분하고 exit0+error를 성공으로 처리하지 않도록 호출 계약을 설명했다. 생성기는 출력 파일을 쓰며 bre는 입력 읽기 전용이라는 역할 차이, 원본과 동일한 shell stdout 경로의 손실 가능성, 실제 환경과 소스 호환성 주장의 차이를 밝혔다.

근거는 같은 모듈의 CLI/fixture/selftest 원본 소스와 로컬 실행이다. 확인 환경 Python3.12.12·NumPy1.26.4,2026-10-04. 소스의 Python3.9+ 문구는 실제3.9 실행/다른 NumPy 호환성을 확인한 사실이 아니다. 자체 회귀가 장비 포맷/전체 오류/파서 정확성을 보증하지 않는다. HERDR_ENV=1 확인 후 pane_not_found로 의견을 받지 못했다. 상위 작업 계약/runbook/입문/도구 문서 통합은 보류하며 아직 검토 대기인 문서의 옛 count/출력 보장은 현 README를 기준으로 후속 대조한다.

### 실행 도구 세 단계 검증

1. 원래32개 Markdown·비Markdown4개 경로와 README의 모든 기존 절/명령을 보존했다. 남은9개·비Markdown4개 SHA가 동일하며 실행 코드3개도 원본과 같다. 현재 원문23/32개와 정리 기록 포함24개를 검토했다.
2. 기존 selftest를 저장소 밖 임시 환경에서 실행해28개 모두PASS와exit0을 확인했다. fixture는 TemporaryDirectory에 생성돼 정리됐다. PythonAST3개와 명령 등록10개/검사 호출28개를 대조했다. missing-file JSON error/exit0,누락 인자 stderr/exit2,help 일반 텍스트/exit0,variance 최소 입력 실패 JSON/exit0을 실제 확인했다. 실제 장비/서버/벤더 파서는 실행하지 않았다.
3. 현재24개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·exact pm_notes CLI properties24개를 통과했다. 도구 README를 Obsidian1.13.7 읽기 모드에서6상태로 끝까지 탐색했고 본문7절이 source와 일치했다. 대표 screenshot에서 source/작업 계약/runbook/정리 기록 상대 링크와 code span 렌더링을 확인했다. 모든 행의 픽셀·다른 renderer·실제 장비 파일 검증은 아니다. 최종 기록 변경 후 상위 목차와 기록을 다시 읽는다.

## 역공학 법률/첫 수순 참고 개별 검토 — 2026-10-04

[03 법률/첫 수순](./binary-reverse-engineering/03-legal-and-first-moves.md)의 모든 원래 절·작성일·미국/EU/상호운용/오류수정/벤더/SDK/export/TIFF/recipe 맥락을 보존했다. “대개 허용”, 미국/EU 예외 범위 순위, 계약의 무조건 우선, 이미지의 위험 최저,EDA가정답이라는 무근거 일반화를 철회했다. 데이터 파싱·코드 분석·통제 우회의 차이와 실제 권한/관할/계약을 구분하고 장비 지원/법적 허용을 미확인으로 남겼다.

2026-10-04 확인 근거: [미국 저작권청§1201(f)](https://www.copyright.gov/title17/92chap12.html) 조문1~4, [공식C-13/20판결](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62020CJ0013) ¶63~69와 주문, [SEMI공개EDA목록](https://www.semi.org/en/products-services/standards/information_and_control). C-13/20은2021-10-06 판결로91/250/EEC를 해석하며 현재 회사 계약을 승인한 판결이 아니다. [2009/24/EC](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32009L0024)는 공식 검색 색인에서5(3)/8과공식요약의 상호운용 조건을 확인했으나 TXT/HTML/PDF/ELI/ALL 및 공식요약LSU 직접 조회는 JavaScript/robot 안내라 전문 확인/최신 개정으로 주장하지 않는다. OLRC직접조회 maintenance·govinfo추측2025URL조회오류 후 copyright.gov대체 조문을 읽었다. CAPTCHA 해결/우회는 하지 않았다. 유료SEMI규격 전문/실제Freeze/장비지원·한국법/회사계약·구체 허용 결론은 미확인이다.

Claude는 이미HERDR_ENV=1/pane_not_found로 연결불가이며 법률·계약 해석을 협의한 의견은 없다. 근거의 범위를 명시하는 정정만 진행하고 실제 계약/관할 결정·유사 문서 통합은 보류했다. 법률/계약은 실행 fixture로 허용을 증명할 대상이 아니며 실제 장비/벤더SW/외부 연락은 실행하지 않았다.

### 법률 참고 세 단계 검증

1. 원래32개Markdown·비Markdown4개 경로, 해당 문서의 모든 원래 절/작성일/고유 맥락을 대조했다. 남은8개·비Markdown4개 SHA는 원본과 동일하다. 원문24/32개,정리기록 포함25개를 검토했다.
2. 법령/판결의 조건과 직접 확인/검색 색인/조회 오류를 구분했다. 법률의 로컬 실행 검증은 적용하지 않으며 회사 계약·장비 실행은 없다.
3. 현재25개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·exact pm_notes CLI properties25개를 통과했다. 법률 참고를 Obsidian1.13.7 읽기 모드에서5상태로 끝까지 탐색했고 본문7절이 source와 일치했다. 대표 screenshot에서 공식 출처·정리 기록 상대 링크와 미확인 범위를 확인했다. 모든 행의 픽셀·외부renderer 검증은 아니다. 최종 기록 변경 후 상위 목차와 기록을 다시 읽는다.

## 역공학 개념·작업 파이프라인 추가 검토 — 2026-10-04

| 문서 | 수정과 역할 | 미확인 |
|---|---|---|
| [개념](./binary-reverse-engineering/reverse-engineering-concepts.md) | byte/encoding/endianness·손상·entropy·시간/checksum·샘플과 필드 의미의 차이 | 실제 장비 포맷/분포·UI/export |
| [작업 계약](./binary-reverse-engineering/00-agent-brief.md) | 모든 대상 필드의 사용 범위·원본/출력·28개 합성 검사·reader/관할 구분 | 실제 계약/완료 규격·Claude 통합 |
| [runbook](./binary-reverse-engineering/00-agent-runbook.md) | JSON/exit·작업 경로·shell 변수와 Python 입력·후보와 경계/개수/의미·writer 조건 | 실제 장비/reader·Kaitai/runtime |
| [task 참고](./binary-reverse-engineering/agent-tasks.md) | template 실행과 문서 작업 구분·quoted BRE·readable null·출력 필드/EOF 조건 | 외부 도구 호출·공유 계약·Claude 통합 |

원래 모든 절·phase/task·작성일·경로와 byte/음식 비유·합성findings JSON·호출의 목적을 보존했다. placeholder가 실제 shell redirection으로 해석되는 runbook을 quoted 숫자 변수로 바꾸고 Python FILE을 정의했다. 실행 코드 변경은 없다. 필요 값/단위/판본의 reader 대조를 종료 조건으로 삼았으며 한 필드 일치로 전체 완성을 확정하지 않는다. `stride`에 coverage가 없는 점, `best`가 None일 수 있는 점, arrays의 절댓값/0 제외/상수 제외와 미확인을 구분했다. `bre.py`의 hint/next_step에 남은 확정 표현/05-formalize-parser 이름은 구현 수정 대상이 아니며 실제 형식화 참고는 현존01 문서다.

2026-10-04 근거: 같은 모듈 [bre.py](./binary-reverse-engineering/scripts/bre.py)·[fixture](./binary-reverse-engineering/scripts/make_fixture.py)·[selftest](./binary-reverse-engineering/scripts/selftest.py), [Python3.12 struct](https://docs.python.org/3.12/library/struct.html)·[codecs](https://docs.python.org/3.12/library/codecs.html), [Kaitai serialization](https://doc.kaitai.io/serialization.html), [tifffile 공식 소스](https://github.com/cgohlke/tifffile). tifffile 확인 소스 표시2026.9.20의 fei_metadata는34680/34682 tag, sem_metadata는34118 tag를 읽으며 특정 CD-SEM 판본의 실제 지원과 다르다. 아직 대기인02 문서의 tag/append 설명은 후속 정정한다. 일반 Kaitai compiler 호출만으로 parse→build writer와 바이트 동일 round-trip을 보장하지 않는다. read-write 옵션/runtime 지원은 별도다. 최신/설치 보증은 없다. HDF5/SER/STDF/unblob 후보의 실제 호출·설치 판본과 Kaitai compiler/runtime는 이 단계에서 미실행/미확인이다.

HERDR_ENV=1 확인 후 pane_not_found로 Claude 의견을 받지 못했다. 문서 역할은 개념→작업 계약→명령 순서→위임 template로 구분하며 완전 통합·업무 완료 계약은 보류했다. task 본문은 읽기 자료이지 이 정리 과정의 새 장비 조사/agent 실행 지시가 아니다.

### 파이프라인 네 문서 세 단계 검증

1. 원래32개 Markdown·비Markdown4개 경로를 보존하고 네 문서의 모든 원래 절/작성일/고유 맥락을 대조했다. 남은4개 문서·비Markdown4개 SHA가 동일하다. 원문28/32개, 정리 기록 포함29개를 검토했다. 기존findings JSON과 byte 숫자/모든phase/task를 보존했다.
2. 문서 Python AST2개(중첩taskB template 포함)·bash syntax9개 통과. 실제Python3.12.12/NumPy1.26.4의 tool에서 비암호화 균등바이트 entropy8, 압축 파일의 낮은entropy, stride coverage 부재·짧은파일error, 빈TLV best=None, 양/음 절댓값 점수 동일·0의 범위계산 제외·유효 상수값의 점수 제외,8byte tail이 있어도 lands_at_eof=True를 확인했다. struct float32=1.0/uint32=1065353216과500의LE/BE,additive checksum 단순변경/충돌 반례를 확인했다. 코드SHA3개 동일. 실제장비/벤더reader/UI·parser생성/round-trip은 아니다.
3. 현재29개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·exact pm_notes CLI properties29개를 통과했다. 네 문서를 Obsidian1.13.7 읽기 화면에서12/7/12/10상태로 끝까지 탐색했고 본문21/9/20/11절이 source와 일치했다. 대표 task 하단 screenshot에서 출처·serialization 조건·상대 링크와 callout 렌더링을 확인했다. 모든 행의 픽셀·다른 renderer·실제 장비 UI 검증은 아니다. 마지막 기록 변경 후 상위 목차와 기록도 다시 읽는다.


## 역공학 목차와 좌표/recipe 추가 검토 — 2026-10-04

[목차](./binary-reverse-engineering/README.md)의 원래 모든 절·문서 목록·실습 목적·작성일을 보존하고 읽기 순서를 추가했다. SEM 전체 TIFF 비율과 reader 지원·측정/recipe만 조사 대상이라는 무근거 일반화를 철회했다. 실제 소스에 맞춰10개 명령·12구간28개 검사·JSON/exit/인자 오류·입출력 경로와 shell 숫자 변수 정의를 설명했다.

[좌표/recipe](./binary-reverse-engineering/04-coordinate-and-recipe-files.md)의 원래 모든 절·도식·예제 맥락·작성일을 보존했다. 좌표 범위/개수/격자/후보 순위만으로 확정하지 않고 단위·원점·축·판본·값 대응을 대조하게 했다. ASCII 비율/signature·offset/string/TLV는 후보이며 best=None·tail 허용·부모 경계·합성 커버를 구분했다. 전체 파일의 --start만 바꾸는 nested TLV는 부모 끝을 제한하지 않으므로 경계가 확인된 value 사본으로 조사하게 설명했다. 실제 장비 파일/외부 parser/실행 코드 변경은 없다.

2026-10-04 확인 근거는 같은 모듈 bre.py/selftest, 앞서 확인한 tifffile 표시2026.9.20, [Microsoft CArchive/msvc-170](https://learn.microsoft.com/en-us/cpp/mfc/reference/carchive-class?view=msvc-170), [BinaryFormatter 공식 지침](https://learn.microsoft.com/en-us/dotnet/standard/serialization/binaryformatter-security-guide)이다. CArchive는 클래스 Serialize/schema에 의존하며 OLE signature를 MFC 증거로 쓰지 않는다. BinaryFormatter의 위험과.NET9 기본 예외를 후보 분석에 적용했으며 실제 벤더 stack/설치 판본으로 추정하지 않았다. HERDR_ENV=1/pane_not_found 재확인으로 Claude 의견은 없고 완전 통합·실제 작업 계약은 보류했다.

### 목차/좌표 세 단계 검증

1. 원래32개 Markdown·비Markdown4개 경로와 두 문서의 원래8/13절·작성일·고유 도식/문서 목록을 보존했다. 남은2개 문서·비Markdown4개 SHA가 동일하다. 원문30/32개와 기록 포함31개를 검토했다. 폴더 이동/삭제/실행 코드 변경은 없다.
2. bash syntax6개 통과. 실제 Python3.12.12/NumPy1.26.4의 serial에서 형식 없는 ASCII 문장의 mostly_text=True·UTF16 텍스트의 False·OLE signature 후보를 확인했다. 합성 TLV에서 전체 파일은 형제까지8record, 부모 value 사본은4record로 읽고 EOF start는 best=None였다. 값/단위·좌표계·의미·실제 장비/표준 parser의 지원 검증은 아니다.
3. 현재31개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·정확한 pm_notes CLI properties31개 통과. CLI eval은 vault 경로 /Users/daeyoung/Codes/pm_notes와 binary README active/preview를 확인했다. 그러나 native AX는 이전 task/전환창 내용을 반복 반환했고 screenshot은 task 하단을 보였다. 앱 재연결/세션 초기화/CLI reveal 후에도 일치하지 않아 새 두 문서와 수정된 상위 목차·기록의 실제 읽기 재검증은 미완료다. CLI open/active만으로 렌더링 완료로 계산하지 않는다. 직전 네 문서의 읽기 근거는 별도로 보존했다.


## 역공학 도구 총람과 벤더 포맷 추가 검토 — 2026-10-04

[01 도구 총람](./binary-reverse-engineering/01-toolkit-reference.md)은 원래19절·도구/연구 후보·MET1/통계/struct 예제 목적·작성일을 보존했다. binwalkv3의 Rust 기반 설치를 공식 README로 정정하고 unblob -e는 공식 guide의 유효 옵션으로 유지했다. 근거 없는 후계/성능/가격/개발 상태·범용 탐지 대체·지원 언어/round-trip 보장을 철회하거나 미확인으로 표시했다. raw DEFLATE와 zlib wrapper, 날짜 후보의 epoch/단위, 의미/경계/stride/count와 휴리스틱을 구분했다. ImHex 초안은 Rust가 아니며 offset64/Construct·Kaitai MET1 header26/CDS1fixture를 같은 형식으로 섞지 않는다.

[02 벤더 포맷](./binary-reverse-engineering/02-cd-sem-formats.md)은 원래11절·벤더/모델/reader/표준/커뮤니티·작성일을 보존했다. FEI34680/34682와 Zeiss34118 tag를 공식 tifffile2026.9.20 소스로 정정하고 EOF append/34119연결/범용 fallback/공개 스펙 없음 확정을 철회했다. dict/None·키 부재·ASCII tail 부재/중복/encoding 오류를 구분했다. reader 구현/이전 HitachiS4800자료를 CG/CV/GS 등 모든 장비로 확대하지 않았다. SEMI E132인증/인가와 P39OASIS/P44mask-tool 범위를 구분했고 실제 transport/Freeze/장비 필드와 규격 전문은 미확인이다.

2026-10-04 일차 근거는 각 문서의 공식 binwalk/unblob/Python3.12/Construct2.10/Kaitai/tifffile/RosettaSciIO/SEMI 목록과2021소개·2023SNARF다. mutable URL은 확인일 상태이지 최신/설치 보증이 아니다. Hitachi8.1.0 API와옛5.2.0PDF 검색 색인 범위, format/Java소스 조회 실패·ExifTool pod403을 구분했다. 유료 규격 전문과 모든 library/연구/GUI/CRC 설치·실제 지원은 확인하지 않았다. HERDR_ENV=1/pane_not_found로 Claude 의견은 없으며 완전 중복 통합과 회사 계약/실제 인터페이스 선택은 보류했다.

### 마지막 두 문서 세 단계 검증

1. 원래32개 Markdown·비Markdown4개 경로와 두 문서의 모든 원래 절/작성일/고유 맥락을 대조했다. 비Markdown4개(실행 코드3/설정1)의 SHA가 동일하다. 원문32/32개와 기록 포함33개를 개별 검토했다. 이동·삭제·실행 코드/첨부 변경은 없다.
2. PythonAST8개·bash syntax1개와 YAMLksy 파싱 통과. 실제 Python3.12.12/NumPy1.26.4/zlib1.2.12에서 NumPy분산/짧은행 절단·struct8byte·MET1header26/payload·zlibwrapper/raw정상/잘못된wrapper오류를 대조했다. 확인일UTC0시 계산은 Unix1791072000/FILETIME134355456000000000/OLE46299다. metadata 예제는 명시적 fake tifffile에서 None/누락/정상키 분기를 검사한 제한된 증거다. 가상 tail parser는 정상·header/key부재2·잘못된ASCII/중복section2를 검사했다. 실제 tifffile/장비/외부 parser/compiler/GUI의 실행 증거가 아니다.
3. 현재33개 상대 링크·앵커·첨부 오류0, unique YAML·검토 metadata·정확한 pm_notes CLI properties33개 통과. CLI dev:screenshot의02 상단에서 정확한 경로·읽기 mode·날짜/태그·H1/결정 순서를 실제 확인했다. 그러나 다음 파일을open해 active path만 바뀌고 DOM/스크린샷이 이전02를 유지하는 상태가 발생했다. 순차scroll의85회 한도 내 source 절이 관측되지 않아 실패를 기록했고, frame대기 프로세스는 진행되지 않아 종료했다. 새4개와 부모 목차/기록의 전체 읽기 검증은 미완료다. screenshot 상단 확인을 전체 문서 검증으로 확대하지 않는다. Native AX도 이전task/전환창이라 증거로 계산하지 않았다.
