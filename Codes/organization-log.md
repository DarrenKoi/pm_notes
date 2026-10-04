---
title: Python 예제 문서 정리 기록
tags: [python, maintenance]
document_type: maintenance
reviewed_on: 2026-10-04
status: partial
---

# Python 예제 문서 정리 기록

## 범위와 변경

기존 Markdown 5건을 검토했다. 실행 코드·requirements·설정·첨부·인증 자료는 변경하지 않았다. README 파일명과 위치를 유지했으며 Codes 최상위 목차를 추가했다.

OpenSearch 패키지 안내와 예제 안내의 동일한 설치·환경변수 설명을 패키지 README로 통합하고 예제 README에서 상대 링크로 참조한다. 주제별 실행 명령과 목적·특이 조건은 예제 안내에 보존했다. 대화 메모리의 기존 타 최상위 폴더 링크 4건은 제거하고 이 폴더의 구현·목차로 대체했다. 원래 링크의 주제는 대화 메모리 이론, 구현 학습, 벡터 및 하이브리드 검색이었다. 다른 폴더 내용을 읽거나 통합하지 않았다.

## 문서별 결과

| 문서 | 결과 |
|---|---|
| [대화 메모리](python/history-opensearch/README.md) | 기존 구조·인덱스·API 예제 보존. 생성 시 인덱스 준비, 요약 200건 한도, 시스템 프롬프트의 최근 메시지 미포함 명시. 실서버 미확인 |
| [OpenSearch 패키지](python/opensearch_handler/README.md) | 한국어로 목적·설정·적용 조건 정리. 설정 대표 문서. 명시 객체 예제를 환경 인증값 사용으로 갱신. nmslib deprecated, bool 결합 검색의 범위 명시 |
| [OpenSearch 예제](python/opensearch_handler/example/README.md) | 중복 설정을 대표 문서로 연결. 7개 실행 명령 유지. Topic 05 삭제·Topic 06 템플릿 잔존 조건 명시 |
| [PowerPoint export](python/drm-pptx-extraction/README.md) | 한국어로 정리. 모든 입력·출력 예제 보존. COM·권한 조건, 동명 출력 충돌과 기존 PNG 삭제, 부분 실패 후 종료코드 한계 명시 |
| [전송 분할](python/zip-split-transfer/README.md) | 기존 직접 분할·ZIP 대안·경로·index 설명 유지. byte/shard 구분, 체크섬 생략, 잔여 조각·덮어쓰기·1,000조각 한계 명시 |
| [Codes 목차](README.md) | 설치 난도와 외부 환경에 따른 읽기 순서. 문서 없는 실행 모듈도 목록화 |
| 이 기록 | 검증·협의·미확인을 기록 |

`requirements-dev.txt`는 의존성 설치 설정이며 학습 문서로 변경하지 않았다. 문서 없는 `org-hierarchy`, `llm-key-rotator`의 소스는 목차 역할을 확인했지만 상세 학습 안내와 외부 SDK 호환은 미완료다.

## 근거

확인일 2026-10-04. 공식 URL의 `latest`·`main`은 이동하는 문서이므로 특정 설치 환경의 호환 보증으로 사용하지 않는다.

- [OpenSearch Python client](https://docs.opensearch.org/latest/clients/python-low-level/): TLS·클라이언트 기본 API. 로컬 pyproject 0.1.0 / Python >=3.10 / opensearch-py >=2.4.0과 대조.
- [OpenSearch hybrid search](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/): 전용 query/pipeline과 로컬 bool 결합의 차이 확인.
- [OpenSearch breaking changes](https://docs.opensearch.org/latest/breaking-changes/): 3.0의 nmslib deprecated. 새 인덱스의 정확한 지원 조건은 실제 버전에서 추가 확인 필요.
- [Microsoft Slide.Export](https://learn.microsoft.com/en-us/office/vba/api/powerpoint.slide.export): 문서 갱신 2022-08-13, 이미지 내보내기 계약. DRM·COM 실행 성공은 입증하지 않음.
- [Transformers big models](https://huggingface.co/docs/transformers/main/big_models): shard와 weight_map의 관계. 실제 모델 로딩 안 함.
- 해당 폴더의 `memory_manager.py`, `connection_settings.py`, `search.py`, Topic 05/06, `main.py`, `export_slides.py`, 분할·join 네 스크립트를 소스 근거로 사용했다.

## Claude 협의

`HERDR_ENV=1`이지만 호출 pane `pane_not_found`. 다른 저장소 pane을 사용하지 않아 협의 결과 없음. 기본 코드 설정 변경, nmslib 엔진 교체, bool 검색의 알고리즘 교체와 API 의미 재설계는 보류했다. 명확한 중복 설정의 대표 위치 정리와 소스로 확인한 사용 조건은 문서에 반영했다.

## 세 단계 검증

1. 기존 5건의 문서 목록과 결과를 대조했다. 고유 실행 명령·API 사용·출력·경로·대안 예제를 보존했다. 통합한 환경 설명은 대표 문서로 참조한다. 초기 해시와 대조하여 Codes의 기존 비-Markdown 파일 변경 0건 확인.
2. 공식 자료·소스 대조 완료. 임시 2,048바이트 파일을 300바이트 조각 7개로 나누고 합쳐 원본 SHA-256 일치 확인. ZIP도 임시 파일을 분할·복원·압축 해제하고 SHA-256·내용 일치 확인. 체크섬 부재 함수가 None을 반환하는 조건 확인. Python AST 구문 검사 실행, 실제 외부 서버·Windows·모델·대용량 검증 없음.
3. Markdown 7건의 상대 링크·앵커·frontmatter 검사에서 신규 깨진 참조 0건. Obsidian 1.13.7 CLI가 Codes 목차 링크 9건과 분할 복원 속성을 인식했다. 창 연결을 한 차례 다시 선택한 뒤 실제 읽기 화면의 Codes 목차에서 분할 복원 문서로 클릭 이동하여 속성·본문·코드 블록을 확인했다. 실제 첨부 생성은 수행하지 않았다.

## 남은 미확인

OpenSearch 버전·클러스터 설정·인증·점수·k-NN 인덱스 생성, 임베딩 차원·LLM 응답, PowerPoint 빌드·DRM 정책, 대용량 전송·모델 로딩. 실제 실행 코드를 고치는 요청이 아니므로 확인된 코드 한계는 문서에 명시하고 구현을 바꾸지 않았다.
