---
title: Python 실행 예제 목차
tags: [python, index]
document_type: index
reviewed_on: 2026-10-04
category_major: "Python 실행 예제"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Python 실행 예제 목차

## 대·중·소분류로 찾기

[주제별 분류 목차](./taxonomy-index.md)에서 **Python 실행 예제 → 중분류 → 소분류**로 탐색한다. 각 문서의 `category_major`·`category_middle`·`category_minor`는 주제, `note_kind`는 용도다. 기존 경로와 아래 읽기 순서는 유지한다.

실행 가능한 Python 예제를 목적과 외부 환경 의존성에 따라 찾는다. 설치·실행은 해당 모듈 디렉터리에서 수행하며 문서 확인만으로 서버나 외부 앱을 실행하지 않는다.

## 권장 읽기 순서

1. [전송용 분할·복원](python/zip-split-transfer/README.md): 바이트 흐름과 무결성을 이해한다. 임시 파일 smoke 검증, 대용량·원격 전송은 미확인.
2. [OpenSearch 헬퍼](python/opensearch_handler/README.md) → [주제별 예제](python/opensearch_handler/example/README.md): 클라이언트와 인덱스·질의 구조를 읽는다. 실서버 검증 미완료.
3. [대화 메모리](python/history-opensearch/README.md): 검색·요약·팩트의 3계층 흐름을 읽는다. OpenSearch·임베딩·LLM 서버 필요.
4. [PowerPoint 이미지 export](python/drm-pptx-extraction/README.md): Windows COM과 앱 권한 조건을 읽는다. 사내 실제 내보내기 미확인.

## 문서가 없는 실행 모듈

- `python/org-hierarchy/`: [OrgTree 구현](python/org-hierarchy/org_tree.py), [데모](python/org-hierarchy/main.py). 조직 경로를 딕셔너리로 만들고 부모·자손·공통 조상을 조회한다. 이름을 키로 쓰므로 서로 다른 조직의 동일 이름을 별개 ID로 표현하는 모델은 아니다. 실제 조직 데이터와 샘플을 혼동하지 않는다.
- `python/llm-key-rotator/`: [키 순환 구현](python/llm-key-rotator/key_rotator.py). 연속 실패 임계에 따라 다음 키로 전환하는 예제다. 실제 API 호출·환경 인증은 미확인이며 `.env`·인증값은 문서 대상이 아니다.

두 모듈의 새 상세 사용 안내는 코드의 실제 환경 검증과 Claude 협의를 확보한 뒤 보강한다. [정리 기록](organization-log.md)에서 기존 문서별 검토와 검증 경계를 확인한다.
