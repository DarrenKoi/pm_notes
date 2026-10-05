---
title: OpenSearch 헬퍼 주제별 예제 읽기 순서
tags: [python, opensearch, examples]
document_type: learning
reviewed_on: 2026-10-04
verification_status: source-verified-live-unverified
category_major: "Python 실행 예제"
category_middle: "검색·메모리"
category_minor: "OpenSearch 헬퍼"
note_kind: "목차"
classified_on: "2026-10-05"
---

# OpenSearch 헬퍼 주제별 예제 읽기 순서

> 연결부터 raw 쿼리까지 기능별로 실행 흐름을 이해하는 예제다. 일부 스크립트는 인덱스를 삭제하고 다시 만든다.

## 준비

설치와 환경변수의 대표 설명은 [패키지 안내](../README.md)에 통합했다. 저장소 루트에서 `cd Codes/python/opensearch_handler`로 이동한 뒤 실행한다. 일반 Markdown과 Obsidian의 상대 링크로 안내를 읽을 수 있다.

실행 전 각 파일의 인덱스 이름·삭제·쓰기 범위를 확인한다. Topic 05는 같은 이름의 기존 인덱스를 삭제하고 종료 시 정리하며 Topic 06은 인덱스를 삭제해도 템플릿을 남긴다. `dry_run=True`는 rollover 요청의 옵션이며 스크립트 전체가 읽기 전용이라는 뜻이 아니다.

## 읽고 실행하는 순서

```bash
python3 -m example.topic_01_connection
python3 -m example.topic_02_index_management
python3 -m example.topic_03_document_crud
python3 -m example.topic_04_search_text_and_agg
python3 -m example.topic_05_vector_and_hybrid
python3 -m example.topic_06_template_alias_rollover
python3 -m example.topic_07_search_raw
```

| 단계 | 파일 | 목적·조건 |
|---|---|---|
| 01 | [연결](topic_01_connection.py) | 환경 설정과 연결 확인 |
| 02 | [인덱스](topic_02_index_management.py) | 생성·mapping·settings·refresh·alias 검사 |
| 03 | [문서](topic_03_document_crud.py) | 단건 CRUD와 bulk 색인 |
| 04 | [검색·집계](topic_04_search_text_and_agg.py) | 텍스트 질의와 집계 |
| 05 | [벡터·결합 검색](topic_05_vector_and_hybrid.py) | k-NN mapping 필요. 현재 예제 nmslib의 서버 버전 호환 미확인 |
| 06 | [템플릿·rollover](topic_06_template_alias_rollover.py) | 번호 인덱스·read/write alias·rollover dry-run. 보존 정책 관리 없음 |
| 07 | [raw 검색](topic_07_search_raw.py) | 헬퍼 밖의 요청 본문 직접 구성 |

Topic 05의 `hybrid_search()`는 전용 hybrid query가 아니라 로컬 bool 결합 구현이다. 엔진 deprecation과 공식 검색 방식의 차이는 [패키지 안내](../README.md)에서 먼저 확인한다.

## 검증 경계

2026-10-04 문서·파일 존재·호출 계약을 확인했다. 서버에 예제를 실행하지 않았으므로 인증·색인·검색·cleanup 성공은 미확인이다. 검증 목적의 실서버 실행은 복제 가능한 실습 데이터와 전용 인덱스에서 수행한다.
