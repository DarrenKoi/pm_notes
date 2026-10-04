---
title: OpenSearch Python 헬퍼 사용 안내
tags: [python, opensearch]
aliases: [opensearch_handler]
document_type: learning
reviewed_on: 2026-10-04
verification_status: source-verified-live-unverified
---

# OpenSearch Python 헬퍼 사용 안내

> 반복되는 연결·인덱스·문서·검색 요청을 작은 함수로 묶는 로컬 예제 패키지다.

## 목적과 작동 방식

`opensearch_handler`는 `opensearch-py` 클라이언트를 만들고 REST 요청 본문을 구성한다. 인덱스 설정, alias·rollover 이름 규칙, 문서 CRUD·bulk, 텍스트·집계·벡터 검색과 raw 요청을 다룬다. 보존 기간 정책 관리는 구현 범위가 아니다. "실무 패턴 예제"라는 설명이 운영 검증을 뜻하지는 않는다.

로컬 패키지 버전은 `pyproject.toml`의 0.1.0, Python 요구 조건은 3.10 이상, 클라이언트 의존성은 `opensearch-py>=2.4.0`이다. 하한만으로 모든 최신 서버와의 호환성이 검증되지는 않는다.

## 설치와 읽기 순서

저장소 루트에서:

```bash
cd Codes/python/opensearch_handler
python3 -m pip install -r requirements.txt
```

재사용 패키지로 개발하려면 같은 디렉터리에서 `python3 -m pip install -e .`로 설치한다. 연결 설정을 준비한 뒤 [주제별 실행 예제](example/README.md)를 읽는다.

## 연결 설정

환경변수 → `load_config()` → `ConnectionConfig` → `create_client()` 순서다. 키워드 override가 환경변수보다 나중에 적용된다. 사용자와 비밀번호는 함께 지정하거나 함께 비워야 한다.

```bash
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER='local-demo-user'
export OPENSEARCH_PASSWORD='<실행 환경에서 제공>'
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=true
export OPENSEARCH_CA_CERTS='/path/to/root-ca.pem'
```

이 값은 예시이며 실제 인증값과 CA 경로로 교체한다. 코드의 기존 기본값은 인증 `admin/admin`, 인증서 검증 `false`다. 이번 작업에서 실행 코드는 바꾸지 않았으므로 설정을 생략하면 그 기본값이 적용된다. 검증을 끄는 `OPENSEARCH_VERIFY_CERTS=false`는 인증서 오류를 무시하는 동작이므로 격리된 로컬 예제에만 적용 조건을 명시한다. [OpenSearch Python 공식 연결 안내](https://docs.opensearch.org/latest/clients/python-low-level/), 확인일 2026-10-04.

```python
from opensearch_handler import create_client, load_config

client = create_client(load_config())
```

환경 대신 명시 객체를 쓰는 패턴도 가능하다.

```python
import os
from opensearch_handler import ConnectionConfig, create_client

client = create_client(ConnectionConfig(
    host="localhost",
    port=9200,
    user=os.environ["OPENSEARCH_USER"],
    password=os.environ["OPENSEARCH_PASSWORD"],
    use_ssl=True,
    verify_certs=True,
    ca_certs=os.environ["OPENSEARCH_CA_CERTS"],
))
```

## 검색에서 확인할 조건

`knn_search()`에는 필드 mapping과 같은 차원의 벡터가 필요하다. `hybrid_search()`는 로컬 구현에서 `bool.should`로 `match`와 `knn`을 결합한다. OpenSearch의 전용 `hybrid` query와 search pipeline 정규화를 설정하는 함수는 아니다. 점수 결합 방식과 필터 결과는 실제 서버에서 확인해야 한다. [공식 hybrid search](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/), 확인일 2026-10-04.

Topic 05의 현재 코드는 `engine: nmslib`를 사용한다. 공식 OpenSearch 3.0 변경 안내는 이 엔진을 deprecated로 표시한다. 실행 대상 버전의 신규 인덱스 허용 여부를 확인하기 전에는 작동을 보장하지 않는다. [공식 breaking changes](https://docs.opensearch.org/latest/breaking-changes/), 확인일 2026-10-04. 코드의 엔진 변경은 이번 문서 작업에 포함하지 않았다.

헬퍼에 없는 기능은 `search_raw()`로 요청 본문을 직접 전달하거나 해당 프로젝트의 작은 래퍼로 확장한다.

## 팀에서 배포하기

개발 중에는 공유 저장소의 이 폴더에서 editable 설치, 작업 환경에는 승인된 Git 저장소 URL로 설치, API가 안정화되면 사내 패키지 저장소 발행을 검토한다. 실제 배포·발행은 실행하지 않았다.

## 검증 상태

2026-10-04 연결·검색 소스와 공식 문서를 대조했다. 실제 OpenSearch 서버 버전·인증·k-NN·검색 점수·rollover 동작은 미확인이다. [정리 기록](../../organization-log.md)에 로컬 검사 결과를 남긴다.
