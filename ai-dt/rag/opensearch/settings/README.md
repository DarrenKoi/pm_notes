---
tags: [opensearch, index-template, mapping, alias, rollover, ism]
level: intermediate
last_updated: 2026-02-12
status: in-progress
---

# OpenSearch Settings 실무 가이드

> 인덱스 매핑, 템플릿, alias, rollover, 오래된 데이터 삭제(ISM)를 안정적으로 운영하기 위한 기준 정리

## 1) 필드 타입 설계 기준 (토큰화 vs 비토큰화)

핵심 원칙은 간단하다.

- 문장 검색이 필요하면 `text`
- 정확 매칭/필터/집계/정렬이면 `keyword`
- 범위 검색이면 숫자/날짜 타입
- 검색은 안 하고 저장만 하려면 `index: false`

### 자주 쓰는 패턴

```json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "fields": {
          "raw": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "doc_id": { "type": "keyword" },
      "category": { "type": "keyword" },
      "price": { "type": "long" },
      "created_at": { "type": "date" },
      "payload": {
        "type": "object",
        "enabled": false
      },
      "raw_message": {
        "type": "text",
        "index": false
      }
    }
  }
}
```

- `title.text + title.raw` 멀티필드: 검색과 정확 매칭을 모두 지원
- `doc_id/category`는 `keyword`로 두어 필터 성능 확보
- `payload.enabled: false`: 원본 저장은 하되 내부 필드 파싱/인덱싱 비용 절감
- `raw_message.index: false`: `_source`에 남기되 검색 인덱스는 만들지 않음

### 실수 방지 팁

- ID, code, enum, email, URL, UUID는 기본적으로 `keyword`
- 집계/정렬할 가능성이 있으면 `text` 단독으로 만들지 말고 `keyword` 서브필드 추가
- 동적 매핑만 믿지 말고, 최소한 핵심 필드는 명시적으로 매핑

## 2) 인덱스 템플릿으로 매핑 표준화

신규 인덱스마다 동일한 설정을 강제하려면 `index template`이 필수다.

```json
PUT _index_template/notes-template-v1
{
  "index_patterns": ["notes-*"],
  "template": {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 1,
      "refresh_interval": "1s",
      "index.plugins.index_state_management.rollover_alias": "notes-write"
    },
    "mappings": {
      "dynamic_templates": [
        {
          "ids_as_keyword": {
            "match": "*_id",
            "mapping": { "type": "keyword" }
          }
        }
      ],
      "properties": {
        "title": {
          "type": "text",
          "fields": {
            "raw": { "type": "keyword", "ignore_above": 256 }
          }
        },
        "created_at": { "type": "date" }
      }
    }
  },
  "priority": 100
}
```

## 3) Alias + Rollover 정석

### 추천 네이밍

- 실제 인덱스: `notes-000001`, `notes-000002`
- 쓰기 alias: `notes-write` (`is_write_index: true`)
- 읽기 alias: `notes-read` (검색 서비스에서 사용)

### 부트스트랩 (처음 1회)

```json
PUT notes-000001
{
  "aliases": {
    "notes-write": { "is_write_index": true },
    "notes-read": {}
  }
}
```

### 롤오버 실행

```json
POST notes-write/_rollover
{
  "conditions": {
    "max_age": "7d",
    "max_primary_shard_size": "30gb",
    "max_docs": 2000000
  }
}
```

운영 포인트:

- 애플리케이션의 인덱스 대상은 항상 `notes-write`
- 검색 대상은 `notes-read` (또는 패턴 + read alias)
- 롤오버 이후 write alias는 자동으로 새 인덱스를 가리킴

## 4) 오래된 데이터 삭제: ISM 정책

Dashboard에서 이미 설정해봤다면, 운영은 그 방식이 가장 직관적이다.
다만 코드 재현성을 위해 JSON도 같이 보관해두는 것을 권장한다.

### 예시 정책 (14일 유지 후 삭제)

```json
PUT _plugins/_ism/policies/notes-retention-v1
{
  "policy": {
    "description": "Delete indices older than 14 days",
    "default_state": "hot",
    "states": [
      {
        "name": "hot",
        "actions": [
          {
            "rollover": {
              "min_index_age": "1d",
              "min_primary_shard_size": "10gb"
            }
          }
        ],
        "transitions": [
          {
            "state_name": "delete",
            "conditions": { "min_index_age": "14d" }
          }
        ]
      },
      {
        "name": "delete",
        "actions": [{ "delete": {} }],
        "transitions": []
      }
    ],
    "ism_template": [
      {
        "index_patterns": ["notes-*"],
        "priority": 100
      }
    ]
  }
}
```

## 5) Dashboard vs Python SDK (실무 권장)

질문한 포인트처럼 ISM은 Python SDK에서 불편할 수 있다. 이유는 간단하다.

- 코어 API(`indices`, `search`)는 SDK 메서드가 좋음
- 플러그인 API(ISM 등)는 `transport.perform_request()`가 더 직접적임

즉, **혼합 전략**이 실무에서 가장 편하다.

- Day-1 설정/검증: Dashboard
- 재현/자동화: Python 스크립트(JSON 정책/템플릿 버전 관리)

### Python 예시 (ISM/Template/Alias)

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False,
)

# 1) Index template
client.indices.put_index_template(
    name="notes-template-v1",
    body={
        "index_patterns": ["notes-*"],
        "template": {
            "settings": {
                "index.plugins.index_state_management.rollover_alias": "notes-write"
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "fields": {"raw": {"type": "keyword"}}
                    },
                    "doc_id": {"type": "keyword"},
                    "created_at": {"type": "date"}
                }
            }
        },
        "priority": 100
    }
)

# 2) Bootstrap first index + aliases
client.indices.create(
    index="notes-000001",
    body={
        "aliases": {
            "notes-write": {"is_write_index": True},
            "notes-read": {}
        }
    }
)

# 3) ISM policy (plugin API: perform_request)
policy_body = {
    "policy": {
        "description": "Delete after 14 days",
        "default_state": "hot",
        "states": [
            {
                "name": "hot",
                "actions": [{"rollover": {"min_index_age": "1d"}}],
                "transitions": [{"state_name": "delete", "conditions": {"min_index_age": "14d"}}]
            },
            {"name": "delete", "actions": [{"delete": {}}], "transitions": []}
        ]
    }
}

client.transport.perform_request(
    method="PUT",
    url="/_plugins/_ism/policies/notes-retention-v1",
    body=policy_body
)
```

## 6) 추천 운영 체크리스트

- 인덱싱 대상은 무조건 write alias로 통일
- 모든 신규 인덱스는 template + naming 규칙으로 생성
- `_id`, `*_id`, 상태값(enum), 코드값은 `keyword` 강제
- 검색 안 할 대형 원본 필드는 `index: false` 또는 `enabled: false`
- 롤오버 조건은 `age + shard_size` 조합으로 설정
- ISM 정책 JSON을 Git으로 관리 (Dashboard에서 만든 정책도 export 보관)

## 관련 문서

- [OpenSearch 기초](../opensearch-basics.md)
- [Python 클라이언트 활용](../python-client.md)
- [성능 최적화](../performance-optimization.md)
