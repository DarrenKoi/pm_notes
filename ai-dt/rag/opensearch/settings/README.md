---
tags: [opensearch, index-template, mapping, alias, rollover, ism]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [OpenSearch 매핑과 인덱스 수명 관리]
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "OpenSearch 검색"
note_kind: "목차"
classified_on: "2026-10-05"
---

# OpenSearch Settings 실무 가이드

> 필드의 검색 목적과 인덱스 생성·쓰기 경로·수명 정책을 분리해서 설계하는 학습 가이드

> [!info] 적용 조건 — 확인 2026-10-04
> 먼저 [기초](../opensearch-basics.md)와 [Python 클라이언트](../python-client.md)를 읽고 필드 설계→템플릿→alias/rollover→ISM 순서로 개념을 익힌다. 서버 OpenSearch3.x와 ISM 플러그인·권한을 가정하고 로컬SDK검증은opensearch-py3.2.0이다. HTTP 예제는 REST Console 형식이다. 별도 fresh 실습 namespace에서 **정책→인덱스 템플릿→부트스트랩** 순서로 명시 실행해야 최초 index에도 ism_template이 적용된다. 삭제 정책은 학습 예시이며 실제 보존 요구를 승인한 것이 아니다. 이번에는 실제 서버에 전송하지 않았다.

## 1) 필드 타입 설계 기준 (토큰화 vs 비토큰화)

핵심 원칙은 간단하다.

- 문장 검색이 필요하면 `text`
- 정확 매칭/필터/집계/정렬이면 `keyword`
- 범위 검색이면 숫자/날짜 타입
- 저장 전용이면 필드 타입에 맞춰 `index: false`와 지원되는 `doc_values: false`, 또는 object의 `enabled: false`를 검토한다. `_source` 설정도 별도 확인한다.

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

- `title`(text) + `title.raw`(keyword) 멀티필드: 분석 검색과 정확 토큰 매칭/집계를 분리. `ignore_above:256`을 넘는 문자열은 raw에 색인되지 않는다
- `doc_id/category`는 `keyword`로 두어 정확 토큰 필터 조건을 표현한다. 성능 개선 폭은 측정 전 미확인
- `payload.enabled: false`: 기본 `_source`에 원문을 남기되 object 내부 파싱/매핑을 생략한다. 비밀 정보 제거/접근 권한을 구현하는 설정은 아니다
- `raw_message.index: false`: `_source`에 남기되 검색 인덱스는 만들지 않음

### 실수 방지 팁

- ID, code, enum, email, URL, UUID는 기본적으로 `keyword`
- 집계/정렬할 가능성이 있으면 `text` 단독으로 만들지 말고 `keyword` 서브필드 추가
- 동적 매핑만 믿지 말고, 최소한 핵심 필드는 명시적으로 매핑

## 2) 인덱스 템플릿으로 매핑 표준화

index template은 이름 패턴에 맞는 **새 index**의 기본 매핑/설정/alias를 재사용한다. 기존 index를 소급 변경하지 않으며 명시 생성 요청의 값이 우선할 수 있어 불변 schema 강제가 아니다. 여러 composable template이 맞으면 최고 priority를 선택하고 같은 priority의 겹치는 패턴은 거부된다. version은 사용자 관리 번호이며 내용 변경을 자동 차단하지 않는다.

```http
PUT /_index_template/notes-template-v1?create=true
{
  "index_patterns": ["notes-*"],
  "template": {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 1,
      "refresh_interval": "1s",
      "index.plugins.index_state_management.rollover_alias": "notes-write"
    },
    "aliases": {"notes-read": {}},
    "mappings": {
      "dynamic_templates": [
        {
          "ids_as_keyword": {
            "match": "*_id",
            "match_mapping_type": "string",
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
  "priority": 100,
  "version": 1
}
```

## 3) Alias + Rollover 정석

### 추천 네이밍

- 실제 인덱스: `notes-000001`, `notes-000002`
- 쓰기 alias: `notes-write` (`is_write_index: true`)
- 읽기 alias: `notes-read` (이 예제의 모든 세대 검색). 위 템플릿이 새 index에도 부착한다. write alias는 각 세대 템플릿에 `is_write_index:true`로 반복 설정하지 않는다.

### 부트스트랩 (처음 1회)

아래 이름은 설명용이다. 기존 동명의 자원/다른 template과 충돌하지 않는 독립 실습 공간에서 먼저4절 정책과2절 템플릿을 준비한다. 정책 생성 전에 만든 기존 index는 자동 적용됐다고 가정하지 말고 Explain으로 확인한다. alias만 같아도 대상 index가 바뀔 수 있어 생성/조회 결과의 concrete 이름을 확인한다.

```http
PUT /notes-000001
{
  "aliases": {
    "notes-write": { "is_write_index": true },
    "notes-read": {}
  }
}
```

### 롤오버 실행

먼저 `POST /notes-write/_rollover?dry_run=true`로 조건을 조회하고 `rolled_over`/`dry_run`/`conditions`를 확인한다. dry_run은 실제 생성/alias 전환 성공·권한·디스크 여유를 보장하지 않는다. 아래 요청의 max_age/max_primary_shard_size/max_docs는 **하나 이상 충족하면(OR)** rollover한다. max_age는 index 생성 이후이고 크기는 최대 primary shard 기준이며 replica 합계가 아니다. 기본 숫자 suffix가 증가한다.

```http
POST /notes-write/_rollover
{
  "conditions": {
    "max_age": "7d",
    "max_primary_shard_size": "30gb",
    "max_docs": 2000000
  }
}
```

운영 포인트:

- 이 시간순 append 예제의 새 쓰기는 `notes-write`로 보낸다. 과거 문서 수정/삭제는 조회 hit의 concrete index와 id/routing을 사용해야 한다. write alias는 과거 세대의 문서를 자동 찾아 수정하지 않는다
- 검색은 `notes-read`를 사용하고 새 세대가 포함됐는지 alias 조회로 확인한다. 읽기 alias는 write alias 전환만으로 자동 추가되지 않는다
- 롤오버 이후 write alias는 자동으로 새 인덱스를 가리킴

## 4) 오래된 데이터 삭제: ISM 정책

Dashboards UI와 API는 같은 정책을 관리하는 다른 조작 경로다. 원문의 사용 경험은 현재 서버 설정 증거가 아니다. JSON·정책 id/seq_no/primary_term·Explain 결과를 기록한다. ism_template은 matching **신규** index의 정책 자동 연결이며 일반 index template의 mapping/alias 적용과 별도다. 기존 index는 명시 add/change API와 결과 확인이 필요하다.

### 예시 정책 (14일 유지 후 삭제)

제목의14일은 **index 생성일부터14일 경과한 index 전체를 삭제하는 후보**다. 문서/event timestamp별 최소14일 보존도 rollover 후14일도 아니다. late-arriving 문서는 훨씬 짧게 남을 수 있다. min_rollover_age/min_state_age는 서로 다른 시계이며 요구 정책을 정하기 전 임의 교체하지 않는다. action 완료 후 transition을 주기적으로 검사하므로 정확14일 시각 삭제가 아니고 rollover 실패/지연이면 삭제 전이도 진행하지 못할 수 있다. rollover action의1d/10gb 역시 OR이며 한도가 정확히 맞을 때 즉시 실행되는 것은 아니다.

```http
PUT /_plugins/_ism/policies/notes-retention-v1
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

Dashboards는 사람이 상태를 탐색하는 데, API는 변경을 재현하는 데 쓸 수 있다. 한 방식이 보편적으로 가장 편하다고 단정하지 않는다. core API는 SDK 메서드를, 이 예제의 ISM은 `transport.perform_request()`를 사용한다. 이 요청 형태가 실제 서버의 ISM 동작을 증명하지는 않는다.

### Python 예시 (ISM/Template/Alias)

```python
from copy import deepcopy
import re
from opensearchpy import OpenSearch, NotFoundError

# 아래 기본 body는 위 REST 예제와 같은 정책/매핑/alias를 표현한다.
NOTES_TEMPLATE = {'index_patterns': ['notes-*'],
 'template': {'settings': {'number_of_shards': 1,
                           'number_of_replicas': 1,
                           'refresh_interval': '1s',
                           'index.plugins.index_state_management.rollover_alias': 'notes-write'},
              'aliases': {'notes-read': {}},
              'mappings': {'dynamic_templates': [{'ids_as_keyword': {'match': '*_id',
                                                                     'match_mapping_type': 'string',
                                                                     'mapping': {'type': 'keyword'}}}],
                           'properties': {'title': {'type': 'text',
                                                    'fields': {'raw': {'type': 'keyword',
                                                                       'ignore_above': 256}}},
                                          'created_at': {'type': 'date'}}}},
 'priority': 100,
 'version': 1}

NOTES_BOOTSTRAP = {'aliases': {'notes-write': {'is_write_index': True}, 'notes-read': {}}}

NOTES_POLICY = {'policy': {'description': 'Delete whole indices at creation age >=14d after rollover; '
                           'demo only',
            'default_state': 'hot',
            'states': [{'name': 'hot',
                        'actions': [{'rollover': {'min_index_age': '1d',
                                                  'min_primary_shard_size': '10gb'}}],
                        'transitions': [{'state_name': 'delete',
                                         'conditions': {'min_index_age': '14d'}}]},
                       {'name': 'delete',
                        'actions': [{'delete': {}}],
                        'transitions': []}],
            'ism_template': [{'index_patterns': ['notes-*'], 'priority': 100}]}}

def settings_plan(prefix: str) -> dict:
    """별도 namespace의 요청 body를 생성할 뿐 서버에 연결하지 않는다."""
    if not isinstance(prefix, str) or not re.fullmatch(r"[a-z][a-z0-9-]{1,39}", prefix):
        raise ValueError("invalid_demo_prefix")
    template = deepcopy(NOTES_TEMPLATE)
    policy = deepcopy(NOTES_POLICY)
    template["index_patterns"] = [f"{prefix}-*"]
    template["template"]["settings"]["index.plugins.index_state_management.rollover_alias"] = f"{prefix}-write"
    template["template"]["aliases"] = {f"{prefix}-read": {}}
    policy["policy"]["ism_template"][0]["index_patterns"] = [f"{prefix}-*"]
    bootstrap = {"aliases": {f"{prefix}-write": {"is_write_index": True},
                             f"{prefix}-read": {}}}
    return {"template_name": f"{prefix}-template-v1", "policy_id": f"{prefix}-retention-v1",
            "first_index": f"{prefix}-000001", "pattern": f"{prefix}-*",
            "write_alias": f"{prefix}-write", "template": template,
            "policy": policy, "bootstrap": bootstrap}


def create_settings_demo(client: OpenSearch, prefix: str) -> dict:
    """동시 관리자 없는 독립 실습에서만 caller가 명시 호출한다."""
    plan = settings_plan(prefix)
    if client.indices.exists(index=plan["pattern"]):
        raise ValueError("demo_namespace_already_exists")
    # 404만 없는 자원으로 취급한다. 403/연결 오류는 그대로 전파한다.
    for read in (
        lambda: client.indices.get_index_template(name=plan["template_name"]),
        lambda: client.transport.perform_request(
            "GET", f"/_plugins/_ism/policies/{plan['policy_id']}"),
    ):
        try:
            read()
        except NotFoundError:
            continue
        raise ValueError("demo_resource_already_exists")
    # 정책이 먼저 있어야 첫 index에도 ism_template 자동 연결을 기대할 수 있다.
    policy_result = client.transport.perform_request(
        "PUT", f"/_plugins/_ism/policies/{plan['policy_id']}", body=plan["policy"])
    template_result = client.indices.put_index_template(
        name=plan["template_name"], body=plan["template"], params={"create": True})
    index_result = client.indices.create(index=plan["first_index"], body=plan["bootstrap"])
    return {"plan": plan, "policy_result": policy_result,
            "template_result": template_result, "index_result": index_result}


def inspect_settings_demo(client: OpenSearch, prefix: str) -> dict:
    plan = settings_plan(prefix)
    return {
        "aliases": client.indices.get_alias(index=plan["pattern"]),
        "ism": client.transport.perform_request(
            "GET", f"/_plugins/_ism/explain/{plan['first_index']}"),
        "rollover_dry_run": client.indices.rollover(
            alias=plan["write_alias"], params={"dry_run": True},
            body={"conditions": {"max_age": "7d", "max_primary_shard_size": "30gb",
                                 "max_docs": 2000000}}),
    }

# 연결은 앞 Python 클라이언트 문서에서 준비한 caller 소유 client를 명시 전달한다.
# 실행 예: result = create_settings_demo(client, "demo-notes-unique")
# 검사 예: status = inspect_settings_demo(client, "demo-notes-unique")
# 이번 문서 검증은 실제 서버에 전송하지 않는다. 종료 시 client.close()는 caller 책임이다.
```

GET→PUT policy 생성은 원자적 create-only가 아니다. 위 함수는 독립 namespace/동시 관리자 없음 조건이며 template만 create=true를 사용한다. 여러 요청은 transaction이 아니어서 중간 실패 시 생성된 정책/템플릿이 남을 수 있다. 이름·서버 응답을 기록해 개별 확인하고 자동 삭제/롤백하지 않는다. 반환했다고 ISM 연결이나 전체 shard acknowledgement가 완료됐다고 단정하지 말고 inspect/Explain에서 구체적으로 확인한다. 코드에는 실제 rollover/삭제 호출이 없다.

## 6) 추천 운영 체크리스트

- 시간순 신규 쓰기와 과거 수정/삭제의 concrete index 경로를 구분했는가?
- 새 index의 template 우선순위·schema·alias 결과와 ISM 연결을 확인했는가?
- `_id`는 특별 metadata다. 일반 doc_id/문자 *_id·enum/code와 구분하고 집계/정렬용 복사 필드는 keyword로 설계했는가?
- 저장 전용 필드의 타입별 index/doc_values/enabled와 _source 조건을 확인했는가?
- age/크기 조건 OR·주기적 실행·문서별 보존과 index 수명 차이를 이해하고 실제 보존 요구를 검토했는가?
- ISM 정책 JSON을 Git으로 관리 (Dashboard에서 만든 정책도 export 보관)

## 참고 자료 (References)

확인일 **2026-10-04**. rolling 자료는 대상 서버 고정 판본/권한/설치·삭제 시각 증거가 아니다.

- [Index parameter](https://docs.opensearch.org/latest/mappings/mapping-parameters/index-parameter/)·[Mappings](https://docs.opensearch.org/latest/mappings/)·[ignore_above](https://docs.opensearch.org/latest/mappings/mapping-parameters/ignore-above/)·[enabled](https://docs.opensearch.org/latest/mappings/mapping-parameters/enabled/)·[_id metadata](https://docs.opensearch.org/latest/mappings/metadata-fields/id/)
- [Index templates](https://docs.opensearch.org/latest/api-reference/index-apis/create-index-template/)
- [Rollover](https://docs.opensearch.org/latest/api-reference/index-apis/rollover/)
- [ISM operations](https://docs.opensearch.org/latest/im-plugin/ism/policies-operations/)·[Policies](https://docs.opensearch.org/latest/im-plugin/ism/policies/)·[Policy examples](https://docs.opensearch.org/latest/im-plugin/ism/policies-examples/)·[ISM API](https://docs.opensearch.org/latest/im-plugin/ism/api/)

## 관련 문서

- [OpenSearch 기초](../opensearch-basics.md)
- [Python 클라이언트 활용](../python-client.md)
- [성능 최적화](../performance-optimization.md)
