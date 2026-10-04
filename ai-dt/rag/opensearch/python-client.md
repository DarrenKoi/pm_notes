---
tags: [opensearch, python, client, performance, async, bulk]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
level: intermediate
last_updated: 2026-02-07
---

# OpenSearch Python 클라이언트 활용 (Advanced Python Client)

> 대량 색인·전체 조회·비동기 요청을 opensearch-py로 구성하는 학습 가이드다.

> [!info] 검토 조건 — 2026-10-04
> 학습 예제다. 역사적 작성일은 유지하고 이번 검토일을 별도로 기록했다. 로컬 확인은 Python3.14.2·opensearch-py3.2.0 및 모의 REST 응답에 한정한다. 실제 OpenSearch/Docker·보안·검색 품질·100GB 부하를 검증하지 않았다. 함수는 대상과 입력을 명시해 호출한다. [정리 기록](../organization-log.md)을 함께 읽는다.


## 왜 필요한가? (Why)

문서 수·문서 크기·갱신 빈도·동시 요청에 따라 단건 호출의 왕복 비용, 메모리 사용, 서버 부하가 달라진다. 100GB는 원래 예제의 규모 가정이며 이 문서의 실행 성능 보장이 아니다.

- **대량 색인**: bulk는 여러 동작을 묶는다. 항목별 실패와 배치 바이트/재시도 정책을 관리한다.
- **대용량 조회**: from/size·search_after+PIT·scroll은 일관성/정렬/자원 유지 조건이 다르다.
- **안정성**: timeout은 성공/실패 확정과 다를 수 있다. 재시도에 따른 중복/충돌을 처리한다.
- **동시 요청**: async는 이벤트 루프의 네트워크 대기를 줄일 때 고려한다. 모든 프레임워크에 필수거나 더 빠른 것은 아니다.

---

## 핵심 기능 (What)

### 1. 클라이언트 설정 (Configuration)

대용량 처리를 위해서는 타임아웃과 커넥션 설정을 튜닝해야 한다.

| 파라미터 | opensearch-py3.2.0에서 확인한 조건 | 예제 값의 의미 |
|---|---|---|
| `timeout` | connection의 요청 timeout. 전체 작업 기한/취소 보장과 다름 | 60초는 제안값. 부하/요구 응답 시간으로 검증 |
| `max_retries` | Transport 기본 3; 설정한 상태/연결 오류에 적용 | 예제 0으로 자동 재전송을 끄고 상위 정책에 맡김 |
| `retry_on_timeout` | 기본 False | timeout 뒤 서버 반영 여부가 미확정일 수 있음 |
| `pool_maxsize` | sync Requests/Urllib3 연결의 pool 설정 | 25는 예시이며 동시성 보장이나 부하 권장값이 아님 |
| `maxsize` | AsyncHttpConnection의 연결 한도 설정 | sync Requests 예제의 `maxsize=25`는 올바른 설정이 아니었음 |

설치 소스의 Transport/RequestsHttpConnection/AsyncHttpConnection을 대조했다. 연결 구현을 바꾸면 파라미터도 다시 확인한다. [공식 client](https://docs.opensearch.org/latest/clients/python-low-level/), [Requests 연결 소스](https://github.com/opensearch-project/opensearch-py/blob/main/opensearchpy/connection/http_requests.py), 확인 2026-10-04.

### 2. Bulk Helpers

OpenSearch는 한 번의 요청으로 여러 문서를 처리하는 `_bulk` API를 제공한다. `opensearch-py`의 `helpers` 모듈은 이를 쉽게 래핑해준다.

- `helpers.bulk()`: streaming 결과를 집계한다. 기본값은 항목 실패에서 예외를 발생시킨다. 실패 목록에는 원문이 포함될 수 있다.
- `helpers.parallel_bulk()`: thread 기반 병렬 처리. 반환 generator를 소진해야 동작하며 서버 부하/실패를 측정한다. 가장 빠르다고 단정하지 않는다.
- `helpers.streaming_bulk()`: 배치를 순차 생성하고 항목 결과를 yield한다. 바이트 한도를 설정해도 한 문서가 그 한도를 넘는 경우의 입력 제한은 별도로 둔다.

### 3. Scroll & Scan

`from+size`는 인덱스의 max_result_window 기본 10,000 조건을 확인한다. 전체 조회는 scroll/scan 또는 일관된 정렬과 PIT를 갖춘 search_after 등을 목적에 맞게 선택한다. scan은 모든 문서를 한 목록에 담지 않는 iterator지만 서버 scroll context를 사용한다. `size`는 helper 설명상 shard별 배치 크기다. 이 예제는 scan만 구현하며 PIT/search_after는 별도 설계다. [페이지네이션](https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/), [helpers 소스](https://github.com/opensearch-project/opensearch-py/blob/main/opensearchpy/helpers/actions.py), 확인 2026-10-04.

---

## 어떻게 사용하는가? (How)

### 1. 견고한 클라이언트 생성

```python
from pathlib import Path
from opensearchpy import OpenSearch, RequestsHttpConnection

def tls_client(host: str, port: int, auth: tuple[str, str], ca_certs: str) -> OpenSearch:
    if not isinstance(host, str) or not host or "/" in host:
        raise ValueError("승인된 서버 hostname을 지정하세요")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("포트가 잘못되었습니다")
    if not isinstance(auth, tuple) or len(auth) != 2 or not all(isinstance(x, str) and x for x in auth):
        raise ValueError("로컬 설정/환경에서 인증값을 공급하세요")
    if not Path(ca_certs).is_file():
        raise ValueError("검증할 CA 파일을 지정하세요")
    return OpenSearch(
        hosts=[{"host": host, "port": port}], http_auth=auth,
        use_ssl=True, verify_certs=True, ca_certs=ca_certs,
        connection_class=RequestsHttpConnection, http_compress=True,
        timeout=60, max_retries=0, retry_on_timeout=False, pool_maxsize=25,
    )
# 이 factory는 자기 호스팅 basic auth 예시다. 관리형 IAM/SigV4/Serverless는 별도 계약.
# 필요 함수만 호출하고 finally에서 client.close(). 인증/접근 권한을 실제로 검증하지 않았다.
```

### 2. 대량 데이터 고속 색인 (Parallel Bulk)

아래는 명시한 실습 인덱스에 가상 데이터를 create하는 예제다. 기존 ID는 충돌하며 덮어쓰지 않는다. 100,000건·thread4/queue4/chunk2000은 측정된 권장값이 아니다. 항목의 HTTP 201/409/429 등을 구분해 요약한다. 전송 오류는 위로 전달하고 실패를 성공으로 바꾸지 않는다. 실제 문서의 source/ID·권한·실패 큐는 별도 구현이다.

```python
from datetime import datetime, timezone
import json
from opensearchpy import helpers

def generate_data(index_name: str, num_docs: int = 100000):
    if not isinstance(index_name, str) or not index_name.strip() or type(num_docs) is not int or num_docs < 0:
        raise ValueError("실습 index와 건수가 잘못되었습니다")
    timestamp = datetime.now(timezone.utc).isoformat()
    for i in range(num_docs):
        yield {"_op_type": "create", "_index": index_name, "_id": f"demo-{i}",
               "_source": {"title": f"Document {i}", "value": i, "timestamp": timestamp}}

def bounded_actions(actions):
    for action in actions:
        meta, source = helpers.expand_action(action)
        serialized = json.dumps(meta, ensure_ascii=False, allow_nan=False)
        if source is not None:
            serialized += "\n" + json.dumps(source, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) + 1 > 1024 * 1024:
            raise ValueError("단일 동작의 실습 바이트 한도 초과")
        yield action

def index_large_data(client: OpenSearch, index_name: str, num_docs: int = 100000) -> dict:
    summary = {"success": 0, "failed": 0, "status_counts": {}}
    for ok, item in helpers.parallel_bulk(
        client, bounded_actions(generate_data(index_name, num_docs)),
        thread_count=4, chunk_size=2000, max_chunk_bytes=5 * 1024 * 1024,
        queue_size=4, raise_on_error=False, raise_on_exception=True,
    ):
        status = next(iter(item.values()))["status"]
        summary["success" if ok else "failed"] += 1
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    return summary
# timestamp는 UTC ISO 문자열. date mapping의 epoch_millis에 초 단위 time.time()를 넣지 않는다.
# 실패 항목 원문을 print하지 않는다. failed>0이면 상위 작업을 실패 처리하고 상태별 정책을 적용한다.
```

### 3. 전체 데이터 조회 (Scan)

10,000건이 넘는 데이터를 모두 가져와야 할 때 (예: 데이터 마이그레이션, 분석).

```python
from collections.abc import Callable

def fetch_all_docs(client: OpenSearch, index_name: str, consume: Callable[[dict], None]) -> int:
    scan_gen = helpers.scan(
        client, query={"query": {"match_all": {}}}, index=index_name,
        size=1000, scroll="5m", _source=["title", "timestamp"],
        request_timeout=60, clear_scroll=True,
    )
    count = 0
    try:
        for hit in scan_gen:
            consume(hit)
            count += 1
    finally:
        scan_gen.close()  # callback 예외/조기 중단에서도 서버 scroll 해제 경로 실행
    return count
# 반환은 처리 건수. 누적 list/원문 전체 print는 하지 않는다. clear_scroll 자체 실패는 위로 전달된다.
```

### 4. 비동기 클라이언트 (Async Client)

이벤트 루프에서 직접 blocking SDK를 호출하지 않도록 async SDK 또는 승인된 thread 실행 경로를 선택한다. async_client는 성공·실패·취소 경로에서도 finally에서 닫는다. 아래는 새 클라이언트를 factory로 공급하고 close하는 단일 요청 예제다. coroutine을 만든 것만으로 요청은 실행되지 않는다.

```bash
python -m pip install "opensearch-py[async]==3.2.0"
```

```python
from opensearchpy import AsyncOpenSearch

async def async_search(client_factory, index_name: str) -> dict:
    async_client = client_factory()  # 승인된 서버/TLS/인증 설정으로 명시 공급
    try:
        return await async_client.search(
            index=index_name, body={"query": {"match_all": {}}}, size=5,
        )
    finally:
        await async_client.close()
# asyncio.run(async_search(factory, "새실습인덱스"))는 이미 event loop가 없는 환경에서만.
# 실행 중인 async 함수에서는 await를 사용한다. 실제 서비스의 factory/권한은 별도 검증한다.
```

### 5. 에러 핸들링 패턴

```python
from opensearchpy import TransportError, ConnectionError, NotFoundError

def search_or_raise(client: OpenSearch, index_name: str) -> dict:
    try:
        return client.search(index=index_name, body={"query": {"match_all": {}}})
    except NotFoundError as exc:
        raise RuntimeError("index_not_found") from exc
    except ConnectionError as exc:
        raise RuntimeError("connection_unconfirmed") from exc
    except TransportError as exc:
        raise RuntimeError(f"transport_status_{exc.status_code}") from exc
# 오류를 빈 검색 결과로 바꾸지 않는다. 사용자 출력은 정해진 코드만 사용하고
# 원래 exception/trace의 raw body·URL·인증값은 허용된 보호 로그에서 별도로 관리한다.
```

---

## 100GB 데이터 처리 시 팁

1. **Chunk Size**: 문서 개수와 실제 NDJSON UTF-8 바이트는 별도다. 원래 5~15MB·1KB당5000건은 미확인 계획값이다. 실제 metadata/멀티바이트/단일 대형 문서·서버 제한을 측정한다. 아래 1MiB/5MiB는 유한 실습 한도다.
2. **Refresh Interval**: refresh를 줄이면 검색 반영이 지연된다. 변경이 필요할 때 현재값/상속값을 보관하고 성공·실패 경로 모두 복원한다. 권한·동시 writer·새 문서 가시성/후속 검증을 갖추기 전 자동 변경하지 않는다. refresh는 백업/flush와 다르다.
3. **Source Filtering**: 필요한 필드만 요청한다. 전달 callback의 처리/로그 정책은 별도다. 원래 query=... placeholder를 실제 match_all로 바꿨다.

```python
def selected_fields(client: OpenSearch, index_name: str, consume: Callable[[dict], None]) -> int:
    return fetch_all_docs(client, index_name, consume)
# 실제 match_all·source 필터·generator 소진/해제는 위 대표 scan 함수에서 관리한다.
```

---

## 참고 자료 (References)

- [opensearch-py Documentation](https://docs.opensearch.org/latest/clients/python-low-level/)
- [Python Bulk Helpers 소스](https://github.com/opensearch-project/opensearch-py/blob/main/opensearchpy/helpers/actions.py)

## 관련 문서

- [OpenSearch 성능 최적화](./performance-optimization.md) - 대용량 처리를 위한 서버 설정
- [OpenSearch 기초](./opensearch-basics.md)
