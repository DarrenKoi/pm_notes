---
tags: [opensearch, python, client, performance, async, bulk]
level: intermediate
last_updated: 2026-02-07
status: in-progress
---

# OpenSearch Python 클라이언트 활용 (Advanced Python Client)

> opensearch-py 라이브러리를 사용하여 대용량 데이터(100GB+)를 효율적으로 처리하고, 비동기 통신 및 에러 핸들링을 구현하는 실무 가이드

## 왜 필요한가? (Why)

기본적인 `index()`나 `search()` 메서드만으로는 100GB 이상의 데이터를 다루기 어렵다.

- **대량 색인**: 루프를 돌며 하나씩 `index()`를 호출하면 너무 느리다. **Bulk API**가 필수다.
- **대용량 조회**: 1만 건 이상의 데이터를 한 번에 가져오면 메모리 초과나 타임아웃이 발생한다. **Scroll/Search After**가 필요하다.
- **안정성**: 네트워크 불안정이나 서버 부하 시 **재시도(Retry)** 및 **타임아웃** 설정이 중요하다.
- **성능**: 높은 동시성을 위해 **비동기(Async) 클라이언트**를 고려해야 한다.

---

## 핵심 기능 (What)

### 1. 클라이언트 설정 (Configuration)

대용량 처리를 위해서는 타임아웃과 커넥션 설정을 튜닝해야 한다.

| 파라미터 | 설명 | 기본값 | 권장 (대용량 시) |
|----------|------|--------|------------------|
| `timeout` | 요청 타임아웃 (초) | 10 | 30~60 |
| `max_retries` | 연결 실패 시 재시도 횟수 | 3 | 3~5 |
| `retry_on_timeout` | 타임아웃 발생 시 재시도 여부 | False | True |
| `maxsize` | 커넥션 풀 크기 | 10 | 25~50 |

### 2. Bulk Helpers

OpenSearch는 한 번의 요청으로 여러 문서를 처리하는 `_bulk` API를 제공한다. `opensearch-py`의 `helpers` 모듈은 이를 쉽게 래핑해준다.

- `helpers.bulk()`: 기본적인 벌크 처리.
- `helpers.parallel_bulk()`: 멀티 스레드로 병렬 처리 (가장 빠름).
- `helpers.streaming_bulk()`: 제너레이터 기반으로 메모리 효율적.

### 3. Scroll & Scan

일반 검색(`from` + `size`)은 `10,000`건(index.max_result_window) 제한이 있다. 전체 데이터를 순회하려면 `scroll` API를 사용해야 한다. `helpers.scan()`은 이를 추상화하여 편리하게 제공한다.

---

## 어떻게 사용하는가? (How)

### 1. 견고한 클라이언트 생성

```python
from opensearchpy import OpenSearch, RequestsHttpConnection

# 100GB 처리를 위한 견고한 클라이언트 설정
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),      # 보안 적용 시
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection,
    timeout=60,                        # 60초 타임아웃
    max_retries=5,                     # 5회 재시도
    retry_on_timeout=True,             # 타임아웃 시에도 재시도
    maxsize=25                         # 커넥션 풀 크기 증가
)
```

### 2. 대량 데이터 고속 색인 (Parallel Bulk)

가장 성능이 좋은 `parallel_bulk`를 사용하는 패턴이다.

```python
import time
from opensearchpy import helpers

def generate_data(num_docs=100000):
    """제너레이터로 데이터 생성 (메모리 절약)"""
    for i in range(num_docs):
        yield {
            "_index": "large-index",
            "_id": i,
            "_source": {
                "title": f"Document {i}",
                "value": i,
                "timestamp": int(time.time())
            }
        }

def index_large_data():
    s_time = time.time()
    
    # parallel_bulk는 제너레이터를 반환하므로 deque 등으로 소진시켜야 실행됨
    # thread_count: CPU 코어 수 * 2 정도 권장
    success, failed = 0, 0
    
    for success_info, error_info in helpers.parallel_bulk(
        client, 
        generate_data(), 
        thread_count=4,
        chunk_size=2000,      # 한 번에 보낼 문서 수 (문서 크기에 따라 조절)
        queue_size=4,         # 대기열 크기
        raise_on_error=False  # 에러 발생해도 중단하지 않음
    ):
        if success_info:
            success += 1
        else:
            failed += 1
            print(f"Error: {error_info}")

    print(f"Indexed {success} docs in {time.time() - s_time:.2f}s (Failed: {failed})")

# 실행
# index_large_data()
```

### 3. 전체 데이터 조회 (Scan)

10,000건이 넘는 데이터를 모두 가져와야 할 때 (예: 데이터 마이그레이션, 분석).

```python
def fetch_all_docs(index_name):
    # helpers.scan은 내부적으로 Scroll API를 사용하며 자동 페이징 처리
    scan_gen = helpers.scan(
        client,
        query={"query": {"match_all": {}}},
        index=index_name,
        size=1000,  # 배치 사이즈 (메모리에 따라 조절)
        scroll="5m" # 스크롤 유지 시간
    )

    count = 0
    for doc in scan_gen:
        # doc 처리 로직
        # print(doc["_source"]["title"])
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} docs...")
            
    print(f"Total processed: {count}")
```

### 4. 비동기 클라이언트 (Async Client)

FastAPI 등 비동기 프레임워크와 함께 사용할 때 필수적이다.

```bash
pip install opensearch-py[async]
```

```python
import asyncio
from opensearchpy import AsyncOpenSearch

async def main():
    async_client = AsyncOpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
        verify_certs=False
    )

    # 비동기 검색
    response = await async_client.search(
        index="large-index",
        body={"query": {"match_all": {}}},
        size=5
    )
    
    print(f"Hits: {response['hits']['total']['value']}")
    
    # 리소스 정리 필수
    await async_client.close()

# 실행
# asyncio.run(main())
```

### 5. 에러 핸들링 패턴

```python
from opensearchpy import TransportError, ConnectionError, NotFoundError

try:
    client.search(index="non-existent-index", body={})
except NotFoundError:
    print("인덱스를 찾을 수 없습니다.")
except ConnectionError:
    print("OpenSearch 서버에 연결할 수 없습니다.")
except TransportError as e:
    print(f"기타 전송 에러: {e.status_code} - {e.error}")
except Exception as e:
    print(f"예상치 못한 에러: {e}")
```

---

## 100GB 데이터 처리 시 팁

1.  **Chunk Size 조절**: `chunk_size`는 문서 크기에 따라 다르다. 보통 5MB~15MB 정도가 한 번의 요청(Payload) 크기가 되도록 설정한다. 문서당 1KB라면 `chunk_size=5000` 정도가 적당하다.
2.  **Refresh Interval**: 대량 색인 중에는 `refresh_interval`을 `-1`로 설정하여 색인 성능을 높이고, 완료 후 복구한다.
3.  **Source Filtering**: `scan`이나 `search` 시 `_source` 파라미터로 필요한 필드만 가져와서 네트워크 대역폭을 절약한다.

```python
# 필요한 필드만 조회
helpers.scan(
    client,
    query=...,
    _source=["title", "timestamp"] # content 같은 큰 필드 제외
)
```

---

## 참고 자료 (References)

- [opensearch-py Documentation](https://opensearch.org/docs/latest/clients/python-low-level/)
- [Python Bulk Helpers](https://opensearch-project.github.io/opensearch-py/helpers.html)

## 관련 문서

- [OpenSearch 성능 최적화](./performance-optimization.md) - 대용량 처리를 위한 서버 설정
- [OpenSearch 기초](./opensearch-basics.md)
