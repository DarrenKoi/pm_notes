---
tags: [redis, cache, queue, python, ttl]
level: intermediate
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# Python에서 Redis 사용하기

> Redis를 Python에서 활용하여 캐싱, TTL 관리, 큐 처리를 구현하는 방법 정리

## 왜 필요한가? (Why)

- 웹 애플리케이션에서 **반복적인 DB 조회나 연산 결과를 캐싱**하여 응답 속도를 크게 개선
- TTL(Time To Live)로 **캐시 수명**을 제한; DB 변경의 즉시 반영·강한 일관성은 별도 설계
- Redis의 List/Stream 자료구조로 **비동기 작업 큐**를 구현하여 무거운 작업을 백그라운드 처리
- 세션 관리, Rate Limiting 등 다양한 실무 패턴에 활용

## 핵심 개념 (What)

### 1. Redis 데이터 타입 요약

| 타입 | 용도 | 주요 명령어 |
|------|------|-------------|
| String | 단순 캐시, 카운터 | `SET`, `GET`, `INCR` |
| Hash | 객체/딕셔너리 저장 | `HSET`, `HGET`, `HGETALL` |
| List | 큐, 스택 | `LPUSH`, `RPOP`, `BRPOP` |
| Set | 고유값 집합 | `SADD`, `SMEMBERS` |
| Sorted Set | 랭킹, 스코어 기반 정렬 | `ZADD`, `ZRANGE` |
| Stream | 이벤트 스트리밍, 고급 큐 | `XADD`, `XREAD`, `XREADGROUP` |

### 2. TTL (Time To Live)

키에 만료 시간을 설정하면 해당 시간이 지난 후 Redis가 자동으로 키를 삭제한다.

- `EXPIRE key seconds` — 초 단위 만료
- `PEXPIRE key milliseconds` — 밀리초 단위 만료
- `EXPIREAT key timestamp` — Unix timestamp 기준 만료
- `TTL key` — 남은 시간 확인 (-1: 만료 없음, -2: 키 없음)
- `PERSIST key` — 만료 제거 (영구 키로 전환)

### 3. 캐시 전략

- **Cache-Aside (Lazy Loading)**: 요청 시 캐시 확인 → 없으면 DB 조회 후 캐시 저장
- **Write-Through**: 데이터 쓸 때 DB와 캐시 동시 업데이트
- **Write-Behind**: 캐시에 먼저 쓰고, 비동기로 DB에 반영

## 어떻게 사용하는가? (How)

### 기본 연결

```python
import redis

# 기본 연결
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Connection Pool 사용 (권장 - 연결 재사용)
pool = redis.ConnectionPool(host="localhost", port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)
```

> `decode_responses=True`를 설정하면 bytes 대신 str로 반환되어 편리하다.

### TTL 관리

```python
# 기본 SET + TTL
r.set("user:1001:name", "Daeyoung", ex=3600)  # 1시간 후 만료
r.set("temp_token", "abc123", px=5000)          # 5초(밀리초) 후 만료

# 기존 키에 TTL 추가/변경
r.expire("user:1001:name", 7200)  # 2시간으로 변경
r.pexpire("user:1001:name", 7200000)  # 밀리초 단위

# TTL 확인
remaining = r.ttl("user:1001:name")   # 남은 초
print(f"남은 시간: {remaining}초")

# TTL 제거 (영구 키로)
r.persist("user:1001:name")

# 특정 시점에 만료 (Unix timestamp)
import time
expire_at = int(time.time()) + 86400  # 24시간 후
r.expireat("user:1001:name", expire_at)

# SET with NX/XX 옵션
r.set("lock:resource", "owner1", ex=30, nx=True)  # 키가 없을 때만 SET (분산 락)
r.set("counter", "10", ex=60, xx=True)             # 키가 있을 때만 SET (업데이트)
```

### Cache-Aside 패턴 구현

```python
import json
from typing import Any

def get_user(user_id: int) -> dict:
    """Cache-Aside 패턴: 캐시 먼저 확인, 없으면 DB 조회 후 캐시 저장"""
    cache_key = f"user:{user_id}"

    # 1. 캐시 확인
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. DB 조회 (캐시 미스)
    user = db.query_user(user_id)  # 실제 DB 조회 함수

    # 3. 캐시 저장 (TTL 30분)
    r.set(cache_key, json.dumps(user), ex=1800)

    return user


def update_user(user_id: int, data: dict) -> None:
    """데이터 변경 시 캐시 무효화"""
    db.update_user(user_id, data)
    r.delete(f"user:{user_id}")  # 캐시 삭제 → 다음 조회 시 갱신됨
```

### Hash를 활용한 객체 캐싱

```python
# 사용자 정보를 Hash로 저장 (필드별 개별 접근 가능)
r.hset("user:1001", mapping={
    "name": "Daeyoung",
    "email": "dy@example.com",
    "role": "engineer"
})
r.expire("user:1001", 3600)

# 특정 필드만 조회
name = r.hget("user:1001", "name")

# 전체 조회
user = r.hgetall("user:1001")
# {'name': 'Daeyoung', 'email': 'dy@example.com', 'role': 'engineer'}

# 특정 필드만 업데이트 (다른 필드는 유지)
r.hset("user:1001", "role", "senior_engineer")
```

### Pipeline으로 성능 최적화

```python
# Pipeline: 여러 명령을 한 번에 보내서 네트워크 왕복 줄이기
pipe = r.pipeline(transaction=False)
for i in range(100):
    pipe.set(f"key:{i}", f"value:{i}", ex=600)
pipe.execute()  # 100개 명령을 한 번에 전송

# Transaction (atomic 실행)
pipe = r.pipeline(transaction=True)
pipe.multi()
pipe.set("balance:A", 900)
pipe.set("balance:B", 1100)
pipe.execute()  # 다른 클라이언트 명령의 끼어들기 없이 실행; 실행 중 오류의 rollback은 없음
```

### 큐 (Queue) 구현

#### 방법 1: List 기반 단순 큐

```python
import json
import time

# --- Producer (작업 등록) ---
def enqueue_task(queue_name: str, task_data: dict):
    """큐에 작업 추가 (LPUSH → 왼쪽에 삽입)"""
    r.lpush(queue_name, json.dumps(task_data))

enqueue_task("task_queue", {"type": "send_email", "to": "user@example.com"})
enqueue_task("task_queue", {"type": "generate_report", "id": 42})


# --- Consumer (작업 처리) ---
def process_queue(queue_name: str):
    """큐에서 작업 꺼내서 처리 (BRPOP → 오른쪽에서 꺼냄, 블로킹)"""
    while True:
        # BRPOP: 큐가 비어있으면 최대 timeout초 대기 (0이면 무한 대기)
        result = r.brpop(queue_name, timeout=5)
        if result is None:
            continue  # timeout, 다시 대기

        _, raw_data = result
        task = json.loads(raw_data)

        print(f"처리 중: {task}")
        handle_task(task)

def handle_task(task: dict):
    if task["type"] == "send_email":
        print(f"이메일 발송: {task['to']}")
    elif task["type"] == "generate_report":
        print(f"리포트 생성: ID {task['id']}")
```

> `BRPOP`은 블로킹 방식으로, 큐가 비어있으면 대기한다. polling보다 효율적이다.

#### 방법 2: List + 처리 중 목록 (복구 절차가 별도로 필요)

```python
def reliable_dequeue(source: str, processing: str, timeout: int = 5):
    """
    BLMOVE: source 오른쪽에서 꺼내 processing 왼쪽으로 원자적 이동
    → 처리 실패 시 processing에서 다시 source로 복구 가능
    """
    raw = r.blmove(source, processing, timeout=timeout, src="RIGHT", dest="LEFT")
    if raw is None:
        return None
    return raw, json.loads(raw)

def complete_task(processing: str, raw_data: str):
    """처리 완료 시 processing 리스트에서 제거"""
    r.lrem(processing, 1, raw_data)

# 사용 예
item = reliable_dequeue("task_queue", "task_processing")
if item is not None:
    raw, task = item
    try:
        handle_task(task)
        complete_task("task_processing", raw)
    except Exception:
        # 실패 시 processing에 남아있으므로 나중에 재처리 가능
        pass
```

#### 방법 3: Redis Stream (고급 큐 — 권장)

```python
# --- Producer ---
def publish_event(stream: str, data: dict):
    """Stream에 이벤트 추가. ID는 자동 생성('*')"""
    r.xadd(stream, data, maxlen=10000)  # maxlen으로 스트림 크기 제한

publish_event("events", {"type": "order_created", "order_id": "1001"})


# --- Consumer Group 설정 ---
# Consumer Group: 여러 워커가 메시지를 나눠서 처리
try:
    r.xgroup_create("events", "workers", id="0", mkstream=True)
except redis.exceptions.ResponseError as exc:
    if not str(exc).startswith("BUSYGROUP"):
        raise  # 실제 권한·타입·서버 오류는 숨기지 않음


# --- Consumer (워커) ---
def consume_stream(stream: str, group: str, consumer: str):
    """Consumer Group으로 스트림 읽기"""
    while True:
        # '>' : 아직 이 그룹에서 읽지 않은 새 메시지만 가져옴
        entries = r.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=10,       # 한 번에 최대 10개
            block=5000      # 5초 대기
        )

        if not entries:
            continue

        for stream_name, messages in entries:
            for msg_id, data in messages:
                print(f"[{consumer}] 처리: {data}")
                handle_event(data)
                # 처리 완료 확인(ACK)
                r.xack(stream, group, msg_id)

# 여러 워커 실행 (각각 다른 프로세스에서)
# consume_stream("events", "workers", "worker-1")
# consume_stream("events", "workers", "worker-2")
```

> **Stream vs List 큐 비교**:
> - List: 여러 워커가 같은 목록을 소비할 수 있다. BRPOP은 꺼낸 뒤 장애가 나면 작업이 유실될 수 있다. BLMOVE 처리 목록도 복구 워커·작업 ID·멱등 처리가 없으면 자동 재처리되지 않는다. 원문을 보존하여 LREM에 쓰지만 같은 본문이 중복되면 정확한 작업 식별이 안 되므로 고유 ID가 필요하다.
> - Stream: 그룹에 전달된 메시지는 ACK 전 PEL에 남는다. `>`는 새 메시지만 읽으므로 실패한 pending은 XAUTOCLAIM/재처리 루프로 회수해야 한다. ACK 자체가 exactly-once나 외부 부수 효과 성공을 보증하지 않는다. 위 예제는 새 메시지 처리 흐름이며 handle_event 구현·복구·trimming 정책은 생략했다.

### 실무 패턴: Rate Limiting

```python
def is_rate_limited(user_id: str, limit: int = 100, window: int = 60) -> bool:
    """첫 요청부터 TTL 동안 세는 고정 윈도우. Sliding Window가 아님."""
    if limit < 1 or window < 1:
        raise ValueError("limit and window must be positive")
    key = f"rate:{user_id}"
    script = """
    local current = redis.call("INCR", KEYS[1])
    if current == 1 then
        redis.call("EXPIRE", KEYS[1], ARGV[1])
    end
    return current
    """
    current = int(r.eval(script, 1, key, window))
    return current > limit
```

### 실무 패턴: 분산 락 (Distributed Lock)

```python
import uuid

def acquire_lock(resource: str, timeout: int = 10) -> str | None:
    """분산 락 획득. 성공 시 lock_id 반환"""
    lock_id = str(uuid.uuid4())
    acquired = r.set(f"lock:{resource}", lock_id, ex=timeout, nx=True)
    return lock_id if acquired else None

def release_lock(resource: str, lock_id: str) -> bool:
    """자신이 획득한 락만 해제 (Lua script로 atomic하게)"""
    script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    result = r.eval(script, 1, f"lock:{resource}", lock_id)
    return bool(result)

# 사용
lock_id = acquire_lock("critical_section")
if lock_id:
    try:
        # 크리티컬 섹션 작업
        pass
    finally:
        release_lock("critical_section", lock_id)
```

## 주의사항

- **메모리 관리**: TTL 없는 키가 쌓이면 메모리 부족 발생. `maxmemory-policy` 설정 확인 (보통 `allkeys-lru`)
- **직렬화**: JSON 또는 필요한 타입을 명시한 형식을 사용한다. pickle 역직렬화는 신뢰할 수 없는 데이터를 실행할 수 있으므로 캐시 데이터라고 무조건 안전하게 취급하지 않는다.
- **Connection Pool**: Redis 클라이언트는 기본 풀을 사용한다. 매 요청마다 새 클라이언트를 만들기보다 수명을 관리해 재사용한다.
- **Key 네이밍**: `서비스:엔티티:ID:필드` 패턴 권장 (예: `myapp:user:1001:profile`)

## 참고 자료 (References)

- [redis-py 공식 문서](https://redis-py.readthedocs.io/)
- [Redis 공식 Commands](https://redis.io/commands/)
- [Redis University](https://university.redis.io/)

## 관련 문서

- [단위 테스트 기초](../testing/unit-testing-basics.md): 외부 Redis 의존성과 순수 로직의 검증 경계

## 적용 조건과 현재 검토

확인일 **2026-10-04**. 서버·redis-py 설치 버전과 실제 서비스 연결은 미확인이다. 위 코드는 독립 완성 앱이 아니라 연결 `r`·DB `db`·handler를 조합하는 학습 조각이다. Cache-Aside는 동시 read/write에서 오래된 값을 다시 채울 수 있어 TTL만으로 해결되지 않는다. JSON 예제와 Hash 예제는 같은 `user:1001` 키를 서로 다른 타입에 쓰므로 각자 분리된 키/임시 DB에서 실행한다. 큐·락용 Redis에 캐시의 allkeys-lru 정책을 그대로 적용하면 중요한 키도 축출될 수 있다. 락은 단일 인스턴스·TTL 안에서 작업 완료를 전제로 하며 작업이 TTL을 넘거나 failover가 나면 배타성을 그대로 보장하지 않는다.

- [redis-py pipeline](https://redis.readthedocs.io/en/stable/examples/pipeline_examples.html): 기본 transaction=True; 단순 배치는 False로 구분. 열람 페이지는 redis-py 8.1.0.
- [Redis transaction](https://redis.io/docs/latest/develop/using-commands/transactions/): 명령 사이 끼어들기는 막지만 실행 오류 rollback은 없다.
- [BRPOPLPUSH](https://redis.io/docs/latest/commands/brpoplpush/)와 [BLMOVE](https://redis.io/docs/latest/commands/blmove/): Redis 6.2부터 기존 명령 deprecated; RIGHT→LEFT 대체.
- [XREADGROUP](https://redis.io/docs/latest/commands/xreadgroup/)·[XAUTOCLAIM](https://redis.io/docs/latest/commands/xautoclaim/): pending 복구는 별도이며 XAUTOCLAIM은 6.2 이상 조건.
- [INCR rate-limit 패턴](https://redis.io/docs/latest/commands/incr/): INCR 뒤 EXPIRE 누락 가능성을 Lua로 제거.
- [락의 유효 시간](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/): 소유 토큰 해제는 다른 소유자의 삭제를 막으며 긴 작업까지 자동 보호하지 않는다.
