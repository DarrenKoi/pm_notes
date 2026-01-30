---
tags: [redis, cache, queue, python, ttl]
level: intermediate
last_updated: 2026-01-31
status: in-progress
---

# Redis with Python

> Redis를 Python에서 활용하여 캐싱, TTL 관리, 큐 처리를 구현하는 방법 정리

## 왜 필요한가? (Why)

- 웹 애플리케이션에서 **반복적인 DB 조회나 연산 결과를 캐싱**하여 응답 속도를 크게 개선
- TTL(Time To Live)로 **캐시 자동 만료**를 관리하여 데이터 정합성 유지
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
pipe = r.pipeline()
for i in range(100):
    pipe.set(f"key:{i}", f"value:{i}", ex=600)
pipe.execute()  # 100개 명령을 한 번에 전송

# Transaction (atomic 실행)
pipe = r.pipeline(transaction=True)
pipe.multi()
pipe.set("balance:A", 900)
pipe.set("balance:B", 1100)
pipe.execute()  # 둘 다 성공하거나 둘 다 실패
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

#### 방법 2: List + 처리 중 보장 (Reliable Queue)

```python
def reliable_dequeue(source: str, processing: str, timeout: int = 5):
    """
    BRPOPLPUSH: source에서 꺼내서 processing 리스트로 이동
    → 처리 실패 시 processing에서 다시 source로 복구 가능
    """
    raw = r.brpoplpush(source, processing, timeout=timeout)
    if raw is None:
        return None
    return json.loads(raw)

def complete_task(processing: str, raw_data: str):
    """처리 완료 시 processing 리스트에서 제거"""
    r.lrem(processing, 1, raw_data)

# 사용 예
task = reliable_dequeue("task_queue", "task_processing")
if task:
    try:
        handle_task(task)
        complete_task("task_processing", json.dumps(task))
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
except redis.exceptions.ResponseError:
    pass  # 이미 존재하면 무시


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
> - List: 단순, 가볍다. 1:1 소비.
> - Stream: Consumer Group으로 여러 워커가 분산 처리. ACK로 처리 보장. 메시지 이력 유지.

### 실무 패턴: Rate Limiting

```python
def is_rate_limited(user_id: str, limit: int = 100, window: int = 60) -> bool:
    """Sliding Window Rate Limiter (분당 100회 제한)"""
    key = f"rate:{user_id}"
    current = r.incr(key)

    if current == 1:
        r.expire(key, window)  # 첫 요청 시 윈도우 시작

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
- **직렬화**: `json.dumps/loads`가 일반적이지만, 큰 데이터는 `msgpack`이나 `pickle` 고려
- **Connection Pool**: 매 요청마다 새 연결 생성하지 말 것. Pool 사용 필수
- **Key 네이밍**: `서비스:엔티티:ID:필드` 패턴 권장 (예: `myapp:user:1001:profile`)

## 참고 자료 (References)

- [redis-py 공식 문서](https://redis-py.readthedocs.io/)
- [Redis 공식 Commands](https://redis.io/commands/)
- [Redis University](https://university.redis.io/)

## 관련 문서

- [FastAPI 관련 학습](./fastapi/) — FastAPI + Redis 캐시 연동
