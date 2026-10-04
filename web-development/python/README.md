---
tags: [web-development, study-index]
aliases: [Python 환경과 Redis 읽기 안내]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# Python 환경과 Redis 읽기 안내

1. [uv 개요](./uv-package-manager.md): 프로젝트 선언·lock·환경·실행의 역할을 구분한다.
2. [pip → uv 이전](./pip-to-uv-migration.md): 기존 핀·빌드·환경·CI를 보존하며 전환한다. 개요의 명령 목록과 용도가 다르다.
3. [Redis 사용](./redis-python.md): 캐시 수명·큐 복구·트랜잭션·rate limiter·락의 적용 조건을 배운다.

FastAPI 전용 학습 문서는 아직 없으며 Redis 예제의 db·handler는 직접 준비해야 한다. flask/job-scheduler에서 발견한 IDE/.nuxt 생성 자료는 실행 가능한 원본 앱으로 간주하지 않는다. 이 목차는 새 앱을 생성하거나 기존 코드를 이동하지 않았다.
