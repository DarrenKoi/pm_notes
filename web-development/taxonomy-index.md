---
tags: [index, taxonomy]
document_type: taxonomy_index
category_major: "웹 개발"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 웹 개발 — 대·중·소분류 목차

**대분류: 웹 개발**. 아래 중분류·소분류에서 문서를 선택한다. [기존 읽기 순서](./README.md)도 함께 사용할 수 있다.

> [!tip] 분류를 읽는 방법
> 대분류는 넓은 분야, 중분류는 기술·업무 영역, 소분류는 구체적인 학습·작업 주제다.
> 문서 유형은 학습·실습·기록·양식 등 용도를 나타내며 주제와 별도로 구분한다.
> 파일 경로는 유지했다. 분류 속성은 기술 검증일·업무 승인·실행 완료를 뜻하지 않는다.

## 분류 요약

| 중분류 | 소분류 수 | 문서 수 |
|---|---|---|
| [백엔드 개발](#%EB%B0%B1%EC%97%94%EB%93%9C%20%EA%B0%9C%EB%B0%9C) | 2 | 4 |
| [프론트엔드·런타임](#%ED%94%84%EB%A1%A0%ED%8A%B8%EC%97%94%EB%93%9C%C2%B7%EB%9F%B0%ED%83%80%EC%9E%84) | 3 | 9 |
| [품질 검증](#%ED%92%88%EC%A7%88%20%EA%B2%80%EC%A6%9D) | 1 | 4 |
| [주제 안내](#%EC%A3%BC%EC%A0%9C%20%EC%95%88%EB%82%B4) | 1 | 1 |
| [문서 관리](#%EB%AC%B8%EC%84%9C%20%EA%B4%80%EB%A6%AC) | 1 | 1 |

## 백엔드 개발

### Python 환경·패키지

| 문서 | 유형 |
|---|---|
| [Python 환경과 Redis 읽기 안내](./python/README.md) | 목차 |
| [pip에서 uv로 마이그레이션 가이드](./python/pip-to-uv-migration.md) | 학습 |
| [uv - 차세대 Python 패키지 매니저](./python/uv-package-manager.md) | 학습 |

### Redis 캐시

| 문서 | 유형 |
|---|---|
| [Python에서 Redis 사용하기](./python/redis-python.md) | 학습 |

## 프론트엔드·런타임

### TypeScript·빌드 도구

| 문서 | 유형 |
|---|---|
| [TypeScript 웹 개발 학습 로드맵](./typescript/README.md) | 목차 |
| [TypeScript 프로젝트 설정 가이드](./typescript/tsconfig-setup.md) | 학습 |
| [Vite 기초 가이드](./typescript/vite-basics.md) | 학습 |

### Bun·API 실습

| 문서 | 유형 |
|---|---|
| [Bun 시작 가이드](./typescript/bun-basics.md) | 학습 |
| [Bun 실행 예제 목차](./typescript/bun/README.md) | 목차 |
| [Bun + TypeScript Task API 예제](./typescript/bun/example-task-api/README.md) | 목차 |

### Vue·상태 관리

| 문서 | 유형 |
|---|---|
| [Vue 학습 목차](./typescript/vue/README.md) | 목차 |
| [Props/Emit vs Pinia: Vue 상태 관리 패턴 선택 가이드](./typescript/vue/props-emit-vs-pinia.md) | 학습 |
| [Vue 3 + TypeScript 시작하기](./typescript/vue/vue3-with-typescript.md) | 학습 |

## 품질 검증

### 소프트웨어 테스트

| 문서 | 유형 |
|---|---|
| [테스트 학습 목차](./testing/README.md) | 목차 |
| [End-to-End Testing (E2E 테스트) 기초](./testing/e2e-testing-basics.md) | 학습 |
| [테스트 프레임워크 비교 및 사용법](./testing/testing-frameworks.md) | 학습 |
| [Unit Testing (단위 테스트) 기초](./testing/unit-testing-basics.md) | 학습 |

## 주제 안내

### 전체 목차

| 문서 | 유형 |
|---|---|
| [웹 개발 지식 목차](./README.md) | 목차 |

## 문서 관리

### 정리·검증 기록

| 문서 | 유형 |
|---|---|
| [웹 개발 문서 정리 기록](./organization-log.md) | 관리 기록 |

## Obsidian에서 찾기

일반 문서의 YAML 속성 `category_major`, `category_middle`, `category_minor`, `note_kind`로 분류를 확인한다. 모두 단일 텍스트 값이다. 기존 tags·aliases·작성일·검토 상태는 유지한다.

검색 패널에서 다음처럼 속성을 조합한다. 대·중·소 값은 위 표의 실제 값을 사용한다.

```text
path:"web-development/" [category_major:"웹 개발"]
path:"web-development/" [note_kind:"학습"]
```

목차·검토·관리 기록은 학습 문서와 유형을 구분했다. 불변 원문과 작업 지침은 속성을 추가하지 않고 이 목차에서 분류한다. 따라서 속성 검색만으로 원본 보존 문서 전체를 찾을 수는 없다.

분류 대상은 기존 Markdown 19개이며, 그중 원본 보존 0개다. 이 목차 자체는 대상 수에 포함하지 않는다.

분류일: 2026-10-05. 링크·문법과 실제 읽기 화면의 확인 범위는 [분류 검증 기록](./classification-review.md)에 남긴다.
