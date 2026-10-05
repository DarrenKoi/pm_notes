---
tags: [index, taxonomy]
document_type: taxonomy_index
category_major: "에이전트 운영 문서"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 에이전트 운영 문서 — 대·중·소분류 목차

**대분류: 에이전트 운영 문서**. 아래 중분류·소분류에서 문서를 선택한다. [기존 읽기 순서](./README.md)도 함께 사용할 수 있다.

> [!tip] 분류를 읽는 방법
> 대분류는 넓은 분야, 중분류는 기술·업무 영역, 소분류는 구체적인 학습·작업 주제다.
> 문서 유형은 학습·실습·기록·양식 등 용도를 나타내며 주제와 별도로 구분한다.
> 파일 경로는 유지했다. 분류 속성은 기술 검증일·업무 승인·실행 완료를 뜻하지 않는다.

## 분류 요약

| 중분류 | 소분류 수 | 문서 수 |
|---|---|---|
| [작업 운영](#%EC%9E%91%EC%97%85%20%EC%9A%B4%EC%98%81) | 3 | 3 |
| [과거 설계 기록](#%EA%B3%BC%EA%B1%B0%20%EC%84%A4%EA%B3%84%20%EA%B8%B0%EB%A1%9D) | 2 | 4 |
| [주제 안내](#%EC%A3%BC%EC%A0%9C%20%EC%95%88%EB%82%B4) | 1 | 1 |
| [문서 관리](#%EB%AC%B8%EC%84%9C%20%EA%B4%80%EB%A6%AC) | 1 | 1 |

## 작업 운영

### 도메인 문맥

| 문서 | 유형 |
|---|---|
| [주제별 도메인 문서를 읽는 방법](./agents/domain.md) | 운영 지침 |

### 이슈 추적

| 문서 | 유형 |
|---|---|
| [GitHub 이슈 추적 운영 안내](./agents/issue-tracker.md) | 운영 지침 |

### 이슈 분류

| 문서 | 유형 |
|---|---|
| [이슈 분류 라벨과 판단 기준](./agents/triage-labels.md) | 운영 지침 |

## 과거 설계 기록

### AIX 폴더 재편

| 문서 | 유형 |
|---|---|
| [AIX_POC 재편 — Smart Align Agent 프로젝트화 Implementation Plan](./superpowers/plans/2026-06-30-smart-align-agent-reorg.md) | 설계·계획 |
| [AIX_POC 폴더 재편 설계 — Smart Align Agent 프로젝트화](./superpowers/specs/2026-06-30-smart-align-agent-reorg-design.md) | 설계·계획 |

### AI 용어 HTML 리더

| 문서 | 유형 |
|---|---|
| [AI Terms HTML Reader Implementation Plan](./superpowers/plans/2026-07-28-ai-terms-html-reader.md) | 설계·계획 |
| [AI 용어 및 기술 HTML 리더 설계](./superpowers/specs/2026-07-28-ai-terms-html-reader-design.md) | 설계·계획 |

## 주제 안내

### 전체 목차

| 문서 | 유형 |
|---|---|
| [에이전트 운영 안내와 과거 설계 기록](./README.md) | 목차 |

## 문서 관리

### 정리·검증 기록

| 문서 | 유형 |
|---|---|
| [에이전트 문서 정리 기록](./organization-log.md) | 관리 기록 |

## Obsidian에서 찾기

일반 문서의 YAML 속성 `category_major`, `category_middle`, `category_minor`, `note_kind`로 분류를 확인한다. 모두 단일 텍스트 값이다. 기존 tags·aliases·작성일·검토 상태는 유지한다.

검색 패널에서 다음처럼 속성을 조합한다. 대·중·소 값은 위 표의 실제 값을 사용한다.

```text
path:"docs/" [category_major:"에이전트 운영 문서"]
path:"docs/" [note_kind:"학습"]
```

목차·검토·관리 기록은 학습 문서와 유형을 구분했다. 불변 원문과 작업 지침은 속성을 추가하지 않고 이 목차에서 분류한다. 따라서 속성 검색만으로 원본 보존 문서 전체를 찾을 수는 없다.

분류 대상은 기존 Markdown 9개이며, 그중 원본 보존 0개다. 이 목차 자체는 대상 수에 포함하지 않는다.

분류일: 2026-10-05. 링크·문법과 실제 읽기 화면의 확인 범위는 [분류 검증 기록](./classification-review.md)에 남긴다.
