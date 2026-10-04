---
title: 에이전트 운영 안내와 과거 설계 기록
tags: [agents, index]
document_type: index
reviewed_on: 2026-10-04
---

# 에이전트 운영 안내와 과거 설계 기록

이 폴더는 에이전트 작업의 운영 규칙과 과거 설계·구현 계획을 보관한다. 학습 순서와 당시 실행 계획을 구분해 읽는다.

## 현재 운영 안내 읽기 순서

1. [도메인 문서 읽기](agents/domain.md): 작업 주제의 정의·결정 범위를 정한다.
2. [이슈 추적](agents/issue-tracker.md): 범위·재현·완료 기준을 읽고 기록한다.
3. [분류 라벨](agents/triage-labels.md): 정보 준비 상태와 다음 담당을 판단한다.

운영 안내는 저장소 관례다. 원격 이슈·라벨의 현재 존재와 접근 권한은 미확인이다.

## 과거 설계·계획 기록

설계는 목적·범위·원칙을 설명하고 계획은 그 설계를 실행하는 단계와 예제를 보관한다. 역할이 달라 통합하지 않는다. 미완료 체크박스만으로 현재 작업 상태를 판단하지 않는다.

| 작성일 | 설계 | 구현 계획 |
|---|---|---|
| 2026-06-30 | [Smart Align 재편 설계](superpowers/specs/2026-06-30-smart-align-agent-reorg-design.md) | [당시 재편 계획](superpowers/plans/2026-06-30-smart-align-agent-reorg.md) |
| 2026-07-28 | [AI 용어 HTML 리더 설계](superpowers/specs/2026-07-28-ai-terms-html-reader-design.md) | [당시 구현 계획](superpowers/plans/2026-07-28-ai-terms-html-reader.md) |

과거 기록의 명령은 실행하지 않는다. 영문 계획 원문은 예제·계약·작성 맥락 보존을 위해 유지했으며 한국어 목차와 상태 안내를 붙였다.

## 검증과 보류

[정리 기록](organization-log.md)에 문서별 결과, 공식 출처, 로컬 확인 범위와 Claude 협의 상태를 남겼다.
