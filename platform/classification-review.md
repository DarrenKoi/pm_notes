---
tags: [taxonomy, review]
document_type: classification_review
category_major: "Agent·Skill 플랫폼"
category_middle: "문서 관리"
category_minor: "분류·호환성 검증"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# Agent·Skill 플랫폼 — 분류·Obsidian 검증 기록

## 적용한 구조

2026-10-05 사용자 요청에 따라 최상위 폴더와 기존 경로를 유지하고 주제별 대·중·소분류를 적용했다. [분류 목차](./taxonomy-index.md)는 주제별 탐색, [기존 목차](./README.md)는 읽기 순서를 제공한다.

- 기존 Markdown 대상: 45개.
- 분류 속성 추가: 45개. `category_major`·`category_middle`·`category_minor`·`note_kind`·`classified_on`을 추가했다.
- 원본 보존: 0개. 지침·불변 출처·과거 작업 원문·로컬 전용 자료는 각 보호 규칙을 따른다.
- 신규 안내: `taxonomy-index.md`와 이 검증 기록. 두 안내는 기존 대상 수에 포함하지 않는다.

기존 작성일·검토 상태와 본문은 보존하며 읽기 입구와 확인된 링크만 보정한다. 분류일은 기술 설명의 재검증일이나 업무 실측일이 아니다. 코드·첨부파일·앱 설정·커밋은 이 작업의 편집 대상 밖이다.

## 검증 기준

YAML 파싱·중복 키, 문서 유형, 태그·기본 속성 타입, 코드 fence, Markdown 링크·이미지·wikilink 대상, 제목 앵커 후보를 정적으로 확인한다. 분류 목차의 링크는 같은 주제 폴더 안의 실파일만 가리킨다. 기존 본문·메타데이터와 보호 대상의 변경 여부를 작업 전 복사본과 대조한다.

중첩 객체 대신 단일 텍스트 속성을 사용하고 기존 Markdown 상대 링크를 유지한다. [Obsidian 속성](https://help.obsidian.md/properties)과 [내부 링크](https://help.obsidian.md/links)의 공식 설명을 2026-10-05 확인했다. 추가 플러그인이나 설정 변경은 필요하지 않다.

## 실행 결과

- 정적 검사: 현재 Markdown 47개, 로컬 링크·이미지·wikilink·앵커 검사 251건. YAML 파싱/중복 키·태그/기본 속성 타입·코드 fence 오류 0건.
- 일반 문서와 신규 안내의 미해결 링크·앵커 후보: 0건. 기존 원문은 보호 여부를 따로 구분했다.
- 보존 대조: 기존 속성값 유지, 보호 원본 바이트 동일, 일반 본문의 변경은 주제 README의 분류 입구와 명시한 링크 보정으로 한정됨을 확인했다.
- 앱 확인: Obsidian 1.13.7의 `pm_notes`에서 이 폴더의 분류 목차를 읽기 모드로 열고 분류 속성·요약표·계층 제목·링크 표시를 확인했다.

## 검증 한계

CLI 실행 파일은 있으나 `obsidian help`·`obsidian version`은 앱 연결 실패를 반환했다. 앱 자체는 연결 가능해 GUI로 확인했다. CLI로 vault 전체의 unresolved 링크 목록이나 메타데이터 캐시를 조회한 결과는 아니다.

읽기 화면 검증은 이 폴더의 분류 목차와 명시한 대표 문서/검색에 한정한다. 기존 문서 전체의 모든 화면·Mermaid·수식·이미지 배치를 하나씩 확인한 것은 아니다. 외부 URL의 현재 응답, 회사 환경·모델·GPU·원격 서비스·실적은 이번 분류 작업에서 재검증하지 않았다. 예제·업무 지시를 실행하지 않았으며 커밋·푸시는 하지 않았다.
