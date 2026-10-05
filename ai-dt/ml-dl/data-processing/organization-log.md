---
tags: [ml, data-processing, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# 데이터 처리 문서 정리 기록

## 개별 검토

원래 4개 Markdown/2,056행을 읽고 검토했다. 파일명·절·포맷별 고유 예제를 유지했다. 공통 적용 조건을 대표 안내에 모았고 문서를 이동하거나 실행 파일을 수정하지 않았다. 원래 작성일 2026-02-14는 보존, 검토일 2026-10-04는 별도 추가.

| 문서 | 수정·검토 결과 |
|---|---|
| data-loading-formats.md | CSV 스키마·포맷 속도·API 계약 일반화 수정. 후보 decode를 자동 판별과 구분. Excel header, Polars API, 문자열/혼합 객체/빈 행 처리와 결측 보존 수정. EDA 상대 링크 복구. |
| eda-recipes.md | 일괄 경고 억제 제거, 스타일 뒤 font 유지. 문자열 선택·빈 plot·crosstab 열·nullable 이상치 mask·z-score 중복 인덱스/상수 입력 수정. 존재하지 않는 전처리 링크를 실제 Pipeline 문서로 연결. |
| feature-engineering.md | 모델 입력·스케일·TF-IDF 일반화 수정. 30세 구간 경계, clone-friendly 날짜 변환기와 출력 피처 순서 수정. 전체 df 데모와 평가 fold 적용 차이를 안내. |
| data-pipeline-template.md | 자동 누수/production/재현 보장 단정 수정. 결측 입력과 AutoPreprocessor 스키마·미지원 타입·clone·피처 이름·set_output·전부 결측 차원 보존 계약 수정. 로그 domain·실제 Python 버전·저장 신뢰 조건 안내. |

## Claude 협의

HERDR_ENV=1을 확인해 `herdr pane current --current`를 재시도했으나 pane_not_found였다. 작업 전용 Claude pane에 연결하지 못했고 다른 pane을 제어하지 않았다. 모델별 인코딩 최적화·대표 예제 재설계·완전 중복 통합은 협의 필요 결정으로 보류했다. 공식 API와 재현되는 실패의 수정, 고유 예제 보존과 적용 조건 설명은 진행했다.

## 검증과 미확인

1. 원래 데이터 처리 4개/2,056행과 루트 README를 대조했다. 원래 절 제목 97개·코드/텍스트 fence 72개를 보존했고 20개 fence의 수정은 실패 재현과 API·입력 계약 수정에 대응한다. 원래 작성일과 ai-dt의 비 Markdown 18개는 그대로다. 고유 예제의 제거·파일 이동은 없다.
2. 공식 자료 15건을 적용 조건 문서에 연결하고 확인일을 기록했다. Python 블록 70개의 AST가 통과했다. 격리된 Python 3.14.2 환경에서 파일 roundtrip·nullable/빈 입력·스키마·clone·피처 순서·미등록 category·중복 인덱스·plot 생성·전체 피처 예제 순차 실행·Pipeline CV/학습/평가/저장 등 18종 검증이 통과했다. 실제 예제의 OpenML 입력만 합성 fixture로 대체했으므로 실제 다운로드 성공이나 성능 증거는 아니다. pandas 3.0.3, scikit-learn 1.9.1, Polars 1.44.2 등 실행 판본은 적용 조건 표에 기록했다. 빈 EDA의 nullable 평균 오류를 실행으로 추가 발견하고 수정·재검증했다. Matplotlib 한글 glyph 경고는 실제 발생했으며 숨기지 않았다.
3. 현재 데이터 처리 문서 7개의 상대 링크·앵커·메타데이터 검사에서 새 오류는 0개였다. 원래 상위 참조 4회는 유지했다. 루트 README/기록을 포함한 9개 YAML의 중복 키가 없고 Obsidian 1.13.7 CLI properties가 검토일을 반환했다. 첫 CLI 응답 timeout 뒤 재시도 9개 모두 통과했다. 실제 pm_notes vault 읽기 화면에서 README → 공통 적용 조건 이동과 한글 본문·속성 렌더링을 확인했다.

외부 OpenML/Titanic 다운로드·실제 API·missingno·Matplotlib plot의 실제 한글 글리프·운영 데이터·최적 인코딩/성능은 미확인이다. Obsidian 본문 화면 확인은 그래프 한글 표시 검증과 다르다. 임시 Python 환경과 합성 데이터 검증은 실제 서비스 배포 증명이 아니다.
