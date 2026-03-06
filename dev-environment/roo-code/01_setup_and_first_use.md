# 01. 설치부터 첫 작업까지 (실전 시작)

## 1) 설치

- VS Code 확장 검색: `Roo Code`
- 권장 설치 경로: VS Code Marketplace
- 설치 후 Activity Bar에서 Roo 아이콘으로 패널 열기

체크:
- VS Code 버전이 충분히 최신인지 확인
- 설치 문제 시 VS Code 재시작 + Output 패널(Roo) 확인

## 2) 첫 LLM Provider 연결

권장 시작점:
- Provider: 사내 OpenAI-compatible API
- 시작 모델: `Kimi-K2.5` 또는 `GLM-4.7`
- Base URL 예시: `https://<company-llm-gateway>/v1`

핵심:
- Roo는 모델 자체가 아니라 "모델을 쓰는 에이전트 확장"이므로 Provider 설정이 필수
- 첫 세팅은 안정적인 모델 1개로 시작하고, 이후 프로필로 분기
- 사내망에서만 동작하도록 endpoint/API key를 내부 값으로 고정

## 3) Profiles(프로필)로 작업 성격 분리

프로필에서 분리할 것:
- Base URL/API Key
- Model
- Temperature
- Rate limit
- Mode별 연결

추천 프로필 예시:
- `kimi-build`: Kimi-K2.5 기반 구현 전용
- `glm-review`: GLM-4.7 기반 리뷰/요약 전용
- `doc-writer`: 문서/설계 전용 (Kimi/GLM 중 더 안정적인 모델 사용)

운영 팁:
- 모드별 마지막 프로필이 기억되므로, 모드-프로필을 세트로 설계
- 태스크 재개 시 원래 프로필이 유지되어 결과 일관성이 높아짐

## 4) 모드 사용 기본

내장 모드:
- `Code`: 구현/수정 중심
- `Ask`: 설명/리서치 중심(편집/명령 제한)
- `Architect`: 설계/계획 중심
- `Debug`: 진단/원인추적 중심
- `Orchestrator`: 하위 작업 위임 중심

전환 방법:
- 입력창 좌측 드롭다운
- `/code`, `/ask`, `/architect`, `/debug`, `/orchestrator`
- 단축키 순환 (`Cmd/Ctrl + .`)

## 5) 첫 작업 프롬프트 템플릿

```text
현재 저장소에서 로그인 에러를 고쳐줘.
조건:
1) 원인 가설 2~3개를 먼저 제시
2) 가장 가능성 높은 원인부터 검증
3) 수정 전/후 동작 차이를 요약
4) 테스트 명령과 결과를 함께 제시
```

잘 되는 요청 패턴:
- 목표 + 제약 + 완료 조건을 함께 말하기
- "바로 코드 수정" 대신 "가설→검증→수정" 흐름 지시

## 6) 바로 써먹는 시작 루틴

1. `Ask` 모드에서 문제 맥락 설명 받기
2. `Architect` 모드에서 작업계획/체크리스트 생성
3. `Code` 모드에서 구현
4. `Debug` 모드에서 실패 케이스 재현/해결
5. 필요하면 `Orchestrator`로 병렬 하위 작업 분리
