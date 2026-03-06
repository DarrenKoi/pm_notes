# 00. 사내 OpenAI-Compatible LLM 연결 가이드

이 문서는 Roo Code를 사내 전용 API 게이트웨이로만 사용하는 설정 기준이다.

대상 모델:
- `Kimi-K2.5`
- `GLM-4.7`

## 1) 전제 조건

- 사내망에서 접근 가능한 OpenAI-compatible endpoint 보유
- API 키 발급 완료
- 모델 ID 확인 완료 (`kimi-k2.5`, `glm-4.7` 등 사내 표기)

## 2) Roo Provider 설정 기본값

권장 설정:
- Provider 타입: OpenAI-compatible
- Base URL: `https://<company-llm-gateway>/v1` 또는 `http://<internal-host>:<port>/v1`
- API Key: 사내 토큰
- Model: `kimi-k2.5` 또는 `glm-4.7`

주의:
- endpoint 끝에 `/v1` 누락 금지
- 사내 인증서 환경이면 OS/VS Code trust store를 먼저 점검

## 3) 프로필 2개를 먼저 만들기

프로필 A: `kimi-build`
- Model: `kimi-k2.5`
- 용도: 구현/리팩터링/긴 코드 생성
- 권장 모드: `Code`, `Architect`

프로필 B: `glm-review`
- Model: `glm-4.7`
- 용도: 코드 리뷰/요약/검증
- 권장 모드: `Ask`, `Debug`

운영 팁:
- 모드별 기본 프로필을 고정해 응답 품질 변동을 줄인다.
- 동일 작업을 두 모델에 교차 검증시키면 품질이 올라간다.

## 4) 사내 전용 운영 규칙

- 외부 URL 참조 금지 정책이 있으면 `@url` 사용 제한
- 명령 실행 권한은 최소화(`Auto-Approve` 엄격 설정)
- 민감 레포에서는 편집 가능 경로를 `fileRegex`로 제한

## 5) 빠른 점검 프롬프트

```text
현재 세션 모델/프로필 기준으로 다음을 수행해줘.
1) 이 저장소의 핵심 엔트리포인트 3개 요약
2) 테스트 실행 명령 제안
3) 실패 시 디버그 우선순위 제시
```

통과 기준:
- 응답 속도/품질이 안정적
- 한글 지시와 코드 지시를 모두 일관되게 해석
- 명령/수정 제안이 저장소 구조와 맞음
