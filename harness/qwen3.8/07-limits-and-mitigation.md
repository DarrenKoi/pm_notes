---
category_major: "에이전트 하네스"
category_middle: "모델별 적용"
category_minor: "Qwen 모델 검토·운영"
note_kind: "학습"
classified_on: "2026-10-05"
---
# 알려진 한계와 극복 전략

## 1. 무한 반복 / 반복 루프

**증상**: 동일 문구 반복, 생성이 끝나지 않음.

**극복**:
- presence_penalty 1.5 (양자화 모델 공식 권장). 0→2 범위에서 튜닝하되 2 근처는 언어 혼용 유발.
- greedy decoding 금지 (특히 thinking 모드).
- repetition_penalty 1.05 수준 병행 (vLLM 예제 기준).

## 2. 환각 (특히 수치·날짜)

**증상**: 그럴듯한 가짜 숫자/타임스탬프. 컨텍스트에 정답이 있어도 창작.

**극복**:
- 팩트는 생성 전 결정론적 주입 (deterministic-first 설계) — DB/도구에서 조회한 값을 프롬프트에 verbatim 삽입.
- "컨텍스트에 없으면 모른다고 답하라" 명시 + 근거 인용(출처 필드) 강제.
- self-verification 턴 추가 (별도 호출로 답 검증).

## 3. 컨텍스트 열화

**증상**: 컨텍스트가 길어질수록 중간 정보 누락, 지시 이탈.

**극복**:
- 핵심 지시/정보를 프롬프트 앞과 뒤에 배치 (lost in the middle 대책).
- 컨텍스트 압축(구조 기반) + 원문 보존 볼륨 유지.
- 장기 대화는 턴 요약/절단. thinking 이력 제거.

## 4. 복잡한 다단계 추론 붕괴

**증상**: 한 번의 호출에 여러 과제를 주면 중간 단계 실수.

**극복**:
- 태스크 분해 → 단계별 호출 → 명시적 전달.
- thinking 모드 + 충분한 max_tokens로 추론 공간 확보.
- 어려운 문제는 `reasoning_effort=xhigh`, 루틴은 `medium/low`로 비용 조절.

## 5. 포맷 불안정 (툴콜/JSON)

**증상**: malformed JSON, 툴콜 포맷 drift (특히 장기 컨텍스트, 커뮤니티 파생 모델).

**극복**:
- guided decoding / structured outputs로 스키마 강제.
- 공식 체크포인트(+공식 FP8) 사용. SFT 파생 모델의 장기 컨텍스트 포맷 drift 주의.
- malformed 툴콜 자체 파싱 + 에러 피드백 재시도 루프.

## 6. 언어 혼용

**증상**: 한국어 요청에 영어 섞임 (presence_penalty 높을 때 특히).

**극복**:
- 시스템 프롬프트에 응답 언어 명시.
- presence_penalty는 1.5 이하 유지.

## 7. 지연/비용 (thinking 모드 남용)

thinking 모드는 출력 토큰을 수 배로 늘리고 실효 처리량을 절반 이하로 깎는다.

**극복**:
- 태스크 라우팅: 복잡한 요청만 thinking ON, 나머지는 OFF.
- `reasoning_effort` 단계별 운영.
- prefix caching 활성화로 공통 시스템 프롬프트 프리필 절감.

## 출처

- https://huggingface.co/Qwen/Qwen3-32B (반복/디코딩)
- https://huggingface.co/Qwen/Qwen3-32B-GGUF (presence_penalty 1.5)
- https://arxiv.org/abs/2609.07663 (환각/deterministic-first)
- https://arxiv.org/abs/2603.14045v2 (컨텍스트 압축)
- https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix (포맷 drift)
