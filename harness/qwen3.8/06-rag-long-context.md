---
category_major: "에이전트 하네스"
category_middle: "모델별 적용"
category_minor: "Qwen 모델 검토·운영"
note_kind: "학습"
classified_on: "2026-10-05"
---
# RAG & 긴 컨텍스트 설계

소형 모델 RAG의 병목은 **검색 품질이 아니라 컨텍스트 활용 능력**이라는 게 2026년 연구의 일관된 결론이다.

## 핵심 연구 결과 (소형 모델의 RAG 함정)

- 7B 이하 모델은 **oracle retrieval(정답이 문맥에 확실히 있는 이상적 검색)**조차 85~100% 실패 — 병목은 retrieval이 아니라 context utilization. (arXiv:2603.11513)
- 검색 컨텍스트의 존재 자체가 원래 맞던 답을 무너뜨리는 **distraction effect**: 42~100%의 정답 파괴. 프롬프트를 바꿔도 회복 안 됨.
- 실패의 지배적 형태는 **무관 생성(61~100%)**: 모델이 주어진 컨텍스트를 완전히 무시.
- 활용 능력은 모델 크기에 log-linearly 비례. >50% 활용에는 7B 초과, 견고한 저항엔 10B+ 필요.

**27B에서의 시사점**: Qwen3.8-27B는 위 연구의 sub-7B보다 훨씬 유리하지만, 여전히 "검색 품질 개선"보다 "컨텍스트 프레젠테이션 설계"가 ROI가 높다.

## 완화 전략 (우선순위 순)

### 1. 컨텍스트 압축 + 구조화

- 그래프 워크 컴프레션(구조 연결 컨텍스트만 유지)으로 입력 토큰 ~60% 절감, 소형 모델 정확도 +6pp. (arXiv:2603.14045)
- SARA 방식: 핵심 패시지는 원문 유지(엔티티/수치 보존), 나머지는 압축 표현으로 커버리지 확보. 압축률을 높이면 수치·고유명사가 유실되어 환각이 늘어난다 — **원문 보존 볼륨을 유지**할 것.

### 2. 답변 중심(answer-centered) 프롬프트

- 검색 결과를 낱개 리스트로 나열하지 말고, 후보 답변별로 지원 근거를 묶어 제시. 소형 모델이 증거를 통합하기 쉬워진다. (RPO-RAG, arXiv:2601.19225)

### 3. 구조화된 CoT 프롬프트

- 자유로운 "step by step" 대신, 질문을 구조적 쿼리(예: SPARQL triple 패턴)로 분해하게 하는 프롬프트가 +2~+14pp. 소형 모델은 개방형 서치보다 템플릿 매칭에 강하다.

### 4. 적응형 검색 (adaptive retrieval)

- 모델이 이미 답할 수 있는 질문엔 검색 컨텍스트를 아예 주지 마라 — distraction으로 오히려 정확도가 떨어진다. 신뢰도 추정 기반으로 검색 여부를 결정.

### 5. 사실성 중요 질의는 deterministic-first

- Noesis (arXiv:2609.07663): 수치·타임스탬프 같은 팩트는 생성 전 단계(producer-side fact layer)에서 결정론적으로 주입. 2B 모델이 35B급 팩트 무결성 달성. "틀린 숫자는 답 없음보다 나쁘다."

## 긴 컨텍스트 운영 (262K native)

- **262,144 토큰 네이티브**, YaRN으로 1M까지. 다만 Qwen 공식: 총 길이가 native 한도를 넘을 때만 YaRN 사용. static YaRN은 짧은 입력 성능도 깎는다.
- 1M 확장 시 factor를 실제 필요에 맞게 조정 (예: 65K 필요 → factor 2.0).
- 소형 모델은 **중간 컨텍스트 열화(lost in the middle)** 가 있음 — 핵심 정보는 프롬프트 앞/뒤에 배치하고, 중간엔 보조 정보.
- 히스토리 관리: 이전 턴 thinking 제거(또는 `preserve_thinking` 끄기), 오래된 턴 요약/절단으로 컨텍스트 예산 관리.

## 출처

- https://arxiv.org/abs/2603.11513 (context utilization 병목)
- https://arxiv.org/abs/2603.14045v2 (SPARQL CoT, graph-walk compression)
- https://arxiv.org/abs/2601.19225 (RPO-RAG, answer-centered prompt)
- https://arxiv.org/abs/2609.07663 (Noesis, deterministic-first)
- https://aclanthology.org/2026.acl-long.661.pdf (SARA)
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8 (262K/YaRN 권장)
