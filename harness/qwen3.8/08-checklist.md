# 운영 체크리스트

## 설정

- [ ] 모드별 샘플링 분리: thinking(1.0/0.95/20 또는 0.6/0.95/20) vs instruct(0.7/0.8/20)
- [ ] thinking 모드에서 greedy decoding(temp=0, top_k=1) 미사용 확인
- [ ] `max_tokens` 여유 할당 확인 — `finish_reason=length` 모니터링
- [ ] `reasoning_effort` 태스크별 운영 (xhigh/medium/low)
- [ ] 루틴 요청 thinking OFF + 필요 시만 ON (비용 최적화 1순위)
- [ ] presence_penalty 1.5 (반복 관찰 시 0~2 튜닝, 2 근처 금지)

## 서빙

- [ ] vLLM/SGLang 최신 버전 + 공식 FP8 체크포인트
- [ ] `--tool-call-parser` 명시 (qwen3_xml 또는 모델 카드 권장값)
- [ ] `--reasoning-parser` 설정
- [ ] `--enable-prefix-caching` 활성화
- [ ] `--max-model-len`을 실제 P95 컨텍스트에 맞게 제한 (무조건 최대값 금지)
- [ ] `--kv-cache-dtype fp8` 검토
- [ ] Ollama 계열이면 `OLLAMA_KEEP_ALIVE=-1` 필수

## 프롬프트/앱

- [ ] 시스템 프롬프트: 역할·규칙·언어를 앞쪽에 불릿으로 고정
- [ ] few-shot 예시 2~5개 (실제 운영 분포 기반)
- [ ] 구조화 출력은 guided decoding으로 강제
- [ ] 멀티턴 히스토리에서 thinking 내용 제거 (preserve_thinking 명시 켠 경우 제외)
- [ ] 툴 20개 이하, description 구체화
- [ ] malformed 툴콜 파싱/재시도 로직
- [ ] 쓰기 연산 휴먼 확인 + 최소 권한 툴

## RAG

- [ ] 검색 품질보다 컨텍스트 프레젠테이션 먼저 개선
- [ ] 답변 중심 배치, 구조 기반 압축(원문 수치 보존)
- [ ] 적응형 검색 (모델이 아는 질문엔 컨텍스트 미주입)
- [ ] 팩트 수치는 결정론적 주입

## 평가

- [ ] 툴 선택/인자 추출/엔드투엔드 성공률 평가셋 운영
- [ ] 출력 형식 표준화 프롬프트로 벤치마크 비교
