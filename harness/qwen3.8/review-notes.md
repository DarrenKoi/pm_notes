---
tags: [qwen, inference, verification]
aliases: [Qwen3.8 설정 검토]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "에이전트 하네스"
category_middle: "모델별 적용"
category_minor: "Qwen 모델 검토·운영"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# Qwen3.8 원문을 읽기 전 확인할 조건

이 디렉터리의 기존 9개 문서는 사용자 작성 중인 자료여서 원문을 수정하지 않았다.
[원래 목차](./README.md)의 01~08을 읽을 때 다음 차이를 함께 확인한다.
[하네스 목차](../README.md)와 같은 최상위 주제 안의 모델별 적용 노트다.

## 공식 모델 카드와 대조한 차이

확인일 **2026-10-04**, 대상은 **Qwen3.8-27B-FP8 공식 카드**다.
BF16·다른 크기·Ollama·클라우드 API 전체에 동일하게 적용한다고 단정하지 않는다.
[공식 근거](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).

| 원문 위치 | 현재 카드 기준 해석 |
|---|---|
| 01 모델 개요 | 기본 컨텍스트 262,144에는 입력과 출력이 함께 포함된다. 1M 확장은 별도 RoPE/YaRN 조건이 필요하다. |
| 01 벤치마크 | Terminal-Bench 2.1은 Terminus 하네스로 측정했다. 모든 코딩 결과가 Claude Code 하네스라는 해석은 맞지 않는다. |
| 02 샘플링 | thinking 권고는 temperature 1.0, presence penalty 0.0; instruct는 0.7, 1.5다. 모드별 설정을 구분한다. |
| 02·07 반복 억제 | instruct용 presence penalty 1.5를 thinking 기본값으로 복사하지 않는다. |
| 03·05 실행 | chat template, reasoning parser, tool parser는 다른 설정이다. 모델 카드가 연결한 엔진별 recipe와 설치 버전을 확인한다. |
| 06 긴 컨텍스트 | 1M 확장 예의 원래 최대 길이는 262,144, factor는 4다. 65K 기준 설정을 무조건 재사용하지 않는다. |

카드는 thinking·`preserve_thinking` 기본 활성화와 `reasoning_effort=xhigh`를 설명한다.
서버가 파라미터를 지원해야 하며 낮은 reasoning effort가 전체 업무 지연을 항상
줄이지는 않는다. reasoning/최종 출력의 별도 상한은 이를 지원하는 프레임워크 조건이다.

## 적용 전에 검증할 것

03의 `<parser>`는 실제 값이 필요한 자리표시자다. 엔진·버전·모델 revision을 기록한
뒤 도구 호출 왕복과 reasoning 분리를 작은 가짜 도구로 시험한다. 이번 작업에서는
GPU 서버·Ollama·vLLM·클라우드 API를 실행하지 않았다.

04의 few-shot 개수는 실험 시작점이다. JSON Schema 일치는 값의 사실성·권한을
보장하지 않는다. 05의 `tool_choice` 제한은 특정 API 계약과 구분해야 한다.
06의 논문 결과를 다른 모델 크기·검색 데이터에 적용할지는 미확인이다.
07·08의 처리량 배수, 양자화 품질 손실률, 모든 업무에 특정 keep-alive가 필수라는
표현은 측정 결과로 확정하지 않았다. 모델 라이선스 설명과 실제 배포 의무도 별도 검토가 필요하다.

원문 수정을 대신하는 확정 설정 파일은 제공하지 않는다. Herdr 현재 pane 연결 실패로
Claude 협의가 필요한 엔진별 파서·API 차이·논문 일반화 판단을 보류했다.
각 원문 문서의 보존·검토 결과는 [정리 기록](../organization-log.md)에 남긴다.
