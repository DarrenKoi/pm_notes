---
title: RAG 벤치마크를 읽고 적용하는 방법
tags: [rag, evaluation]
aliases: [RAG 평가 해석]
document_type: learning
reviewed_on: 2026-10-04
verification_status: partially-verified
category_major: "RAG 사례 분석"
category_middle: "벤치마크 해석"
category_minor: "평가 적용 조건"
note_kind: "학습"
classified_on: "2026-10-05"
---

# RAG 벤치마크를 읽고 적용하는 방법

> 목적은 높은 순위를 복제하는 것이 아니라 자신의 문서·질문·비용 조건에서 검색과 답변의 실패를 구분하는 것이다.

## 왜 필요한가?

검색 순위가 좋아도 답변이 근거를 잘못 해석할 수 있다. 다른 데이터셋의 1위 모델이나 청크 크기를 그대로 적용하면 문서 길이, 언어, 질문 분포 차이를 놓친다. RAG는 외부 검색 결과를 생성에 결합하며 원 논문은 학습 가능한 검색기와 seq2seq 생성기를 함께 다룬다. 모든 현대 RAG가 원 논문의 학습 구성을 그대로 사용하는 것은 아니다. [Lewis 등, arXiv v4](https://arxiv.org/abs/2005.11401v4), 확인일 2026-10-04.

## 어떻게 작동하는가?

문서를 분할·색인하고 질문으로 근거를 검색한 뒤 생성기에 전달한다. 검색 평가에는 질문별 관련 문서 정답이 필요하고 답변 평가에는 정답 또는 별도 채점 기준이 필요하다.

- MRR: 첫 관련 결과 순위의 역수를 질문별로 평균한다. 첫 결과가 없는 질문은 0이며, 여러 관련 문서의 회수 정도를 모두 표현하지 못한다.
- Hit@K: 상위 K개 안에 관련 결과가 하나라도 있는 질문의 비율이다. 관련 결과를 얼마나 모두 가져왔는지는 별도로 살핀다.
- Judge 점수: 평가 모델·프롬프트·척도에 따른 평가값이다. 여러 모델의 합의를 인간 정답과 같은 것으로 취급하지 않는다. 위치, 장황함, 자기 선호 편향이 보고되어 있다. [Zheng 등, arXiv v4](https://arxiv.org/abs/2306.05685v4), 확인일 2026-10-04. 이 논문의 MT-Bench 결과는 이 폴더의 한국어 RAG 수치를 검증한 결과가 아니다.

## 어떻게 사용하는가?

1. 질문, 관련 근거, 기대 답변을 고정하고 개발용 질문과 최종 평가용 질문을 나눈다.
2. 문서 스냅샷, 청크 크기의 단위(문자/토큰), overlap, 임베딩·생성 모델의 정확한 ID와 버전, 검색 K를 기록한다.
3. 같은 질문에서 검색 결과 ID·순위와 답변·채점 근거를 저장한다. 서로 다른 judge의 점수는 별개 표로 비교한다.
4. 실패 질문을 직접 읽고 검색 누락, 문맥 손실, 생성 오류, 채점 오류를 구분한다.
5. 후보를 개발용 데이터에서 선택하고 고정한 평가용 데이터에서 다시 확인한다. 지연, API 요금, 로컬 GPU 비용도 함께 기록한다.

다음은 외부 호출 없는 지표 계산 예제다. 정답 집합과 검색 순위가 이미 주어진 경우에만 의미가 있다. Python 3 표준 라이브러리 예제이며 설치가 필요 없다.

```python
relevant = [{"a", "b"}, {"c"}, {"d"}]
rankings = [["x", "a", "b"], ["c", "y"], ["x", "y"]]
k = 2
reciprocal_ranks = []
hits = []
for truth, ranking in zip(relevant, rankings):
    first = next((i for i, item in enumerate(ranking[:k], 1) if item in truth), None)
    reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
    hits.append(first is not None)
print(sum(reciprocal_ranks) / len(relevant))  # MRR@2 = 0.5
print(sum(hits) / len(relevant))              # Hit@2 = 2/3
```

## 적용 조건과 미확인

이 예제는 순위 계산을 보여주며 실제 검색 품질을 입증하지 않는다. 동일 문서의 여러 청크를 관련 결과로 인정할지, 검색 결과 중복을 어떻게 처리할지는 평가 전에 정한다. 300문항, 300~500자, 특정 하이브리드 비율과 같은 값은 보편적 최소 조건이 아니다.

[사이트 관찰 기록](rag_baeum_site_analysis.md)의 모델명·요금·수치·재현 데이터는 2026-10-04 미확인이다. 공개 페이지 응답만으로 로그인 뒤 대시보드의 존재나 내용을 확인하지 못했다. 관찰 기록의 수치는 현재 도구 선택의 근거로 단독 사용하지 않는다.
