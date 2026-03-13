---
tags: [foundation-model, llm, transformer, attention, encoder, decoder]
level: beginner-to-advanced
last_updated: 2026-03-14
status: in-progress
---

# Foundation Model / LLM 기초

> Transformer가 왜 LLM의 표준이 되었는지, attention / encoder / decoder가 무엇인지, foundation LLM이 어떤 과정을 거쳐 만들어지는지 정리한 학습 노트

## 왜 필요한가? (Why)

- LLM을 제대로 이해하려면 프롬프트 사용법보다 먼저 **모델 구조와 학습 방식**을 알아야 한다
- `attention`, `transformer`, `encoder`, `decoder`, `foundation model`은 자주 쓰이지만 서로 다른 층위의 개념이라 한 번에 헷갈리기 쉽다
- BERT, GPT, T5를 한 축으로 보면 "이해 모델", "생성 모델", "통합 모델"의 차이가 정리된다

## 추천 학습 순서

```text
1. Attention
   -> 2. Transformer가 LLM으로 이어진 이유
   -> 3. Encoder / Decoder 구조 차이
   -> 4. Foundation LLM 제작 파이프라인
```

## 문서 구성

- [Attention이란 무엇인가?](./attention.md) - attention의 직관, Q/K/V, self-attention, multi-head, causal mask
- [Transformer가 어떻게 LLM으로 이어졌는가?](./transformer-to-llm.md) - RNN/attention/transformer/BERT/GPT/T5/foundation model로 이어지는 계보
- [Encoder와 Decoder란 무엇인가?](./encoder-and-decoder.md) - encoder-only, decoder-only, encoder-decoder의 차이와 대표 모델
- [Foundation LLM은 어떻게 만들어지는가?](./how-foundation-llms-are-built.md) - 데이터, 토크나이저, 사전학습, 스케일링, 정렬(alignment), 배포

## 핵심 요약

- **Attention**: 지금 계산 중인 토큰이 다른 토큰들 중 어디를 얼마나 볼지 정하는 가중합 메커니즘
- **Transformer**: recurrence 없이 attention 중심으로 시퀀스를 처리하는 구조
- **Foundation model**: 대규모 broad data로 자기지도 사전학습을 수행하고 여러 다운스트림 작업에 적응 가능한 모델
- **Chat / Instruct model**: foundation model 위에 instruction tuning, preference optimization, safety tuning을 추가한 모델

## 이 문서 묶음에서 구분하는 층위

| 층위 | 질문 | 예시 |
|------|------|------|
| 메커니즘 | 정보는 어떻게 참고되는가? | attention |
| 아키텍처 | 블록은 어떻게 쌓이는가? | transformer |
| 구조 역할 | 입력을 읽는가, 생성하는가? | encoder / decoder |
| 학습 체계 | 어떻게 큰 범용 모델이 되는가? | foundation LLM pipeline |

## 참고 자료 (Primary Sources)

- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014): <https://arxiv.org/abs/1409.0473>
- Vaswani et al., *Attention Is All You Need* (2017): <https://arxiv.org/abs/1706.03762>
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018): <https://arxiv.org/abs/1810.04805>
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019): <https://arxiv.org/abs/1910.10683>
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020): <https://arxiv.org/abs/2005.14165>
- Bommasani et al., *On the Opportunities and Risks of Foundation Models* (2021): <https://arxiv.org/abs/2108.07258>
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022): <https://arxiv.org/abs/2203.15556>
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022): <https://arxiv.org/abs/2203.02155>
