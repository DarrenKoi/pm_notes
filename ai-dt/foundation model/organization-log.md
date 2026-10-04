---
type: review-log
tags: [foundation-model, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
---

# Foundation model 문서 정리 기록

## 범위와 결정

기존 Markdown 5개를 모두 읽고 같은 경로에 유지했다. 원래 `last_updated: 2026-03-14`는 보존하고 검토일을 별도로 기록했다. 실행 코드·첨부가 없는 문서 묶음이다. 이동·삭제·다른 주제와의 통합은 없다.

개념 재등장은 입문 요약, 구조 비교, 역사 설명, 제작 공정이라는 서로 다른 사용 목적을 가진다. 전체 문서를 합치는 판단은 하지 않았다. 수식·mask의 상세 설명은 attention, 구조 선택 조건은 encoder/decoder, 파이프라인은 제작 문서를 대표 설명으로 삼고 README의 기존 읽기 순서를 유지했다. 제품 시장 비중과 범용 최적성을 확인한 듯한 문장은 적용 사례·미확인 조건으로 바꿨다.

| 원래 문서 | 개별 검토 결과 |
|---|---|
| [README](./README.md) | 네 층위와 읽기 순서 유지. Foundation model에 특정 후처리를 필수로 넣지 않으며 chat 공정도 선택 사항임을 명시 |
| [Attention](./attention.md) | soft lookup이 전체 Key 계산을 피한다는 오해, 양방향/causal mask, head 역할의 보장, 학습 병렬화와 생성 순차성, dense 연산/메모리 비용을 수정. 수치 예제 추가 |
| [Encoder와 Decoder](./encoder-and-decoder.md) | 2017 post-LN의 sublayer별 residual/normalization과 입력 embedding 위치 수정. 현재 입력 위치와 미래 정답을 구분. BERT 검색 embedding의 추가 학습 조건 보완 |
| [Transformer에서 LLM으로](./transformer-to-llm.md) | 공개 사례 중심 계보로 범위 한정. GPT-3 few-shot 결과를 보편적인 갑작스러운 능력 출현으로 해석하지 않음. 현재 시장 점유율·최적성 단정 제거 |
| [제작 파이프라인](./how-foundation-llms-are-built.md) | Foundation 정의와 assistant 후처리 구분. tokenizer 재사용/ID 대응, 학습 compute 조건의 Chinchilla, InstructGPT와 DPO 절차 차이 보완 |

## 기술 근거 · 확인 2026-10-04

원 논문의 공개 본문·초록을 확인했다. 아래 연도는 발표 연도이며 최신 제품 버전이라는 의미가 아니다.

| 일차 자료 | 확인한 주장과 적용 범위 |
|---|---|
| [Bahdanau 2014](https://arxiv.org/abs/1409.0473) | recurrent seq2seq의 입력 상태 참조. attention 도입만으로 RNN 순차 계산이 없어지지 않음 |
| [Transformer 2017, arXiv v7(2023), §3.1–3.5](https://arxiv.org/html/1706.03762v7) | scaled dot product, mask, shifted output, 각 sublayer의 post-LN, 위치 정보. 후속 모델의 모든 block 순서로 일반화하지 않음 |
| [BERT 2018, v2(2019)](https://arxiv.org/abs/1810.04805v2) | 양방향 표현과 downstream fine-tuning |
| [T5 2019, v4(2023)](https://arxiv.org/abs/1910.10683v4) | text-to-text 프레임워크와 학습 조건 비교 |
| [GPT-3 2020, v4](https://arxiv.org/abs/2005.14165v4) | 자동회귀 LM의 prompt 기반 few-shot 평가와 실패 과제. 모든 모델·과제에 대한 보장 아님 |
| [Foundation Models 2021, v3(2022)](https://arxiv.org/abs/2108.07258v3) | broad data, scale, downstream 적응 가능성. 정의에 Transformer 또는 assistant alignment를 강제하지 않음 |
| [Chinchilla 2022, v1](https://arxiv.org/abs/2203.15556v1) | 고정된 학습 compute와 실험 범위의 모델/토큰 배분. 모든 제품 비용의 최적 비율 아님 |
| [InstructGPT 2022, v1](https://arxiv.org/abs/2203.02155v1) | 시범 답변의 지도 학습과 인간 선호를 이용한 추가 학습. 오류가 사라진다는 보장 없음 |
| [DPO 2023, v3(2024)](https://arxiv.org/abs/2305.18290v3) | 선호 쌍으로 직접 정책 최적화. 별도 reward model/PPO 절차와 구분 |
| [FlashAttention 2022, v2](https://arxiv.org/abs/2205.14135v2) | exact attention의 타일 계산/메모리 접근 절감. dense 모든 위치 쌍 연산을 제거하는 방법 아님 |
| [Sentence-BERT 2019, v1](https://arxiv.org/abs/1908.10084v1) | 의미 유사도 문장 embedding을 위한 별도 학습 구조. 원래 BERT 표현의 단순 pooling을 좋은 검색 성능으로 보장하지 않음 |

## Claude 협의

`HERDR_ENV=1`에서 `herdr pane current --current`가 `pane_not_found`를 반환했다. 이 저장소의 작업 전용 Claude pane을 확보하지 못했으며 다른 저장소 pane은 조작하지 않았다. Claude 의견은 받지 못했다. 논문으로 직접 확인 가능한 사실과 조건만 수정했다. 불확실한 분할·이동·전체 통합 결정은 보류한다.

## 세 차례 검증

1. **목록·고유 내용:** 작업 전 ai-dt 원문 snapshot의 5개 경로와 현재 문서를 대조한다. 모든 원래 문서, 제목과 설명 예제·텍스트 도식을 보존한다. 조건을 바로잡은 문장은 수정 내역으로 위 표에 남긴다. 삭제·이동 없음.
2. **기술·예제:** 일차 자료 11건을 확인했다. Python 표준 `math`로 두 위치 attention의 softmax 합, 반올림 출력과 causal mask의 0 기여를 실행 검증했다. 결과는 `[0.6697615493, 0.3302384507]`, 첫 출력 `[6.6976154933, 6.6047690135]`, mask 첫 출력 `[10, 0]`다. 실제 모델 학습·GPU kernel·검색 품질 검증은 수행하지 않았다.
3. **탐색·메타데이터:** 상대 링크, anchor, 첨부 참조와 YAML을 검사하고 정확한 `pm_notes` vault에서 CLI properties와 읽기 화면을 확인한다. 결과는 아래 최종 확인에 기록한다.

## 남은 미확인

2026년 개별 상용 모델의 미공개 학습 공정·구조·시장 비중, 이 저장소 예제의 실제 모델 품질, Claude 협의는 미확인이다. 이는 공개 원 논문 수준의 개념 검토이며 모든 후속 구현에 대한 인증은 아니다.

## 최종 확인

- 원래 5개 경로, 제목/소제목과 fenced 예제를 snapshot과 자동 대조: 모두 보존. 문장 수정은 위 개별 결과에 기록했다.
- 현재 6개 문서의 YAML 검토일·상태 확인, 상대 링크·anchor 검사: 기존/신규 깨진 참조 0. 첨부·wiki·reference-style 링크는 이 묶음에 없다. `git diff --check` 통과.
- 정확한 `pm_notes` vault의 Obsidian 1.13.7 CLI properties: 6개 모두 검토일을 읽었다. 읽기 화면에서 README 목차 → Attention 상대 링크 클릭이 같은 주제 경로로 이동함을 확인했다. 한국어 본문·tags·원래 날짜·검토 날짜·callout을 화면으로 확인했다. 모든 문서의 전체 화면·아래쪽 표/수식을 개별 시각 검증한 것은 아니다.
- Claude 협의와 실제 모델 실험이 남아 `partial`로 표시한다. 공개 자료의 개념 검토와 로컬 수치 예제 검증은 완료했다.
