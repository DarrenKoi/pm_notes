---
tags: [itc, aix, roadmap, ch3, drafting]
level: drafting
last_updated: 2026-05-15
version: 0.1
related: [./itc-aix-roadmap-outline.md, ./ch1-context-asis-draft.md, ./ch2-vision-draft.md]
---

# Ch.3 To-Be — Active를 거쳐 Proactive로 — 슬라이드 드래프트 (v0.1)

> [Outline §Ch.3](./itc-aix-roadmap-outline.md#chapter-3-to-be--active를-거쳐-proactive로-2p) 본문 드래프트. **2슬라이드 분량**.
> 작성 원칙: Ch.2 Vision의 *정성적 그림*을 *시간축 + 능력 + KPI*의 정량 좌표로 옮긴다.

## 슬라이드 흐름

```
Slide 1 — 3단계 전환 척추               Slide 2 — KPI Scorecard
──────────────────────────              ───────────────────────────
Passive → Active → Proactive            5 stream × L1 KPI + Enabler
시간축 + 능력 추가 다이어그램            Hero 3개 ★1·3·5 + E1·E2
"우리는 이렇게 진화한다"                 "이렇게 측정한다"
```

Slide 1 = narrative (3.1 + 3.2 통합). Slide 2 = measurement (3.3).

---

## Slide 1 — 3단계 전환 — Passive에서 Proactive까지 (Slide 3.1 + 3.2)

### Slide title
**3단계 전환 — 사람의 일과 AI의 일이 어떻게 나뉘는가**

### Headline (1줄)
> **"Active = 사람 옆에 AI 보조자. Proactive = AI가 먼저 제안하고 사람이 검증한다."**

### 본문 — 3단계 정의표 (좌측 1/2)

| 단계 | 시점 | 사람의 일 | AI의 일 | 5 stream 작동 능력 |
|------|------|-----------|---------|-------------------|
| **Passive** | As-Is | 전부 사람 | 없음 / 보조 | — |
| **Active** | **1~2년** | 판단·결정 | 분류·선별·RAG·초안·이상탐지 | 5 stream 전반 |
| **Proactive** | **3~5년** | 검증·승인 | 예측·시뮬레이션·자동 제안 | 5 stream 전반 + 능동 제안 |

### 본문 — 단계별 ITC 모습 (우측 1/2)

#### Active ITC (1~2년)
> *"엔지니어 옆에 AI 보조자가 있다."*

- 의뢰 자동 분류·라우팅 → 분석가 워크 큐로 직접 흐름
- 분석 보고서를 AI가 초안 작성, 사람은 편집·검증
- 검측 이미지가 자동 분류되어 *우선순위 큐*에 정렬
- 사내 RAG에서 *과거 사례·운영 매뉴얼*이 자동 회수
- 한정 범위 FDC 이상 시그널이 사람 검증과 함께 측정

#### Proactive ITC (3~5년)
> *"AI가 먼저 제안하고, 사람은 검증·승인한다."*

- *"이 장비 2주 내 이상 가능성 87%"* — 알람 전에 예지보전 알림
- *"이 소재 후보가 우선순위 1위"* — 실험 전에 후보 추천
- *"이 의뢰는 X 패턴 — 사전 대응 가이드 자동 생성"* — 요청 오기 전 제안
- OPC 보조·검사 우선순위가 시뮬레이션 기반으로 자동 산출
- 암묵지가 5 stream 전반에 흐르는 *기본 동작*

### 본문 — 척추 다이어그램 (슬라이드 중앙 또는 하단)

```
   Passive               Active                    Proactive
   (As-Is)               (1~2년)                  (3~5년)
   ─────                 ──────                    ──────────
   요청 → 처리           요청 → AI 보조 → 결정     예측 → AI 제안 → 검증
   
   능력: 없음            +분류·선별                +이상·예측
                         +회수·요약(RAG)           +추천·시뮬레이션
                         +초안 생성                
                         +이상 시그널(시드)
                                              ◀ Active→Proactive 전환선 ▶
```

### Speaker notes
> "Active와 Proactive를 가르는 결정적 차이는 *능동성* 입니다. Active에서는 사람이 *요청하면* AI가 응답합니다. Proactive에서는 AI가 *요청 전에* 제안합니다. 1~2년차에 우리는 *분류·RAG·초안*이라는 4개 능력을 5 stream 전반에 배치하고, 3년차부터 *예측·추천*을 더해 능동 영역으로 진화합니다. 우측에 단계별 *구체 장면*을 보여드리는 이유는, 이 전환이 슬로건이 아니라 *엔지니어가 일하는 방식이 실제로 바뀌는* 변화이기 때문입니다."

### 슬라이드 시각요소 제안
- 좌측: 3단계 정의표 (Passive·Active·Proactive 가로축, 사람/AI 분업 컬럼)
- 우측: 단계별 ITC 모습 — Active와 Proactive 각각 5개 구체 장면 (작은 아이콘)
- 하단 또는 중앙: 시간축 다이어그램 — 능력이 점진적으로 추가되는 *층층* 표현. 2~3년차 사이 *전환선* (간트 챕터에서도 동일하게 등장 — visual coherence)

### [SLOT]
- *"87%"*, *"우선순위 1위"* 같은 구체 수치 표현 — Ch.2 Slide A *"3년 후 모습"* 과 동일 표현 사용해 보고서 내부 visual repetition 의도
- Active 5개 / Proactive 5개 장면을 각각 5 stream과 1:1 매핑하면 narrative가 더 단단해짐. 검토 필요

---

## Slide 2 — 성공을 어떻게 측정할 것인가 (Slide 3.3) ★ Scorecard

### Slide title
**성공 측정 — 2-Layer KPI Scorecard**

### Headline (1줄)
> **"5가지 흐름마다 1개 outcome KPI. 그중 3개를 Hero로 강조한다. 횡단 Enabler KPI 2개를 추가한다."**

### 본문 좌측 (2/3) — Layer 1: 5 Stream × CEO Outcome KPI

| Stream | L1 KPI | 1년 목표 | 3년 목표 | Hero |
|--------|--------|---------|---------|------|
| 1. 의뢰 대응 | **의뢰 평균 TAT** | ↓30% | ↓50% | ★ |
| 2. 분석 | 분석 보고서 평균 작성시간 | ↓40% | ↓60% | |
| 3. 계측 데이터 생산 | **검측→공정 의사결정 lead time** | ↓30% | ↓50% | ★ |
| 4. 장비 운영 | 신규 인력 ramp-up 시간 | ↓30% | ↓50% | |
| 5. 안정성 모니터링 | **예지보전 hit rate / 알람 대응시간** | hit ≥40%, 대응 ↓25% | hit ≥70%, 대응 ↓50% | ★ |

→ **Hero 3개 (★1·3·5)** 가 CEO 보고에서 가장 자주 인용되는 수치. 나머지 2개는 *동행 KPI*.

### 본문 우측 (1/3) — Layer 2: Enabler 횡단 KPI

```
┌──────────────────────────────┐
│ E1 — 인당 AI 활용 빈도         │
│  1년: 핵심 인력 ≥10회/월       │
│  3년: 전 직원 ≥10회/월         │
│  의미: AI가 일에 침투했는가     │
└──────────────────────────────┘

┌──────────────────────────────┐
│ E2 — 암묵지 캡처 사이클 횟수   │
│  1년: 1개 영역 1회전          │
│  3년: 분기 ≥3회 정례화        │
│  의미: 순환고리가 살아 있는가  │
└──────────────────────────────┘
```

### 본문 하단 — 운영 원칙 박스

> **Baseline 우선:** Hero KPI 3개 모두 6개월 PoC 단계에서 baseline 측정 필수. *측정 가능성이 ↓30% 표현의 전제조건*.
>
> **Layer 3 (technical metrics):** 분류 정확도·RAG hit rate·precision-recall 등은 [Appendix B](./itc-aix-roadmap-outline.md#appendix)에 격리.
>
> **Fallback 보존:** Stream 3 lead time baseline 측정 자체가 PoC 과제 수준 — 1년차 미달성 시 Hero에서 E1 승격 옵션.

### Speaker notes
> "KPI는 *너무 많으면 흐려지고, 너무 적으면 약해진다* 의 trade-off가 있습니다. 우리 답은 *5 stream 전부 노출, 그중 3개를 Hero로 강조* 입니다. Hero 3개는 의뢰 TAT, 검측 lead time, 예지보전 hit rate — 모두 *CEO에게 직접 의미 있는 outcome*이고, baseline 측정도 가능합니다. 우측 Enabler 2개는 *흐름이 살아 있는가*의 proxy입니다. 모델 정확도 같은 기술 지표는 Appendix로 격리해서 본문이 흐려지지 않도록 했습니다."

### 슬라이드 시각요소 제안
- 좌측: L1 KPI 표 — Hero 3행은 음영 처리 또는 ★ 표시. 1년·3년 목표는 작은 화살표(↓·↑)로 시각화
- 우측: E1·E2 카드 2개 — 1년/3년 목표 + *"의미"* 한 줄
- 하단 박스: *Baseline 우선* 메시지를 시각적으로 분리 — 이게 *측정 못 하면 표현도 못 한다* 의 정직성 표명

### [SLOT]
- Hero 표시 방식 — ★ vs 음영 vs 색상 강조. 디자이너와 협의
- Layer 1 표의 *근거(Baseline 출처)* 컬럼을 추가할지 — A1/A2/A3 anchor와 호응. Ch.1 Slide 2와 visual 연결 강화 가능
- Fallback 표현 위치 — 본문 박스 vs speaker notes only. *"E1 승격 옵션"* 을 공개적으로 노출하면 CEO에게 *"왜 미리 못 정하느냐"* 반박 위험 — speaker notes only가 안전할 가능성

---

## 슬라이드 간 흐름

```
Slide 1 (전환 척추)          Slide 2 (KPI Scorecard)
──────────                  ──────────────────────
"이렇게 진화한다"      →    "이렇게 측정한다"
정성 → 정량 bridge          5 stream × L1 + E1·E2
```

Ch.3 전체가 **정성(Vision)에서 정량(추진 트랙·Quick Win)으로 가는 다리**. Slide 1이 *그림*, Slide 2가 *눈금*.

---

## 보고서 전체 narrative arc 위치

```
Ch.1 (As-Is)            Ch.2 (Vision)         Ch.3 (To-Be) ← 여기      Ch.4·5·7 (실행)
─────────────────       ─────────────         ─────────────────       ────────────────
"우리는 Passive다"  →   "Proactive로 간다"  →  "이렇게 측정 가능"  →   "이렇게 실행"
```

Ch.3는 *측정 가능성*의 약속 — 다음 챕터들의 실행 디테일이 *얼마나 진척됐는지 알 수 있다* 는 보증.

---

## Round 4 후속 검토 포인트

1. **Active 5개 / Proactive 5개 장면의 5 stream 매핑** — Slide 1 우측을 stream과 1:1로 묶을지
2. **Hero KPI 시각 강조 방식** — ★ vs 음영 vs 색상
3. **Fallback (Stream 3 → E1 승격)의 공개 수준** — 본문 박스 / speaker notes / 비공개
4. **Slide 2에 baseline 출처 컬럼** — Ch.1 Slide 2 anchor 카드와 visual 연결 강화 여부
5. **Active·Proactive 단계의 시간 정의** — *1~2년·3~5년*이 정확한지, *2년·5년*으로 단순화할지

## 관련 문서
- [itc-aix-roadmap-outline.md](./itc-aix-roadmap-outline.md) Ch.3 (high-level 구조)
- [ch1-context-asis-draft.md](./ch1-context-asis-draft.md) — Ch.1 (As-Is anchor 3개와 Ch.3 KPI baseline 호응)
- [ch2-vision-draft.md](./ch2-vision-draft.md) — Ch.2 (Passive→Active→Proactive narrative spine 공유)
- [quick-win-cards.md](./quick-win-cards.md) — Ch.7 (PoC별 KPI가 Ch.3 Layer 1·2의 sub-set)
- [CONTEXT.md](./CONTEXT.md) — Passive/Active/Proactive 정의
