---
tags: [itc, aix, roadmap, ch2, drafting]
level: drafting
last_updated: 2026-05-15
version: 0.1
related: [./itc-aix-roadmap-outline.md, ./CONTEXT.md, ./docs/adr/0001-aix-tf-ax-part-boundary.md]
---

# Ch.2 Vision & Positioning — 슬라이드 드래프트 (v0.1)

> [Outline §Ch.2](./itc-aix-roadmap-outline.md#chapter-2-itc-aix-vision--positioning-2p) 본문 드래프트. 2슬라이드 분량.
> 작성 원칙: 본문 팀명·도메인 라벨 ✕ (완전 추상화). 5 stream만 노출.

---

## Slide A — ITC AIX Vision (Slide 2.1)

### Slide title (대)
**ITC AIX Vision — 요청 대응에서 예측 대응으로**

### Headline (1줄, 청자가 가장 먼저 읽는 문장)
> **"공정·장비·소재 기반기술을 AI로 가속하고, 요청 대응을 예측 대응으로 진화시키는 ITC"**

### Narrative spine — 다이어그램 본문 (Passive → Active → Proactive)

```
  Passive (As-Is)       Active (1~2년)         Proactive (3~5년)
  ─────────────         ──────────────          ────────────────
  요청 오면 처리         빠르고 정확한 응답       요청 오기 전 제안
  사람이 전부            AI가 보조               사람은 검증·승인
  의뢰 수기 분류         자동 분류·RAG·초안      예지보전·후보 추천
  보고서 수작업          이상 탐지               시뮬레이션·자동 제안
```

### 3년 후 ITC의 모습 (CEO가 머릿속에 그릴 장면)

- **엔지니어 옆에 AI 보조자** — 분류·선별·RAG·초안 능력이 5 stream 전반에서 일상 작동
- **요청 오기 전 제안** — *"이 장비 2주 내 이상 가능성 87%"*, *"이 소재 후보가 우선순위 1위"* 같은 사전 알림이 표준
- **암묵지가 흐른다** — 엔지니어의 노하우가 *일하는 부산물*로 캡처되어 다음 결정에 재활용

### 슬라이드 우측 (시각요소 제안)
- 위쪽: 화살표 다이어그램 (Passive → Active → Proactive) + 각 단계의 시간축
- 아래쪽: 5 stream 아이콘 5개 (의뢰·분석·계측·운영·안정성) — 각 단계에서 능력이 추가되며 색이 진해지는 표현

### Speaker notes (구두 설명)
> "이 한 문장이 5년 후 ITC의 자기 정의입니다. 핵심은 두 가지 — *기반기술을 AI로 가속*하고, *요청 대응*을 *예측 대응*으로 바꾼다. Passive에서 Active까지 1~2년, Active에서 Proactive까지 추가 2~3년. 이건 단순한 도구 도입이 아니라 **일하는 방식의 단계적 전환**입니다."

### [SLOT]
- 슬라이드에 *"3년 후 모습"* 의 구체 장면을 picture-in-picture 으로 넣을지 — 사내 사용 사례 mockup 1개 추가 검토

---

## Slide B — 왜 ITC인가 (Slide 2.2)

### Slide title (대)
**왜 ITC가 AIX의 중심인가 — 5 Stream × 현장 결합**

### Headline (1줄)
> **"전사 AIX TF가 만드는 공통 능력을, ITC가 5가지 흐름과 결합해 공정·장비 성과로 전환한다."**

### 본문 좌측 (2/3) — 5 Stream Differentiator

ITC는 **다음 5가지 흐름을 모두 가진 유일한 조직 단위**입니다.

| # | Stream | ITC가 가진 자산 |
|---|--------|----------------|
| 1 | **의뢰 대응** | 의뢰 채널·과거 사례 (Biz 지원 채널의 집중점) |
| 2 | **분석** | 분석 보고서·해석 로그 |
| 3 | **계측 데이터 생산** | **공정 개선 의사결정의 원천 데이터** |
| 4 | **장비 운영 관리** | 셋업·운영 매뉴얼·교대 로그 |
| 5 | **장비 안정성 모니터링** | **FDC 시계열** |

→ Stream 3·5는 *데이터 자산 자체가 ITC 외부에 존재하지 않는 자원*. 이걸 가지지 않은 단위는 *"공통 AI 능력"*을 받아도 공정 개선·장비 안정성으로 **전환할 도구가 없습니다**.

### 본문 우측 (1/3) — 분업 구조 (Platform/Consumer)

```
   전사 AIX TF  (DT 주도)
   │   공통 능력 제공
   │   GPU · 플랫폼 · LLM
   │   과제 우선순위 · 표준
   ▼
   ITC AX Part
   │   5 stream 도메인 결합
   │   암묵지 순환고리
   │   ITC 로드맵 owning
   ▼
   ITC AI 실행 TF
       각 팀 현업 확산 채널
```

> *"전사 TF = 무엇을 만들 것인가"*
> *"ITC = 어디에 결합할 것인가"*

### 보조 메시지 (2.3, 슬라이드 하단 박스)

> **"RAG 품질의 상한은 암묵지 채굴이 결정한다."**
>
> 단순 문서 RAG로는 한계가 있다. ITC는 *일하는 흐름의 부산물로 암묵지가 명시화되는* **순환고리**를 갖춘다 — 이게 사내 AI 인프라를 *현장 성과*로 전환하는 결정적 차이.

### Speaker notes (구두 설명)
> "CEO께서 가장 먼저 묻고 싶으실 질문이 *'이거 결국 전사 TF 일 아닌가'* 일 것입니다. 답은 *공통 능력 ≠ 현장 결합*입니다. 전사 TF가 GPU·플랫폼·LLM을 만들면, 그건 *공통 도구*입니다. 그 도구가 *공정 수율*이나 *장비 downtime 감소*로 전환되려면 5가지 흐름 모두에 손이 닿아 있는 단위가 필요합니다. ITC가 그 유일한 단위입니다.
>
> 두 번째로, 사내 AI의 진짜 상한은 *모델 성능*이 아니라 *암묵지를 얼마나 흘리느냐*입니다. 엔지니어의 머릿속 노하우가 RAG로 들어오지 않으면, 아무리 좋은 LLM이 와도 답이 얕습니다. ITC는 이 *암묵지 순환고리*를 일하는 흐름 안에 박는 1년차 PoC를 시작합니다."

### 슬라이드 우측 (시각요소 제안)
- 5 stream 아이콘 5개 + ITC 로고 중심
- 우측 3-tier 다이어그램 (전사 TF → AX Part → 실행 TF)
- 하단 박스: *"RAG 품질의 상한"* 인용 박스 (시각적으로 분리)

### CEO 예상 반박 & 답변 (speaker note backup)

| CEO 반박 | 답변 (10초 내) |
|----------|--------------|
| *"이거 결국 전사 TF 일 아닌가?"* | *공통 능력은 TF, 현장 결합은 ITC. 둘은 함께 작동해야 성과가 나온다."* (→ [ADR-0001](./docs/adr/0001-aix-tf-ax-part-boundary.md)) |
| *"5 stream 모두에 1년 안에 적용 가능한가?"* | *"1년차는 4 stream + Enabler PoC. 5 stream 전부는 3~5년 horizon."* (→ Ch.7) |
| *"DT 사업부가 개발 인력 파견해주는데 왜 ITC가 자체 보유?"* | *"ITC는 자체 broker(AX Part) + 실행 TF로 자력 확산 단위. DT 파견은 옵션으로 보존하되 default 아님."* (→ ADR-0001) |
| *"암묵지 순환고리는 추상적이지 않은가?"* | *"1년차 PoC #5 — 1개 영역에서 캡처→RAG→정제 사이클 1회전 검증. 3년차 정례화."* (→ Ch.7 #5) |

### [SLOT]
- 5 stream 아이콘 디자인 — Appendix E 용어집 아이콘과 통일 필요
- 3-tier 다이어그램이 너무 조직도처럼 보이면 정치적 부담 — 추상화 수준 검토

---

## 슬라이드 간 흐름

```
Slide A (Vision)               Slide B (왜 ITC인가)
─────────────                   ─────────────────
"우리는 이렇게 진화한다"   →    "그래서 왜 우리여야 하는가"
Passive→Active→Proactive       5 stream × 현장 결합
ITC의 미래상                    ITC의 차별점
```

A가 *"무엇을"*, B가 *"왜 우리가"*. 청자는 A에서 *"이건 듣고 싶다"*, B에서 *"이건 들어야 한다"* 를 느낌.

---

## Round 4 후속 검토 포인트

1. **5 stream 아이콘 디자인 통일** — Vision 슬라이드부터 Quick Win 슬라이드까지 일관 사용
2. **3-tier 다이어그램의 추상화 수준** — *"AX Part 권한 과대"* 정치적 우려 ↔ *"명확한 분업"* 메시지 강도 trade-off
3. **CEO 예상 반박 4개** — 보고 리허설 시 검증할 핵심 질문
4. **Slide A의 "3년 후 모습"** — picture-in-picture mockup 추가 여부

## 관련 문서
- [itc-aix-roadmap-outline.md](./itc-aix-roadmap-outline.md) Ch.2 (high-level 구조)
- [CONTEXT.md](./CONTEXT.md) — 5 stream / 거버넌스 3-tier / 암묵지 순환고리 용어
- [ADR-0001](./docs/adr/0001-aix-tf-ax-part-boundary.md) — Platform/Consumer 분업의 정식 결정 문서
- [quick-win-cards.md](./quick-win-cards.md) — Ch.7 카드 (Slide B 예상 반박 #4와 연결)
