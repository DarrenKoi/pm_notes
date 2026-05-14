---
id: ADR-0001
title: AIX TF ↔ ITC AX Part ↔ ITC AI 실행 TF 역할 경계
status: Accepted
date: 2026-05-15
related: [../../CONTEXT.md, ../../itc-aix-roadmap-outline.md]
tags: [governance, organization, itc, aix]
---

# ADR-0001. AIX TF ↔ ITC AX Part ↔ ITC AI 실행 TF 역할 경계

## Status

Accepted (2026-05-15)

## Context

ITC AI/DT 로드맵(CEO 보고용)을 수립하면서, AIX 추진 거버넌스에 **3개의 조직 단위**가 동시에 등장한다는 점이 드러났다:

1. **전사 AIX TF (DT 주도)** — 여러 Biz가 참여하는 전사 TF. DT 사업부가 주도. GPU 투자, 사내 AI 플랫폼·LLM 운영, 과제 우선순위, 각 Biz 과제 관리를 담당하며, 개발 역량이 부족한 Biz에 개발 인력을 파견·협업 형태로 지원한다.
2. **ITC AX Part (기반기술전략팀 내)** — ITC 단위에서 본 로드맵을 owning하는 조직. 단순 기획·관리만 하는 것이 아니라 AI Agent 일부 자체 개발 + ITC AI 실행 TF와의 협업으로 현업 확산을 broker 한다.
3. **ITC AI 실행 TF** — ITC 산하 6팀(MASK기술개발·소재개발·DMI·AT·ME·환경제어)에서 선출된 AI/coding에 비교적 익숙한 인력으로 구성. 각 현업 팀으로 AI 확산을 실행하는 채널. 본 로드맵 Ch.6.1의 *"트랙별 AI 챔피언 10~20명"* 과 동일 인력군.

이 세 단위의 책임 경계가 정의되지 않은 채로 로드맵을 진행하면 다음 문제가 발생한다:

- CEO 보고에서 *"이거 결국 전사 TF 일 아닌가? 왜 ITC가 별도로?"* 라는 반박에 대응 불가
- 1년차 운영 시점에 *"GPU 투자는 TF가 하는데 ITC가 왜 자체 RAG 운영을 결정하느냐"*, 반대로 *"AX Part가 도메인 결합을 결정한다는 권한은 어디서 나왔느냐"* 같은 분쟁
- 산하 6팀 AI 실행 TF가 *AX Part의 협업 대상*인지 *전사 TF의 confidence channel*인지 모호하면 인력·예산이 두 갈래로 흩어짐

세 가지 거버넌스 패턴이 후보로 검토되었다:

| 패턴 | TF 역할 | AX Part 역할 | 경계 |
|------|---------|--------------|------|
| **P1 Platform/Consumer** | 공통 능력·인프라·표준 제공 | 도메인 결합·현장 배포 | "무엇을" vs "어디에" |
| P2 Steering/Execution | 우선순위·예산·KPI 표준 결정 | 실행, 결과 보고 | "왜·얼마나" vs "어떻게" |
| P3 Parallel Scope | Biz 간 횡단 과제 | ITC 단독 과제 | "공유" vs "전용" |

## Decision

**패턴 P1 (Platform/Consumer) 모델을 3-tier로 채택한다.**

```
전사 AIX TF (DT 주도)
   │  공통 능력 제공: GPU·플랫폼·표준 LLM·과제 우선순위·개발 인력 파견 풀
   ▼
ITC AX Part (기반기술전략팀)
   │  ITC 단위 broker: 로드맵 owning + 일부 AI Agent 개발 + 실행 TF 연결
   ▼
ITC AI 실행 TF (산하 6팀 선출 인력)
      현업 확산 채널: 각 팀 도메인 적용·전파
```

**역할 경계 (상세 표):**

| 의사결정 항목 | 전사 AIX TF | ITC AX Part | ITC AI 실행 TF |
|--------------|------------|-------------|---------------|
| 사내 표준 LLM 스택 (Kimi/Qwen/BGE-M3) | ◎ | ○ 참여 | — |
| GPU 투자·플랫폼 운영 | ◎ | ○ ITC 수요 제출 | — |
| 전사 과제 우선순위 | ◎ | ○ ITC 안건 제출 | — |
| ITC 로드맵 (본 문서) | ○ 정합성 확인 | ◎ | ○ 영역 의견 |
| ITC 도메인 데이터 결합 방식 | — | ◎ | ○ 영역 적용 |
| 암묵지 순환고리 설계·운영 | — | ◎ | ○ 캡처 실행 |
| ITC 5 stream KPI 설정 | ○ 표준 가이드 | ◎ | ○ 측정 |
| Quick Win 5선 선정·실행 | — | ◎ 선정 | ◎ 영역 실행 |
| 개발 인력 (AI Agent 개발) | ○ DT 파견 옵션 | ◎ 자체 보유 | — |
| 현업 팀 확산·교육 | — | ○ 표준 제공 | ◎ |
| 보안·정보 유출 정책 | ◎ | ○ 준수 | ○ 준수 |

◎ = 결정권 / ○ = 참여·기여

**핵심 원칙:**
- 전사 AIX TF의 *DT 파견 모델*은 옵션으로 열어두되 **default가 아니다**. ITC는 AX Part + AI 실행 TF로 자체 개발 역량을 보유한 단위로 작동한다.
- AX Part는 *실행조직(P2)*이 아니라 **broker** — 기획·관리·일부 개발을 모두 수행하되, 핵심은 산하 6팀 실행 TF와의 협업으로 AI를 현업에 침투시키는 것.

## Consequences

### Positive
- **CEO 반박에 정합:** *"왜 ITC인가, 전사 TF 일 아닌가"* 질문에 *"공통 능력(TF) ≠ 현장 결합(ITC). 그리고 ITC는 자체 broker + 실행 TF로 자력 확산이 가능한 단위"* 로 답변 가능.
- **Vision과 정합:** Ch.2.2의 *"공통 능력 × 현장 도메인 데이터·암묵지 결합"* 진술이 거버넌스 수준에서 backing 됨.
- **5 stream 차별점 보호:** AX Part가 도메인 결합·암묵지 순환고리 결정권을 가지므로, *"공정·장비 anomaly 영역까지 전사 TF가 직접 들어와 결정"* 같은 침투를 막을 근거가 생김.
- **인력 흐름 명확화:** 산하 6팀에서 선출된 인력이 *AI 실행 TF*라는 명시적 단위로 정례화되어, 기존의 *"AI 챔피언 10~20명"* 같은 모호한 표현이 사라짐.

### Negative / Trade-offs
- **TF 인프라 품질이 ITC 성과의 상한이 됨.** 표준 LLM 스택·플랫폼 안정성이 부족하면 ITC가 자체 우회할 권한이 없음. 1년차에 사내 LLM 품질에 명시적 의존.
- **AX Part가 권한 과대로 비춰질 위험.** 본문 노출은 한 줄 요약만, 상세 경계표는 Appendix F로 격리하는 정책으로 완화.
- **3-tier 협업의 오버헤드.** AX Part가 위로(전사 TF)·아래로(실행 TF) 양방향 broker라 인지 부하와 회의량 증가. AX Part 인력 capacity가 병목이 될 가능성.
- **DT 파견 거부의 정치적 비용.** DT 입장에서는 ITC가 *"우리 모델을 안 받는 Biz"* 로 보일 수 있음. 1년 운영 후 *전사 TF 동등 파트너* 위상이 인정되는지 점검 필요.

### Follow-ups
- 1년차 운영 후 *"전사 TF ↔ AX Part 협업 빈도·만족도"*, *"AX Part ↔ 실행 TF 협업 모델 성숙도"* 를 측정해 P1이 실제로 작동하는지 검증.
- 만약 AX Part의 capacity 병목이 심각하면, 향후 ADR-XXXX로 *"AX Part 인력 확대"* 또는 *"실행 TF 일부 권한 위임"* 의사결정 필요 가능성.
- 본 로드맵 Ch.6.1 *"AI 챔피언"* 명칭은 *"ITC AI 실행 TF"* 로 통일 (이미 v0.3 반영).

## Alternatives considered

- **P2 (Steering/Execution):** AX Part를 단순 실행조직으로 격하. 거버넌스가 단순해지지만 ITC 5 stream 차별점(도메인 결합·암묵지)이 살아남기 어려움. CEO 반박 *"전사 TF만 있으면 되는 거 아닌가"* 에 약함. **기각.**
- **P3 (Parallel Scope):** TF는 횡단 과제, AX Part는 단독 과제로 영역 분리. *"왜 둘 다 필요한가"* 의 답이 약하고, 횡전개 시 매번 협의 부담. **기각.**
- **명시 안 함 (default):** *"협력"* 같은 모호한 표현으로 둠. CEO 첫 반박에 즉시 무너짐. **기각.**

## References

- [../../CONTEXT.md](../../CONTEXT.md) — 전사 AIX TF / ITC AX Part / ITC AI 실행 TF 용어 정의
- [../../itc-aix-roadmap-outline.md](../../itc-aix-roadmap-outline.md) — Ch.5.4 거버넌스 표, Appendix F
- [../../grilling-session-2026-05-14.md](../../grilling-session-2026-05-14.md) — Q5(원안) / Q18(해결) 흐름
