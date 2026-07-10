---
tags: [legal, reverse-engineering, vendor, dmca, nda, cd-sem]
level: beginner
last_updated: 2026-07-10
---

# 역공학 적법성 & 첫 수순

> 법률 자문 아님. 구체 사안은 사내 법무·벤더 apps 팀에 확인. 아래는 일반 지형과 실무 권장 순서.

## 왜 먼저 읽나 (Why)

역공학은 대개 허용되지만 **계약(NDA/구매계약)이 저작권법보다 더 강하게 묶을 수 있다.** 시작 전에 리스크가 낮은 경로가 있는지부터 확인하는 것이 빠르고 안전하다.

## 지형 (What)

- **상호운용(interoperability) 목적 역공학은 폭넓게 허용, 조건부.**
  - **US — DMCA §1201(f):** 독립 제작 프로그램과의 **상호운용** 달성을 *유일한 목적*으로 하는 기술적 보호 우회·역공학을 명시 허용. 미국 예외는 대체로 EU보다 넓게 해석.
  - **EU — Software Directive 2009/24/EC:** **Art. 5(3)** 적법 사용자가 아이디어/원리 파악을 위해 관찰·연구·시험 허용; **Art. 6** 디컴파일은 *상호운용에 불가결*하고 정보를 달리 얻을 수 없을 때만, 적법 라이선시에 의해. 2021 CJEU는 라이선스가 금지해도 **오류 수정** 목적 디컴파일 허용.
- **그러나 계약이 더 좁게 묶는다.** 장비 **구매계약·EULA·NDA**에 anti-reverse-engineering 조항이 흔하다. 그것이 법정 상호운용 예외를 무력화하는지는 관할·사실관계 의존. Fab에서는 저작권 조문보다 **NDA/PO 조항이 실질 제약**이다.

## 실무 권장 순서 (How)

1. **벤더 apps 엔지니어에게 먼저 요청** — (a) 포맷 스펙, (b) SDK, (c) 문서화된 **export**(CSV/XML/DB), (d) **EDA/Interface A** 피드. 더 빠르고 계약상 안전.
2. **자사 데이터를 자사가 파싱**(내가 생성한 TIFF 읽기)은 그들 SW 바이너리를 디컴파일하는 것과 전혀 다른, 훨씬 낮은 리스크. 그래도 PO/NDA는 확인.
3. **디컴파일이나 clean-room 전에는 법무 + 벤더를 개입**시키고, 목적이 상호운용임을 문서화.

## CD-SEM 맥락 적용

- **이미지 파일**: 자사가 찍은 TIFF를 `tifffile`로 읽는 것 → 리스크 최저. Phase 2로 바로 진행.
- **측정결과·recipe binary**: 파싱 자체는 대개 문제없으나, **EDA/host export가 있으면 그게 정답**이다(구조화·표준·벤더중립). [02-cd-sem-formats §4](./02-cd-sem-formats.md) 참조.
- 결론: 역공학은 export 경로가 없을 때의 수단. 있으면 export를 쓴다.

## 참고 자료 (References)

- 17 U.S.C. §1201(f) (DMCA reverse engineering exception)
- EU Directive 2009/24/EC Art. 5·6 · CJEU C-13/20 (2021, 오류수정 디컴파일)
- SEMI EDA/Interface A: semi.org/en/next-gen-semi-eda-standards
