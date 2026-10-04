---
tags: [legal, reverse-engineering, vendor, dmca, nda, cd-sem]
level: beginner
last_updated: 2026-07-10
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: work_reference
---

# 역공학 적법성 & 첫 수순

> 법률 자문 아님. 구체 사안은 사내 법무·벤더 apps 팀에 확인. 아래는 일반 지형과 실무 권장 순서.

## 왜 먼저 읽나 (Why)

이 문서는 출력 데이터의 파싱, 프로그램 코드의 관찰/디컴파일, 접근 통제 우회를 구분하고 사용할 수 있는 공식 export 경로를 조사하는 참고다. 허용 여부는 행위·이용 권한·관할·계약·다른 권리에 따라 달라진다. “역공학은 대개 허용”이나 “계약이 법보다 항상 우선”으로 일반화하지 않는다. 실제 회사 계약과 한국 등 적용 관할의 결론은 미확인이다.

## 지형 (What)

- **미국 — 17 U.S.C. §1201(f)**: 프로그램 사본을 사용할 권리를 적법하게 얻은 사람이 독립 제작 프로그램과의 상호운용에 필요한 요소를 식별·분석하는 제한된 조건의 예외다. 필요한 정보가 이전에 쉽게 제공되지 않았고 행위가 저작권 침해가 아니어야 한다. 도구 개발·정보 제공에도(f)(2)·(3)의 필요성/목적/다른 법 준수 조건이 있다. 상호운용이라는 표어만으로 모든 디컴파일·우회·배포가 허용되지 않는다. [미국 저작권청 조문(f)(1)~(4)](https://www.copyright.gov/title17/92chap12.html),2026-10-04 확인.
- **EU — Directive2009/24/EC**: 원문의Art.5(3) 관찰·연구·시험은 이용권자가 허용받은 로딩/표시/실행/전송/저장 행위 중 원리 파악을 하는 조건이다. Art.6은 독립 프로그램 상호운용에 필요한 코드 재현/형태 변환에 조건을 둔다. Art.8은Art.6과Art.5(2)/(3)에 반하는 계약을 무효로 정한다. 따라서 “계약이 예외를 항상 무력화”한다는 설명은 부정확하다. [EUR-Lex 원문](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32009L0024)의공식 검색 색인에서Art.5(3)/8과[공식 요약](https://eur-lex.europa.eu/legal-content/EN/LSU/?uri=celex:32009L0024)의상호운용 조건을 확인했다. 원문과 공식 요약 직접 조회는 JavaScript/robot 안내였고 위 설명은 공식 검색 색인 범위다.2009지침 전문·회원국 적용/개정은 미확인이다. 확인일2026-10-04.
- **CJEU C-13/20, Top System,2021-10-06**: 판결은Directive91/250/EEC Art.5(1)의 적법 구매자 오류 수정 디컴파일을 해석했다. Art.6의 상호운용 조건을 그대로 충족해야 하는 것은 아니지만 오류 수정에 필요한 범위와 적용 가능한 계약 조건을 따른다. 계약이 모든 오류 수정 가능성을 막을 수는 없지만 수행 방식은 계약으로 정할 수 있다고 설명한다(¶63~69,판결 주문). “라이선스가 금지해도 항상 디컴파일 허용”으로 줄이지 않는다. [공식 판결](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62020CJ0013),2026-10-04 확인.

미국 예외가EU보다 넓다는 원문 비교는 근거가 확인되지 않아 철회했다. NDA/구매계약/EULA는 실제 조항을 확인할 자료이며 저작권·영업비밀·접근 통제 등의 결론을 대신하지 않는다. 위 외국 법령/판결을 한국 회사의 구체 허용 근거로 전용하지 않는다.

## 실무 권장 순서 (How)

1. **벤더 apps 엔지니어에게 먼저 요청** — (a) 포맷 스펙, (b) SDK, (c) 문서화된 **export**(CSV/XML/DB), (d) **EDA/Interface A** 피드. 지원되는 데이터 범위·판본·권한·라이선스·출력 정확도를 확인한다. 계약상 안전이나 더 빠른 결과를 자동 보장하지 않는다.
2. **자사 데이터를 자사가 파싱**(내가 생성한 TIFF 읽기)은 그들 SW 바이너리를 디컴파일하는 것과 프로그램 자체 디컴파일과 행위가 다르다. 파일에 접근/분석할 권한, 데이터에 포함된 비밀/제3자 권리, PO/NDA 범위를 확인하고 어느 쪽도 소유만으로 적법성을 확정하지 않는다.
3. **디컴파일이나 clean-room 전에는 법무 + 벤더를 개입**시키고, 실제 목적·대상·필요 범위를 문서화한다. 실제 오류 수정/데이터 해석 목적을 형식적으로 상호운용이라고 바꾸지 않는다.

## CD-SEM 맥락 적용

- **이미지 파일**: 권한이 확인된 출력 TIFF의 기존 reader 적용 가능성을 조사한다. 이미지 생성 주체나 확장자만으로 법적 허용/메타데이터 지원을 확정하지 않는다. 이 문서는 TIFF 지원을 검사한 결과가 아니다.
- **측정결과·recipe binary**: EDA/host export가 필요한 값과 의미를 제공하는지 먼저 조사한다. EDA(Equipment Data Acquisition,Interface A)는 장비 데이터 수집 표준 묶음이며 특정 장비의 모든 recipe/원시 결과를 제공한다는 보증이 아니다. 실제 구현·라이선스·공급 데이터는 미확인이다. [02-cd-sem-formats §4](./02-cd-sem-formats.md) 참조.
- 결론: 필요한 값을 의미·권한·정확도까지 충족하는 공식 export/SDK가 있으면 우선 검토한다. 없다면 형식 조사 범위와 승인 근거를 기록한다. 실제 장비에 대한 실행 지시는 아니다.

## 참고 자료 (References)

- [미국 저작권청 Chapter12,§1201(f)](https://www.copyright.gov/title17/92chap12.html). OLRC uscode 직접 조회는 maintenance 안내였고 공식 대체 조문으로 확인했다.
- [Directive2009/24/EC](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32009L0024), [EUR-Lex 공식 요약](https://eur-lex.europa.eu/legal-content/EN/LSU/?uri=celex:32009L0024). 전문 직접 조회 미완료, 검색 색인/요약 범위와 구분한다.
- [C-13/20,ECLI:EU:C:2021:811](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62020CJ0013),2021판결과 그대상91/250/EEC를 현재 회사 계약으로 바꾸지 않는다.
- [SEMI Information and Control의 EDA 표준 목록](https://www.semi.org/en/products-services/standards/information_and_control). 공개 목록 확인은 유료 규격 전문·Freeze별 구현·회사 장비 지원 확인이 아니다.

## 검토 결과

2026-10-04 원래 절/관할/상호운용·오류수정·벤더/export 조사·CD-SEM 맥락을 보존하며 무근거 위험 순위/광범위 허용/계약 우선 비교를 정정했다. 법률·계약은 로컬 실행으로 검증할 대상이 아니며 장비/SW 조사와 외부 연락은 실행하지 않았다. 구체 관할/계약 해석과 유사 문서의 완전 통합은 Claude 연결 실패로 보류했다. [정리 기록](../organization-log.md)에 근거·조회 실패·미확인을 남긴다.
