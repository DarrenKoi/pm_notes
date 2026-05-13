# ITC AI/DT Roadmap

기반기술센터(ITC)의 3~4년향 AI/DT 로드맵을 수립하기 위한 컨텍스트.
1차 목적은 센터장이 CEO에게 보고할 수 있는 통합 narrative 확보, 2차 목적은 팀 간 AIX/DX 수준 상향 평준화.

## Language

**기반기술센터 (ITC, Infra. Tech Center)**:
양산총괄·제조/기술·연구소 등 다른 Biz의 목표(수율·개발 속도 등)를 지원하는 사업부.
_성격_: 자체 사업 KPI보다 "지원 품질"이 핵심인 연합체(federation) 조직.

**Biz**:
회사 내 사업부 단위. 양산총괄, 제조/기술, 미래기술연구원, ITC, DT 등.
_Avoid_: "사업부"는 한국어 동의어로 허용하나 영어 표기에서는 Biz로 통일.

**DT (사업부)**:
Data 관리를 담당하는 별도 Biz. ITC와 다른 조직.
_Avoid_: "AI/DT 로드맵"의 DT(Digital Transformation)와 혼동 금지.

**AI/DT (로드맵 맥락)**:
이 로드맵의 산출물 범위를 가리키는 용어. AI 적용 + 데이터/디지털 역량(DX)을 합친 개념.
_Avoid_: DT 사업부와 혼용.

**AIX**:
AI를 적용해 업무 자동화·생산성 극대화·노동력 절감을 달성하는 활동/지향.
_관계_: AI/DT 로드맵의 핵심 추진 동력. DX는 그 기반 역량(데이터·시스템).

**AIX TF**:
여러 Biz가 모여 만든 전사 TF. AIX 확산을 위한 기획·개발·과제 관리 수행. AX Part가 참여.

**기반기술전략팀 AX Part**:
ITC 내에서 이 로드맵을 owning하는 조직 단위. 본 문서의 저자 소속.

**ITC 산하 팀**:
MASK기술개발, 소재개발, DMI(Defect Metrology Inspection), AT(Analysis Technology), ME(Machine Engineering), 환경제어팀. 각 팀이 다른 Biz로부터 다른 종류의 의뢰/요청을 받음.

**의뢰/요청 (Request)**:
다른 Biz가 ITC 팀에 보내는 업무 요청. 반복적·정형적인 것부터 비정형 분석 요청까지 스펙트럼 존재. 본 로드맵의 자동화/AI 대응 대상의 기본 단위.
_상태_: 단위(unit of work)·정형성 스펙트럼·현재 채널은 아직 미확정 → grilling 진행 중.

**상향 평준화**:
팀별로 큰 AIX/DX 격차를 좁혀 하위 팀의 역량을 위로 끌어올리는 것. 2차 목적.

## Relationships

- **ITC**는 여러 **팀**의 연합체이며, 각 **팀**은 서로 다른 **Biz**로부터 **의뢰/요청**을 받음
- 각 **팀**은 이미 개별 AI/DT 로드맵(주로 AI Agent 개발 우선순위)을 보유
- **AX Part**가 팀별 로드맵을 통합한 ITC 차원의 로드맵을 작성해야 함
- **AIX TF**는 전사 단위, **AX Part**는 ITC 단위 → 둘의 관계는 아직 미정의

## Thesis (검증 중)

> "ITC는 다른 Biz의 의뢰/요청에 대해 AI를 통해 빠르고 정확한 대응을 달성한다.
> 반복되는 의뢰/요청은 AI로 자동화한다."

이 thesis가 (1) Biz 지원 외에 (2) 공정 기술 개발, (3) 장비 운영/관리 까지 포괄하는지 미확정.

## Flagged ambiguities

- "DT"가 사업부 이름이자 Digital Transformation 약어로 동시 사용됨 → 본 문서에서 사업부는 **DT(사업부)**, 변화 활동은 **DX**로 분리.
- "의뢰/요청"의 정확한 unit·정형성 수준 미정의 → 자동화율 측정 기준 불명확.
- 팀별 개별 로드맵 vs ITC 통합 로드맵의 관계가 roll-up인지 abstract layer인지 미정.

## Example dialogue

> **Daeyoung:** "AX Part 입장에서는, 다른 Biz의 **의뢰/요청**에 AI로 빠르고 정확하게 대응하는 게 핵심이야."
> **Claude:** "그러면 로드맵의 단위 성과는 '대응 TAT 단축'이나 '자동 대응 비율' 같은 형태가 되겠네요. (2) 공정 기술 개발이나 (3) 장비 운영은 별도 트랙으로 빠지나요?"
