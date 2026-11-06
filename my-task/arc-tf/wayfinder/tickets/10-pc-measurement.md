---
type: task
status: open
assignee:
blocked_by: []
---

# 엔지니어 PC 자원 실측과 Windows 버전 확인

## Question

[PC 자원 조사](02-client-footprint.md)의 예산(평균 CPU 2%, RAM 150MB, 업로드 300kbps 이하)은 추정치다. 회사의 대표 엔지니어 PC에서 실제로 지킬 수 있는가? 회사에서만 할 수 있는 일이라 사람이 한다 (HITL).

확인 목록:
- [ ] 엔지니어 PC의 Windows 버전 (10인가 11인가). 10이면 캡처 테두리를 끌 수 없어 엔지니어 화면에 노란 테두리가 보인다
- [ ] 사양이 다른 대표 PC 2~3종에서, 장비 뷰어 창 하나를 WGC로 캡처하는 간단한 시험 프로그램을 30분 돌려 CPU·RAM·업로드 측정 (로컬 RCS 1종, SecureVDI 1종 이상)
- [ ] 측정 중 엔지니어가 느낀 지연이 있는가
- [ ] 회사 PC에 MSIX 패키지 설치가 허용되는가

결과로 적을 것: Windows 버전, PC별 실측값, 예산을 지켰는가. 2027 통과 기준의 "PC 자원 한도 준수" 지표가 이 값에 기댄다.
