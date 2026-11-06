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
- [ ] 사양이 다른 대표 PC 2~3종에서, 장비 뷰어 창 하나를 캡처하는 시험 프로그램을 30분 돌려 CPU·RAM·업로드 측정 (로컬 RCS 1종, SecureVDI 1종 이상)
- [ ] 측정 중 엔지니어가 느낀 지연이 있는가
- [ ] Thickness 경로(SecureVDI → RCS/Radmin)에서 입력 주입 시험: 리드 본인 PC, 저장 없이 승인 불필요 조작만. 2027-01 Thickness Align Fail 착수 조건 ([2027 분기별 계획](07-phase1-plan.md))
- [ ] 회사 PC에 MSIX 패키지 설치가 허용되는가
- [ ] [observe_probe.py](../tools/observe_probe.py)로 상시 관찰 모드 시험: 30분 동안 저장 장수가 실제 화면 전환 횟수와 비슷한가, 아무것도 안 할 때 장수가 늘지 않는가 (라이브 영역 마스크 적용 전후 비교)
- [ ] 녹화 모드(초당 10장 낱장) 시험: CPU, PC 중복 거름 뒤 남는 비율, 실제 업로드량, 1GB 대기열이 녹화 1건(약 45분)을 버티는가

결과로 적을 것: Windows 버전, PC별 실측값, 예산을 지켰는가. 2027 통과 기준의 "PC 자원 한도 준수" 지표가 이 값에 기댄다.
