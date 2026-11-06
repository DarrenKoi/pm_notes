---
type: research
status: closed
assignee:
blocked_by: []
---

# 엔지니어 PC에서 상시 관찰할 때 자원을 얼마나 쓰는가

## Question

Windows 엔지니어 PC에서 **장비 뷰어 창 하나를 상시 캡처하고 화면이 바뀔 때만 서버로 보내는** 프로그램이 CPU·메모리·네트워크를 얼마나 쓰는가? 엔지니어 작업에 영향을 주지 않으려면 어떤 방식과 한도를 잡아야 하는가?

조건: FHD 모니터, 캡처 대상은 RCS 창 또는 SecureVDI(웹 VDI) 브라우저 창, 추론은 하지 않음(서버에서), 설치형 프로그램.

알고 싶은 것:
1. 창 캡처 방식 비교 (Windows Graphics Capture, DXGI Desktop Duplication, GDI BitBlt): 자원 사용, 가려진 창·최소화 창 캡처 가능 여부
2. 변화 감지 방식 (픽셀 차이, 지각 해시 등)과 비용
3. 비슷한 상시 화면 기록 도구(작업 기록·화면 녹화 도구)의 공개된 자원 사용 수치
4. 캡처 주기·해상도·압축에 따른 CPU·네트워크 어림값과, 엔지니어 PC 영향을 묶어 둘 수 있는 한도 제안 (예: CPU 몇 % 이하)

공개 문서와 공개 측정 자료로 확인한다.

## Resolution

조사 결과: 브랜치 `research/02-client-footprint`의 `my-task/arc-tf/wayfinder/research/02-client-footprint.md` (2026-11-06, 병합하지 않음)

- 캡처 방식: **Windows.Graphics.Capture(WGC) 창 단위 캡처**. 가려진 창도 찍히고, 하드웨어 가속 브라우저(SecureVDI)에서도 검은 화면이 드묾. DXGI Desktop Duplication은 모니터 전체라 가린 창이 찍힘. BitBlt/PrintWindow는 예비용
- 노란 캡처 테두리 끄기: Windows 11부터 가능, MSIX 패키지와 사용자 동의 필요. Windows 10에서는 끌 수 없음
- 변화 감지: GPU로 줄인 회색조 프레임을 타일 단위로 비교하고, 화면이 멈추면 원본 1장 전송. 지각 해시는 숫자 한두 자리 변화를 놓침
- 자원 예산 (추정치, 실측 전): 평균 CPU 2% 이하, RAM 150MB 이하, GPU 2% 이하, 업로드 평균 300kbps 이하, 오프라인 대기열 1GB. 비교 근거는 screenpipe 공식 자료(CPU 5~10%, RAM 약 600MB, PC에서 OCR 수행)의 1/3 이하
- 가장 큰 위험: 원격 제어·VDI 프로그램이 화면 캡처 보호를 켜 두면 어떤 공개 API로도 캡처할 수 없음. 다만 2026-11-02에 로컬 RCS와 SecureVDI 두 경로 모두 캡처가 된다는 것을 확인했으므로, 지금 쓰는 프로그램에서는 걸리지 않음. 원격 제어 프로그램이 업데이트되면 다시 확인해야 함
- 남는 확인: 회사 PC의 Windows 버전(10·11), 대표 PC 2~3종에서 실측 → 새 티켓 "엔지니어 PC 자원 실측과 Windows 버전 확인"
