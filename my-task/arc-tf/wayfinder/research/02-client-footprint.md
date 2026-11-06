---
tags: [my-task, arc-tf, research, screen-capture, windows, footprint]
note_kind: "조사 티켓"
last_updated: 2026-11-06
status: 조사 완료 — 실측 전 추정치 포함
---

# 엔지니어 PC에서 상시 관찰할 때 자원을 얼마나 쓰는가

> 하나의 창(원격 제어 뷰어 또는 웹 VDI를 띄운 브라우저)을 하루 종일 캡처해 화면 변화만 서버로 보내는 Windows 프로그램이 PC 자원을 얼마나 쓰는지, 어떤 캡처 방식을 써야 하는지 정리함. PC에서는 추론을 하지 않음. 모니터는 FHD(1920x1080) 기준임.

## 결론

- **캡처 방식**: Windows.Graphics.Capture(WGC)의 창 단위 캡처를 기본으로 함. 창을 직접 지정해 가려진 창도 캡처되고, GPU 텍스처로 받기 때문에 하드웨어 가속 브라우저에서도 검은 화면 문제가 적음. DXGI Desktop Duplication은 모니터 전체를 복제하므로 대상 창이 가려지면 가린 창이 찍힘. GDI BitBlt/PrintWindow는 예비 수단으로만 둠.
- **노란 테두리**: `GraphicsCaptureSession.IsBorderRequired`로 끌 수 있음. 이 속성은 Windows 10 build 20348(Server 2022) 계열 API 계약 v12.0 이상에서 존재하며, 클라이언트 OS로는 Windows 11부터 해당함. 문서상 절차는 패키지 매니페스트에 `graphicsCaptureWithoutBorder` 선언 후 사용자 동의 프롬프트(`RequestAccessAsync(Borderless)`)를 거치는 방식임. Windows 10(19041~19045) 클라이언트에서는 테두리를 끌 수 없음.
- **변화 감지**: GPU에서 1/4~1/8로 축소한 회색조 프레임의 타일 단위 픽셀 차이를 기본으로 함. 프레임당 CPU 비용은 1ms 미만 수준(추정)임. 퍼셉추얼 해시는 숫자 한두 자리 변경을 놓치기 쉬워 보조로만 씀. dirty rect(WGC는 Windows 11 24H2 SDK부터, DXGI는 모니터 전체 기준)는 "아무 변화 없음"을 싸게 거르는 1차 필터로만 씀.
- **권장 예산(추정치)**: 평균 CPU 전체 대비 2% 이하, 순간 최대 1코어 50% 이하 1초 미만, 작업 메모리 150MB 이하, GPU 사용률 2% 이하, 업로드 평균 300kbps 이하·순간 5Mbps 이하, 오프라인 대기열 디스크 1GB 상한.
- **비교 대상**: 상시 화면 기록 도구인 screenpipe는 공식 문서에서 CPU 약 5~10%, RAM 약 600MB를 "일반적"이라고 밝힘. 이 프로그램은 OCR·추론이 없으므로 그 1/3 이하를 목표로 잡는 것이 타당함.
- **주요 주의점**: (1) VDI·원격 클라이언트가 `SetWindowDisplayAffinity`로 화면 캡처 보호를 켜 두면 어떤 공개 API로도 내용이 안 찍힘. (2) 최소화된 창은 새 프레임이 오지 않음(추정, 실측 필요). (3) 노란 테두리 제거는 OS 버전과 배포 방식(패키지 여부)에 묶임. (4) 수치 대부분이 추정이므로 대표 PC 2~3종에서 실측해 예산을 확정해야 함.

## 왜 필요한가? (Why)

엔지니어 PC는 장비 원격 제어와 문서 작업을 동시에 하는 업무용 PC입니다. 관찰 프로그램이 CPU나 네트워크를 눈에 띄게 쓰면 원격 화면이 끊기거나 입력이 늦어지고, 그 순간부터 현장은 프로그램을 끄고 싶어 합니다. 그래서 캡처 방식과 변화 감지 방식을 정하기 전에 "얼마까지 써도 되는가"라는 예산을 먼저 정해야 합니다.

## 핵심 개념 (What)

### 1. 캡처 방식 비교

| 항목 | Windows.Graphics.Capture (WGC) | DXGI Desktop Duplication | GDI BitBlt / PrintWindow |
|---|---|---|---|
| 캡처 단위 | 창(HWND) 또는 모니터 | 모니터(출력) 전체 | 창 DC 또는 화면 DC |
| 최소 OS | 창 지정 `CreateForWindow`는 Windows 10 1903(18362) | Windows 8 | Windows XP 이상 |
| 결과 형식 | D3D11 텍스처(GPU) | DXGI 표면(GPU), B8G8R8A8 | CPU 비트맵 |
| CPU 비용 | 낮음. CPU로 옮기는 복사(FHD 1장 약 8.3MB)만 CPU 부담 | 낮음. 동일하게 읽어오기 비용만 있음 | 높음. 매 호출이 동기식이고 대상 앱이 다시 그리는 경우도 있음 |
| 가려진 창 | 캡처됨(창의 합성 결과를 받음) | 가린 창이 그대로 찍힘 | BitBlt는 가려진 부분이 깨짐. PrintWindow는 대상 앱이 그리기에 응해야 함 |
| 최소화된 창 | 새 프레임이 오지 않음(추정, 실측 필요) | 화면에 없으므로 불가 | PrintWindow는 제목 표시줄만 나오는 사례가 보고됨 |
| 하드웨어 가속 브라우저 | 정상 캡처가 일반적임 | 화면에 보이는 대로 캡처됨 | BitBlt는 검은 화면이 흔함. `PW_RENDERFULLCONTENT`(비공식 플래그)로 우회하는 사례가 있음 |
| 변경 영역 정보 | `Direct3D11CaptureFrame.DirtyRegions` (Windows 11 24H2 SDK, build 26100부터) | `GetFrameDirtyRects`, `GetFrameMoveRects` | 없음 |
| 갱신 주기 조절 | `MinUpdateInterval` (build 26100부터) | `AcquireNextFrame` 타임아웃으로 조절 | 호출 주기로 조절 |
| 사용자 표시 | 노란 테두리(끌 수 있음, 아래 참고) | 없음 | 없음 |

근거와 해석은 다음과 같습니다.

- Desktop Duplication은 "데스크톱 이미지에 대한 프레임 단위 갱신"을 DXGI 표면으로 주는 API이며 형식은 항상 `DXGI_FORMAT_B8G8R8A8_UNORM`임. 단위가 모니터이므로 창 하나만 보려면 창 영역을 잘라야 하고, 그 창이 다른 창에 가려지면 가린 창이 찍힘.
- WGC는 "디스플레이 또는 애플리케이션 창"에서 프레임을 받는 API이며, 시스템이 캡처 중인 항목 둘레에 노란 테두리를 그린다고 문서에 명시됨.
- `PrintWindow`는 대상 창을 소유한 앱이 `WM_PRINT`를 처리해 그리는 동기 함수라서, 문서가 "UI 스레드에서 부르면 앱이 멈춘 것처럼 보일 수 있다"고 경고함. 상시 캡처의 주 경로로는 부적합함.
- OBS의 창 캡처 문서는 BitBlt를 "덜 효율적이지만 특정 프로그램에서 도움이 될 수 있다"고 설명함. 즉 업계 표준 도구도 WGC 계열을 기본으로, BitBlt를 호환용 예비로 둠.

### 2. 노란 캡처 테두리를 끄는 조건

| 조건 | 내용 |
|---|---|
| 속성 | `GraphicsCaptureSession.IsBorderRequired = false` |
| API 도입 | Windows 10 version 2104(build 20348), UniversalApiContract v12.0 |
| 사용자 동의 | `GraphicsCaptureAccess.RequestAccessAsync(GraphicsCaptureAccessKind.Borderless)` 호출 시 프롬프트 표시. 거부하면 `false` 설정은 성공하지만 무시되고 테두리가 그려짐 |
| 매니페스트 | `graphicsCaptureWithoutBorder` 기능 선언 필요(`uap11:Capability`). 즉 패키지(MSIX) 앱 기준 절차임 |
| 다른 앱 영향 | 같은 창을 다른 앱이 `IsBorderRequired = true`로 캡처 중이면 테두리 표시 |
| 창 프로그래밍 지정 | `WindowId`로 캡처 항목을 직접 만드는 경로는 `graphicsCaptureProgrammatic` 기능이 필요함. 비패키지 Win32는 `IGraphicsCaptureItemInterop::CreateForWindow`(1903+)를 씀 |

비패키지 Win32 앱에서 프롬프트 없이 테두리를 끄는 방법이 커뮤니티에 보고되어 있지만 Microsoft 문서에는 없는 동작입니다. 향후 OS 업데이트에서 막힐 수 있으므로 설계 근거로 삼지 않고, 사내 배포를 MSIX로 하고 최초 1회 동의를 받는 문서상 절차를 기준으로 잡는 편이 안전합니다. 테두리는 오히려 "관찰 중"을 엔지니어에게 알리는 표시로 남겨 두는 선택지도 있습니다.

### 3. 캡처가 아예 안 되는 경우

- 원격 클라이언트나 VDI 클라이언트가 `SetWindowDisplayAffinity(WDA_MONITOR 또는 WDA_EXCLUDEFROMCAPTURE)`를 설정하면 캡처 결과에서 창 내용이 빠지거나 창 자체가 사라짐. `WDA_EXCLUDEFROMCAPTURE`는 Windows 10 2004부터 지원함.
- Microsoft Recall 관리 문서도 Azure Virtual Desktop의 화면 캡처 보호처럼 "클라이언트가 화면 캡처 보호를 구현하면 저장되지 않는다"고 밝힘. 웹 VDI·원격 뷰어 정책에 캡처 보호가 켜져 있는지가 이 프로그램의 성립 조건임.

### 4. 변화 감지 방식 비교

| 방식 | 프레임당 비용(추정) | 장점 | 약점 |
|---|---|---|---|
| 전체 해상도 픽셀 차이(CPU) | FHD 2장 비교 시 메모리 약 16MB 읽기, 수 ms | 가장 정확함 | 매 프레임 GPU→CPU 전체 복사 필요 |
| 축소 프레임 타일 차이(GPU 축소 후 CPU 비교) | 1/4 축소(480x270) 회색조 약 130KB 비교, 1ms 미만 | 싸고 작은 글자 변화도 타일 단위로 잡힘 | 임계값 조정 필요. 커서·시계 깜빡임은 마스크 처리 필요 |
| 퍼셉추얼 해시(aHash·dHash·pHash) | 해시 자체는 1ms 미만. 축소 비용이 대부분 | 압축 잡음·미세 이동에 강함 | 64비트 요약이라 숫자 한두 자리 변경을 놓치기 쉬움. "값이 바뀌었는가"를 봐야 하는 장비 화면에는 부적합 |
| DXGI dirty rects | 메타데이터 조회만, 사실상 0 | OS가 바뀐 영역을 알려 줌 | 모니터 전체 기준이라 다른 창 변화도 섞임. 영역이 합쳐지며 실제보다 넓게 보고될 수 있다고 문서가 명시함 |
| WGC DirtyRegions | 메타데이터 조회만 | 대상 창 기준 | build 26100(Windows 11 24H2) 이상만 해당. 원격 뷰어는 화면 전체를 다시 그리는 경우가 많아 영역이 넓게 나올 수 있음(추정) |

권장 조합은 "dirty 정보가 비어 있으면 건너뜀 → 축소 회색조 타일 차이로 판정 → 일정 시간(예: 1~2초) 화면이 안정되면 원본 해상도로 1장 인코딩해 전송"입니다. 화면이 연속으로 바뀌는 동안 매 프레임을 보내지 않고 안정된 마지막 화면을 보내면 업로드량이 크게 줄어듭니다.

### 5. 비교 가능한 상시 캡처 도구의 공개 수치

| 도구 | 공개 수치 | 출처 성격 | 이 프로그램과의 차이 |
|---|---|---|---|
| screenpipe | RAM 약 600MB, CPU 약 5~10%, 저장 월 5~10GB. 이벤트 기반 캡처, 동일 프레임 중복 제거, JPEG 저장 | 공식 문서(자체 보고). 측정 조건은 MacBook Pro M3 32GB 또는 400달러급 Windows 노트북이라고만 밝힘 | 전체 화면 + OCR·검색 인덱스를 PC에서 처리함. 이 프로그램보다 무거운 것이 정상 |
| Microsoft Recall | 최소 요건 16GB RAM, 8 논리 프로세서, 40 TOPS NPU, 저장 공간 256GB(활성화에 여유 50GB, 25GB 미만이면 저장 중지). 스냅샷 저장 상한 10~150GB | Microsoft 공식 관리 문서. CPU 사용률 수치는 공개하지 않음 | 화면이 이전 스냅샷과 달라질 때 주기적으로 저장하고 NPU로 로컬 분석함. 변화 기반 캡처라는 점은 같음 |
| OBS Studio 창 캡처 | 공식 CPU 수치 없음. 캡처 방식 선택지(자동, BitBlt, Windows 10 1903+)만 설명 | 공식 KB | 초당 30~60프레임 영상 인코딩이 목적이라 비교 기준으로는 과함 |
| ActivityWatch | 공식 자원 수치 없음. 활성 창 제목·URL만 수집 | 공식 문서·사용자 보고 | 화면 픽셀을 캡처하지 않으므로 하한선 참고용 |
| DXcam(Desktop Duplication 파이썬 라이브러리) | 평균 약 239fps 캡처(5900X+RTX 3090, 자체 측정) | 프로젝트 README 자체 보고 | 캡처 API 자체가 초당 수백 장도 가능할 만큼 싸다는 근거로만 씀 |

공개 수치가 있는 도구는 screenpipe뿐이고 그것도 자체 보고입니다. 따라서 이 표는 "상시 화면 기록 도구가 OCR까지 해도 CPU 한 자릿수 %, RAM 수백 MB"라는 상한 감각을 주는 용도이며, 이 프로그램의 예산은 실측으로 확정해야 합니다.

## 어떻게 사용하는가? (How)

### 6. 주기·해상도·압축별 CPU·네트워크 추정 (모두 추정치)

전제는 다음과 같습니다. 대상은 FHD 전체 화면 크기의 창 1개이고, 장비 UI·원격 데스크톱처럼 글자와 단색 면이 많은 화면입니다. JPEG은 libjpeg-turbo 계열, WebP는 libwebp 기본 설정 근처를 가정합니다.

**프레임 1장 크기(추정)**

| 해상도 | JPEG q75 | JPEG q90 | WebP q75 | 비고 |
|---|---|---|---|---|
| 1920x1080 | 150~300KB | 250~500KB | 100~200KB | WebP는 같은 SSIM에서 JPEG보다 25~34% 작다는 Google 연구 결과를 적용함 |
| 1280x720 | 70~150KB | 120~250KB | 50~100KB | 작은 글자 판독성 저하 우려 |
| 960x540 | 40~80KB | 60~130KB | 30~60KB | 서버 측 글자 판독에는 부족할 가능성 큼 |

**인코딩 CPU(추정)**

| 해상도 | JPEG (SIMD 가속) | WebP (method 4 기본) | WebP (method 0~2) |
|---|---|---|---|
| 1920x1080 | 1코어 기준 10~25ms | 60~150ms | 30~60ms |

libjpeg-turbo 공식 성능 페이지는 절대 처리량 대신 기존 libjpeg 대비 압축 3.5~6.9배 같은 상대 수치만 제시하므로, 위 ms 값은 실측으로 바꿔야 할 추정입니다. WebP는 cwebp 문서가 `-m`이 "인코딩 속도와 크기·품질의 절충"을 조절한다고 밝히므로, 상시 캡처에서는 낮은 method를 쓰는 편이 맞습니다.

**업로드량(추정, FHD JPEG q75 1장 약 250KB 가정)**

| 전송 방식 | 전송 빈도 | 평균 업로드 | 하루 8시간 |
|---|---|---|---|
| 고정 1초 주기, 변화 무관 | 1장/1초 | 약 2Mbps | 약 7GB |
| 고정 5초 주기 | 1장/5초 | 약 400kbps | 약 1.4GB |
| 변화 감지 + 안정화 후 전송, 변화가 분당 6회 | 1장/10초 | 약 200kbps | 약 0.7GB |
| 변화 감지 + 안정화 후 전송, 변화가 분당 2회 | 1장/30초 | 약 70kbps | 약 0.24GB |

**CPU 총량(추정, 8 논리 코어 PC 기준)**

| 구성 | 평균 CPU(전체 대비) |
|---|---|
| WGC 프레임 수신 + GPU 축소 + 타일 차이, 2fps 감시 | 0.2~0.5% |
| 위 + JPEG 인코딩 1장/10초 | 0.3~0.6% |
| 위 + WebP(method 4) 인코딩 1장/10초 | 0.5~1% |
| BitBlt로 전체 해상도 CPU 복사 + CPU 전체 비교, 2fps | 2~5% |

감시는 초당 1~2장이면 충분합니다. 원격 장비 화면은 사람이 조작하는 속도로 바뀌므로 초당 수십 장을 볼 필요가 없고, WGC의 `MinUpdateInterval`(build 26100 이상)이나 자체 타이머로 감시 주기를 묶어 두면 됩니다.

### 7. 권장 예산과 근거 (추정치)

| 항목 | 권장 상한 | 근거 |
|---|---|---|
| 평균 CPU | 전체 대비 2% 이하 | 위 추정상 정상 동작은 1% 안팎임. 두 배 여유를 두되 OCR까지 하는 screenpipe(5~10%)의 1/3 이하로 둠 |
| 순간 CPU | 1코어 50% 이하, 1초 미만 | FHD JPEG 1장 인코딩이 수십 ms 수준이므로 충분함. 인코딩 스레드는 낮은 우선순위로 둠 |
| 작업 메모리 | 150MB 이하 | FHD BGRA 1장 약 8.3MB. 캡처 버퍼 2~3장 + 축소본 + 인코딩 버퍼 + 런타임을 합쳐도 100MB 안팎임. screenpipe 600MB의 1/4 수준 |
| GPU | 사용률 2% 이하, 전용 메모리 50MB 이하 | 축소 셰이더와 텍스처 몇 장 수준. 원격 뷰어·브라우저 렌더링과 GPU를 나눠 쓰므로 작게 잡음 |
| 업로드 평균 | 300kbps 이하 | 변화 기반 전송이면 70~200kbps로 추정됨. 원격 제어 세션 자체의 대역폭을 침범하지 않는 수준 |
| 업로드 순간 | 5Mbps 이하로 속도 제한 | 연속 변화 구간에서도 원격 세션 품질 저하를 막음 |
| 디스크 | 오프라인 대기열 1GB 상한, 오래된 것부터 삭제 | 서버 장애 시 PC 디스크를 채우지 않게 함. Recall도 상한 도달 시 오래된 스냅샷부터 삭제하는 방식을 씀 |
| 전송 빈도 상한 | 1장/2초 | 화면이 계속 바뀌는 구간의 폭주 방지 |

자기 보호 장치도 같이 두어야 합니다. 프로세스가 자기 CPU·메모리를 주기적으로 재서 상한을 넘으면 감시 주기를 늘리고, 그래도 넘으면 캡처를 멈추고 상태를 서버에 알리는 방식입니다. 예산은 숫자를 정하는 것보다 넘었을 때 스스로 물러나는 동작이 있어야 현장에서 지켜집니다.

### 8. 실측 계획 (예산 확정 전 필요)

- 대표 PC 2~3종(저사양 노트북 포함)에서 원격 뷰어 창과 웹 VDI 브라우저 창 각각 측정.
- 측정 항목: 프로세스 CPU %(Windows 성능 카운터), 작업 집합 MB, GPU 엔진 사용률(작업 관리자 GPU 열), 업로드 kbps, 프레임 1장 크기 분포.
- 확인 항목: 대상 VDI·원격 클라이언트의 캡처 보호 여부, 최소화·잠금 화면·화면 보호기 상태에서의 프레임 수신 여부, 노란 테두리 제거 가능 OS 비율.

## 참고 자료 (References)

모든 링크의 접근일은 2026-11-06입니다.

- Microsoft Learn, Screen capture (Windows.Graphics.Capture): https://learn.microsoft.com/en-us/windows/uwp/audio-video-camera/screen-capture
- Microsoft Learn, GraphicsCaptureSession.IsBorderRequired: https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturesession.isborderrequired
- Microsoft Learn, GraphicsCaptureAccess.RequestAccessAsync: https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscaptureaccess.requestaccessasync
- Microsoft Learn, GraphicsCaptureAccessKind: https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscaptureaccesskind
- Microsoft Learn, GraphicsCaptureSession.MinUpdateInterval: https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturesession.minupdateinterval
- Microsoft Learn, Direct3D11CaptureFrame.DirtyRegions: https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.direct3d11captureframe.dirtyregions
- Microsoft Learn, IGraphicsCaptureItemInterop::CreateForWindow: https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.capture.interop/nf-windows-graphics-capture-interop-igraphicscaptureiteminterop-createforwindow
- Microsoft Learn, App capability declarations (graphicsCaptureWithoutBorder, graphicsCaptureProgrammatic): https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/app-capability-declarations
- Microsoft Learn, Desktop Duplication API: https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api
- Microsoft Learn, PrintWindow function: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-printwindow
- Microsoft Learn, SetWindowDisplayAffinity function: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowdisplayaffinity
- Microsoft Learn, Manage Recall for Windows clients: https://learn.microsoft.com/en-us/windows/client-management/manage-recall
- OBS Knowledge Base, Window Capture Sources: https://obsproject.com/kb/window-capture-sources
- screenpipe 공식 문서, Architecture: https://docs.screenpipe.com/architecture
- ActivityWatch 공식 문서, Watchers: https://docs.activitywatch.net/en/latest/watchers.html
- DXcam README(자체 벤치마크): https://github.com/ra1nty/DXcam
- libjpeg-turbo, Performance: https://www.libjpeg-turbo.org/About/Performance
- Google, WebP Compression Study: https://developers.google.com/speed/webp/docs/webp_study
- Google, cwebp 문서: https://developers.google.com/speed/webp/docs/cwebp
- 비공식 참고(설계 근거로 쓰지 않음), 비패키지 앱 테두리 제거 보고: https://handmade.network/forums/t/8701-wcap/2

## 관련 문서

- [플랫폼 1차 PRD](../../06-platform-prd.md)
