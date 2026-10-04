---
title: PowerPoint 슬라이드를 PNG로 내보내기
tags: [python, powerpoint, windows]
aliases: [DRM PPTX extraction]
document_type: learning
reviewed_on: 2026-10-04
verification_status: source-verified-windows-unverified
---

# PowerPoint 슬라이드를 PNG로 내보내기

> Windows PowerPoint가 열 수 있는 PPTX의 각 슬라이드를 PNG 파일로 저장하는 COM 자동화 예제다.

## 목적과 작동 방식

슬라이드를 이미지로 보관하거나 사람이 읽을 자료를 만들 때 사용한다. 별도 PPTX에 슬라이드를 복사하거나 텍스트·도형을 추출하지 않는다. `main.py`가 입력 파일을 찾고 `export_slides.py`가 파일마다 PowerPoint 인스턴스를 열어 `Slide.Export(..., "PNG")`를 호출한 뒤 닫는다.

프로젝트 이름의 DRM은 입력 문서의 맥락이다. 사용자의 PowerPoint 계정과 문서 정책이 열기·내보내기를 허용하는 경우에만 동작하며 DRM 해제 또는 모든 보호 문서의 성공을 보장하지 않는다. 실제 사내 환경에서는 미확인이다.

PNG는 텍스트가 많은 슬라이드의 저장에 사용하는 형식이다. 출력 크기는 코드가 명시하지 않아 PowerPoint의 실제 내보내기 환경을 확인해야 한다. [Microsoft Slide.Export 공식 문서](https://learn.microsoft.com/en-us/office/vba/api/powerpoint.slide.export), 문서 갱신 2022-08-13, 확인일 2026-10-04. 공식 문서는 이미지 export 계약의 근거이며 DRM 문서의 권한이나 성공을 검증한 자료는 아니다.

## 요구 환경과 설치

- Windows와 설치된 Microsoft PowerPoint.
- Python 3.10 이상, `requirements.txt`의 COM 관련 의존성.
- 내보내기가 허용된 입력 PPTX와 충분한 디스크 공간.

모듈 디렉터리에서 실행한다.

```bash
pip install -r requirements.txt
```

## 사용 방법

`input/` 아래 모든 `*.pptx`:

```bash
python main.py
```

특정 파일:

```bash
python main.py input/drm_file.pptx
```

여러 파일 또는 glob 패턴:

```bash
python main.py input/a.pptx "input/team-*.pptx"
```

다른 출력 루트:

```bash
python main.py "input/*.pptx" -o captured_slides
```

## 출력과 재실행 조건

```text
output/
├── file_a/
│   ├── slide_001.png
│   ├── slide_002.png
│   └── ...
└── file_b/
    ├── slide_001.png
    └── ...
```

폴더 이름은 PPTX 확장자를 제외한 이름이다. 입력 경로가 달라도 이름이 같으면 같은 출력 폴더를 사용한다. 내보내기 준비 단계에서 그 폴더의 기존 `slide_*.png`를 지우므로 보존할 결과는 별도 출력 루트에 둔다. 번호 자릿수는 최소 3자리이며 슬라이드 수가 더 크면 늘어난다.

슬라이드별 실패는 경고 후 다음 슬라이드로 계속한다. 파일 실패도 로그를 남기고 다음 파일로 진행하며 최종 종료코드가 0일 수 있다. 따라서 종료코드만으로 전체 성공을 판단하지 말고 원본 슬라이드 수와 생성 파일 수, WARN/ERROR 로그를 대조한다.

## 파일과 검증 상태

- [main.py](main.py): 입력 해석과 실행 흐름.
- [export_slides.py](export_slides.py): COM 열기·export·정리.
- `requirements.txt`, `input/`, `output/`: 설치·입출력 환경.

2026-10-04 소스와 공식 메서드 계약을 대조했다. macOS에서 Windows COM·PowerPoint·DRM 내보내기는 실행하지 않았다. 지원되는 실제 PowerPoint 빌드와 정책 조건은 미확인이다.
