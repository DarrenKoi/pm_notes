# drm-pptx-extraction

> DRM 보호된 PPTX에서 텍스트와 이미지를 추출하는 도구

## 왜 필요한가? (Why)

회사 문서의 대부분이 DRM 보호 상태이므로 python-pptx로 직접 열 수 없다.
PowerPoint COM 자동화로 DRM PPTX의 슬라이드를 빈 non-DRM PPTX에 복사-붙여넣기한 뒤, python-pptx로 텍스트/이미지를 추출한다.

## 전제 조건

- **Windows** + **Microsoft PowerPoint** 설치 필요 (COM 자동화용)
- Python 3.10+

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 준비

1. `input/` 폴더에 두 파일을 배치:
   - DRM 보호된 PPTX 파일
   - 빈 non-DRM PPTX 파일 (PowerPoint에서 새로 만들기 → 저장)

### Step 0: 가능 여부 테스트 (먼저 실행)

```bash
python test_copy_paste.py input/drm_file.pptx input/empty.pptx
```

이 스크립트는 단계별로 가능 여부를 검증한다:
1. PowerPoint COM으로 두 파일을 열 수 있는지
2. DRM 슬라이드를 복사-붙여넣기할 수 있는지
3. 저장된 파일을 python-pptx로 읽을 수 있는지 (DRM lock 전파 여부)

각 단계에서 `[OK]` / `[FAIL]`로 결과를 표시한다.

### 전체 파이프라인 (copy + extract)

```bash
python main.py full input/drm_file.pptx input/empty.pptx -o output
```

### 추출만 (이미 non-DRM인 경우)

```bash
python main.py extract input/non_drm_file.pptx -o output
```

## 출력

```
output/
├── extraction_result.json    # 슬라이드별 구조화된 데이터
├── copied.pptx               # (full 모드) 복사된 non-DRM PPTX
└── images/
    ├── slide_01_img_01.png
    ├── slide_01_img_02.jpg
    └── ...
```

### JSON 형식

```json
{
  "source": "copied.pptx",
  "total_slides": 10,
  "slides": [
    {
      "slide_number": 1,
      "title": "슬라이드 제목",
      "texts": ["본문 텍스트1", "본문 텍스트2"],
      "images": ["images/slide_01_img_01.png"]
    }
  ]
}
```

## 파일 구조

```
├── main.py            # CLI 진입점 (full / extract 서브커맨드)
├── copy_slides.py     # PowerPoint COM으로 DRM → non-DRM 슬라이드 복사
├── extract.py         # python-pptx로 텍스트/이미지 추출
├── requirements.txt
├── input/             # 입력 파일 배치
└── output/            # 추출 결과
```
