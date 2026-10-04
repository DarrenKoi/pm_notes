---
tags: [rag, tokenization, powerpoint, pptx, slides]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [PPTX 추출과 슬라이드 청킹]
---

# PowerPoint 문서 토큰화 전략 (PPTX Tokenization Strategy)

> PPTX의 텍스트·표·발표자 노트와 시각 자료를 구분하고 슬라이드 출처를 유지해 검색 단위를 만든다. 파일 추출·청킹·모델 토큰화는 서로 다른 단계다.

> [!info] 검토 범위 — 2026-10-04
> python-pptx 1.0.2의 생성 PPTX로 재귀 그룹·동일 제목 본문·표/병합·노트·빈 슬라이드·원파일 보존을 확인한다. 공식 rolling 문서 표시 버전은 1.0.0이다. OpenAI SDK 3.24.0 요청은 모의 HTTP 검증이다. LibreOffice 실제 변환·Unstructured 파서·사내 VLM/업무 자료·검색 품질은 미확인이다. 작성일 2026-02-12를 보존하며 다섯 방법의 품질 우위는 단정하지 않는다.

## 왜 필요한가? (Why)

엔지니어링 분야에서 PowerPoint의 특징:
- **기술 발표 자료**: 공정 설명, 장비 소개, 프로젝트 리뷰
- **짧은 bullet point**: 핵심만 요약된 텍스트 → 맥락 부족
- **다이어그램/차트 의존**: 텍스트만으로 내용 파악 어려움
- **표(Table)**: 데이터 비교, 스펙 정리에 자주 사용
- **Speaker Notes**: 발표자 노트에 상세 설명이 있는 경우

**핵심 과제**: 텍스트와 시각 자료에서 무엇을 추출했는지 밝히고, 노트가 원래 슬라이드에 대한 설명인지 구분하여 출처를 보존하는 것. 짧은 문구가 있다는 이유만으로 일반 splitter가 실패하거나 VLM이 필요하다고 단정하지 않는다.

## 핵심 개념 (What)

### PowerPoint 구조 이해

```
PPTX 파일
├── 슬라이드 (Slide)
│   ├── 제목 (Title)
│   ├── 본문 텍스트 (Body Text)
│   ├── 테이블 (Table)
│   ├── 차트 (Chart)
│   ├── 이미지 (Image)
│   └── 도형 내 텍스트 (Shape Text)
├── 발표자 노트 (Speaker Notes)
├── 슬라이드 마스터 (Master/Layout)
└── 미디어 파일 (images, videos)
```

### 청킹 단위 선택

| 전략 | 설명 | 적합한 경우 |
|------|------|-------------|
| **슬라이드 단위** | 1 슬라이드 = 1 청크 | 각 슬라이드가 독립적 토픽 |
| **섹션 단위** | 여러 슬라이드를 섹션으로 묶음 | 연속된 슬라이드가 하나의 토픽 |
| **요소 단위** | 테이블, 차트 등을 개별 청크 | 테이블/차트가 독립적으로 검색되어야 할 때 |

## 어떻게 사용하는가? (How)

### 방법 1: python-pptx - 기본 텍스트 추출

```python
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def extract_pptx_by_slide(pptx_path: str) -> list[dict]:
    """파일을 저장하지 않고 슬라이드 shape tree와 노트 본문을 읽는다."""
    prs = Presentation(pptx_path)
    result = []
    for slide_num, slide in enumerate(prs.slides, 1):
        title_shape = slide.shapes.title
        data = {
            "slide_num": slide_num, "slide_id": slide.slide_id, "source": pptx_path,
            "title": title_shape.text if title_shape is not None else "",
            "body_texts": [], "tables": [], "notes": "", "shapes_text": [],
            "elements": [], "unextracted_shapes": [], "notes_status": "absent",
        }

        def visit(shapes, parents: tuple[int, ...] = ()) -> None:
            for shape in shapes:
                path = (*parents, shape.shape_id)
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    visit(shape.shapes, path)
                    continue
                common = {"shape_path": path, "name": shape.name,
                          "bbox_emu": (shape.left, shape.top, shape.width, shape.height)}
                if shape.has_text_frame:
                    text = shape.text_frame.text
                    # 제목과 문자열이 같아도 다른 shape의 본문을 버리지 않는다.
                    if not parents and title_shape is not None and shape.shape_id == title_shape.shape_id:
                        continue
                    if text.strip():
                        key = "shapes_text" if parents else "body_texts"
                        data[key].append(text)
                        data["elements"].append({**common, "type": "text", "text": text})
                elif shape.has_table:
                    table = shape.table
                    rows = [[cell.text for cell in row.cells] for row in table.rows]
                    merges = [{"row": r, "column": c,
                               "is_merge_origin": cell.is_merge_origin,
                               "is_spanned": cell.is_spanned,
                               "span_height": cell.span_height, "span_width": cell.span_width}
                              for r, row in enumerate(table.rows)
                              for c, cell in enumerate(row.cells)
                              if cell.is_merge_origin or cell.is_spanned]
                    data["tables"].append(rows)
                    data["elements"].append({**common, "type": "table", "rows": rows,
                                             "merged_cells": merges})
                else:
                    data["unextracted_shapes"].append({
                        **common, "shape_type": str(shape.shape_type),
                        "has_chart": shape.has_chart,
                        "status": "content_not_extracted",
                    })
        visit(slide.shapes)
        if slide.has_notes_slide:
            frame = slide.notes_slide.notes_text_frame
            data["notes_status"] = "missing_placeholder" if frame is None else "present"
            if frame is not None:
                data["notes"] = frame.text
        result.append(data)
    return result
```

텍스트/표를 그룹 깊이와 함께 읽고 shape ID로 제목을 제외한다. 같은 문자열의 다른 본문을 버리지 않는다. 도형 순서는 z-order이며 시각적 읽기 순서가 아니다. bbox는 EMU 좌표이고 그룹 변환을 펼친 페이지 좌표가 아니다. 표 병합 origin/spanned 정보와 원 grid를 보존하며 표시 표가 완전한 병합 구조는 아니다. [Shapes API](https://python-pptx.readthedocs.io/en/latest/api/shapes.html)·[표/병합](https://python-pptx.readthedocs.io/en/latest/user/table.html), 확인일 2026-10-04.

노트는 `has_notes_slide`를 먼저 검사하고 `notes_text_frame=None`을 처리한다. 노트 본문 placeholder만 읽으며 다른 노트 도형/마스터·레이아웃·숨김/애니메이션/SmartArt/미디어·차트 값은 읽지 않는다. `missing_placeholder`와 노트 없음/빈 본문을 구분한다. 차트 API는 존재하므로 python-pptx가 차트를 전혀 지원하지 않는다는 뜻이 아니다. 이 예제는 차트/이미지 내용을 `content_not_extracted`로 기록한다. [Slide/Notes API](https://python-pptx.readthedocs.io/en/latest/api/slides.html)·[노트](https://python-pptx.readthedocs.io/en/latest/user/notes.html)·[차트 API](https://python-pptx.readthedocs.io/en/latest/api/chart.html), 확인일 2026-10-04.

### 방법 2: 슬라이드 단위 청킹 (권장 기본 전략)

```python
import html


def pptx_table_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("표 열 수를 확인하세요")
    def escape(value: str) -> str:
        return html.escape(value).replace("\\", "\\\\").replace("|", "\\|").replace(
            "\n", "<br>"
        ).replace("\v", "<br>")
    display = [[escape(cell) for cell in row] for row in rows]
    lines = ["| " + " | ".join(display[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in display[1:])
    return "\n".join(lines)


def chunk_pptx_by_slide(pptx_path: str) -> list[dict]:
    chunks = []
    for slide in extract_pptx_by_slide(pptx_path):
        parts = [f"# {slide['title']}"] if slide["title"].strip() else []
        for element in slide["elements"]:
            if element["type"] == "text":
                parts.append(element["text"])
            elif element["type"] == "table":
                parts.append(pptx_table_markdown(element["rows"]))
        if slide["notes"].strip():
            parts.append(f"발표자 노트:\n{slide['notes']}")
        text = "\n\n".join(parts)
        # 빈/이미지 전용 슬라이드도 번호와 누락 상태를 유지한다.
        chunks.append({
            "text": text,
            "metadata": {
                "source": pptx_path, "slide_num": slide["slide_num"],
                "slide_id": slide["slide_id"], "title": slide["title"],
                "has_table": bool(slide["tables"]), "has_notes": bool(slide["notes"].strip()),
                "notes_status": slide["notes_status"], "has_content": bool(text.strip()),
                "extraction_method": "text", "unextracted_shapes": slide["unextracted_shapes"],
            },
            "original_elements": slide["elements"],
            "original_notes": slide["notes"],
        })
    return chunks
```

표 첫 행을 Markdown 헤더로 표시하는 예제 가정이다. 첫 행이 헤더인지 업무 문서로 확인한다. 한 행 표도 버리지 않으며 pipe/줄바꿈을 escape한다. 원 셀/병합정보는 `original_elements`에 보존한다. 이 청크는 한 슬라이드의 **조립 단위**이며 문자/토큰 한도를 강제하지 않는다. 긴 본문/노트/표는 이후 모델 입력에 맞춰 분할하고 slide_num/source/요소 범위를 유지한다. 빈 슬라이드는 색인에서 제외할 수 있으나 원 번호 기록은 유지한다.

### 방법 3: Unstructured 활용

```python
def unstructured_pptx_chunks(pptx_path: str) -> list:
    # 추가 의존성을 준비한 뒤 호출. 실제 파서 실행은 이번 검토에서 미완료.
    from unstructured.partition.pptx import partition_pptx
    from unstructured.chunking.title import chunk_by_title

    elements = partition_pptx(filename=pptx_path, include_page_breaks=True)
    return chunk_by_title(
        elements,
        max_characters=1500,
        combine_text_under_n_chars=100,
        multipage_sections=False,
    )

# 준비된 환경에서만 실행: chunks = unstructured_pptx_chunks("presentation.pptx")
# metadata.orig_elements로 원래 슬라이드/요소·표 HTML을 확인한다.
```

`include_page_breaks`만으로 한 슬라이드 한 청크가 보장되지는 않는다. `multipage_sections=False`와 원요소 page metadata를 대조한다. Title은 추정 분류이며 작은 섹션 병합 100은 짧은 bullet을 반드시 합치는 규칙이 아니다. 1500은 문자 hard maximum이며 표는 별도/큰 표는 분할될 수 있다. [공식 PPTX partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pptx)·[청킹](https://docs.unstructured.io/open-source/core-functionality/chunking), 확인일 2026-10-04. 실제 노트 포함/슬라이드 순서/표 HTML·설치판본은 미확인이다.

### 방법 4: Vision LLM으로 슬라이드 이미지 분석

다이어그램/차트의 보이는 내용을 설명하는 비교 후보다. 원 데이터/정확한 수치와 생성 설명을 구분한다. LibreOffice 변환은 기존 고정 `/tmp` 출력 대신 매번 새 출력/프로필에 수행하고 결과 존재와 timeout을 검사한다. 명령 반환 성공만으로 PDF가 만들어졌다고 보지 않는다. [공식 실행 매개변수](https://help.libreoffice.org/latest/en-US/text/shared/guide/start_parameters.html), 확인일 2026-10-04. 현재 PATH에서 libreoffice를 찾지 못해 실제 변환/판본은 미확인이다.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import pymupdf
import base64
from openai import OpenAI


def pptx_to_images(pptx_path: str, *, executable: str = "libreoffice") -> list[dict]:
    """PPTX→임시 PDF→PNG. 렌더링 페이지와 원 슬라이드 대응은 아직 미확인."""
    source = Path(pptx_path).resolve(strict=True)
    if source.suffix.lower() != ".pptx":
        raise ValueError("PPTX 입력을 제공하세요")
    with TemporaryDirectory(prefix="pptx-render-") as temp:
        output = Path(temp) / "output"
        output.mkdir()
        profile = Path(temp) / "profile"
        subprocess.run([
            executable, f"-env:UserInstallation={profile.as_uri()}",
            "--headless", "--convert-to", "pdf", "--outdir", str(output), str(source),
        ], check=True, timeout=60, capture_output=True, text=True)
        pdf = output / f"{source.stem}.pdf"
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise RuntimeError("PDF 변환 결과가 없습니다")
        result = []
        with pymupdf.open(pdf) as doc:
            for page in doc:
                result.append({
                    "source": str(source), "rendered_page": page.number + 1,
                    "source_slide_num": None, "mapping_status": "unverified",
                    "image_bytes": page.get_pixmap(
                        matrix=pymupdf.Matrix(2, 2), alpha=False,
                    ).tobytes("png"),
                })
        return result


def analyze_slide_with_vision(
    image_bytes: bytes, slide_num: int, *, client: OpenAI, model: str,
) -> str:
    if isinstance(slide_num, bool) or not isinstance(slide_num, int) or slide_num < 1:
        raise ValueError("슬라이드 번호는 1-based 정수입니다")
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n") or not model.strip():
        raise ValueError("PNG 입력과 확인된 모델 ID가 필요합니다")
    image = base64.b64encode(image_bytes).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                f"원 슬라이드 {slide_num}의 제목·본문·표·다이어그램을 구조화하세요. "
                "다이어그램/차트 설명과 보이는 데이터를 구분하고, "
                "판독할 수 없는 수치/내용은 미확인으로 표시하세요."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]}],
        max_tokens=2048,
    )
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise ValueError("완료된 추출 응답이 아닙니다")
    text = response.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError("추출 텍스트가 없습니다")
    return text

# client/인증/모델은 확인된 환경 설정에서 주입하고 호출자가 닫는다.
# 원문 Qwen3-VL-30B 별칭은 실제 배포/지원이 확인되지 않은 후보다.
```

변환된 PDF의 페이지 번호가 원 슬라이드 번호라고 추측하지 않는다. 숨김 슬라이드·출력 설정·순서·폰트 대체·잘림/차트·노트 포함 여부를 원 PPTX와 대조한 뒤 호출자가 `images_by_slide` 대응을 만든다. `pptx_to_images`의 `source_slide_num=None`은 미확인을 뜻하며 임시 PDF 파일이 삭제되어도 PNG bytes/원 출처는 반환한다. 대량 자료는 전체 이미지를 메모리에 두는 대신 페이지별 처리 방식도 비교한다. 이 검토에서는 명령 조립/실패 경계와 생성 PDF 렌더링만 검사한다.

이미지 data URL과 SDK 요청 형식은 [공식 이미지 입력 안내](https://developers.openai.com/api/docs/guides/images-vision)를 기준으로 한다(확인일2026-10-04). compatible 서버의 모델 ID·지원·사내 배포는 미확인이다. 2배 이미지/2048 응답은 예제 설정이며 `length`/빈 content/choice 없음은 성공으로 저장하지 않는다. `stop`도 내용 정확성 증거는 아니다.

### 방법 5: 하이브리드 전략 (권장)

텍스트 추출을 보존하고 호출자가 선택한 원 슬라이드에 Vision 결과를 별도 필드로 추가한다. 원문 50자 threshold는 미검증 후보 기준으로 남기며 자동 선택에 쓰지 않는다. 텍스트가 많아도 그림이 핵심일 수 있다. 빈 슬라이드를 누락한 뒤 enumerate 인덱스로 이미지를 연결하던 원문 오류를 제거했다. 선택 번호와 대조된 이미지 대응이 없으면 중단하며 기존 본문/노트를 덮어쓰지 않는다.

```python
from copy import deepcopy
from collections.abc import Callable


def hybrid_pptx_chunking(
    pptx_path: str, *, vision_slide_nums: set[int], images_by_slide: dict[int, bytes],
    analyzer: Callable[[bytes, int], str],
) -> list[dict]:
    """호출자가 대조한 이미지/원 슬라이드 대응에만 Vision을 추가한다."""
    chunks = deepcopy(chunk_pptx_by_slide(pptx_path))
    known = {chunk["metadata"]["slide_num"] for chunk in chunks}
    if any(isinstance(n, bool) or not isinstance(n, int) for n in vision_slide_nums):
        raise ValueError("선택 슬라이드 번호는 정수입니다")
    if not vision_slide_nums <= known or not vision_slide_nums <= images_by_slide.keys():
        raise ValueError("선택 슬라이드/확인된 이미지 대응이 없습니다")
    for chunk in chunks:
        number = chunk["metadata"]["slide_num"]
        if number not in vision_slide_nums:
            continue
        description = analyzer(images_by_slide[number], number)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Vision 보강 결과가 없습니다")
        # 원문/노트와 생성 설명을 구분하고 덮어쓰지 않는다.
        chunk["vision_text"] = description
        chunk["metadata"]["extraction_method"] = "text+vision" if chunk["text"].strip() else "vision"
    return chunks
```

## 메타데이터 보강 전략

전체 제목과 이웃 제목을 출처 맥락으로 추가하는 후보이며 검색 성능 개선을 보장하지 않는다. 첫 슬라이드 제목을 전체 제목으로 보는 것은 예제 가정이다. 원 PPTX에서 이웃을 찾으므로 일부 청크가 필터링되어도 번호가 어긋나지 않는다. 입력 청크는 deepcopy로 보존하고 없는 이웃은 None, 제목 없는 실제 이웃은 빈 문자열이다. metadata를 실제 검색/임베딩 입력에 사용하는 방식은 별도 설계/평가가 필요하다.

```python
def enrich_slide_metadata(chunks: list[dict], pptx_path: str) -> list[dict]:
    prs = Presentation(pptx_path)
    titles = {n: slide.shapes.title.text if slide.shapes.title is not None else ""
              for n, slide in enumerate(prs.slides, 1)}
    result = deepcopy(chunks)
    for chunk in result:
        metadata = chunk["metadata"]
        number = metadata["slide_num"]
        if isinstance(number, bool) or not isinstance(number, int) or number not in titles:
            raise ValueError("원 PPTX의 슬라이드 번호가 아닙니다")
        metadata.update({
            "presentation_title": titles.get(1),  # 첫 제목을 전체 제목으로 가정
            "prev_slide_title": titles.get(number - 1),
            "next_slide_title": titles.get(number + 1),
        })
    return result
```

## 도구 비교

| 도구 | 읽거나 생성하는 것 | 검증 범위/한계 |
|------|-------------------|----------------|
| python-pptx | 텍스트/표/노트·차트 API 등 | 본 예제는 텍스트/표/노트만 추출; 차트/시각 의미 미추출 |
| Unstructured | 요소 분류/Title 청킹 후보 | 공식 문서/AST만; 실제 파서·노트·품질 미확인 |
| Vision LLM | 이미지로부터 구조화 설명 생성 | SDK mock만; 사내 모델·수치/설명 정확도 미확인 |
| 하이브리드 | 원문과 별도 생성 설명 조립 | 선택/대응/보존 fixture; 개선 효과·비용 미측정 |

별점/무료/높음·중간 비용은 비교 조건과 측정 근거가 없어 제거했다. 패키지/모델/서비스 라이선스·운영 비용은 각각 확인한다. 평가용 질의/정답으로 원문 누락·슬라이드 대응·표/수치·근거성·검색 recall·지연/비용을 비교한 뒤 선택한다.

## 검증 결과와 남은 조건

원래 절과 다섯 방법·1500/100·50자 후보·2배 이미지/2048·전체/전후 제목 맥락을 보존한다. 실제 생성 PPTX로 재귀 그룹·같은 제목 본문·병합/한 행 표·노트 placeholder 없음·빈 슬라이드·원파일 bytes를 검사하고 SDK 모의 요청/하이브리드 번호/metadata 입력 불변을 검사한다. 기존 첨부는 처리하지 않는다. LibreOffice 명령 대역은 실제 변환/폰트/슬라이드 품질의 증거가 아니다. 기술/참조/읽기 화면 검사 결과는 폴더 정리 기록에 남긴다.

HERDR_ENV=1에서 현재 pane_not_found로 Claude 연결이 되지 않았다. 문서 통합·도구 우위/50자 기준·업무 VLM 운영 판단은 보류한다. 공식 API와 재현되는 오류만 수정했다.

## 참고 자료 (References)

- [python-pptx Documentation](https://python-pptx.readthedocs.io/)
- [Unstructured PPTX Partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pptx)
- [LibreOffice CLI](https://help.libreoffice.org/latest/en-US/text/shared/guide/start_parameters.html)

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
