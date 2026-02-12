---
tags: [rag, drm, vlm, chunking, markdown]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# VLM 추출 결과물의 청킹 전략

> VLM이 이미지에서 추출한 Markdown 텍스트는 일반 문서 파싱 결과와 특성이 다르다. VLM 출력에 최적화된 청킹 전략을 다룬다.

## 왜 필요한가? (Why)

VLM 추출 결과물의 특수성:

| 특성 | 일반 파싱 결과 | VLM 추출 결과 |
|------|---------------|--------------|
| **구조 정확도** | 100% (파일 구조 그대로) | 90-95% (VLM 해석에 의존) |
| **텍스트 정확도** | 100% | 95-99% (오인식 가능) |
| **페이지 경계** | 명확 | 스크린샷 단위로 명확 |
| **메타데이터** | 풍부 (스타일, 좌표 등) | 최소 (페이지 번호 정도) |
| **테이블 구조** | 프로그래밍적 접근 가능 | Markdown 텍스트로만 존재 |
| **연속성** | 페이지 간 텍스트 연결 가능 | 페이지 단위로 독립 추출 |

**핵심 과제**: VLM 출력의 이러한 특성을 고려한 맞춤 청킹

## 핵심 개념 (What)

### VLM 출력 → 청킹의 3가지 전략

```
전략 1: 페이지 단위 청킹 (Page-Level)
  → 각 페이지 추출 결과 = 1 청크
  → 단순하지만 효과적, 특히 PPT

전략 2: 구조 기반 재조합 (Structure-Aware)
  → VLM 출력의 Markdown 헤더를 파싱하여 섹션 단위 재조합
  → Word, 긴 보고서에 적합

전략 3: 요소 분리 (Element Separation)
  → 텍스트/테이블/다이어그램을 개별 청크로 분리
  → 테이블이 많은 Excel, 혼합 문서에 적합
```

## 어떻게 사용하는가? (How)

### 전략 1: 페이지 단위 청킹

**가장 간단하고 안전한 방법**. VLM은 이미 페이지 단위로 구조화된 출력을 생성하므로, 그대로 활용.

```python
def chunk_by_page(
    page_results: list[dict],
    source_file: str,
    doc_type: str,
) -> list[dict]:
    """VLM 추출 결과를 페이지 단위 청크로 변환"""
    chunks = []

    for i, result in enumerate(page_results):
        text = result["extracted_text"].strip()
        if not text:
            continue

        chunks.append({
            "text": text,
            "metadata": {
                "source": source_file,
                "page": i + 1,
                "total_pages": len(page_results),
                "doc_type": doc_type,
                "extraction_method": "vlm_screenshot",
                "quality_score": result.get("quality_score", 100),
            }
        })

    return chunks
```

**적합한 경우**: PowerPoint (1 슬라이드 = 1 토픽), 짧은 문서

**부적합한 경우**: 한 섹션이 여러 페이지에 걸친 Word 보고서

### 전략 2: 페이지 간 섹션 재조합

VLM 출력의 Markdown 헤더를 파싱하여, 같은 섹션에 속하는 페이지들을 하나의 청크로 묶음.

```python
import re

def chunk_by_section_across_pages(
    page_results: list[dict],
    source_file: str,
    max_chunk_size: int = 2000,
    split_level: int = 2,
) -> list[dict]:
    """여러 페이지의 VLM 추출 결과를 섹션 단위로 재조합"""

    # 1. 모든 페이지를 하나로 합침
    combined = []
    for i, result in enumerate(page_results):
        combined.append({
            "text": result["extracted_text"],
            "page": i + 1,
        })

    # 2. Markdown 헤더 기준으로 섹션 분리
    sections = []
    current_section = {
        "headers": {},
        "texts": [],
        "pages": [],
    }

    for page_data in combined:
        lines = page_data["text"].split("\n")

        for line in lines:
            # Markdown 헤더 감지
            header_match = re.match(r'^(#{1,4})\s+(.+)', line)

            if header_match:
                level = len(header_match.group(1))

                if level <= split_level and current_section["texts"]:
                    # 새 섹션 시작 → 이전 섹션 저장
                    sections.append(dict(current_section))
                    current_section = {
                        "headers": {
                            k: v for k, v in current_section["headers"].items()
                            if k < level
                        },
                        "texts": [],
                        "pages": [],
                    }

                current_section["headers"][level] = header_match.group(2)

            current_section["texts"].append(line)
            if page_data["page"] not in current_section["pages"]:
                current_section["pages"].append(page_data["page"])

    # 마지막 섹션
    if current_section["texts"]:
        sections.append(current_section)

    # 3. 청크 생성
    chunks = []
    for section in sections:
        text = "\n".join(section["texts"]).strip()
        if not text:
            continue

        header_path = " > ".join(
            section["headers"][k]
            for k in sorted(section["headers"])
        )

        # 크기 초과 시 추가 분할
        if len(text) > max_chunk_size:
            sub_chunks = split_section(text, max_chunk_size)
            for j, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "text": sub_text,
                    "metadata": {
                        "source": source_file,
                        "pages": section["pages"],
                        "section_path": header_path,
                        "part": j + 1,
                        "extraction_method": "vlm_screenshot",
                    }
                })
        else:
            chunks.append({
                "text": text,
                "metadata": {
                    "source": source_file,
                    "pages": section["pages"],
                    "section_path": header_path,
                    "extraction_method": "vlm_screenshot",
                }
            })

    return chunks


def split_section(text: str, max_size: int) -> list[str]:
    """큰 섹션을 문단 경계에서 분할"""
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks
```

### 전략 3: 요소 분리 (텍스트/테이블/다이어그램)

VLM 출력에서 테이블, 다이어그램 설명, 일반 텍스트를 분리하여 각각 청크로 생성.

```python
def chunk_by_element_type(
    page_results: list[dict],
    source_file: str,
) -> list[dict]:
    """VLM 출력을 요소 유형별로 분리하여 청킹"""
    chunks = []

    for i, result in enumerate(page_results):
        text = result["extracted_text"]
        page_num = i + 1

        # Markdown 테이블 영역 추출
        tables = extract_markdown_tables(text)
        for j, table in enumerate(tables):
            chunks.append({
                "text": table["text"],
                "metadata": {
                    "source": source_file,
                    "page": page_num,
                    "type": "table",
                    "table_title": table.get("title", ""),
                    "extraction_method": "vlm_screenshot",
                }
            })

        # [Figure: ...] 설명 추출
        figures = extract_figure_descriptions(text)
        for j, fig in enumerate(figures):
            chunks.append({
                "text": fig,
                "metadata": {
                    "source": source_file,
                    "page": page_num,
                    "type": "figure",
                    "extraction_method": "vlm_screenshot",
                }
            })

        # 나머지 텍스트 (테이블, 그림 설명 제거 후)
        remaining = remove_tables_and_figures(text)
        if remaining.strip():
            chunks.append({
                "text": remaining.strip(),
                "metadata": {
                    "source": source_file,
                    "page": page_num,
                    "type": "text",
                    "extraction_method": "vlm_screenshot",
                }
            })

    return chunks


def extract_markdown_tables(text: str) -> list[dict]:
    """Markdown 텍스트에서 테이블 영역을 추출"""
    lines = text.split("\n")
    tables = []
    current_table = []
    table_title = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            # 테이블 바로 윗줄이 제목일 수 있음
            if not current_table and i > 0:
                prev = lines[i-1].strip()
                if prev and not prev.startswith("|"):
                    table_title = prev
            current_table.append(line)
        else:
            if current_table:
                tables.append({
                    "text": "\n".join(current_table),
                    "title": table_title,
                })
                current_table = []
                table_title = ""

    if current_table:
        tables.append({"text": "\n".join(current_table), "title": table_title})

    return tables


def extract_figure_descriptions(text: str) -> list[str]:
    """[Figure: ...] 패턴의 그림 설명 추출"""
    pattern = r'\[Figure:\s*(.+?)\]'
    return re.findall(pattern, text, re.DOTALL)


def remove_tables_and_figures(text: str) -> str:
    """테이블과 그림 설명을 제거한 나머지 텍스트"""
    lines = text.split("\n")
    result = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            in_table = True
            continue
        else:
            if in_table:
                in_table = False
            # Figure 설명도 제거
            if not re.match(r'\[Figure:', stripped):
                result.append(line)

    return "\n".join(result)
```

### 문서 유형별 권장 조합

```python
def chunk_vlm_output(
    page_results: list[dict],
    source_file: str,
    doc_type: str,
) -> list[dict]:
    """문서 유형에 따라 최적 청킹 전략 자동 선택"""

    if doc_type == "pptx":
        # PPT: 페이지(슬라이드) 단위가 자연스러움
        return chunk_by_page(page_results, source_file, doc_type)

    elif doc_type == "xlsx":
        # Excel: 테이블을 독립 청크로 분리
        return chunk_by_element_type(page_results, source_file)

    elif doc_type == "docx":
        # Word: 섹션 재조합 (헤더 기반)
        return chunk_by_section_across_pages(
            page_results, source_file, max_chunk_size=1500
        )

    else:
        # PDF 등: 페이지 단위 + 테이블 분리 하이브리드
        page_chunks = chunk_by_page(page_results, source_file, doc_type)
        element_chunks = chunk_by_element_type(page_results, source_file)

        # 테이블 청크만 element에서 가져오고, 나머지는 page 단위
        table_chunks = [c for c in element_chunks if c["metadata"]["type"] == "table"]
        return page_chunks + table_chunks
```

## 페이지 경계 문제 해결

VLM은 페이지 단위로 추출하므로, **페이지 경계에서 문장이 잘리는 문제**가 발생.

### 페이지 간 텍스트 연결

```python
def merge_page_boundaries(page_results: list[dict]) -> list[dict]:
    """인접 페이지 경계에서 잘린 문장/문단을 연결"""
    if len(page_results) < 2:
        return page_results

    merged = [page_results[0].copy()]

    for i in range(1, len(page_results)):
        prev_text = merged[-1]["extracted_text"]
        curr_text = page_results[i]["extracted_text"]

        # 이전 페이지의 마지막 줄이 불완전한 문장인지 확인
        prev_lines = prev_text.rstrip().split("\n")
        last_line = prev_lines[-1].strip() if prev_lines else ""

        # 불완전 문장 감지: 마침표/물음표/느낌표로 끝나지 않음
        if last_line and not re.search(r'[.?!。]$', last_line):
            # 현재 페이지의 첫 줄을 이전 페이지 끝에 연결
            curr_lines = curr_text.lstrip().split("\n")
            first_line = curr_lines[0].strip() if curr_lines else ""

            # 이전 페이지 끝에 연결
            if first_line and not first_line.startswith("#"):
                prev_lines[-1] = last_line + " " + first_line
                merged[-1]["extracted_text"] = "\n".join(prev_lines)
                # 현재 페이지에서 첫 줄 제거
                curr_text = "\n".join(curr_lines[1:])

        merged.append({
            **page_results[i],
            "extracted_text": curr_text,
        })

    return merged
```

### 슬라이딩 윈도우 컨텍스트

인접 페이지의 내용을 일부 포함하여 맥락 보존:

```python
def add_sliding_context(
    chunks: list[dict],
    context_lines: int = 3,
) -> list[dict]:
    """각 청크에 이전/다음 페이지의 마지막/처음 N줄을 컨텍스트로 추가"""
    enriched = []

    for i, chunk in enumerate(chunks):
        parts = []

        # 이전 청크의 마지막 N줄
        if i > 0:
            prev_lines = chunks[i-1]["text"].split("\n")
            context = "\n".join(prev_lines[-context_lines:])
            parts.append(f"[이전 페이지 끝]\n{context}\n[---]")

        parts.append(chunk["text"])

        # 다음 청크의 처음 N줄
        if i < len(chunks) - 1:
            next_lines = chunks[i+1]["text"].split("\n")
            context = "\n".join(next_lines[:context_lines])
            parts.append(f"[---]\n[다음 페이지 시작]\n{context}")

        enriched.append({
            **chunk,
            "text": "\n\n".join(parts),
        })

    return enriched
```

## 메타데이터 보강

VLM 추출은 메타데이터가 부족하므로, 텍스트 LLM(Kimi-K2.5)으로 보강:

```python
async def enrich_chunk_metadata(
    chunk: dict,
    client: AsyncOpenAI,
) -> dict:
    """LLM으로 청크에 키워드/요약 메타데이터 추가"""
    response = await client.chat.completions.create(
        model="Kimi-K2.5",  # 사내 텍스트 LLM (메타데이터 보강에 충분)
        messages=[{
            "role": "user",
            "content": f"""다음 문서 청크를 분석하여 JSON으로 응답하세요:

```
{chunk['text'][:1000]}
```

응답 형식:
{{
    "summary": "1-2문장 요약",
    "keywords": ["키워드1", "키워드2", ...],
    "topic": "주제 분류",
    "has_data": true/false,
    "language": "ko/en/mixed"
}}"""
        }],
        response_format={"type": "json_object"},
        max_tokens=256,
    )

    import json
    metadata_extra = json.loads(response.choices[0].message.content)
    chunk["metadata"].update(metadata_extra)

    return chunk
```

## 참고 자료 (References)

- [LangChain MarkdownHeaderTextSplitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)
- [Chunking Strategies for RAG](https://www.pinecone.io/learn/chunking-strategies/)

## 관련 문서

- [스크린샷 + VLM 파이프라인](./screenshot-vlm-pipeline.md)
- [DRM 해제 후 하이브리드 전략](./post-drm-hybrid.md)
- [청킹 방법론 총론](../overview-chunking-methods.md)
