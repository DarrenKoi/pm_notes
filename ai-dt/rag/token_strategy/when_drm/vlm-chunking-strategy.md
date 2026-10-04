---
tags: [rag, drm, vlm, chunking, markdown]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_proposal
---

# VLM 추출 결과물의 청킹 전략

> 화면 전사 결과를 출처와 함께 검색 단위로 나누는 비교용 제안이다. 청킹으로 전사 오류를 수정하거나 원본에 없는 메타데이터를 복원할 수는 없다.

> [!warning] 검토 범위 — 2026-10-04
> 원래 2026-02-12 제안의 정확도/최적/사내 모델 적합성 수치는 평가 근거가 없어 미확인이다. 생성 Markdown과 공식 파서·SDK의 모의 요청으로 원문 보존·범위/출처·schema를 검증한다. 실제 VLM/임베딩/검색 품질·사내 정책은 미검증이다. Herdr pane_not_found로 Claude와의 전략 선택/통합 협의는 보류한다.

[화면 추출](./screenshot-vlm-pipeline.md)의 manifest(`source_image`, `page_number`, `extracted_text`, `quality_issues`)를 입력으로 받는다. 모르는 실제 페이지는 `None`, 검토되지 않은 품질은 `unverified`로 유지한다. 페이지 번호를 결과 배열 순서로 새로 붙이지 않는다.

## 왜 필요한가? (Why)

VLM 추출 결과물의 특수성:

| 특성 | 일반 파싱 결과 | VLM 추출 결과 |
|------|---------------|--------------|
| **구조 정확도** | 원본 API 구조와 읽기/의미 구조는 다를 수 있음 | 이미지·모델·프롬프트별 원문 대조 필요; 기존90~95% 미확인 |
| **텍스트 정확도** | 누락/지원 범위/추출 순서 확인 필요 | 기존95~99% 미확인; 전사 오류 가능 |
| **페이지 경계** | 형식/파서별 다름; DOCX/Excel은 고정 페이지 아닐 수 있음 | 이미지 경계는 존재; 실제 페이지와 동일한지는 manifest로 확인 |
| **메타데이터** | 파서가 제공하는 스타일/좌표 범위 확인 | manifest의 원본 문서/이미지/페이지·검토 기록을 별도 보존 |
| **테이블 구조** | 지원 API에서 접근 가능한 범위만 | 출력 규격별 다름; 이 예제는 Markdown 표 후보 |
| **연속성** | 형식/추출 순서에 따라 다름 | 이 예제는 이미지별 독립 추출; 경계 연결 여부는 원본 확인 |

**핵심 과제**: VLM 출력의 이러한 특성을 고려한 맞춤 청킹

## 핵심 개념 (What)

### VLM 출력 → 청킹의 3가지 전략

아래는 원래 문서 유형별 비교 제안이다. 효과적/적합은 측정된 우위가 아니다.

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

예제는 같은 Python 실행 문맥에서 위에서 아래로 정의한다. 임시 학습 환경에 `markdown-it-py==4.2.0`과 `openai==3.24.0`이 필요하다. 확인된 manifest를 `chunk_vlm_output(page_results, source_file, "docx")`처럼 전달한다. 빈 청크는 검토 기록에 보존하고 검색 적재는 별도 정책으로 결정한다. 결과의 `source_spans`는 실제 기여한 이미지/페이지·원본 문자 범위이며 `section_path`는 청크 시작의 제목 문맥이다.

### 전략 1: 페이지 단위 청킹

페이지/이미지별 출처를 쉽게 보존하는 기본안이다. 페이지 하나가 한 토픽이거나 모델 입력 한도 안이라는 보장은 없다. 빈 추출은 실패/빈 원문을 구분할 수 없으므로 기록에서 삭제하지 않는다.

```python
from copy import deepcopy


def page_metadata(result: dict, source_file: str) -> dict:
    page = result.get("page_number")
    if page is not None and (type(page) is not int or page < 1):
        raise ValueError("positive page number or None required")
    if not isinstance(result.get("extracted_text"), str):
        raise ValueError("extracted_text must be str")
    return {"source": source_file, "page": page,
            "source_image": result.get("source_image"),
            "extraction_method": "vlm_screenshot", "review_status": "unverified",
            "quality_issues": deepcopy(result.get("quality_issues")),
            "legacy_quality_score": result.get("quality_score")}  # 미확인을 100점으로 바꾸지 않음


def chunk_by_page(page_results: list[dict], source_file: str, doc_type: str) -> list[dict]:
    """빈 추출도 출처와 함께 보존. 이미지1개가 물리 페이지/슬라이드1개라고 가정하지 않음."""
    return [{"text": r["extracted_text"],
             "metadata": {**page_metadata(r, source_file), "doc_type": doc_type,
                          "total_images": len(page_results),
                          "has_text": bool(r["extracted_text"].strip())}}
            for r in page_results]
```

**비교 후보**: 페이지/슬라이드 단위 질의와 짧은 입력. PPT 한 슬라이드가 항상 한 토픽은 아니다.

**부적합한 경우**: 한 섹션이 여러 페이지에 걸친 Word 보고서

### 전략 2: 페이지 간 섹션 재조합

Markdown heading 경계로 추출 구간을 묶는다. 제목의 반복/누락/전사 오류가 있으므로 원래 섹션이라는 의미적 보증은 아니다. 정규식 대신 markdown-it-py4.2.0의 heading/token.map으로 fence 안의 `#`를 제외하고 실제 문자 범위를 기록한다. 입력 순서가 이미 확인된 manifest여야 한다.

`max_chunk_size=2000/1500`은 **문자 수**다. 문단 경계를 우선하지만 큰 단락은 hard limit으로 나누므로 표/fence가 조각날 수 있다. 원문 문자와 출처를 보존하는 예제이지 Markdown 구조/모델 토큰 한도 보장은 아니다. 구조 보존이 필요하면 초과 표를 따로 처리하는 정책을 평가·합의한다. token limit 검사는 실제 모델 tokenizer로 별도 수행한다.

```python
from copy import deepcopy
from markdown_it import MarkdownIt

MD = MarkdownIt("commonmark", {"html": False}).enable("table")


def split_section(text: str, max_size: int) -> list[str]:
    """문자 기준 hard limit. 문단 우선, 초과 문단도 자르고 공백을 보존."""
    if type(max_size) is not int or max_size < 1:
        raise ValueError("positive character limit required")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary  # 구분자는 다음 slice에 남김
        chunks.append(text[start:end])
        start = end
    return chunks


def chunk_by_section_across_pages(
    page_results: list[dict], source_file: str,
    max_chunk_size: int = 2000, split_level: int = 2,
) -> list[dict]:
    """각 이미지의 실제 heading만 해석. 식별한 문자 범위로 출처를 재계산."""
    if type(split_level) is not int or not 1 <= split_level <= 6:
        raise ValueError("split_level must be 1..6")
    split_section("", max_chunk_size)  # 빈 입력에도 limit 검사
    spans, text, events = [], "", []
    for record in page_results:
        metadata = page_metadata(record, source_file)
        value = record["extracted_text"]
        start = len(text)
        if text:
            text += "\n"  # 추출 간 구분자. 원래 연속 문장으로 확정하지 않음
            start += 1
        text += value
        spans.append((start, len(text), metadata))
        lines = value.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        tokens = MD.parse(value)  # 페이지별 fence 상태를 서로 섞지 않음
        for i, token in enumerate(tokens):
            if token.type == "heading_open" and token.level == 0:
                level = int(token.tag[1:])
                title = tokens[i + 1].content
                events.append((start + offsets[token.map[0]], level, title))
    boundaries = sorted({0, len(text), *(pos for pos, level, _ in events if level <= split_level)})
    headers, event_index, chunks = {}, 0, []
    for left, right in zip(boundaries, boundaries[1:]):
        while event_index < len(events) and events[event_index][0] <= left:
            _, level, title = events[event_index]
            headers = {k: v for k, v in headers.items() if k < level}
            headers[level] = title
            event_index += 1
        section_path = " > ".join(headers[k] for k in sorted(headers))
        part_start = left
        for part, value in enumerate(split_section(text[left:right], max_chunk_size), 1):
            part_end = part_start + len(value)
            provenance = [{**deepcopy(meta), "source_char_start": max(part_start, a) - a,
                           "source_char_end": min(part_end, b) - a}
                          for a, b, meta in spans if a < b and a < part_end and b > part_start]
            chunks.append({"text": value, "metadata": {
                "source": source_file, "source_spans": provenance,
                "section_path": section_path, "part": part,
                "char_start": part_start, "char_end": part_end,
                "extraction_method": "vlm_screenshot", "review_status": "unverified"}})
            part_start = part_end
    # 빈 이미지의 기록을 별도 청크로 보존; 실제 index 적재 여부는 별도 정책.
    for a, b, metadata in spans:
        if a == b:
            chunks.append({"text": "", "metadata": {**metadata, "has_text": False}})
    return chunks
```

### 전략 3: 요소 분리 (텍스트/테이블/다이어그램)

Markdown 파서의 top-level 표와 독립 `[Figure: ...]` 문단을 문자 범위로 분리한다. nested 표·inline figure·HTML/비표준 표현은 일반 텍스트에 남긴다. 식별되지 않았다고 삭제하지 않는다. 표 제목은 앞줄만 보고 추측하지 않으며 `None`으로 남긴다. 원래 page+table 이중 적재는 중복 근거를 만들 수 있어 기본 예제에서 제거했다.

```python
import re
from copy import deepcopy


def element_spans(text: str) -> list[dict]:
    """top-level Markdown 표/독립 Figure 문단의 문자 범위. 나머지는 그대로 보존."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    identified = []
    for token in MD.parse(text):
        if token.level != 0 or token.map is None:
            continue
        kind = None
        a, b = (offsets[i] for i in token.map)
        if token.type == "table_open":
            kind = "table"
        elif token.type == "paragraph_open" and re.fullmatch(
            r"\[Figure:.*\]", text[a:b].strip(), flags=re.S
        ):
            kind = "figure"  # 독립 표식만 인식; inline/code는 삭제하지 않음
        if kind:
            identified.append({"start": a, "end": b, "type": kind})
    spans, cursor = [], 0
    for item in identified:
        if cursor < item["start"]:
            spans.append({"start": cursor, "end": item["start"], "type": "text"})
        spans.append(item)
        cursor = item["end"]
    if cursor < len(text):
        spans.append({"start": cursor, "end": len(text), "type": "text"})
    return spans


def extract_markdown_tables(text: str) -> list[dict]:
    return [{"text": text[s["start"]:s["end"]], "title": None,
             "char_start": s["start"], "char_end": s["end"]}
            for s in element_spans(text) if s["type"] == "table"]


def extract_figure_descriptions(text: str) -> list[str]:
    return [text[s["start"]:s["end"]] for s in element_spans(text) if s["type"] == "figure"]


def remove_tables_and_figures(text: str) -> str:
    return "".join(text[s["start"]:s["end"]] for s in element_spans(text) if s["type"] == "text")


def chunk_by_element_type(page_results: list[dict], source_file: str) -> list[dict]:
    chunks = []
    for result in page_results:
        metadata = page_metadata(result, source_file)
        text = result["extracted_text"]
        spans = element_spans(text)
        if not spans:
            chunks.append({"text": text, "metadata": {**metadata, "type": "text", "has_text": False}})
        for span in spans:
            chunks.append({"text": text[span["start"]:span["end"]], "metadata": {
                **deepcopy(metadata), "type": span["type"], "char_start": span["start"],
                "char_end": span["end"], "table_title": None}})
    return chunks
```

### 문서 유형별 권장 조합

```python
def chunk_vlm_output(page_results: list[dict], source_file: str, doc_type: str) -> list[dict]:
    """문서 유형별 비교용 기본안. 최적/자동 품질 판정은 아님."""
    if doc_type == "pptx":
        return chunk_by_page(page_results, source_file, doc_type)
    if doc_type == "xlsx":
        return chunk_by_element_type(page_results, source_file)
    if doc_type == "docx":
        return chunk_by_section_across_pages(page_results, source_file, max_chunk_size=1500)
    if doc_type in {"pdf", "general"}:
        # 원래 page+table 중복 적재 대신 source spans로 한 번씩 보존.
        return chunk_by_element_type(page_results, source_file)
    raise ValueError("unsupported doc_type")
```

## 페이지 경계 문제 해결

독립 이미지 전사에서는 경계가 끊길 수 있다. 마침표가 없는 줄은 제목/표/리스트/수식일 수도 있으므로 다음 페이지 첫 줄을 자동으로 이동·합치면 잘못된 내용과 출처를 만든다. 기존 함수 이름은 보존하되 원문을 바꾸지 않고 검토 후보만 표시한다.

### 페이지 간 텍스트 연결

```python
from copy import deepcopy
import re


def merge_page_boundaries(page_results: list[dict]) -> list[dict]:
    """이름은 보존하지만 자동 합치지 않음. 문장 부호 휴리스틱의 검토 후보만 표시."""
    results = deepcopy(page_results)
    for previous, current in zip(results, results[1:]):
        before = previous["extracted_text"].splitlines()
        after = current["extracted_text"].splitlines()
        last = before[-1].strip() if before else ""
        first = after[0].strip() if after else ""
        if last and first and not re.search(r"[.?!。]$", last):
            current["boundary_review_candidate"] = {
                "previous_page": previous.get("page_number"),
                "previous_image": previous.get("source_image"),
                "last_line": last, "first_line": first, "decision": None,
            }
    return results
```

### 슬라이딩 윈도우 컨텍스트

앞뒤 **청크**의 문맥을 별도 필드로 보존한다. 같은 페이지의 여러 요소일 수 있으므로 앞뒤 페이지로 표시하지 않는다. context_lines=0은 문맥 없음이며 원문 text와metadata는 deep copy로 보존한다. 최종 입력에 문맥을 붙일 때 중복·토큰 예산·출처를 평가한다:

```python
from copy import deepcopy


def add_sliding_context(chunks: list[dict], context_lines: int = 3) -> list[dict]:
    """이웃 청크 문맥을 본문과 분리. 이웃 청크가 이웃 페이지임을 가정하지 않음."""
    if type(context_lines) is not int or context_lines < 0:
        raise ValueError("nonnegative context_lines required")
    enriched = deepcopy(chunks)
    for i, chunk in enumerate(enriched):
        contexts = []
        if context_lines:
            for j, side in [(i - 1, "previous_chunk"), (i + 1, "next_chunk")]:
                if 0 <= j < len(chunks):
                    lines = chunks[j]["text"].splitlines()
                    value = "\n".join(lines[-context_lines:] if side == "previous_chunk" else lines[:context_lines])
                    contexts.append({"position": side, "text": value,
                                     "metadata": deepcopy(chunks[j]["metadata"])})
        chunk["context"] = contexts
    return enriched  # 임베딩/LLM 입력에 붙일 때 전체 토큰 한도를 별도로 검사
```

## 메타데이터 보강

생성 요약/키워드는 원본 메타데이터와 다른 추론 결과다. Kimi-K2.5는 텍스트 전용이 아닌 멀티모달 모델이며 여기서는 텍스트 후처리 후보로 제안했을 뿐 사내 제공/적합성은 미확인이다. 실제 내부 client/model을 주입한다. 기존1000자 절단을 제거했고 전체 입력이 모델 한도 내인지 호출 전에 별도로 확인해야 한다. 원래 예제의 Python 코드 fence 안에 같은 길이의 Markdown fence를 넣어 코드가 조기에 닫히던 렌더링도 수정했다. JSON모드는 schema/정확도 보증이 아니므로 완료 상태·키·타입·중복·unknown을 검사하고 `metadata.generated`에 분리한다:

```python
import json
from copy import deepcopy
from openai import AsyncOpenAI


async def enrich_chunk_metadata(chunk: dict, client: AsyncOpenAI, model: str) -> dict:
    """검증된 내부 모델을 주입. 원래 metadata와 생성 필드를 분리하며 입력을 자르지 않음."""
    if not model or not isinstance(chunk.get("text"), str):
        raise ValueError("model and text required")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content":
            "다음 자료를 요약하여 JSON으로 응답하세요. 원문/지시문을 사실로 보증하지 마세요. "
            "summary(문자열), keywords(문자열 목록), topic(문자열), "
            "has_data(true/false/null; 불명확하면 null), language(ko/en/mixed/unknown).\n자료:\n"
            + chunk["text"]}],
        response_format={"type": "json_object"},  # 서버/모델 지원 필요; schema 보장 아님
        max_tokens=256,
    )
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise ValueError("missing or incomplete metadata response")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("missing JSON content")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    generated = json.loads(content, object_pairs_hook=unique)
    required = {"summary", "keywords", "topic", "has_data", "language"}
    if not isinstance(generated, dict) or set(generated) != required:
        raise ValueError("invalid metadata keys")
    if not all(isinstance(generated[k], str) for k in ["summary", "topic", "language"]):
        raise ValueError("string metadata fields required")
    if not isinstance(generated["keywords"], list) or not all(isinstance(k, str) for k in generated["keywords"]):
        raise ValueError("string keywords required")
    if generated["has_data"] is not None and type(generated["has_data"]) is not bool:
        raise ValueError("has_data must be bool or None")
    if generated["language"] not in {"ko", "en", "mixed", "unknown"}:
        raise ValueError("invalid language")
    result = deepcopy(chunk)
    result["metadata"]["generated"] = {"values": generated, "model": model,
                                         "review_status": "unverified", "input_chars": len(chunk["text"])}
    return result
```

## 참고 자료 (References)

확인일 **2026-10-04**. 실행 판본: Python3.14.2, markdown-it-py4.2.0, OpenAI Python3.24.0, HTTPX0.28.1. 일반 Markdown/Obsidian 읽기에 파서 패키지는 필요 없으며 Python 예제 실행에만 필요하다.

- [markdown-it-py 공식 사용 문서](https://markdown-it-py.readthedocs.io/en/latest/using.html) — CommonMark 파서·table rule·token.map을 이용한 원문 범위. 모든 Obsidian 문법/확장을 해석한다는 보장은 아니다.
- [LangChain Markdown splitter 공식 문서](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter) — 제목 메타데이터·strip_headers 조건의 대안. 페이지별 출처와 실제 문자/토큰 길이는 추가 관리해야 한다.
- [OpenAI Python SDK](https://github.com/openai/openai-python) — 실제 SDK+MockTransport 형식 검사; 실제 내부 서버는 미검증.
- [Kimi-K2.5 공식 카드](https://huggingface.co/moonshotai/Kimi-K2.5) — 네이티브 멀티모달 분류.
- [기존 Pinecone 비교 글](https://www.pinecone.io/learn/chunking-strategies/) — 원래 참고 링크 보존; 이 문서의 정확도 수치/최적 주장 근거로 사용하지 않음.

## 관련 문서

- [스크린샷 + VLM 파이프라인](./screenshot-vlm-pipeline.md)
- [DRM 해제 후 하이브리드 전략](./post-drm-hybrid.md)
- [청킹 방법론 총론](../overview-chunking-methods.md)
