---
tags: [rag, document-loaders, pdf, web, vlm, drm, metadata]
level: advanced
last_updated: 2026-07-06
---

# 12. PDF, 웹 문서, 내부 지식 적용 사례 실습

> PDF·웹·사내 문서를 RAG에 넣기 위한 로더와 전처리를 다룬다. 특히 DRM으로 텍스트 추출이 막힌 사내 문서는 스크린샷+VLM 파이프라인으로 텍스트화한다.

## 왜 필요한가? (Why)

- RAG의 입력은 결국 **다양한 포맷의 실문서**다. 포맷마다 로더와 전처리가 다르다.
- 사내 문서의 **99%가 DRM**이라 일반 PDF 파서로는 텍스트가 안 뽑힌다 → **스크린샷 + VLM(Qwen3-VL)** 이 사실상 유일한 추출 경로.
- 메타데이터(출처/날짜/문서유형)를 잘 붙여야 [필터 검색](./10-retriever-tuning.md)과 인용이 가능하다.

## 핵심 개념 (What)

### Document 객체
LangChain의 모든 소스는 `Document(page_content=str, metadata=dict)` 리스트로 정규화된다. 로더의 역할은 "어떤 포맷 → Document 리스트" 변환.

### 소스별 로더
| 소스 | 로더 |
|------|------|
| PDF(텍스트) | `PyPDFLoader`, `PyMuPDFLoader` |
| PDF(스캔/DRM) | 이미지화 후 **VLM OCR** (아래) |
| 웹 | `WebBaseLoader`, `AsyncHtmlLoader`+`BeautifulSoupTransformer` |
| 디렉터리 | `DirectoryLoader` |
| Markdown/텍스트 | `TextLoader`, `UnstructuredMarkdownLoader` |

## 어떻게 사용하는가? (How)

### 1) 일반 PDF
```python
from langchain_community.document_loaders import PyPDFLoader
docs = PyPDFLoader("manual.pdf").load()     # 페이지별 Document, metadata에 page 포함
```

### 2) 웹 문서
```python
from langchain_community.document_loaders import WebBaseLoader
docs = WebBaseLoader(["https://example.com/guide"]).load()

# 본문만 정제하고 싶으면 HTML 변환기 사용
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
html = AsyncHtmlLoader(["https://example.com/guide"]).load()
docs = BeautifulSoupTransformer().transform_documents(html, tags_to_extract=["p","li","h1","h2"])
```

### 3) DRM 문서 → 스크린샷 + VLM (사내 핵심 파이프라인)
DRM으로 텍스트 추출이 막힌 문서는 **페이지를 이미지로 캡처**한 뒤 VLM에게 "이 페이지의 텍스트를 그대로 옮겨라"라고 시킨다.
```python
import base64, glob
from openai import OpenAI
from langchain_core.documents import Document

# 사내 VLM (OpenAI 호환). 가벼운 8B 또는 정확한 30B 선택
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")

def ocr_page(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model="Qwen3-VL-30B",     # 정확도 우선. 대량은 Qwen3-VL-8B-Instruct
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "이 페이지의 모든 텍스트를 표/수식 포함해 그대로 마크다운으로 옮겨라. 설명 금지."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=0,
    )
    return resp.choices[0].message.content

# 페이지 이미지들 → Document 리스트
docs = []
for i, p in enumerate(sorted(glob.glob("captures/doc1_page_*.png"))):
    text = ocr_page(p)
    docs.append(Document(page_content=text,
                         metadata={"source": "doc1", "page": i+1, "extractor": "vlm"}))
```
> 팁: 표·수식이 많은 페이지는 30B, 단순 텍스트는 8B로 **비용/정확도**를 조절. OCR 결과는 원본 이미지 경로를 metadata에 남겨 검증 가능하게 한다.

### 4) 메타데이터 부여와 정규화
검색 필터·인용을 위해 일관된 메타데이터를 붙인다.
```python
for d in docs:
    d.metadata.setdefault("doc_type", "manual")
    d.metadata.setdefault("lang", "ko")
    # d.metadata["date"] = "2026-06-01"  # 날짜 필터용
```

### 5) 통합 인덱싱 (여러 소스 → 하나의 벡터스토어)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="BGE-M3",
                              base_url="http://llm-gateway.internal/v1", api_key="EMPTY")

all_docs = docs  # + pdf_docs + web_docs ...
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)\
         .split_documents(all_docs)
vs = FAISS.from_documents(chunks, embeddings)
vs.save_local("faiss_kb")
```

### 6) 증분 갱신과 중복 방지 (운영)
문서가 갱신될 때 전체 재구축은 낭비다. LangChain **indexing API**는 소스 해시로 변경분만 갱신한다.
```python
# from langchain.indexes import index, SQLRecordManager  (개념: 변경분만 upsert/삭제)
```

## 사내 적용 정리
1. DRM 문서 → 캡처 → **Qwen3-VL OCR** → Document 화 (가장 큰 병목이자 차별화 포인트).
2. 표/수식 페이지는 30B, 단순 페이지는 8B로 라우팅해 비용 최적화.
3. metadata에 source/page/extractor/date를 남겨 **인용·필터·검증** 가능하게.
4. 임베딩은 BGE-M3 하나로 통일해 소스가 달라도 같은 공간에서 검색.

## 관련 문서
- [09. 문서 임베딩 & FAISS](./09-document-embedding-faiss.md) · [10. Retriever 튜닝](./10-retriever-tuning.md)
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md)
- [13. Mini Project](./13-mini-project.md) — 이 파이프라인으로 실제 프로젝트 구성

## 참고 자료 (References)
- Document loaders: https://docs.langchain.com/oss/python/integrations/document_loaders/
- Qwen3-VL (사내 서빙), OpenAI 호환 vision 메시지 포맷
