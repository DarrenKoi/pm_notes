---
tags: [rag, document-loaders, pdf, web, vlm, drm, metadata]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 12. PDF, 웹 문서, 내부 지식 적용 사례 실습

> PDF·웹·사내 문서를 RAG에 넣기 위한 로더와 전처리를 다룬다. 특히 DRM으로 텍스트 추출이 막힌 사내 문서는 스크린샷+VLM 파이프라인으로 텍스트화한다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- RAG의 입력은 결국 **다양한 포맷의 실문서**다. 포맷마다 로더와 전처리가 다르다.
- 원래 “사내99% DRM/유일한 경로” 주장은 출처·측정이 없어 미확인이다. 승인된 export/OCR/텍스트 추출 가능 여부를 먼저 확인하고 캡처+VLM은 후보 경로로 평가한다.
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
import os
import base64, glob, re
from openai import OpenAI
from langchain_core.documents import Document

# 사내 VLM (OpenAI 호환). 8B/30B는 원래 라우팅 후보; 정확도·속도·서빙 alias는 미확인
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

def ocr_page(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model=os.environ["VLM_MODEL"],     # 정확도 우선. 대량은 Qwen3-VL-8B-Instruct
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "이 페이지의 모든 텍스트를 표/수식 포함해 그대로 마크다운으로 옮겨라. 설명 금지."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=0,
    )
    content = resp.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OCR 텍스트 미관측; 빈 페이지로 기록하지 않음")
    return content

# 페이지 이미지들 → Document 리스트
docs = []
for i, p in enumerate(sorted(glob.glob("captures/doc1_page_*.png"),
                           key=lambda p: int(re.search(r"_page_(\d+)\.png$", p).group(1)))):
    text = ocr_page(p)
    docs.append(Document(page_content=text,
                         metadata={"source": "doc1", "page": int(re.search(r"_page_(\d+)\.png$", p).group(1)), "extractor": "vlm", "image_path": p,
                                   "review_status": "unverified_ocr"}))
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
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"],
                              check_embedding_ctx_length=False, encoding_format="float",
                              base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

all_docs = docs  # + pdf_docs + web_docs ...
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)\
         .split_documents(all_docs)
vs = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
vs.save_local("faiss_kb")
```

### 6) 증분 갱신과 중복 방지 (운영)
문서가 갱신될 때 전체 재구축은 낭비다. LangChain **indexing API**는 record manager와 content/metadata hash·source id·cleanup mode로 변경분을 관리한다. 아래는 import 개념만 제시하며 실제 인덱싱/삭제는 미구현이다.
```python
# from langchain_classic.indexes import index, SQLRecordManager  (개념: 변경분만 upsert/삭제)
```

## 사내 적용 정리
1. DRM 문서 → 캡처 → **Qwen3-VL OCR** → Document 화 (가장 큰 병목이자 차별화 포인트).
2. 표/수식 페이지는 30B, 단순 페이지는 8B로 라우팅해 비용/표·수식/누락률을 실측할 후보.
3. metadata에 source/page/extractor/date를 남겨 **인용·필터·검증** 가능하게.
4. 임베딩은 BGE-M3 같은 모델/판본·전처리·차원·정규화 계약을 유지한다. 변경 시 전체 재임베딩 조건을 검토한다.

## 관련 문서
- [09. 문서 임베딩 & FAISS](./09-document-embedding-faiss.md) · [10. Retriever 튜닝](./10-retriever-tuning.md)
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md)
- [13. Mini Project](./13-mini-project.md) — 이 파이프라인으로 실제 프로젝트 구성

## 참고 자료 (References)
- Document loaders: https://docs.langchain.com/oss/python/integrations/document_loaders/
- Qwen3-VL (사내 서빙), OpenAI 호환 vision 메시지 포맷
