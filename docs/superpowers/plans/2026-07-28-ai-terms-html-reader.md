# AI Terms HTML Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 11 Markdown documents in `ai-dt/ai-terms-and-technologies/` into a polished, fully offline HTML learning portal that opens directly from `html/index.html`.

**Architecture:** A standard-library Python builder parses the repository's current Markdown subset into semantic HTML, rewrites collection links, and generates one checked-in page per source document. Shared CSS provides the editorial, responsive, theme-aware visual system; a small shared JavaScript file adds filtering, mobile navigation, progress, theme persistence, and active-table-of-contents behavior without fetching content.

**Tech Stack:** Python 3 standard library, semantic HTML5, CSS custom properties and media queries, dependency-free browser JavaScript, Python `unittest`.

## Global Constraints

- All reading functions must work from `file://` without a server, CDN, package install, or network request.
- Preserve all 11 source Markdown files without modifying their content.
- Generate exactly 11 HTML documents: `README.md` becomes `index.html`; the other filenames change only from `.md` to `.html`.
- Use only local `styles.css` and `app.js`; do not add external fonts, icons, images, or libraries.
- Escape source HTML before applying the supported Markdown transformations.
- JavaScript is progressive enhancement: article content and basic navigation remain usable when it is disabled.
- Keep all implementation files under `ai-dt/ai-terms-and-technologies/html/`.
- Stage and commit only the files named in each task; preserve unrelated working-tree changes.

---

## File Map

- `ai-dt/ai-terms-and-technologies/html/build.py`
  - Defines document order and metadata.
  - Parses front matter and the Markdown subset used by the current sources.
  - Generates anchors, table of contents, navigation, search data, and final pages.
  - Rewrites collection links and validates generated local references.
- `ai-dt/ai-terms-and-technologies/html/tests/test_build.py`
  - Covers parser behavior, escaping, link rewriting, full-site generation, deterministic output, and local reference validation.
- `ai-dt/ai-terms-and-technologies/html/styles.css`
  - Owns the complete light/dark visual system, portal and reader layouts, responsive behavior, accessibility states, and print rules.
- `ai-dt/ai-terms-and-technologies/html/app.js`
  - Owns theme switching, search/filtering, mobile navigation, reading progress, and active table-of-contents state.
- `ai-dt/ai-terms-and-technologies/html/index.html`
  - Generated home portal for `README.md`.
- `ai-dt/ai-terms-and-technologies/html/01-ai-model-families.html` through `09-prompt-rag-finetuning-and-evaluation.html`
  - Generated chapter reader pages.
- `ai-dt/ai-terms-and-technologies/html/all-in-one.html`
  - Generated integrated reader page.

---

### Task 1: Build the Safe Markdown Conversion Core

**Files:**

- Create: `ai-dt/ai-terms-and-technologies/html/build.py`
- Create: `ai-dt/ai-terms-and-technologies/html/tests/test_build.py`

**Interfaces:**

- Produces: `DocumentSpec`, `Heading`, and `ParsedDocument` dataclasses.
- Produces: `parse_front_matter(text: str, source_name: str) -> tuple[dict[str, str], str]`.
- Produces: `slugify(text: str, seen: dict[str, int]) -> str`.
- Produces: `render_inline(text: str) -> str`.
- Produces: `convert_markdown(text: str, spec: DocumentSpec) -> ParsedDocument`.
- Consumes: only Python standard-library modules.

- [ ] **Step 1: Create focused parser tests**

Create `html/tests/test_build.py` with a path-safe import and the first behavior tests:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HTML_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HTML_ROOT.parent
sys.path.insert(0, str(HTML_ROOT))

import build  # noqa: E402


class MarkdownParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = build.DocumentSpec(
            source_name="sample.md",
            output_name="sample.html",
            number="01",
            short_title="샘플",
            summary="샘플 설명",
            accent="indigo",
        )

    def test_front_matter_is_removed_and_returned(self) -> None:
        metadata, body = build.parse_front_matter(
            "---\ntags: [ai, rag]\nlevel: beginner\n---\n\n# 제목\n본문",
            "sample.md",
        )

        self.assertEqual(metadata["level"], "beginner")
        self.assertEqual(metadata["tags"], "[ai, rag]")
        self.assertEqual(body, "# 제목\n본문")

    def test_unclosed_front_matter_reports_the_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample.md"):
            build.parse_front_matter("---\nlevel: beginner\n# 제목", "sample.md")

    def test_duplicate_korean_headings_receive_stable_ids(self) -> None:
        seen: dict[str, int] = {}

        self.assertEqual(build.slugify("핵심 요약", seen), "핵심-요약")
        self.assertEqual(build.slugify("핵심 요약", seen), "핵심-요약-2")

    def test_supported_blocks_are_preserved_and_raw_html_is_escaped(self) -> None:
        parsed = build.convert_markdown(
            """---
level: beginner
---

# 샘플 문서

> 중요한 요약입니다.

## 비교

| 구분 | 설명 |
|---|---|
| RAG | 문서를 찾고 답합니다. |

- 첫 항목
- 둘째 항목

```text
검색 -> 생성
```

<script>alert("unsafe")</script>
""",
            self.spec,
        )

        self.assertEqual(parsed.title, "샘플 문서")
        self.assertEqual(parsed.lead, "중요한 요약입니다.")
        self.assertIn("<table>", parsed.body_html)
        self.assertIn("<ul>", parsed.body_html)
        self.assertIn('class="diagram-block"', parsed.body_html)
        self.assertIn("&lt;script&gt;", parsed.body_html)
        self.assertNotIn("<script>alert", parsed.body_html)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: `ERROR` because `build.py` does not exist.

- [ ] **Step 3: Implement document types, front matter, slugs, inline rendering, and block parsing**

Create `html/build.py` with these public types and entry points:

```python
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit


@dataclass(frozen=True)
class DocumentSpec:
    source_name: str
    output_name: str
    number: str
    short_title: str
    summary: str
    accent: str


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    anchor: str


@dataclass(frozen=True)
class ParsedDocument:
    spec: DocumentSpec
    metadata: dict[str, str]
    title: str
    lead: str
    body_html: str
    headings: tuple[Heading, ...]
    search_text: str


def parse_front_matter(text: str, source_name: str) -> tuple[dict[str, str], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized.strip()
    end = normalized.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{source_name}: front matter closing delimiter is missing")
    metadata: dict[str, str] = {}
    for line in normalized[4:end].splitlines():
        if ":" not in line:
            raise ValueError(f"{source_name}: invalid front matter line: {line}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, normalized[end + 5 :].strip()


def slugify(text: str, seen: dict[str, int]) -> str:
    base = re.sub(r"[^\w가-힣]+", "-", text.casefold()).strip("-") or "section"
    count = seen.get(base, 0) + 1
    seen[base] = count
    return base if count == 1 else f"{base}-{count}"
```

Implement `render_inline()` as a token-preserving scanner rather than applying Markdown regular expressions to already generated tags:

1. Protect inline code spans as escaped placeholders.
2. Protect Markdown links as placeholders with escaped targets; Task 2 extends this step by passing targets through `rewrite_link()`.
3. Escape all remaining source text with `html.escape()`.
4. Convert paired `**strong**` and `*emphasis*`.
5. Restore the protected code and link fragments.

Implement `convert_markdown()` as a line-oriented state machine. It must:

- remove front matter;
- use the first `h1` as `ParsedDocument.title` without repeating it in `body_html`;
- use the first top-level blockquote as `lead` without repeating it in `body_html`;
- emit headings with stable anchors and `Heading` entries;
- group adjacent list items into `ul` or `ol`;
- recognize a table only when a header row is followed by a Markdown delimiter row;
- render fenced `text` blocks with `diagram-block` and other fences with `code-block`;
- flush paragraphs at blank lines and block boundaries;
- preserve unrecognized lines as escaped paragraph text;
- derive `search_text` by stripping Markdown punctuation from the source body.

- [ ] **Step 4: Run the parser tests**

Run:

```bash
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: all four parser tests pass.

- [ ] **Step 5: Commit the conversion core**

```bash
git add ai-dt/ai-terms-and-technologies/html/build.py ai-dt/ai-terms-and-technologies/html/tests/test_build.py
git diff --cached --check
git commit -m "Add AI terms Markdown converter"
```

---

### Task 2: Generate the Portal and Reader Pages

**Files:**

- Modify: `ai-dt/ai-terms-and-technologies/html/build.py`
- Modify: `ai-dt/ai-terms-and-technologies/html/tests/test_build.py`
- Create: `ai-dt/ai-terms-and-technologies/html/index.html`
- Create: `ai-dt/ai-terms-and-technologies/html/01-ai-model-families.html`
- Create: `ai-dt/ai-terms-and-technologies/html/02-vector-embedding-and-rag.html`
- Create: `ai-dt/ai-terms-and-technologies/html/03-zero-shot-and-few-shot.html`
- Create: `ai-dt/ai-terms-and-technologies/html/04-vibe-coding-harness-agent-orchestration.html`
- Create: `ai-dt/ai-terms-and-technologies/html/05-ai-literacy-bias-and-slop.html`
- Create: `ai-dt/ai-terms-and-technologies/html/06-jailbreak-and-ai-security.html`
- Create: `ai-dt/ai-terms-and-technologies/html/07-ax-and-ai-data-centers.html`
- Create: `ai-dt/ai-terms-and-technologies/html/08-token-context-window-and-hallucination.html`
- Create: `ai-dt/ai-terms-and-technologies/html/09-prompt-rag-finetuning-and-evaluation.html`
- Create: `ai-dt/ai-terms-and-technologies/html/all-in-one.html`

**Interfaces:**

- Consumes: Task 1's `convert_markdown()` and document dataclasses.
- Produces: `DOCUMENTS: tuple[DocumentSpec, ...]`.
- Produces: `rewrite_link(target: str, spec: DocumentSpec, source_root: Path) -> str`.
- Produces: the extended `render_inline(text: str, spec: DocumentSpec, source_root: Path) -> str`.
- Produces: the extended `convert_markdown(text: str, spec: DocumentSpec, source_root: Path) -> ParsedDocument`.
- Produces: `render_document_page(document: ParsedDocument, documents: tuple[ParsedDocument, ...]) -> str`.
- Produces: `render_home_page(home: ParsedDocument, chapters: tuple[ParsedDocument, ...], integrated: ParsedDocument) -> str`.
- Produces: `build_site(source_root: Path, output_root: Path) -> tuple[Path, ...]`.

- [ ] **Step 1: Add full-site generation tests**

Extend `test_build.py`:

```python
import tempfile


class SiteBuildTests(unittest.TestCase):
    def test_build_generates_the_complete_offline_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generated = build.build_site(SOURCE_ROOT, output_root)

            html_names = sorted(path.name for path in generated if path.suffix == ".html")
            self.assertEqual(
                html_names,
                sorted(
                    [
                        "index.html",
                        "all-in-one.html",
                        *[
                            spec.output_name
                            for spec in build.DOCUMENTS
                            if spec.number.isdigit()
                        ],
                    ]
                ),
            )
            index = (output_root / "index.html").read_text(encoding="utf-8")
            chapter = (output_root / "02-vector-embedding-and-rag.html").read_text(
                encoding="utf-8"
            )
            self.assertIn('href="styles.css"', index)
            self.assertIn('src="app.js"', index)
            self.assertIn('data-search-card', index)
            self.assertIn("통합본으로 읽기", index)
            self.assertIn('href="03-zero-shot-and-few-shot.html"', chapter)
            self.assertNotIn('href="./03-zero-shot-and-few-shot.md"', chapter)

    def test_unknown_collection_document_link_fails_with_source_name(self) -> None:
        bad_spec = build.DocumentSpec(
            "bad.md", "bad.html", "10", "오류", "오류 문서", "rose"
        )
        with self.assertRaisesRegex(ValueError, "bad.md"):
            build.rewrite_link("./10-missing.md", bad_spec, SOURCE_ROOT)
```

- [ ] **Step 2: Run the tests and verify the new tests fail**

Run:

```bash
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: failures because `DOCUMENTS`, `build_site()`, and page renderers are not defined.

- [ ] **Step 3: Define the ordered collection**

Add one `DocumentSpec` for the home source, nine numbered chapters, and the integrated edition. Use these exact output mappings:

```python
DOCUMENTS = (
    DocumentSpec("README.md", "index.html", "", "AI 용어 및 기술 소개", "처음 시작하는 AI 학습 포털", "indigo"),
    DocumentSpec("01-ai-model-families.md", "01-ai-model-families.html", "01", "AI 모델의 종류", "LLM, SLM, VLM, VLA의 차이", "indigo"),
    DocumentSpec("02-vector-embedding-and-rag.md", "02-vector-embedding-and-rag.html", "02", "벡터·임베딩·RAG", "사내 지식을 찾고 연결하는 방법", "cyan"),
    DocumentSpec("03-zero-shot-and-few-shot.md", "03-zero-shot-and-few-shot.html", "03", "제로샷과 퓨샷", "예시로 모델의 답을 조정하는 방법", "violet"),
    DocumentSpec("04-vibe-coding-harness-agent-orchestration.md", "04-vibe-coding-harness-agent-orchestration.html", "04", "하네스·에이전트", "모델이 도구를 사용하고 협업하는 구조", "blue"),
    DocumentSpec("05-ai-literacy-bias-and-slop.md", "05-ai-literacy-bias-and-slop.html", "05", "AI 리터러시", "편향과 저품질 결과를 판단하는 기준", "amber"),
    DocumentSpec("06-jailbreak-and-ai-security.md", "06-jailbreak-and-ai-security.html", "06", "탈옥과 AI 보안", "지시와 데이터를 안전하게 구분하는 법", "rose"),
    DocumentSpec("07-ax-and-ai-data-centers.md", "07-ax-and-ai-data-centers.html", "07", "AX와 AI 데이터 센터", "조직 변화와 연산 기반 이해하기", "emerald"),
    DocumentSpec("08-token-context-window-and-hallucination.md", "08-token-context-window-and-hallucination.html", "08", "토큰·컨텍스트·환각", "모델의 입력 한계와 오류 이해하기", "orange"),
    DocumentSpec("09-prompt-rag-finetuning-and-evaluation.md", "09-prompt-rag-finetuning-and-evaluation.html", "09", "AI 적용 방법 선택", "프롬프트부터 평가까지의 결정 순서", "teal"),
    DocumentSpec("all-in-one.md", "all-in-one.html", "ALL", "한 편으로 읽는 통합본", "아홉 주제를 하나의 흐름으로 읽기", "indigo"),
)
```

- [ ] **Step 4: Implement link rewriting and semantic page shells**

Implement `rewrite_link()` with `urlsplit()`:

- return `http`, `https`, `mailto`, and fragment-only targets unchanged;
- decode the path before matching collection source names;
- map a collection Markdown filename to its output filename and retain query/fragment parts;
- for another existing local source, resolve it from `source_root / spec.source_name`, then calculate a relative path from the HTML output directory;
- raise `ValueError(f"{spec.source_name}: unresolved collection link: {target}")` when the link names a missing numbered collection Markdown file.

Implement a shared document shell containing:

```html
<!doctype html>
<html lang="ko" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{escaped_title} · AI 용어 및 기술</title>
  <link rel="stylesheet" href="styles.css">
  <script defer src="app.js"></script>
</head>
<body data-page="{escaped_output_name}" data-accent="{escaped_accent}">
  <a class="skip-link" href="#main-content">본문으로 건너뛰기</a>
  <div class="reading-progress" aria-hidden="true"><span></span></div>
  {top_bar}
  <div class="reader-layout">
    {chapter_navigation}
    <main id="main-content" class="article-shell">{document_content}</main>
    {table_of_contents}
  </div>
</body>
</html>
```

The home renderer must add:

- a hero with the collection description and statistics;
- a featured integrated-edition card;
- nine `data-search-card` chapter cards;
- a search field with `aria-controls="chapter-grid glossary-table"`;
- an initially hidden `data-search-empty` status;
- the converted `README.md` body after the portal cards.

The reader renderer must add:

- document metadata chips;
- the lead callout exactly once;
- left chapter navigation with `aria-current="page"`;
- right `h2`/`h3` table of contents;
- previous and next links based on `DOCUMENTS` order.

- [ ] **Step 5: Implement `build_site()` and generate the pages**

`build_site()` must:

1. verify every `DocumentSpec.source_name` exists;
2. reject duplicate source names and output names;
3. parse every document before rendering any output;
4. create `output_root`;
5. render home, chapters, and integrated edition in document order;
6. write UTF-8 files with Unix newlines;
7. return generated HTML paths in document order.

Add:

```python
def main() -> int:
    output_root = Path(__file__).resolve().parent
    source_root = output_root.parent
    generated = build_site(source_root, output_root)
    print(f"Generated {len(generated)} HTML documents in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```bash
python3 ai-dt/ai-terms-and-technologies/html/build.py
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: `Generated 11 HTML documents ...` and all tests pass.

- [ ] **Step 6: Commit generated semantic pages**

```bash
git add ai-dt/ai-terms-and-technologies/html/build.py ai-dt/ai-terms-and-technologies/html/tests/test_build.py ai-dt/ai-terms-and-technologies/html/*.html
git diff --cached --check
git commit -m "Generate AI terms HTML reader pages"
```

---

### Task 3: Add the Visual System and Offline Interactions

**Files:**

- Create: `ai-dt/ai-terms-and-technologies/html/styles.css`
- Create: `ai-dt/ai-terms-and-technologies/html/app.js`
- Modify: `ai-dt/ai-terms-and-technologies/html/tests/test_build.py`

**Interfaces:**

- Consumes: the semantic classes and data attributes generated in Task 2.
- Produces: `data-theme` state on `<html>`.
- Produces: live filtering for `[data-search-card]` and `#glossary-table tbody tr`.
- Produces: `.is-current` state for active table-of-contents links.
- Produces: `--reading-progress` updates or equivalent progress-bar width.

- [ ] **Step 1: Add asset and offline-safety tests**

Extend `SiteBuildTests`:

```python
    def test_local_assets_have_no_runtime_network_dependency(self) -> None:
        css = (HTML_ROOT / "styles.css").read_text(encoding="utf-8")
        javascript = (HTML_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertNotIn("@import", css)
        self.assertNotRegex(css, r"url\\(['\"]?https?://")
        self.assertNotIn("fetch(", javascript)
        self.assertNotIn("XMLHttpRequest", javascript)
        self.assertIn("localStorage", javascript)
        self.assertIn("textContent", javascript)

    def test_generated_pages_expose_accessible_interaction_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            build.build_site(SOURCE_ROOT, output_root)
            index = (output_root / "index.html").read_text(encoding="utf-8")
            chapter = (output_root / "01-ai-model-families.html").read_text(
                encoding="utf-8"
            )

            self.assertIn('class="skip-link"', index)
            self.assertIn('aria-label="색상 테마 전환"', index)
            self.assertIn('aria-controls="chapter-grid glossary-table"', index)
            self.assertIn('aria-current="page"', chapter)
```

- [ ] **Step 2: Run tests and verify the asset test fails**

Run:

```bash
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: `FileNotFoundError` for `styles.css` or `app.js`.

- [ ] **Step 3: Implement the shared editorial visual system**

Create `styles.css` beginning with an explicit token set:

```css
:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --surface: #ffffff;
  --surface-muted: #edf2f8;
  --ink: #122033;
  --muted: #5e6c80;
  --line: #dbe3ee;
  --accent: #4f46e5;
  --accent-soft: #eef2ff;
  --focus: #0891b2;
  --shadow: 0 18px 50px rgb(30 41 59 / 10%);
  --radius-lg: 24px;
  --radius-md: 14px;
  --content-width: 760px;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR",
    "Apple SD Gothic Neo", sans-serif;
  --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}

[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0b1220;
  --surface: #121c2d;
  --surface-muted: #1a2639;
  --ink: #edf4ff;
  --muted: #a4b2c7;
  --line: #2a3950;
  --accent: #8b9cff;
  --accent-soft: #202b52;
  --focus: #67e8f9;
  --shadow: 0 18px 50px rgb(0 0 0 / 28%);
}
```

Implement styles for:

- skip link and `:focus-visible`;
- sticky top bar and progress line;
- home hero, statistics, featured card, search field, chapter grid, and chapter cards;
- three-column reader layout with sticky rails;
- article typography with a readable 760px maximum line width;
- lead callouts, responsive table wrappers, striped rows, inline code, code panels, and diagram panels;
- chapter accent colors via `[data-accent="..."]`;
- mobile navigation at `max-width: 980px` and single-column card layout at `max-width: 640px`;
- `prefers-reduced-motion: reduce`;
- print rules that hide navigation and controls while flattening colors and shadows.

- [ ] **Step 4: Implement progressive interactions**

Create `app.js` as an IIFE in strict mode. Use named functions with these responsibilities:

```javascript
(() => {
  "use strict";

  const root = document.documentElement;

  function safeStorageGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function safeStorageSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      // Theme still applies to the current page.
    }
  }

  function setTheme(theme) {
    root.dataset.theme = theme;
    const button = document.querySelector("[data-theme-toggle]");
    if (button) {
      const isDark = theme === "dark";
      button.setAttribute("aria-pressed", String(isDark));
      button.querySelector("[data-theme-label]").textContent = isDark
        ? "밝은 테마"
        : "어두운 테마";
    }
  }
```

Continue the IIFE with:

- initial theme from `localStorage`, otherwise `prefers-color-scheme`;
- click handling for `[data-theme-toggle]`;
- case-insensitive filtering of card `data-search-text` and glossary row `textContent`;
- safe empty-state output through `textContent`;
- progress updates on `scroll` and `resize`, clamped between 0 and 1;
- mobile menu buttons that toggle `hidden` and synchronize `aria-expanded`;
- `IntersectionObserver` for heading anchors, with a scroll-position fallback when unavailable.

Do not use `innerHTML`, `fetch`, `XMLHttpRequest`, or external module imports.

- [ ] **Step 5: Run automated verification**

Run:

```bash
python3 ai-dt/ai-terms-and-technologies/html/build.py
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
git diff --check -- ai-dt/ai-terms-and-technologies/html
```

Expected: 11 pages generated, all tests pass, and `git diff --check` reports no errors.

- [ ] **Step 6: Commit the visual and interaction layer**

```bash
git add ai-dt/ai-terms-and-technologies/html/styles.css ai-dt/ai-terms-and-technologies/html/app.js ai-dt/ai-terms-and-technologies/html/tests/test_build.py ai-dt/ai-terms-and-technologies/html/*.html
git diff --cached --check
git commit -m "Style AI terms offline reader"
```

---

### Task 4: Validate Determinism, Links, Responsive Layout, and Direct-File Use

**Files:**

- Modify: `ai-dt/ai-terms-and-technologies/html/build.py`
- Modify: `ai-dt/ai-terms-and-technologies/html/tests/test_build.py`
- Regenerate: `ai-dt/ai-terms-and-technologies/html/*.html`

**Interfaces:**

- Consumes: all builder and asset interfaces from Tasks 1–3.
- Produces: `validate_local_references(output_root: Path, generated: tuple[Path, ...]) -> None`.
- Produces: deterministic checked-in output matching a fresh build.

- [ ] **Step 1: Add deterministic-build and local-reference tests**

Add:

```python
    def test_build_is_deterministic_and_local_references_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            first_paths = build.build_site(SOURCE_ROOT, first_root)
            second_paths = build.build_site(SOURCE_ROOT, second_root)

            first_output = {
                path.name: path.read_text(encoding="utf-8") for path in first_paths
            }
            second_output = {
                path.name: path.read_text(encoding="utf-8") for path in second_paths
            }
            self.assertEqual(first_output, second_output)
            build.validate_local_references(first_root, first_paths)

    def test_every_source_heading_is_present_in_generated_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            build.build_site(SOURCE_ROOT, output_root)

            for spec in build.DOCUMENTS:
                source = (SOURCE_ROOT / spec.source_name).read_text(encoding="utf-8")
                output = (output_root / spec.output_name).read_text(encoding="utf-8")
                source_headings = re.findall(r"^#{1,3}\\s+(.+)$", source, re.MULTILINE)
                for heading in source_headings:
                    self.assertIn(html.escape(heading), output)
```

Add the missing standard-library imports `html` and `re` to the test module.

- [ ] **Step 2: Run tests and verify the validator test fails**

Run:

```bash
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: failure because `validate_local_references()` is not defined.

- [ ] **Step 3: Implement local-reference validation**

Use `html.parser.HTMLParser` to collect `href` and `src` attributes from every generated page. `validate_local_references()` must:

- ignore fragments and `http`, `https`, and `mailto` targets;
- strip query and fragment parts;
- URL-decode local paths;
- resolve them relative to each generated page;
- verify files exist;
- for generated HTML fragments, verify the target `id` exists;
- aggregate every failure and raise one `ValueError` listing source page and broken target.

Call the validator at the end of `build_site()` after all documents have been written.

- [ ] **Step 4: Run the full build and tests twice**

Run:

```bash
python3 ai-dt/ai-terms-and-technologies/html/build.py
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
python3 ai-dt/ai-terms-and-technologies/html/build.py
python3 -m unittest discover -s ai-dt/ai-terms-and-technologies/html/tests -v
```

Expected: both runs generate 11 documents and every test passes.

- [ ] **Step 5: Perform direct-file visual verification**

Open:

```text
file:///Users/daeyoung/Codes/pm_notes/ai-dt/ai-terms-and-technologies/html/index.html
```

Verify at desktop width:

- hero, integrated-edition card, nine chapter cards, glossary, and related links are visually distinct;
- searching `RAG` leaves relevant cards and glossary rows;
- searching a nonexistent term shows the empty state without inserting HTML;
- light/dark theme changes and persists after reload;
- chapter navigation, previous/next links, and table-of-contents links work.

Verify at 360px width:

- the page has no viewport-wide horizontal overflow;
- tables scroll inside their own wrapper;
- chapter and table-of-contents menus open by keyboard and pointer;
- body text remains legible without zooming.

Verify `01-ai-model-families.html`, `02-vector-embedding-and-rag.html`, and `all-in-one.html`:

- lead text appears once;
- tables, diagram blocks, lists, and external links are styled correctly;
- reading progress and active table-of-contents state update;
- disabling JavaScript still leaves the article and basic links readable.

Capture desktop and mobile screenshots for visual inspection during the implementation session; do not add screenshots to the repository.

- [ ] **Step 6: Check scope and commit final validation changes**

Run:

```bash
git diff --check -- ai-dt/ai-terms-and-technologies/html
git status --short
git diff --name-only HEAD~3..HEAD
```

Confirm that the implementation touched only:

- `docs/superpowers/plans/2026-07-28-ai-terms-html-reader.md`
- `ai-dt/ai-terms-and-technologies/html/`

Then commit any validator or regenerated-page changes:

```bash
git add ai-dt/ai-terms-and-technologies/html/build.py ai-dt/ai-terms-and-technologies/html/tests/test_build.py ai-dt/ai-terms-and-technologies/html/*.html
git diff --cached --check
git commit -m "Validate AI terms offline reader"
```

If no final changes remain after validation, do not create an empty commit.
