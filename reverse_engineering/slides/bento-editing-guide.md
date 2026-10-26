---
tags: [bento, slides, agent-guide, reverse-engineering]
level: intermediate
last_updated: 2026-10-26
document_type: guide
category_major: "리버스 엔지니어링"
category_middle: "발표 자료"
category_minor: "Bento deck 편집 지침"
note_kind: "에이전트 지침"
---

# RE·REA 팀 공유 deck 편집 지침

> `re-rea-team-share.bento.html`을 이어서 고치는 LLM과 사람을 위한 지침. 고칠 곳은 파일 안의 JSON 블록 하나뿐이고, 내용의 원본은 이 폴더의 `.md` 문서들이다.

## 왜 필요한가? (Why)

이 deck은 Claude가 만들고 codex가 원본 대조 리뷰를 했다. 다음에 다른 모델이 슬라이드를 고치거나 추가할 때 세 가지가 깨지기 쉽다.

- **파일 구조**: deck은 압축된 앱 런타임과 문서 JSON이 한 HTML에 들어 있다. JSON 밖을 건드리거나 파일 전체를 다시 만들면 앱이 깨진다.
- **디자인 일관성**: 색, 글꼴, 머리글 위치, morph용 id가 슬라이드 사이에서 맞물려 있다.
- **내용의 정확성**: 원본 문서의 단서("후보 단서", "실험 단계", 버전 번호)를 슬라이드로 줄이다 보면 지워진다. 리뷰에서 실제로 지적된 문제다.

## 핵심 개념 (What)

### 파일 구성

| 파일 | 역할 |
|---|---|
| `re-rea-team-share.bento.html` | deck 본체. 브라우저로 열면 편집기가 뜬다. 약 700KB |
| `bento-editing-guide.md` | 이 문서 |
| `../*.md` | 내용의 원본. 슬라이드의 모든 사실은 여기에 근거가 있어야 한다 |

deck 안의 문서는 아래 블록 하나에 JSON으로 들어 있다. 이 블록만 고친다.

```html
<script type="application/bento+json" id="bento-doc"> { "format": "bento/slides", ... } </script>
```

Bento 형식 전체 스키마는 <https://bento.page/agents.md>에 있다. 처음 편집한다면 먼저 읽는다. 이 문서는 그 위에 이 deck만의 규칙을 더한다.

### 바꾸면 안 되는 것

- JSON 블록 밖의 내용(런타임). 파일을 처음부터 다시 만들지 않는다.
- HTML 블록에 넣는 JSON의 모든 `<`는 여섯 글자 `\u003c`로 이스케이프해서 저장한다. JSON 문자열 안에서는 같은 문자로 읽히고, 블록 안에 `</script>`가 생겨 블록이 일찍 닫히는 일을 막는다. 아래 3절의 스크립트가 이 처리를 한다.
- 최상위에 `docId`와 `collab`이 없다. 일부러 뺀 것이다. 앱이 처음 열 때 새로 만든다. 편집기에서 저장한 파일에 `collab`(특히 `ownerPriv`, `writerPriv`)이 생겼다면 그 파일을 채팅이나 외부 LLM에 넘기지 않는다. 공유용은 편집기의 *Save → Save read-only copy*로 만든다.

### 디자인 토큰

| 이름 | 값 | 쓰는 곳 |
|---|---|---|
| 배경 | `#0E1217` | 모든 슬라이드 `background` |
| 본문 | `#ECE8E1` | 기본 글자 |
| 보조 | `#8A939E` | 설명, 각주, 비활성 |
| 강조(유일한 accent) | `#F2A541` | 머리글, 번호, 경고 테두리, 강조 줄 |
| 면 | `#171D25`, `#202833` | 카드 배경, 표 머리 |
| 선 | `rgba(236,232,225,0.14)` | 카드·표 테두리 |
| 강조 위 글자 | `#14110B` | 강조색으로 채운 상자 안 글자 |
| 본문 글꼴 | `'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', system-ui, sans-serif` | 모든 텍스트 |
| 코드 글꼴 | `Consolas, 'SF Mono', Menlo, monospace` | 코드 블록, 명령어, 큰 숫자 |

강조색은 하나만 쓴다. 새 색을 추가하지 않는다. 글꼴은 두 개(본문, 코드)로 끝이다. 글꼴은 파일에 넣지 않고 시스템 글꼴을 쓴다. 회사 Windows PC에서는 `Malgun Gothic`으로 보인다.

### 슬라이드 공통 틀

표지를 뺀 모든 슬라이드에 아래 네 요소가 같은 `id`, 같은 위치로 있다. `transition: "morph"`이므로 이 요소들이 슬라이드 사이에서 제자리에 머문다.

| id | 위치(x, y, w, h) | 내용 |
|---|---|---|
| `kicker` | 96, 40, 700, 26 | 절 번호와 이름. 예: `03 · REA 사용법`. 16px, 강조색 |
| `title` | 96, 72, 1088, 84 | 슬라이드 제목. 40px, 800, `role: "title"` |
| `accent-bar` | 96, 160, 48, 4 | 강조색 막대. 표지에서는 크게 있다가 2장에서 작아지며 morph |
| `pg` | 1024, 668, 160, 24 | `{{page}} / {{pages}}`. 14px, 오른쪽 정렬 |

본문은 y 196~208에서 시작해 y 648 안에서 끝낸다. 좌우 여백은 96px이고 오른쪽 끝은 x 1184다. 열 나누기는 아래 값을 쓴다.

| 나누기 | 너비 | x |
|---|---|---|
| 2열 | 528 | 96, 656 |
| 3열 | 340 | 96, 470, 844 |
| 4열 | 254 | 96, 374, 652, 930 |

### 이 deck에서 쓰는 Bento 기능

| 기능 | 슬라이드 | 고칠 때 주의 |
|---|---|---|
| morph(같은 id 공유) | 4~6장 `rail0~2`, `rail-hl`, `src*`, `code-bg` / 9~10장 `st0~6`, `flow`, `loop` | 세 층 슬라이드의 하이라이트(`rail-hl`)는 y만 바뀐다. 7단계 상자는 채우기 색만 바뀐다. id를 바꾸면 애니메이션이 끊긴다 |
| state 슬라이드 | `prompt-example`(`stateOf: "prompt"`) | 14장의 `ex-btn`, `ex-btn-t`가 `link`로 연결한다. `dismiss` 투명 사각형이 부모로 돌아간다 |
| dash-march 반복 | 9~10장 `flow`, `loop`, 13장 연결선 | `strokeStyle: "dashed"`가 있어야 움직임이 보인다 |
| motion-path 반복 | 표지 `hex`, `orbit` | 진입 효과(`fx.enter`)를 같이 넣지 않는다 |
| countUp | 20장 `big-num`(330) | 숫자만 넣는다. 단위 `초`는 별도 요소 |
| 표(table) | 3, 8, 11, 15, 19장 | 표 스타일은 `tbl-*` 요소의 `style` 객체를 그대로 복사해 쓴다 |

### 런타임에서 확인한 제약

이 deck을 만들면서 실제 렌더링으로 확인했다.

- **인라인 `style` 속성은 지워진다.** 텍스트 `html` 안의 `<span style="color:...">`, `<b style="...">`는 모양 없이 글자만 남는다. 쓸 수 있는 것은 `<b>`, `<i>`, `<br>` 정도다. 한 줄 전체를 강조색으로 만들려면 요소의 `color`를 바꾼다. 한 문장 안에서 일부만 색을 바꿀 수는 없으므로 `<b>`로 강조한다.
- **텍스트 상자 높이는 자동으로 늘지 않는다.** 글이 넘치면 아래 요소와 겹친다. 고친 뒤에는 반드시 `validate()`의 `text-overflow`를 확인한다. 미리 재려면 `window.bento.measure({ html, w, fontSize, lineHeight })`를 쓴다.
- **브라우저로 열기만 해도 앱이 메모리에 collab 키를 만든다.** `validate()`가 `collab-secrets-present`를 info로 보고한다. 편집기에서 저장하지 않는 한 디스크 파일에는 들어가지 않는다.

## 어떻게 사용하는가? (How)

### 1. 내용부터 원본 문서에서 확인한다

슬라이드에 넣을 사실은 이 폴더의 `.md`에서 찾아 근거로 삼는다. 폴더 밖 문서를 근거로 쓰지 않는다. 슬라이드별 원본은 아래와 같다.

| # | 슬라이드 id | 주제 | 원본 문서 |
|---|---|---|---|
| 1 | `cover` | 표지 | README.md |
| 2 | `why` | 왜 필요한가 | re-fundamentals.md |
| 3 | `compile` | 컴파일은 정보를 버린다 | re-fundamentals.md |
| 4~6 | `layer-bytes`, `layer-asm`, `layer-pseudo` | 기계어·어셈블리·의사 코드 | re-fundamentals.md |
| 7 | `static-dynamic` | 정적·동적 분석 | re-fundamentals.md |
| 8 | `evidence` | 증거의 등급, 한계·커버리지 | re-fundamentals.md, rea-usage-guide.md |
| 9~10 | `steps`, `steps-human` | 7단계 절차, 사람의 일, 분석 기록 | re-process.md, templates/investigation-log.md |
| 11 | `questions` | 좋은 질문 | re-process.md |
| 12 | `safety` | 허가와 실행 환경 | re-process.md |
| 13 | `rea-arch` | REA 구성, 결과는 모델로 간다 | rea-usage-guide.md, README.md |
| 14 | `prompt` (+ `prompt-example`) | 프롬프트 네 가지 | rea-usage-guide.md |
| 15 | `targets` | 대상별 남는 것과 첫 도구 | re-fundamentals.md, rea-usage-guide.md |
| 16 | `windows` | Windows + pi 되는 것·안 되는 것 | rea-windows-pi-setup.md |
| 17 | `setup` | pi에 REA 연결 | rea-windows-pi-setup.md |
| 18 | `web` | 웹 앱 관찰 순서와 소스 맵 | rea-usage-guide.md, source-map-discovery.md |
| 19 | `installed` | 설치형 프로그램 7단계 | installed-program-analysis.md |
| 20 | `pitfalls` | 자주 밟는 함정 | rea-usage-guide.md |
| 21 | `community` | 커뮤니티 작업법 | community-resources.md |
| 22 | `next` | 실습과 읽는 순서 | re-fundamentals.md, README.md |

원본 `.md`가 바뀌면 해당 슬라이드도 같이 고친다. 반대로 슬라이드에만 있는 사실을 만들지 않는다.

### 2. 줄이면 안 되는 단서를 지킨다

codex 리뷰(2026-10-26)에서 지적되어 반영한 항목이다. 문구를 줄일 때 아래 단서가 사라지지 않게 한다.

- Windows의 Ghidra 연동은 **실험 단계(P0)이고 읽기 전용**이다. "읽기 전용"만 쓰면 정식 지원처럼 보인다.
- 함수 이름이 사라지는 것은 **스트립된 바이너리**의 경우다.
- 15바이트, `cmovl` 예제는 **Windows x64, clang -O1** 결과다. 의사 코드는 **실제 Ghidra 출력이 아니라 형태 예시**다.
- **.NET NativeAOT**는 Ghidra 경로이고, CLI 메타데이터가 없다는 것만으로 NativeAOT라고 확정하지 않는다.
- 임포트는 **후보 단서**이고 실제 호출을 증명하지 않는다.
- `trace_native_values`는 메모리 별칭·전역 상태를 거치는 값을 미확인으로 남긴다. `match_native_functions`는 **대응 후보**만 낸다.
- 설정 절차는 `/reload`로 끝나지 않는다. **`binary_session`으로 연결을 확인**해야 끝난다. 설치 전제(Node.js, Ghidra 12.1.x, JDK 21)는 setup이 설치하지 않는다.
- 웹 관찰은 **관찰을 시작한 뒤에** 버튼을 누르거나 새로고침한다. REA는 연결 전 요청을 기록하지 않는다.
- HAR에는 쿠키·토큰이 있다. 저장소·공유 폴더에 올리지 않는다.
- 회사 LLM의 컨텍스트 크기 같은 내부 구성 수치는 공유용 deck에 넣지 않는다.

장비명, 내부 시스템 상세, 디컴파일 코드 원문, 실제 사내 URL은 넣지 않는다. 예시는 `RecipeTool.exe`, `internal-app.example`처럼 가상 이름을 쓴다.

### 3. JSON을 꺼내서 고치고 다시 넣는다

아래 스크립트는 표준 라이브러리만 쓴다. 저장소 루트에서 실행한다.

```python
#!/usr/bin/env python3
"""Extract or write back the #bento-doc JSON of a Bento deck.

Usage:
  python3 bento_doc.py extract DECK.bento.html doc.json
  python3 bento_doc.py inject  DECK.bento.html doc.json
"""
import json
import re
import sys

BLOCK = re.compile(r'(<script type="application/bento\+json" id="bento-doc">)(.*?)(</script>)', re.S)


def main(mode, deck, doc_path):
    page = open(deck, encoding="utf-8").read()
    match = BLOCK.search(page)
    if not match:
        sys.exit("bento-doc block not found")
    if mode == "extract":
        doc = json.loads(match.group(2))
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"{len(doc['slides'])} slides -> {doc_path}")
        return
    doc = json.load(open(doc_path, encoding="utf-8"))
    for key in ("docId", "collab"):
        if key in doc:
            sys.exit(f"refusing to write '{key}' into the shared deck; remove it first")
    payload = json.dumps(doc, ensure_ascii=False, indent=1).replace("<", "\\u003c")
    page = page[:match.start(2)] + payload + page[match.end(2):]
    open(deck, "w", encoding="utf-8").write(page)
    print(f"{len(doc['slides'])} slides written into {deck}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
```

```bash
python3 bento_doc.py extract reverse_engineering/slides/re-rea-team-share.bento.html /tmp/doc.json
# /tmp/doc.json 편집
python3 bento_doc.py inject reverse_engineering/slides/re-rea-team-share.bento.html /tmp/doc.json
```

`doc.json`은 작업용 임시 파일이다. 저장소에 올리지 않는다. 사람은 브라우저 편집기에서 직접 고쳐도 된다. 그때는 편집기의 저장 기능으로 같은 파일에 덮어쓰고, 위 `collab` 주의를 지킨다.

### 4. 새 슬라이드를 추가할 때

1. 비슷한 슬라이드 하나를 복사해서 시작한다. 공통 틀 네 요소(`kicker`, `title`, `accent-bar`, `pg`)는 그대로 둔다.
2. 슬라이드 `id`와 요소 `id`를 새로 정한다. 요소 id는 슬라이드 안에서 겹치지 않게 하고, morph로 이어질 요소만 앞 슬라이드와 같은 id를 쓴다.
3. `transition: "morph"`, `background: "#0E1217"`을 둔다.
4. `notes`(발표자 노트)를 반드시 쓴다. 발표 원고 역할을 한다.
5. `kicker`의 절 번호(`01 · 기초` … `08 · 시작하기`)를 앞뒤 슬라이드와 맞춘다.
6. 위 슬라이드 표와 이 문서의 슬라이드 목록을 갱신한다.

### 5. 확인한다

JSON만 보고 끝내지 않는다. 브라우저로 열어서 본다.

```bash
cd reverse_engineering/slides && python3 -m http.server 8765 --bind 127.0.0.1
# 브라우저에서 http://127.0.0.1:8765/re-rea-team-share.bento.html
```

브라우저 개발자 도구 콘솔에서 실행한다.

```js
const { counts, findings } = window.bento.validate()
findings.filter(f => f.severity !== 'info')
```

`error`와 `warning`이 0이어야 한다. info 세 가지(`collab-secrets-present`, `font-not-embedded`, 표지 `hex`의 `past-margin`)는 의도된 것이다. 그다음 모든 슬라이드를 넘겨 보면서 글자 겹침, 어색한 줄바꿈, 화면 밖으로 나간 요소를 확인한다. `validate()`는 겹침을 다 잡지 못한다.

내용을 크게 바꿨다면 다른 모델에게 원본 `.md`와 대조하는 리뷰를 맡긴다. 이 deck은 "원본보다 강하게 말하지 않았는가"를 기준으로 리뷰했다.

## 참고 자료 (References)

- Bento 에이전트용 형식 문서: <https://bento.page/agents.md>
- Bento 앱과 템플릿: <https://bento.page>

## 관련 문서

- [리버스 엔지니어링 노트 목차](../README.md)
- [커뮤니티 자료에서 가져온 작업법](../community-resources.md)
