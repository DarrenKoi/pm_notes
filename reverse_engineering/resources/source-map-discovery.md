---
tags: [reverse-engineering, source-map, javascript, web, devtools]
level: beginner
last_updated: 2026-10-24
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "웹 앱"
category_minor: "소스 맵"
note_kind: "사용법"
---

# 소스 맵 찾고 원본 꺼내는 방법

> 번들 끝의 `sourceMappingURL` 주석, 응답 헤더 `SourceMap`, 관례상 경로 `번들.js.map` 순서로 찾는다. 찾은 맵에 `sourcesContent`가 있으면 원본 파일을 그대로 꺼낼 수 있다.

## 왜 필요한가? (Why)

사내 웹 앱의 프런트엔드 코드는 브라우저에 이미 다 내려와 있다. 다만 압축(Minify)되어 있어서 `function a(e,t){return e.n>t?-1:...}` 같은 한 줄짜리 코드로 보인다. 이 상태로도 읽을 수는 있지만 느리다.

소스 맵이 있으면 사정이 달라진다. 변수 이름, 파일 구조(`src/views/RecipeList.vue`), 주석까지 남은 원본을 얻을 수 있다. 웹 앱 분석에서 소스 맵 확인이 첫 단계인 이유다. 몇 분 만에 찾으면 몇 시간짜리 번들 읽기를 건너뛴다.

## 핵심 개념 (What)

### 소스 맵 파일의 구조

소스 맵은 JSON 파일이다. 빌드 도구가 번들과 함께 만든다.

```json
{
  "version": 3,
  "file": "app.4f2a1c.js",
  "sources": ["webpack:///./src/main.js", "webpack:///./src/views/RecipeList.vue"],
  "sourcesContent": ["import { createApp } from 'vue'\n...", "<template>\n..."],
  "names": ["createApp", "recipeList"],
  "mappings": "AAAA,SAASA,..."
}
```

| 필드 | 뜻 | 분석에서 쓰는 곳 |
|---|---|---|
| `sources` | 원본 파일 경로 목록 | 프로젝트 폴더 구조. 이것만 있어도 화면과 모듈 이름을 안다 |
| `sourcesContent` | 원본 파일 내용. `sources`와 같은 순서 | 있으면 원본 소스를 그대로 얻는다. 선택 필드라 없을 수도 있다 |
| `names` | 압축 전 변수, 함수 이름 | 압축된 코드에서 원래 이름 복원 |
| `mappings` | 압축된 코드의 위치와 원본 위치의 대응표(Base64 VLQ 인코딩) | DevTools가 쓴다. 사람이 직접 읽을 일은 없다 |
| `sections` | 여러 맵을 이어 붙인 색인 맵(Index Map)일 때만 있다 | 맵이 안에 여러 개 들어 있다 |

`sourcesContent`가 비어 있으면 원본 내용은 없고 경로와 이름만 얻는다. 그래도 `sources` 목록 하나로 프로젝트 구조를 파악할 수 있다.

### 번들이 소스 맵을 가리키는 세 가지 방법

1. **파일 끝의 주석**: 가장 흔하다. 번들 마지막 줄에 이렇게 붙는다. CSS는 `/*# sourceMappingURL=... */` 형태다.

   ```js
   //# sourceMappingURL=app.4f2a1c.js.map
   ```

   주소가 상대 경로면 번들의 URL을 기준으로 해석한다. `data:application/json;base64,...`로 시작하면 맵이 번들 안에 통째로 들어 있는 인라인 맵이다.

2. **HTTP 응답 헤더**: 서버가 번들을 내려줄 때 `SourceMap: /maps/app.js.map` 헤더를 붙인다. 오래된 서버는 `X-SourceMap`을 쓴다. 주석이 없어도 헤더가 있을 수 있다.

3. **아무 표시 없음(숨긴 소스 맵)**: 빌드 도구의 `hidden` 옵션은 `.map` 파일은 만들고 주석만 뺀다. 에러 추적 서비스에 올리려고 만든 맵이 서버에 그대로 남아 있는 경우가 많다. 관례상 `번들 주소 + .map`에 있으므로 직접 요청해 본다.

### 빌드 도구별 기본값

배포본에 소스 맵이 있을지 미리 짐작하는 데 쓴다. 기본값은 버전마다 바뀔 수 있으므로, 프로젝트 설정 파일이 있으면 그쪽을 믿는다.

| 도구 | 운영 빌드 기본값 | 설정 위치 |
|---|---|---|
| Vue CLI | 만든다 | `vue.config.js`의 `productionSourceMap` |
| Create React App | 만든다 | 환경 변수 `GENERATE_SOURCEMAP` |
| Vite | 만들지 않는다 | `vite.config`의 `build.sourcemap` (`true`, `'inline'`, `'hidden'`) |
| Nuxt 3 | 클라이언트는 만들지 않는다 | `nuxt.config`의 `sourcemap.client` |
| Next.js | 브라우저용은 만들지 않는다 | `next.config`의 `productionBrowserSourceMaps` |
| webpack | `devtool` 설정에 따라 다르다 | `devtool: 'source-map'`, `'hidden-source-map'` 등 |

오래된 사내 Vue 2 앱(Vue CLI)은 소스 맵이 남아 있을 가능성이 높다.

## 어떻게 사용하는가? (How)

아래 순서로 찾는다. 앞 단계에서 찾으면 뒤는 건너뛴다.

### 1단계: DevTools에서 바로 확인한다

가장 빠르다. 로그인한 상태 그대로 볼 수 있다.

1. 분석할 화면을 열고 DevTools(F12)를 연다.
2. 설정(톱니바퀴) → **Preferences** → **Sources**에서 **Enable JavaScript source maps**가 켜져 있는지 확인한다. 꺼져 있으면 맵이 있어도 안 보인다.
3. **Sources** 탭 왼쪽 파일 트리를 본다. `webpack://`, `src/`, `.vue` 파일이 보이면 소스 맵이 로드된 것이다. 트리 위 `⋮` 메뉴의 **Group by Authored/Deployed**를 켜면 원본(Authored)과 배포본(Deployed)이 나뉘어 보인다.
4. 원본이 안 보이면 **More tools** → **Developer resources** 패널을 연다. DevTools가 시도한 소스 맵 요청과 성공, 실패 여부가 나온다. 소스 맵 요청은 페이지가 아니라 DevTools가 보내기 때문에 **Network** 탭에는 나오지 않는다.

Developer resources에 실패(404, 403)로 나오면 맵을 가리키는 표시는 있는데 파일이 없거나 막힌 것이다. 아무것도 없으면 표시가 없는 것이므로 2단계로 간다.

### 2단계: 번들 주소를 모으고 표시를 직접 확인한다

**Network** 탭에서 필터를 **JS**로 두고 새로고침해서 번들 주소를 모은다. `app.4f2a1c.js`, `chunk-vendors.js`, `index-BkX9.js`처럼 해시가 붙은 파일이 대상이다. 화면을 옮길 때 새로 로드되는 청크(Chunk)도 있으므로 분석할 화면을 한 번씩 다 연다.

번들마다 끝부분과 응답 헤더를 본다. Windows PowerShell에서는 `curl` 대신 `curl.exe`로 쓴다(`curl`은 `Invoke-WebRequest`의 별칭이다).

```bash
URL=http://internal-app.example/assets/index-BkX9.js

# 파일 끝의 주석
curl -s "$URL" | tail -c 300

# 응답 헤더의 SourceMap, X-SourceMap
curl -sI "$URL" | grep -i sourcemap
```

로그인이 필요한 앱이면 Network 탭에서 요청을 우클릭해 **Copy** → **Copy as cURL**로 쿠키까지 복사해서 쓴다. 복사한 명령에는 세션 쿠키가 들어 있으므로 파일이나 채팅에 남기지 않는다.

### 3단계: 표시가 없으면 관례 경로를 요청한다

```bash
curl -s -o app.js.map -w "%{http_code} %{content_type}\n" "$URL.map"
head -c 100 app.js.map
```

결과를 볼 때 한 가지를 조심한다. SPA(Single Page Application) 서버는 없는 경로에도 `200`과 함께 `index.html`을 돌려주는 경우가 많다. 상태 코드만 보고 "맵이 있다"고 판단하지 않는다. 내용이 `{"version":3`으로 시작해야 진짜 소스 맵이다.

경로를 바꿔 볼 곳도 있다.

- 번들과 같은 폴더의 `번들이름.map` (`.js`를 뺀 형태)
- 빌드 도구가 맵을 따로 모으는 폴더 (`/maps/`, `/sourcemaps/`)

이 이상 경로를 추측해서 대량으로 요청하지 않는다. 그건 탐색이 아니라 스캔이 되고, 허가 범위를 넘을 수 있다.

### 4단계: 파일로 받은 앱에서 찾는다

Electron 앱이나 설치 폴더에 웹 리소스가 들어 있는 경우다. 네트워크 없이 파일 검색으로 끝난다.

```bash
# Electron 앱은 app.asar 압축부터 푼다
npx @electron/asar extract "C:/re-work/tool/resources/app.asar" C:/re-work/tool/asar-out

# .map 파일과 sourceMappingURL 주석을 찾는다
find C:/re-work/tool/asar-out -name "*.map"
grep -rl "sourceMappingURL" C:/re-work/tool/asar-out --include="*.js"
```

REA를 쓰면 `analyze_javascript_application`(파일)과 `analyze_web_bundle`(브라우저에 열린 페이지)이 소스 맵 유무를 함께 보고한다. 사용법은 [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md#사내-웹-앱-vue-nuxt-react-프런트엔드)에 있다.

### 5단계: 원본 파일을 꺼낸다

DevTools의 Sources 탭에서 파일을 하나씩 열어 봐도 되지만, 파일이 많으면 맵에서 한 번에 꺼낸다. 아래 스크립트는 표준 라이브러리만 쓴다.

```python
#!/usr/bin/env python3
"""Write each original file stored in a source map's sourcesContent to disk.

Usage: python3 extract_sourcemap.py app.js.map out_dir
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def safe_relpath(source: str) -> Path:
    """webpack:///./src/App.vue -> src/App.vue. Never escapes out_dir."""
    url = urlparse(source)
    raw = f"{url.netloc}/{url.path}" if url.scheme else source
    parts = [p for p in re.split(r"[\\/]+", raw) if p not in ("", ".", "..")]
    parts = [re.sub(r'[<>:"|?*]', "_", p) for p in parts]  # Windows-invalid chars
    return Path(*parts) if parts else Path("unnamed")


def extract(smap: dict, out_dir: Path) -> tuple[int, int]:
    """Return (files written, sources listed)."""
    if "sections" in smap:  # index map: several maps glued together
        totals = [extract(s["map"], out_dir) for s in smap["sections"] if "map" in s]
        return sum(t[0] for t in totals), sum(t[1] for t in totals)
    sources = smap.get("sources") or []
    contents = smap.get("sourcesContent") or []
    written = 0
    for i, source in enumerate(sources):
        text = contents[i] if i < len(contents) else None
        if text is None:
            continue
        base = out_dir / safe_relpath(source or f"source_{i}")
        target, n = base, 1
        while target.exists():  # Vue splits one .vue file into several entries
            target = base.with_name(f"{base.stem}.{n}{base.suffix}")
            n += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written += 1
    return written, len(sources)


if __name__ == "__main__":
    map_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    smap = json.loads(map_path.read_text(encoding="utf-8"))
    written, listed = extract(smap, out_dir)
    print(f"{written}/{listed} files written to {out_dir}")
    if written == 0:
        print("sourcesContent is empty. Paths only:")
        for source in smap.get("sources") or []:
            print("  ", source)
```

```bash
python3 extract_sourcemap.py app.js.map C:/re-work/web-app/src-out
```

스크립트가 막는 것이 두 가지 있다. 소스 맵은 남이 만든 파일이므로, `sources`에 `../../`가 들어 있어도 출력 폴더 밖에 파일을 쓰지 않도록 `..`을 버린다. 그리고 Vue는 `.vue` 파일 하나를 `App.vue?vue&type=script` 같은 항목 여러 개로 나누므로, 이름이 겹치면 `App.1.vue`처럼 번호를 붙인다.

꺼낸 파일 중 `node_modules/` 아래는 외부 라이브러리다. 읽지 않는다. 동료가 짠 코드는 대개 `src/` 아래에 있다.

### 결과를 기록한다

[분석 기록 양식](./templates/investigation-log.md)에 아래를 남긴다.

- 소스 맵을 찾은 방법(주석, 헤더, 관례 경로, 파일)과 맵 URL 또는 경로
- `sourcesContent` 유무와 꺼낸 파일 수 (예: "212/240")
- 번들 파일의 해시. 다음 배포에서 번들이 바뀌면 맵도 다시 받아야 한다

소스 맵에서 꺼낸 원본은 배포 시점의 소스다. 저장소의 소스와 다를 수 있다는 점을 기록에 함께 적는다. 꺼낸 원본은 이 저장소에 올리지 않는다.

### 소스 맵이 없을 때

찾지 못했다면 세 가지 길이 있다.

1. **소유자에게 묻는다.** 에러 추적 서비스(Sentry 등)에만 맵을 올리고 배포본에서는 지운 경우가 있다. 소유 팀에 맵이나 소스 저장소를 요청하는 편이 가장 빠르다.
2. **DevTools의 Pretty print로 읽는다.** Sources 탭에서 압축된 파일을 열고 `{}` 버튼을 누르면 줄바꿈이 들어간다. 이름은 없지만 문자열, API 경로, 라우트 이름은 그대로 남아 있다.
3. **REA로 번들을 분석한다.** `analyze_web_bundle`이 맵 없이도 라우트와 API 엔드포인트 후보를 뽑아 준다.

## 참고 자료 (References)

- Source Map 형식(TC39 ECMA-426): <https://tc39.es/ecma426/>
- Chrome DevTools 소스 맵 안내: <https://developer.chrome.com/docs/devtools/javascript/source-maps>
- Chrome DevTools Developer resources 패널: <https://developer.chrome.com/docs/devtools/developer-resources>
- Vite `build.sourcemap`: <https://vite.dev/config/build-options#build-sourcemap>
- webpack `devtool`: <https://webpack.js.org/configuration/devtool/>

## 관련 문서

- [리버스 엔지니어링 기초 개념](./re-fundamentals.md)
- [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)
- [리버스 엔지니어링 작업 절차](./re-process.md)
