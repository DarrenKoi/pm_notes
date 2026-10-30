"""Build one mobile-friendly HTML page from a pm_notes folder's Markdown docs.

Usage: python scripts/build_folder_html.py <folder>   (needs the `markdown` package)

Reads <folder>/README.md, then every .md file it links inside the folder, in link
order. Writes <folder>/reading.html, which is published as a claude.ai artifact.
"""
import html
import re
import sys
from pathlib import Path

import markdown
from urllib.parse import quote, unquote

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
OUT = ROOT / "reading.html"
REPO = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/DarrenKoi/pm_notes/blob/main"


def readme_order():
	readme = (ROOT / "README.md").read_text(encoding="utf-8")
	docs = ["README.md"]
	for target in re.findall(r"\]\((?:\./)?([^)#:\s]+\.md)(?:#[^)]*)?\)", readme):
		target = Path(target).as_posix()
		if target not in docs and (ROOT / target).is_file() and ".." not in target:
			docs.append(target)
	return docs


DOCS = readme_order()


def doc_id(path):
	return re.sub(r"[^\w.-]", "-", path.removesuffix(".md").replace("/", "-")).lower()


def slugify(text, sep="-"):
	# GitHub-style anchors: lowercase, drop punctuation, spaces become hyphens.
	text = re.sub(r"<[^>]+>", "", text).strip().lower()
	text = re.sub(r"[^\w\- ]", "", text)
	return text.replace(" ", sep)


def heading_slug(anchor):
	# Obsidian writes heading links percent-encoded ("#실행%20환경"); GitHub writes slugs ("#실행-환경").
	return slugify(unquote(html.unescape(anchor)))


def render(path):
	src = (ROOT / path).read_text(encoding="utf-8")
	src = re.sub(r"\A---\n.*?\n---\n", "", src, flags=re.S)
	did = doc_id(path)
	md = markdown.Markdown(
		extensions=["tables", "fenced_code", "toc"],
		extension_configs={"toc": {"slugify": lambda t, s: f"{did}--{slugify(t)}"}},
		tab_length=4,
	)
	body = md.convert(src)
	title = re.search(r"^# (.+)$", src, flags=re.M).group(1)

	def fix_link(m):
		target, anchor = m.group(1), m.group(2)
		resolved = (Path(path).parent / target).as_posix().removeprefix("./")
		resolved = re.sub(r"[^/]+/\.\./", "", resolved)
		if resolved not in DOCS:
			# Not on this page: link to the file on GitHub instead.
			repo_path = quote((ROOT.resolve() / resolved).relative_to(REPO).as_posix())
			return f'href="{GITHUB}/{repo_path}{anchor or ""}"'
		rid = doc_id(resolved)
		frag = f"{rid}--{heading_slug(anchor[1:])}" if anchor else rid
		return f'href="#{frag}"'

	body = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{did}--{heading_slug(m.group(1))}"', body)
	body = re.sub(r'href="([^"#:]+\.md)(#[^"]*)?"', fix_link, body)
	body = re.sub(
		r"<blockquote>\s*<p>\[!(\w+)\]\s*(.*?)</p>",
		lambda m: f'<blockquote class="callout callout-{m.group(1).lower()}"><p class="callout-label">{m.group(1).upper()}</p><p>{m.group(2)}</p>',
		body,
		flags=re.S,
	)
	body = re.sub(r"<table>", '<div class="tbl"><table>', body)
	body = body.replace("</table>", "</table></div>")
	body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)
	return did, title, body


def main():
	parts, nav = [], []
	for i, path in enumerate(DOCS):
		did, title, body = render(path)
		num = f"{i:02d}"
		nav.append(f'<li><a href="#{did}"><span class="num">{num}</span>{html.escape(title)}</a></li>')
		parts.append(
			f'<section class="doc" id="{did}">'
			f'<p class="file"><span class="num">{num}</span>{html.escape(path)}</p>'
			f"<h1>{html.escape(title)}</h1>{body}"
			f'<p class="back"><a href="#top">목차로</a></p></section>'
		)
	site = html.escape(re.sub(r"\(.*?\)", "", render("README.md")[1]).strip())
	page = TEMPLATE.replace("{{NAV}}", "\n".join(nav)).replace("{{DOCS}}", "\n".join(parts)).replace("{{SITE}}", site)
	OUT.write_text(page, encoding="utf-8")
	print(f"wrote {OUT}")


TEMPLATE = """<meta charset="utf-8">
<title>{{SITE}}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
/* Layout: one reading column; a sticky bar names the docs, each doc is a section headed by its file path like a listing header. */
:root {
	--bg: #f3f5f4; --surface: #ffffff; --ink: #18201d; --muted: #5b6862; --line: #d5ddd9;
	--accent: #a63a2b; --code-bg: #e8edea; --note: #2f6690; --warn: #a65e00;
	--sans: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
	--mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
	--bg: #121715; --surface: #1a201e; --ink: #e3e9e6; --muted: #98a59f; --line: #2c3532;
	--accent: #e07a62; --code-bg: #222a27; --note: #7fb3dd; --warn: #e3a54a; color-scheme: dark } }
:root[data-theme="dark"] {
	--bg: #121715; --surface: #1a201e; --ink: #e3e9e6; --muted: #98a59f; --line: #2c3532;
	--accent: #e07a62; --code-bg: #222a27; --note: #7fb3dd; --warn: #e3a54a; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 16px/1.75 var(--sans); }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--surface); }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--surface); border-bottom: 1px solid var(--line);
	padding-block: 10px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.bar-inner { max-width: 760px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.bar strong { font-size: 15px; }
.bar a { color: var(--accent); font-size: 14px; text-decoration: none; }
main { max-width: 760px; margin: 0 auto; padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.toc { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 20px; }
.toc h2 { margin: 0 0 8px; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); border: 0; padding: 0; }
.toc ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.toc a { display: flex; gap: 12px; padding: 8px 4px; color: var(--ink); text-decoration: none; border-radius: 4px; }
.toc a:hover, .toc a:focus-visible { background: var(--code-bg); }
.num { font-family: var(--mono); color: var(--accent); font-size: 13px; padding-top: 3px; flex: none; }
.doc { margin-top: 56px; scroll-margin-top: 64px; }
.file { font-family: var(--mono); font-size: 13px; color: var(--muted); margin: 0; display: flex; gap: 10px; border-bottom: 2px solid var(--accent); padding-bottom: 6px; }
h1, h2, h3, h4 { text-wrap: balance; line-height: 1.35; scroll-margin-top: 64px; }
h1 { font-size: 1.65rem; margin: 14px 0 8px; }
h2 { font-size: 1.3rem; margin: 40px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 1.1rem; margin: 28px 0 8px; }
h4 { font-size: 1rem; margin: 22px 0 6px; color: var(--muted); }
p, li { max-width: 68ch; }
a { color: var(--accent); overflow-wrap: anywhere; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
code { font-family: var(--mono); font-size: .88em; background: var(--code-bg); padding: 1px 5px; border-radius: 3px; overflow-wrap: anywhere; }
pre { background: var(--code-bg); border-radius: 6px; padding: 14px 16px; overflow-x: auto; tab-size: 4; line-height: 1.55; }
pre code { background: none; padding: 0; font-size: 13.5px; overflow-wrap: normal; }
.tbl { overflow-x: auto; margin: 16px 0; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; line-height: 1.55; }
th, td { text-align: left; vertical-align: top; padding: 9px 12px; border-bottom: 1px solid var(--line); min-width: 7em; }
th { background: var(--code-bg); font-weight: 600; }
tr:last-child td { border-bottom: 0; }
blockquote { margin: 16px 0; padding: 10px 16px; border-left: 3px solid var(--line); color: var(--muted); }
.callout { background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--note); border-radius: 6px; color: var(--ink); }
.callout-warning { border-left-color: var(--warn); }
.callout-label { font-family: var(--mono); font-size: 12px; letter-spacing: .08em; margin: 0 0 2px; color: var(--note); }
.callout-warning .callout-label { color: var(--warn); }
.callout p { margin: 6px 0; }
.back { margin-top: 32px; font-size: 14px; }
@media (max-width: 480px) { body { font-size: 15.5px; } h1 { font-size: 1.4rem; } th, td { padding: 8px 10px; } }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
</style>
<header class="bar"><div class="bar-inner"><strong>{{SITE}}</strong><a href="#top">목차</a></div></header>
<main id="top">
<nav class="toc" aria-label="문서 목록"><h2>문서 · 읽는 순서</h2><ol>
{{NAV}}
</ol></nav>
{{DOCS}}
</main>
"""

if __name__ == "__main__":
	main()
