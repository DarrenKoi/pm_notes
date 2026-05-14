# Context Map

Index of domain contexts in this repo. Each entry points to a `CONTEXT.md` that defines the vocabulary, relationships, and example dialogue for one subproject.

Skills like `/grill-with-docs`, `/diagnose`, `/tdd`, and `/improve-codebase-architecture` read this file first to find the relevant glossary for the area they're working in.

## Contexts

| Subproject | CONTEXT.md | Scope |
| --- | --- | --- |
| ITC AI/DT Roadmap | [ai-dt/roadmap/CONTEXT.md](ai-dt/roadmap/CONTEXT.md) | 기반기술센터(ITC) 3~4년향 AI/DT 로드맵 수립 — Biz·5 stream·암묵지 순환고리 등 도메인 언어 |

## Adding a new context

When a subproject's vocabulary starts mattering (terms get reused, ambiguity surfaces, the same concept gets named differently in different places), run `/grill-with-docs` from inside that subproject. It'll create a `CONTEXT.md` at the subproject root. Add a row to the table above pointing at it.

Don't pre-create empty `CONTEXT.md` files — they earn their existence by being written during a real grilling session.
