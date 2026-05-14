# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo uses a **multi-context** layout. Each subproject that has earned its own domain language gets its own `CONTEXT.md` (and, when needed, its own `docs/adr/`). A root-level `CONTEXT-MAP.md` lists where they live.

## Before exploring, read these

1. **`CONTEXT-MAP.md`** at the repo root — the index of all known contexts in this repo.
2. The **`CONTEXT.md`** for the subproject you're about to work in (path is in the map).
3. The **`docs/adr/`** for that subproject, if one exists — read ADRs that touch the area you're about to change.

If `CONTEXT-MAP.md` doesn't exist, or the relevant subproject has no `CONTEXT.md` yet, **proceed silently**. Don't flag absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT-MAP.md                  # index of contexts
├── ai-dt/
│   └── roadmap/
│       ├── CONTEXT.md              # ITC AI/DT 로드맵 도메인 언어
│       └── docs/adr/               # (created lazily)
├── web-development/
│   └── (no CONTEXT.md yet — add one when this subproject earns its own glossary)
└── (rest of repo)
```

## Picking the right context

When the user's request maps clearly to one subproject's path (e.g. they're editing files under `ai-dt/roadmap/`), use that subproject's `CONTEXT.md`. When the request spans multiple subprojects, read each relevant `CONTEXT.md` — but prefer the more specific one if terms collide.

If no context covers the area you're working in, that's fine. Don't invent one. Note the gap for `/grill-with-docs` if domain terminology starts mattering.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids (e.g. `CONTEXT.md` may say "use `Biz`, not `사업부`" — honor that).

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (some-decision) — but worth reopening because…_
