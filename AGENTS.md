# Repository Guidelines

## Project Structure & Module Organization
Mixed knowledge base plus runnable examples. **Each top-level folder is a standalone topic** — see the 폴더 독립성 원칙 in [`CLAUDE.md`](CLAUDE.md) before you start; it is this repo's first rule.
- `ai-dt/`: AI/DT study notes (RAG, MCP, OpenSearch, LLMOps, roadmap).
- `web-development/`: web learning notes and sample projects (most complete app: `python/flask/job-scheduler`, Flask + Nuxt).
- `Codes/python/`: executable Python examples (`opensearch_handler`, `history-opensearch`, `drm-pptx-extraction`, ...).
- `dev-environment/`: terminal/tooling notes (codex, pi, terminal, warp, vlm).
- `my-task/`: in-flight deliverables (`AIX_POC`, `2026_report`) — has its own `CLAUDE.md`.
- `RAG/`, `docs/`, `_workspace/`: one-off analysis, agent-ops docs, scratch.

Scope every search, edit, and commit to a single top-level folder. Keep docs next to their topic (`README.md` per module) and runnable code inside its module folder. No cross-folder links.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` functions/variables, `PascalCase` classes, type hints for new/changed code.
- Vue/TS (Nuxt): follow ESLint + Nuxt defaults; use 2-space indentation (see `frontend/.editorconfig`).
- Markdown: concise sections, clear headings, and relative links.
- File/folder names: use descriptive `kebab-case` or `snake_case`; match existing local conventions in each module.

## Testing Guidelines
Automated tests are not yet standardized across the repository. For code changes:
- Run lint/type-check where available (Nuxt frontend).
- Perform module-level smoke tests (run the affected script/app end-to-end).
- When adding non-trivial logic, add a local `tests/` folder in that module and prefer `pytest` with `test_*.py` naming.

## Commit & Pull Request Guidelines
Recent history follows short, imperative commit subjects (for example: `Add ...`, `Refactor ...`, `Update ...`, `Clean up ...`).
- Keep subject lines concise and action-first.
- One logical change per commit.
- PRs should include: purpose, affected paths, commands run for validation, and screenshots for UI changes (`job-scheduler/frontend`).

## Security & Configuration Tips
- Do not commit secrets, API keys, or internal endpoints.
- Keep environment-specific values in local config overrides or environment variables.
- Review `config.py`/`config.yaml` changes carefully before opening a PR.
