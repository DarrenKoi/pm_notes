# Repository Guidelines

## Project Structure & Module Organization
This repository is a documentation-first workspace with a small number of focused areas.
- `.remember/`: working memory, daily notes, and short-term planning files.
- `2026_report/`: retrospective and planning documents for 2026.
- `AIX_POC/`: AI transformation proof-of-concept material, including `README.md`, `CLAUDE.md`, lectures, guides, tools, and project-specific folders such as `프로젝트_smart_align_agent/`.

Keep new documents close to the topic they explain. Use local `README.md` files when a folder needs orientation, and prefer relative links between related notes.

## Build, Test, and Development Commands
There is no repository-wide build system at this level. Use lightweight checks before committing:
- `git status --short`: review changed files.
- `find . -maxdepth 2 -type f | sort`: inspect nearby document structure.
- `rg "search term" AIX_POC 2026_report`: search notes before adding duplicate content.

If a subfolder adds runnable code, document its exact setup and smoke-test commands in that subfolder's `README.md`.

## Coding Style & Naming Conventions
Markdown should be concise, structured with clear headings, and written for future readers. Prefer descriptive filenames using existing local conventions: `kebab-case`, `snake_case`, or established Korean folder names already present in the repository.

For Python or shell examples added under project/tool folders, use 4-space Python indentation, `snake_case` names, type hints for changed Python code, and short comments only where they clarify non-obvious behavior.

## Testing Guidelines
Automated tests are not standardized for the root workspace. For documentation changes, verify links and paths manually. For code added in a module, create a local `tests/` directory when logic is non-trivial and prefer `pytest` with `test_*.py` files.

Run the narrowest meaningful smoke test for the changed module, then record the command in the PR or commit notes.

## Commit & Pull Request Guidelines
Recent commits use short, scoped, action-first subjects, often like `docs(aix-poc): ...` or `docs(my-task): ...`. Keep one logical change per commit.

PRs should include the purpose, affected paths, validation performed, and screenshots only when visual artifacts or rendered documents changed.

## Security & Configuration Tips
Do not commit secrets, private endpoints, credentials, or local machine paths unless they are intentionally documented examples. Keep environment-specific values in local overrides or clearly marked templates.
