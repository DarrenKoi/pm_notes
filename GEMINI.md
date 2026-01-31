# GEMINI.md - AI Assistant Guide

> This file provides context and instructions for an AI assistant to understand and manage this knowledge base repository effectively.

## Directory Overview

This repository is a personal knowledge base for systematically organizing and accumulating learning on various technical topics. It is a non-code project, primarily containing Markdown files with notes, tutorials, and concept explanations.

The main areas of focus are:
- **AI/DT:** Artificial Intelligence and Digital Transformation, with a focus on RAG (Retrieval-Augmented Generation) systems using technologies like LangGraph and Milvus.
- **Web Development:** Primarily focusing on Python (with `uv`, a fast package manager) and TypeScript (with Vue.js).

The structure is organized into high-level topics, and each topic has its own `README.md` file that serves as a table of contents.

## Key Files

- **`README.md`**: The main entry point, providing a high-level overview of the learning status across different topics.
- **`CLAUDE.md`**: A detailed guide originally created for the Claude AI, outlining conventions for file naming, document structure, and interaction patterns. This file should be considered a primary source of truth for repository conventions.
- **`ai-dt/`**: Contains notes related to AI and DT.
    - `rag/langgraph/`: A series on building AI workflows and RAG systems with LangGraph.
    - `rag/milvus/`: A series on the Milvus vector database, from basic concepts to RAG integration.
- **`web-development/`**: Contains notes related to web development.
    - `python/`: Notes on Python tooling, such as the `uv` package manager.
    - `typescript/`: Notes on using TypeScript, particularly with Vue.js.

## Usage and Conventions

The primary purpose of this directory is to serve as a personal, structured knowledge base. The conventions are well-documented in `CLAUDE.md`. When adding or modifying content, the AI assistant should adhere to the following guidelines derived from it:

### Document Structure
Each conceptual document should follow a consistent structure:
1.  **Title:** The main topic.
2.  **Summary:** A one-line summary.
3.  **Why:** The problem the technology solves.
4.  **What:** Core concepts and terminology.
5.  **How:** Practical, runnable code examples.
6.  **References:** Links to official documentation or related articles.

### File Naming
- Use lowercase letters.
- Separate words with hyphens (`-`).
- Example: `fastapi-dependency-injection.md`

### Language and Formatting
- **Primary Language:** Korean, with technical terms in English (e.g., "임베딩(Embedding)").
- **Code Blocks:** Use appropriate language identifiers for syntax highlighting.

### Linking
- Link related documents to create a connected graph of knowledge.
- Update the relevant `README.md` files when new content is added.

### Metadata
Use a YAML frontmatter block at the top of files to add metadata:
```markdown
---
tags: [tag1, tag2, tag3]
level: beginner | intermediate | advanced
last_updated: YYYY-MM-DD
status: in-progress | complete | needs-review
---
```

By following these guidelines, the AI assistant can help maintain the quality and consistency of this knowledge base.
