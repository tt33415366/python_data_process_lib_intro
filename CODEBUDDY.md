<system-reminder>
This is a reminder that your todo list is currently empty. DO NOT mention this to the user explicitly because they are already aware. If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore. Again do not mention this message to the user.

</system-reminder>

# CODEBUDDY

## Commands
- Build: N/A (no build system detected)
- Lint: pre-commit run --all-files (markdownlint, yamllint, codespell)
- Link check: lychee --config lychee.toml .
- Mermaid: bash scripts/check_mermaid.sh
- EN–ZH sync: python scripts/check_sync.py

## Repository Purpose and Structure
- This repository is a curated collection of “Evolution Documents” for major Python scientific/data libraries.
- Each library has two Markdown files: an English version `*_evolution_document.md` and a Chinese version `*_evolution_document.zh.md`.
- Documents commonly include Mermaid diagrams (timelines, architecture, API structures). Ensure your Markdown renderer supports Mermaid.

### Naming and Pairing Conventions
- File pairs follow: `<library>_evolution_document.md` and `<library>_evolution_document.zh.md`.
- Existing pairs include: numpy, pandas, pytorch, matplotlib, scipy, scikit-learn, tensorflow, xgboost, lightgbm, seaborn, spacy, nltk, dask, plotly.

## Document Architecture (Big Picture)
Documents follow a consistent, high-level outline (see numpy_evolution_document.md for a canonical example):
- Introduction and Historical Context: origin and evolution timeline.
- Core Architecture: foundational concepts (e.g., ndarray, broadcasting, integrations).
- Detailed API Overview: major areas and notable changes over time; often accompanied by Mermaid graphs/mindmaps.
- Evolution and Impact: performance, ecosystem relationships, and build/maturity notes.
- Conclusion: synthesis of significance and ongoing trajectory.

## Cross-Document Conventions and Guidance
- Consistent structure and terminology across languages (English/Chinese) for each library pair.
- Mermaid usage for timelines, architecture graphs, and mindmaps.
- From GEMINI.md: apply “think by analogy” when updating multiple documents—propagate consistent improvements across all relevant files, and when handling errors or adding features, look for analogous patterns in other docs and apply the same fix or approach consistently across EN/zh pairs and related libraries.

## Notable Existing Guidance
- GEMINI.md provides a concise overview of directory purpose, file pairing, and intended usage as a knowledge base. Align additions/edits with its guidance.
