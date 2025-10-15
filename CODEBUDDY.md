---
description: "Core guidelines for maintaining the 'Evolution Documents' repository, emphasizing consistency, architectural integrity, and XP/SOLID principles adapted for documentation."
globs: ["**/*_evolution_document.md", "**/*_evolution_document.zh.md"]
alwaysApply: true
---

# CODEBUDDY (v2.0 XP Edition)

@ai, this document contains the core principles and rules for maintaining this repository. Your primary role is to act as an XP (eXtreme Programming) partner to ensure the quality and consistency of the documentation.

## Commands
- Build: N/A (no build system detected)
- Lint: N/A (no lint tooling detected)
- Test: N/A (no tests or test runner detected)
- Single test: N/A

## Part 1: Core Principles and Rules

### 1.1 Development Methodology: XP for Documentation

Our collaboration will follow XP principles adapted for documentation.

- **Communication & Collaboration:** As my "pair-writing" partner, you must communicate your plan before making changes.
  - **AI-Instruction:** Before writing, state your plan: "This is my plan: [steps]. Does this seem correct?"
  - **AI-Instruction:** After implementing, ask for review: "Let's review the changes and look for improvements."

- **Simple Design (YAGNI):** Only implement what is currently needed. The design should be the simplest possible.
  - **AI-Instruction:** For every decision, ask: "Is this the simplest thing we need right now?"

- **Refactoring:** Continuously improve the internal structure (clarity, accuracy, consistency) without altering the external behavior (factual information).
  - **AI-Instruction:** After making changes, scan for "document smells" like inconsistent terminology, unclear diagrams, or overly complex sentences. Suggest specific refactorings.

- **Collective Ownership & Coding Standard:** Anyone can improve any document. All code must follow a uniform style.
  - **AI-Instruction:** Propose improvements anywhere you see an opportunity, regardless of file boundaries. Strictly adhere to the conventions in this document.

### 1.2 Document Architecture: SOLID for Docs

Apply SOLID principles to maintain the integrity and structure of the documents. The canonical example is `numpy_evolution_document.md`.

- **SRP (Single Responsibility Principle):** A document should have only one reason to change: an evolution in the library it describes. Each section must have a single, clear purpose.
  - The high-level outline is:
    - Introduction and Historical Context
    - Core Architecture
    - Detailed API Overview
    - Evolution and Impact
    - Conclusion

- **OCP (Open-Closed Principle):** The core document architecture is **closed** for modification but **open** for extension (e.g., adding details about a new library version).

- **LSP (Liskov Substitution Principle):** The Chinese version (`*_evolution_document.zh.md`) must be a perfect substitute for the English version (`*_evolution_document.md`) in terms of structure, meaning, and information content.

- **DIP (Dependency Inversion Principle):** High-level summaries should depend on abstractions (key concepts of the library), not on obscure implementation details.

### 1.3 File Conventions: Packaging Principles

We treat document pairs as "packages" that are released together.

- **Common Closure Principle (CRP):** A change to a library's evolution history is a single conceptual change. All classes/files affected by that change should be bundled. The English and Chinese documents are closed together against such changes.
  - **AI-Instruction:** When updating one document in a pair (e.g., the English version), you **must** apply the equivalent change to its counterpart (the Chinese version).

- **Common Reuse Principle (CCP):** The English and Chinese files are always consumed and updated together. They form an inseparable pair.
  - **File Naming Convention:** `<library>_evolution_document.md` and `<library>_evolution_document.zh.md`.
  - **Existing Pairs:** numpy, pandas, pytorch, matplotlib, scipy, scikit-learn, tensorflow, xgboost, lightgbm, seaborn, spacy, nltk, dask, plotly.

### 1.4 Cross-Document Conventions: Think by Analogy

- **Consistent Structure:** All documents (across all libraries) must follow the same high-level structure and use consistent terminology.
- **Mermaid Usage:** Use Mermaid diagrams for timelines, architecture graphs, and mindmaps to visually represent complex information.
- **Think by Analogy:**
  - **AI-Instruction:** When you make an improvement to one document (e.g., improve a Mermaid diagram), you should ask: "Should I apply this same improvement to other documents like pandas and scipy?"
  - **AI-Instruction:** When adding a new library, you must use the existing documents as a template to ensure consistency.