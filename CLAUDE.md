# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **documentation-only repository** containing "Evolution Documents" for major Python scientific computing and data science libraries. Each document provides a comprehensive overview of a specific library, including its history, core architecture, and detailed API breakdown.

**Important:** This is NOT a code repository. It contains only Markdown documentation files with embedded Mermaid diagrams. There is no build system, no tests, no linting, and no dependencies.

## Repository Structure

```
docs/
├── computer-graphics/    # Geometric modeling (NURBS, Lie Algebra, G2 blending)
├── concurrency/          # Async programming patterns
├── context/              # Project metadata (GEMINI.md)
├── data-processing/      # NumPy, Pandas, SciPy, Dask, RMSE/R-squared
├── deep-learning/        # PyTorch, TensorFlow, Keras, CNN, Transformer, RL concepts
├── gpu-computing/        # CUDA
├── ml/                   # Scikit-learn, XGBoost, LightGBM
├── nlp/                  # NLTK, spaCy
└── visualization/        # Matplotlib, Plotly, Seaborn
```

Each subdirectory contains `en/` (English) and `zh/` (Chinese) language versions.

## File Conventions

**Document Pairs:** Every library document exists as an inseparable pair:
- English: `<library>_evolution_document.md`
- Chinese: `<library>_evolution_document.zh.md`

**CRITICAL:** When updating one document in a pair, you MUST apply the equivalent change to its counterpart. The Chinese version must be a perfect substitute for the English version in terms of structure, meaning, and information content.

## Standard Document Structure

All evolution documents follow this architecture (SRP - Single Responsibility Principle):

1. **Introduction and Historical Context** - Library origins, evolution timeline, community impact
2. **Core Architecture** - Fundamental concepts, design principles, integration points
3. **Detailed API Overview** - Major functional areas, key classes/methods, evolution notes
4. **Evolution and Impact** - Performance improvements, ecosystem relationships
5. **Conclusion** - Current significance, future trajectory

## XP/SOLID Principles for Documentation

This repository follows XP (eXtreme Programming) principles adapted for documentation:

- **Simple Design (YAGNI):** Only implement what is currently needed
- **Refactoring:** Continuously improve clarity, accuracy, consistency without altering factual information
- **Collective Ownership:** Anyone can improve any document
- **Pair-Writing:** Communicate plans before making changes, ask for review after implementing

Apply SOLID principles to documents:
- **SRP:** Each section has a single, clear purpose
- **OCP:** Core architecture is closed for modification but open for extension
- **LSP:** Chinese and English versions must be interchangeable in structure/content
- **DIP:** High-level summaries depend on key concepts, not obscure implementation details

## "Think by Analogy" Practice

When making an improvement to one document (e.g., enhancing a Mermaid diagram, restructuring a section), ask: **"Should I apply this same improvement to other similar documents?"**

For example, if you improve the timeline diagram in the NumPy document, consider whether the same improvement should be propagated to Pandas, SciPy, and other data-processing documents.

## Mermaid Diagrams

All documents use Mermaid for visual elements:
- **Timelines:** Library evolution history
- **Architecture graphs:** Core components and relationships
- **Mindmaps:** API structure overview

Ensure Mermaid syntax is valid. Test diagrams in your Markdown viewer before committing.

## Key Files

- **README.md** - Project overview and navigation
- **README.zh.md** - Chinese version of README
- **CODEBUDDY.md** - Core guidelines for maintaining the repository (XP/SOLID principles)
- **GEMINI.md** - Gemini-specific operational guidelines
- **.mermaid_checks/** - Empty directory (placeholder for diagram validation tools)

## No Development Commands

This repository has:
- **Build:** N/A (pure Markdown, no compilation)
- **Lint:** N/A (no lint tooling)
- **Test:** N/A (no test suite)
- **Dependencies:** None (self-contained documentation)

## Adding New Library Documentation

When adding a new library:

1. Create both English (`*_evolution_document.md`) and Chinese (`*_evolution_document.zh.md`) versions
2. Follow the standard document structure exactly
3. Include Mermaid diagrams for timeline, architecture, and API overview
4. Use existing documents (especially `numpy_evolution_document.md`) as a template
5. Update README.md to include the new library in the "Covered Libraries" section
6. Update both README.md and README.zh.md (bilingual updates required)

## Document Quality Checklist

Before considering a document change complete:

- [ ] Structure matches standard format (Intro → Core → API → Evolution → Conclusion)
- [ ] Mermaid diagrams render correctly
- [ ] English and Chinese versions are synchronized (if updating a pair)
- [ ] Terminology is consistent across similar documents
- [ ] No "document smells" (unclear sentences, inconsistent formatting, missing diagrams)
