# Gemini Context File

This file provides an overview of the current directory's contents and purpose, to be used as instructional context for future interactions with Gemini.

## Directory Overview

This directory contains a collection of "Evolution Documents" for major libraries within the Python scientific computing and data science ecosystem. Each document provides a comprehensive overview of a specific library, including its history, core architecture, and a detailed breakdown of its API.

The documents are provided in both English (`.md`) and Chinese (`.zh.md`) versions. They follow a consistent structure, making them easy to compare and use as reference material. The documents also include Mermaid diagrams to visually represent timelines, architectures, and API structures.

## Key Files

The directory contains the following pairs of documents:

*   **`numpy_evolution_document.md` / `.zh.md`**: Details the history and API of the NumPy library, the foundation for numerical computing in Python.
*   **`pandas_evolution_document.md` / `.zh.md`**: Covers the Pandas library, the primary tool for data manipulation and analysis in Python.
*   **`pytorch_evolution_document.md` / `.zh.md`**: Describes the PyTorch framework, a leading open-source machine learning library.
*   **`matplotlib_evolution_document.md` / `.zh.md`**: Explains the Matplotlib library, the most widely used plotting and visualization library in Python.
*   **`cnn_evolution_document.md` / `.zh.md`**: Details the history and architecture of Convolutional Neural Networks (CNNs), a foundational deep learning architecture for computer vision.
*   **`transformer_evolution_document.md` / `.zh.md`**: Describes the Transformer architecture, a revolutionary model for sequence modeling, especially in natural language processing.

## Usage

The contents of this directory are intended to be used as a detailed knowledge base on the specified Python libraries. This information can be used for:

*   Understanding the history and architectural evolution of these key libraries.
*   Referencing detailed API descriptions, including context, parameters, and return values for common functions.
*   Providing context for future AI-driven tasks related to these libraries, such as code generation, explanation, or modification.
*   **Think by Analogy:** When encountering errors or implementing features, apply a 'think by analogy' approach. If a similar issue or feature has been addressed in one document or library, consider how that solution or approach might be adapted and applied to other relevant documents or libraries, ensuring consistency and preventing recurring errors.

## Gemini's Operational Guidelines for this Project

As an AI assistant working within this project, I will adhere to the following guidelines, drawing inspiration from established software engineering principles to ensure high-quality and efficient task execution.

### Core Principles

*   **Contextual Understanding:** Before any action, I will thoroughly analyze the codebase, existing documentation, and project conventions.
*   **Iterative Development (Inspired by XP):** For tasks involving code generation or modification, I will adopt an iterative approach, prioritizing:
    *   **Test-Driven Development (TDD):** If applicable, I will aim to write tests first, then implement the minimum functionality to pass them, followed by refactoring.
    *   **Simple Design:** I will strive for the simplest solution that meets the requirements, avoiding premature optimization or over-engineering.
    *   **Refactoring:** I will continuously look for opportunities to improve the internal structure of code without altering its external behavior.
*   **Continuous Verification:** After any significant change, I will verify the integrity of the project by running tests, build commands, and linting checks as appropriate.

### Tool Utilization

*   **Codebase Intelligent Analysis:** I will leverage my `search_file_content`, `glob`, `read_file`, and `read_many_files` tools to understand the entire codebase, identify existing patterns, conventions, and dependencies.
*   **Multi-file Coordinated Modification:** When a task requires changes across multiple files, I will:
    *   Analyze all affected files before proposing modifications.
    *   Provide a clear plan outlining the scope and impact of changes.
    *   Ensure the atomicity and consistency of modifications.
    *   Only proceed with modifications after explicit approval.

### Design Considerations (for Code-Related Tasks)

If future tasks involve generating or modifying code within these libraries, I will consider the following design principles:

*   **SOLID Principles:** I will strive to apply Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion principles to promote maintainable and flexible code.
*   **Package Design Principles:** I will consider principles like Reuse/Release Equivalence, Common Reuse, Common Closure, Acyclic Dependencies, Stable Dependencies, and Stable Abstractions to ensure well-organized and manageable module dependencies.
