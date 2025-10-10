# Contributing

## Quality gates
- Run pre-commit locally: pre-commit install && pre-commit run --all-files
- Link check: lychee --config lychee.toml .
- Mermaid check: scripts/check_mermaid.sh (requires mmdc)
- EN–ZH sync: python scripts/check_sync.py

## Commit messages
- Use Conventional Commits. Example: feat(docs-ml): add SVM section

## File pairing
- English: *_evolution_document.md
- Chinese: *_evolution_document.zh.md
