#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_DIRS = [
    (ROOT / 'docs' / 'data-processing' / 'en', ROOT / 'docs' / 'data-processing' / 'zh'),
    (ROOT / 'docs' / 'ml' / 'en', ROOT / 'docs' / 'ml' / 'zh'),
    (ROOT / 'docs' / 'deep-learning' / 'en', ROOT / 'docs' / 'deep-learning' / 'zh'),
    (ROOT / 'docs' / 'visualization' / 'en', ROOT / 'docs' / 'visualization' / 'zh'),
    (ROOT / 'docs' / 'nlp' / 'en', ROOT / 'docs' / 'nlp' / 'zh'),
]

HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)')


def headings(path: Path):
    hs = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            m = HEADER_RE.match(line)
            if m:
                hs.append((len(m.group(1)), m.group(2).strip(), i))
    return hs


def compare(en: Path, zh: Path):
    en_h = headings(en)
    zh_h = headings(zh)
    issues = []
    if not en_h or not zh_h:
        return [("empty_headings", f"{en} or {zh} has no headings")] 
    if len(en_h) != len(zh_h):
        issues.append(("count_mismatch", f"{en} vs {zh}: {len(en_h)} != {len(zh_h)}"))
    for i in range(min(len(en_h), len(zh_h))):
        e, z = en_h[i], zh_h[i]
        if e[0] != z[0]:
            issues.append(("level_mismatch", f"{en}:{e[2]}(h{e[0]}) vs {zh}:{z[2]}(h{z[0]})"))
    return issues


def main():
    problems = 0
    for en_dir, zh_dir in PAIR_DIRS:
        if not en_dir.exists() or not zh_dir.exists():
            print(f"missing dir: {en_dir} or {zh_dir}")
            problems += 1
            continue
        for en_file in sorted(en_dir.glob('*.md')):
            zh_file = zh_dir / (en_file.stem + '.zh.md')
            if not zh_file.exists():
                print(f"missing zh: {zh_file}")
                problems += 1
                continue
            issues = compare(en_file, zh_file)
            for code, msg in issues:
                print(f"{code}: {msg}")
                problems += 1
    if problems:
        print(f"SYNC_CHECK_FAILED: {problems} issues")
        sys.exit(1)

if __name__ == '__main__':
    main()
