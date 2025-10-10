#!/usr/bin/env bash
set -euo pipefail
# Finds Mermaid code blocks in Markdown and renders them via mmdc to validate syntax.
# Requires: mermaid-cli (mmdc)

TMP_DIR=".mermaid_checks"
rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

mapfile -t files < <(rg -n "^```mermaid$" -l -- '**/*.md')

for f in "${files[@]}"; do
  # Extract blocks and render sequentially
  awk 'BEGIN{inblk=0; idx=0} \
  /^```mermaid$/ {inblk=1; idx++; next} \
  /^```$/ {if(inblk){inblk=0} else {print}; next} \
  { if(inblk){print > ("'$TMP_DIR'/" idx ".mmd");} }' "$f"

done

status=0
for m in "$TMP_DIR"/*.mmd; do
  [ -e "$m" ] || continue
  out="$m.png"
  if ! mmdc -i "$m" -o "$out" >/dev/null 2>&1; then
    echo "Mermaid render failed: $m"
    status=1
  fi
  rm -f "$out"
done

rm -rf "$TMP_DIR"
exit $status
