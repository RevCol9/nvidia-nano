#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVELOP="${DEVELOP_ROOT:-$(cd "$ROOT/../../develop" 2>/dev/null && pwd || echo "")}"
WT="$ROOT/weights"
mkdir -p "$WT"
if [[ -z "$DEVELOP" || ! -f "$DEVELOP/detect640.pt" ]]; then
  echo "develop/detect640.pt 不存在，请设置 DEVELOP_ROOT" >&2
  exit 1
fi
ln -sf "$DEVELOP/detect640.pt" "$WT/detect640.pt"
[[ -f "$DEVELOP/detect1088.pt" ]] && ln -sf "$DEVELOP/detect1088.pt" "$WT/detect1088.pt" || true
echo "weights 已链接到 $DEVELOP"
