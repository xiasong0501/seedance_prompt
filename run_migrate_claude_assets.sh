#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/migrate_claude_assets_to_series_prompts.py "$@"
