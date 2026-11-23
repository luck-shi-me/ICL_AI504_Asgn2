#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# --fast skips tests that need the dataset and download models.
if [ "${1:-}" = "--fast" ]; then python -m pytest tests -q -m "not model"; else python -m pytest tests -q; fi
