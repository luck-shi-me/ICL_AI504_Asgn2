#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Report Table 4. Extra shuffles with random offsets: add --random-shuffles 5
python analysis/probe_document_use.py "$@"
