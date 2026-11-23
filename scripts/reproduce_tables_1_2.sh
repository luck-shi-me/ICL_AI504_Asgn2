#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Report Tables 1 and 2. Add --quick for the reduced run (2 languages, 100 pairs, writes results/quick/).
python src/evaluate_classifiers.py "$@"
