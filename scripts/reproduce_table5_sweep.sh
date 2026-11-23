#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Report Table 5. Needs results/table10/*_preds.csv from reproduce_tables_1_2.sh.
python analysis/threshold_sweep.py
