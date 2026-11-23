#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/download_data.sh
bash scripts/reproduce_tables_1_2.sh
bash scripts/reproduce_table3_confusion.sh
bash scripts/reproduce_table4_probe.sh
bash scripts/reproduce_table5_sweep.sh
