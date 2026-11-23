#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://drive.google.com/drive/folders/1WbTqPWLGTGkJFgVxVehPMjM0rYGUwwG7"
python -m pip show gdown >/dev/null 2>&1 || python -m pip install gdown
gdown --folder "$URL" -O dataset/xsum-translation
python src/check_data.py   # fails if a test file is missing, has wrong columns, or is not 500 rows
