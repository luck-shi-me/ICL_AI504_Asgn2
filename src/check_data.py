"""Check that the downloaded dataset has the expected files, columns and row counts."""
import sys

import pandas as pd

from common import LANGS, DATA_DIR, EXPECTED_TEST_ROWS, DOC_COL, SUM_COL, LABEL_COL, labels_to_binary, find_test_file


def main(data_dir=DATA_DIR):
    """Validate each language's test file; exit with status 1 if anything is wrong."""
    ok = True
    for lang in LANGS:
        try:
            d = pd.read_csv(find_test_file(lang, data_dir))
            labels_to_binary(d[LABEL_COL])
            cols_ok = {DOC_COL, SUM_COL, LABEL_COL} <= set(d.columns)
            rows_ok = len(d) == EXPECTED_TEST_ROWS
            print(f"{lang}: rows={len(d)} columns_ok={cols_ok} rows_ok={rows_ok}")
            ok &= cols_ok and rows_ok
        except Exception as e:  # report every problem, then fail
            print(f"{lang}: ERROR {e}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
