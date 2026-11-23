"""Make src/ and analysis/ importable from the tests."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for sub in ("src", "analysis"):
    sys.path.insert(0, str(ROOT / sub))
