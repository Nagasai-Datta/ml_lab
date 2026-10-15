"""Shared paths for the prototypes: our copy of sir's Topic 7 code and the real dataset, inside topic7/."""
import sys
from pathlib import Path

TOPIC7 = Path(__file__).resolve().parents[1]
CODE = TOPIC7 / "code"
CSV = TOPIC7 / "data" / "KaggleV2-May-2016.csv"

for p in (str(CODE), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
