"""Rebuild data/Physical_Core_Speed_Sequences/ from the raw NASA C-MAPSS files.

The per-engine CSVs ship with the repo, so this step is optional. It extracts
sensor measurement 9 (physical core speed, rpm; column 14 of the raw file) of
every engine in the FD001 training set and writes one CSV per engine.

Usage:
    python prepare_data.py --raw path/to/train_FD001.txt
    python prepare_data.py --raw path/to/train_FD001.txt --out-dir some/dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dataset import COLUMN


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data" / "Physical_Core_Speed_Sequences"

# Raw layout: unit, cycle, 3 operational settings, sensors 1..21.
UNIT_COL, CYCLE_COL, SENSOR_9_COL = 0, 1, 2 + 3 + 8


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True,
                    help="train_FD001.txt from the NASA C-MAPSS archive")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw, sep=r"\s+", header=None)
    if raw.shape[1] != 26:
        raise ValueError(f"expected 26 columns, got {raw.shape[1]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for unit, df in raw.groupby(UNIT_COL, sort=True):
        df = df.sort_values(CYCLE_COL)
        pd.DataFrame({COLUMN: df[SENSOR_9_COL].values}).to_csv(
            args.out_dir / f"{int(unit)}.csv", index=False)
    print(f"Wrote {raw[UNIT_COL].nunique()} engines to {args.out_dir}")


if __name__ == "__main__":
    main()
