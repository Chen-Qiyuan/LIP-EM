"""
Anonymize the K=10 integer-gravity HDF5s into source_NN.csv files for the
LLM elicitation prompt.

Reads data/sources_replay/g{1..10}.hdf5 and writes elicitation/data/source_NN.csv
where the (g -> NN) mapping is a seeded random permutation. The truth mapping
is saved to elicitation/_truth.json (NOT shown to the LLM).

NOTE: fit_lip.py contains a hardcoded SOURCE_TO_G that matches the mapping
used to produce the published elicitation/results.json. If you re-run this
script with a different --seed you will produce a NEW mapping; update
SOURCE_TO_G in fit_lip.py to match the new elicitation/_truth.json before
fitting the LIP from new responses.

Usage (from publish/):
    py src/setup_elicitation.py --seed 42 --max-rows 2500
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import h5py
import numpy as np


STATE_COLS = ["z", "pitch", "q_thigh", "q_leg", "q_foot",
              "x_dot", "z_dot", "pitch_dot",
              "qd_thigh", "qd_leg", "qd_foot"]
ACTION_COLS = ["a_thigh", "a_leg", "a_foot"]


def hdf5_to_rows(path: Path, max_rows: int | None):
    with h5py.File(path, "r") as f:
        obs  = np.array(f["observations"], dtype=np.float32)
        act  = np.array(f["actions"], dtype=np.float32)
        term = np.array(f["terminals"], dtype=bool)
    if max_rows is not None and len(obs) > max_rows:
        obs, act, term = obs[:max_rows], act[:max_rows], term[:max_rows]
    rows, ep_id, step = [], 0, 0
    for i in range(len(obs)):
        rows.append([ep_id, step]
                    + [f"{x:.4f}" for x in obs[i]]
                    + [f"{x:.4f}" for x in act[i]]
                    + [int(term[i])])
        step += 1
        if term[i]:
            ep_id += 1; step = 0
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=str, default="data/sources_replay",
                   help="Where g{1..10}.hdf5 live.")
    p.add_argument("--out-dir", type=str, default="elicitation/data",
                   help="Where to write source_NN.csv files.")
    p.add_argument("--truth-out", type=str, default="elicitation/_truth.json")
    p.add_argument("--max-rows", type=int, default=2500,
                   help="Rows per source CSV (~2500 keeps 4-source prompts "
                        "below Gemini's 1M-token input limit).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    hdf5s = sorted(src_dir.glob("g*.hdf5"), key=lambda p: int(p.stem[1:]))
    if not hdf5s:
        sys.exit(f"No g*.hdf5 found under {src_dir}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(hdf5s))

    cols = ["episode", "step"] + STATE_COLS + ACTION_COLS + ["terminal"]
    mapping, g_by_anon = {}, {}

    for anon_idx, src_idx in enumerate(perm):
        src_path = hdf5s[src_idx]
        with h5py.File(src_path, "r") as f:
            g_value = float(f["metadata/g"][...])
        anon_id = f"source_{anon_idx + 1:02d}"
        dst_path = out_dir / f"{anon_id}.csv"
        rows = hdf5_to_rows(src_path, max_rows=args.max_rows)
        with dst_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow(r)
        mapping[anon_id] = src_path.stem  # "g3" etc.
        g_by_anon[anon_id] = g_value
        print(f"  {anon_id} <- {src_path.stem}  (g={g_value}, {len(rows):,} rows)")

    Path(args.truth_out).write_text(json.dumps({
        "seed": args.seed,
        "max_rows": args.max_rows,
        "anonymized_to_original": mapping,
        "g_by_anonymized": g_by_anon,
    }, indent=2))
    print(f"\nTruth saved -> {args.truth_out}")


if __name__ == "__main__":
    main()
