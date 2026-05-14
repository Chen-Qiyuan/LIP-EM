"""
Generate synthetic EM-result.json files for the two non-LIP baselines used
by weighted_iql.py:

  - target_only:   final_weights = [0]*K          (sample target only)
  - pool:          final_weights = [1]*K          (sample target + sources by size)

These directories satisfy the schema weighted_iql.py expects (target_g, N0,
final_weights), so the same training script reproduces the corresponding
rows of the published table without needing a separate codepath.

Usage (from publish/):
    py src/make_baseline_em_results.py --out-dir results/baselines \
        --target-g 8.87 --N0-list 125,250,500,1000,2000,4000
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

K = 10  # always 10 source gravities


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="results/baselines")
    p.add_argument("--target-g", type=float, required=True)
    p.add_argument("--N0-list", type=str, default="128,256,512,1024,2048,4096",
                   help="Comma-separated list of N0 values.")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    n0s = [int(x) for x in args.N0_list.split(",")]
    gravities = list(range(1, K + 1))

    for kind, weights in [("target_only", [0.0] * K), ("pool", [1.0] * K)]:
        for N0 in n0s:
            d = out_root / f"{kind}_N{N0}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "result.json").write_text(json.dumps({
                "target_g": args.target_g,
                "gravities": gravities,
                "lip_mode": f"baseline_{kind}",
                "N0": N0,
                "final_weights": weights,
                "final_argmax_g": -1,
            }, indent=2))
            print(f"wrote {d}/result.json")


if __name__ == "__main__":
    main()
