"""
Fit a Language-Induced Prior (LIP) pi over the K=10 sources from LLM
elicitation responses (paper eq:LIP_opt, conditional logit with null option),
using the shared fit in lip/lip.py.

Reads a response file (written by lip_query_gemini.py, or the Claude agent's
elicitation/claude_responses.json), where each query is a dict
{subgroup: [source_NN, ...], matching: [source_MM] or []}, and maps
source_NN -> integer gravity g via SOURCE_TO_G (the seed-42 anonymization of
setup_elicitation.py).

With the cached elicitation/gemini_responses.json this reproduces the LIP-G
prior lips/lip_gemini.json (argmax at g = 7); with --responses
elicitation/claude_responses.json --compare lips/lip_claude.json it reproduces
the LIP-C prior (argmax at g = 9; see elicitation/claude_agent/README.md).
Prints the fitted pi and its difference to --compare; writes a JSON only when
--out is given.

Usage (from hopper/):
    python src/fit_lip.py                                  # defaults p0=0.01, eps=1.0
    python src/fit_lip.py --responses <file> --out lips/my_lip.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

HOPPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOPPER_DIR.parent))          # repository root
from lip.lip import fit_lip

GRAVITIES = list(range(1, 11))

# source_NN -> integer gravity g (elicitation/_truth.json of
# `setup_elicitation.py --seed 42`).
SOURCE_TO_G = {
    "source_01": 6,
    "source_02": 7,
    "source_03": 1,
    "source_04": 8,
    "source_05": 4,
    "source_06": 3,
    "source_07": 5,
    "source_08": 10,
    "source_09": 2,
    "source_10": 9,
}


def load_queries(path: Path) -> list[tuple[list[int], list[int]]]:
    """(S, C) pairs over source indices 0..9 (index = g - 1); |C| <= 1."""
    with open(path) as f:
        queries = json.load(f)["queries"]
    out = []
    for q in queries:
        S = sorted(SOURCE_TO_G[s] - 1 for s in q["subgroup"])
        C = [SOURCE_TO_G[s] - 1 for s in q["matching"]]
        out.append((S, C))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", type=str,
                   default=str(HOPPER_DIR / "elicitation" / "gemini_responses.json"),
                   help="LLM responses (from lip_query_gemini.py).")
    p.add_argument("--p0", type=float, default=0.01,
                   help="Prior probability for an arbitrary source (default 0.01).")
    p.add_argument("--eps", type=float, default=1.0,
                   help="L2 regularization on alpha_k (default 1.0).")
    p.add_argument("--compare", type=str,
                   default=str(HOPPER_DIR / "lips" / "lip_gemini.json"),
                   help="LIP JSON to compare the fit against.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    responses = load_queries(Path(args.responses))
    n_null = sum(1 for _, C in responses if not C)
    print(f"Loaded {len(responses)} responses from {Path(args.responses).name} "
          f"(null picks: {n_null})")

    pi = fit_lip(responses, K=len(GRAVITIES), p0=args.p0, eps=args.eps)

    print(f"\n=== Fitted LIP (p0={args.p0}, eps={args.eps}) ===")
    for g, pg in zip(GRAVITIES, pi):
        print(f"  g={g:<2}: pi = {pg:.4f}")
    print(f"  argmax g = {GRAVITIES[int(np.argmax(pi))]}")

    if args.compare and Path(args.compare).exists():
        ref = np.array(json.load(open(args.compare))["pi"])
        print(f"\nmax |pi - pi[{Path(args.compare).name}]| = {np.abs(pi - ref).max():.2e}")

    if args.out:
        out = {"p0": args.p0, "eps": args.eps, "pi": [float(x) for x in pi],
               "gravities": GRAVITIES, "source_to_g": SOURCE_TO_G,
               "n_queries": len(responses)}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
