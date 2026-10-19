"""Collect LIP subgroup responses from a Gemini judge (optional; the responses
used in the paper are cached in data/llm_subgroup_queries_gemini.json).

Issues `n_queries` random subgroup queries against all 100 engines and saves
the (S, C, reasoning) triples to JSON. The same JSON serves every target:
`lip.load_responses(path, target_basename=...)` drops the subgroups that
contain the chosen target and re-indexes the rest.

The paper's collection used the defaults below (gemini-3-flash-preview,
200 queries, subgroups of 3-5 engines, seed 42, 10 parallel workers) with the
NASA report and target description in data/. Attachments are passed without
text labels (`label_attachments=False`).

Usage:
    export GEMINI_API_KEY=...
    python collect_responses.py --out llm_subgroup_queries.json
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip.lip import (LLMSubgroupJudge, collect_llm_responses,  # noqa: E402
                     save_responses)

from dataset import COLUMN, save_as_clean_text  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SEQ_DIR = DATA_DIR / "Physical_Core_Speed_Sequences"
DEFAULT_REPORT_PATH = DATA_DIR / "Damage_Propagation_Modeling.pdf"
DEFAULT_TARGET_DESC_PATH = DATA_DIR / "target_description.txt"
DEFAULT_OUT = HERE / "llm_subgroup_queries.json"


def _upload_source_files(client, source_files: list[Path]) -> list:
    """Upload each source's y-values as a plain-text file."""
    print(f"\n--- Uploading {len(source_files)} source files to Gemini ---",
          flush=True)
    objs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, f in enumerate(source_files):
            y = pd.read_csv(f)[COLUMN].values
            tmp = Path(tmp_dir) / f"source_{idx}_{f.name}.txt"
            save_as_clean_text(y, tmp)
            uploaded = client.files.upload(
                file=str(tmp), config={"mime_type": "text/plain"})
            objs.append(uploaded)
            if (idx + 1) % 20 == 0 or idx == len(source_files) - 1:
                print(f"  uploaded {idx+1}/{len(source_files)}", flush=True)
    return objs


def _delete_uploaded(client, file_objs) -> None:
    for fo in file_objs:
        try:
            client.files.delete(name=fo.name)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", type=str, default="gemini-3-flash-preview",
                    help="Gemini model name")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH,
                    help="technical report uploaded as LLM context")
    ap.add_argument("--target-desc-path", type=Path,
                    default=DEFAULT_TARGET_DESC_PATH,
                    help="file containing the target description")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--subgroup-min", type=int, default=3)
    ap.add_argument("--subgroup-max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-parallel", type=int, default=10,
                    help="concurrent in-flight queries (>=1); the judge's "
                         "429 retry handles the per-minute token limit")
    ap.add_argument("--sleep-between", type=float, default=0.0,
                    help="ignored when --n-parallel > 1")
    ap.add_argument("--abort-after", type=int, default=3,
                    help="abort sequential collection after this many "
                         "consecutive API failures (0 to disable; "
                         "ignored in parallel mode)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep-uploads", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable.")

    if not args.target_desc_path.exists():
        raise RuntimeError(
            f"Missing target description file: {args.target_desc_path}")
    target_desc = args.target_desc_path.read_text(encoding="utf-8")

    source_files = sorted(SEQ_DIR.glob("*.csv"), key=lambda f: f.name)
    K = len(source_files)
    if K == 0:
        raise RuntimeError(f"no CSVs found in {SEQ_DIR}")

    print(f"Source pool: {K} engines")
    print(f"Plan: {args.n_queries} queries, size {args.subgroup_min}-{args.subgroup_max}")
    print(f"Output: {args.out}")
    print(f"Target description:\n{target_desc.strip()}\n")

    from google import genai
    client = genai.Client(api_key=api_key)
    source_file_objs = _upload_source_files(client, source_files)
    try:
        judge = LLMSubgroupJudge(api_key=api_key,
                                 manual_path=str(args.report_path),
                                 model_name=args.model,
                                 label_attachments=False)

        print("\n--- Running subgroup queries ---", flush=True)
        t0 = time.time()
        responses = collect_llm_responses(
            judge=judge, source_file_objs=source_file_objs,
            target_desc=target_desc, n_queries=args.n_queries,
            subgroup_size_range=(args.subgroup_min, args.subgroup_max),
            seed=args.seed, sleep_between=args.sleep_between,
            abort_after_consecutive_failures=args.abort_after,
            n_parallel=args.n_parallel)
        elapsed = time.time() - t0
        print(f"\nCollected {len(responses)} responses in {elapsed:.1f}s "
              f"({elapsed/max(len(responses),1):.1f}s per query)")

        if responses:
            sizes_S = [len(r[0]) for r in responses]
            sizes_C = [len(r[1]) for r in responses]
            n_empty = sum(1 for r in responses if len(r[1]) == 0)
            n_with_rsn = sum(1 for r in responses if len(r) >= 3 and r[2].strip())
            avg_rsn_chars = (sum(len(r[2]) for r in responses if len(r) >= 3)
                             / max(len(responses), 1))
            print(f"  |S|: min={min(sizes_S)}  median={int(np.median(sizes_S))}  "
                  f"max={max(sizes_S)}")
            print(f"  |C|: min={min(sizes_C)}  median={int(np.median(sizes_C))}  "
                  f"max={max(sizes_C)}  empty={n_empty}")
            print(f"  reasoning: {n_with_rsn}/{len(responses)} non-empty, "
                  f"avg {avg_rsn_chars:.0f} chars")
        else:
            print("  (no valid responses to summarize)")
    finally:
        if not args.keep_uploads:
            print("\nCleaning up uploaded source files...", flush=True)
            _delete_uploaded(client, source_file_objs)

    if not responses:
        print(f"\nNo new responses; not creating {args.out}.")
        return

    if args.out.exists():
        bak = args.out.with_name(args.out.name + f".bak.{int(time.time())}")
        os.replace(args.out, bak)
        print(f"\nBacked up existing output -> {bak}")

    save_responses(str(args.out), target_desc=target_desc, responses=responses,
                   source_basenames=[f.name for f in source_files],
                   meta={"model": args.model, "n_queries": args.n_queries,
                         "subgroup_size_range": [args.subgroup_min, args.subgroup_max],
                         "seed": args.seed, "n_parallel": args.n_parallel})
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
