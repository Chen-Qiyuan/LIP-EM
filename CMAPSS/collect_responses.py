"""Collect LIP subgroup responses from a Gemini judge.

Issues `n_queries` random subgroup queries against all 100 machines and saves
the (S, C, reasoning) triples to JSON. The same JSON serves every per-target
experiment via `lip.load_responses(path, target_basename=...)` — subgroups
containing the chosen target are dropped at load time.

Usage (defaults reproduce the v2 collection in `llm_subgroup_queries_gemini_v2.json`):
    GEMINI_API_KEY=... python collect_responses.py \
        --n-queries 200 --n-parallel 10 \
        --out llm_subgroup_queries_gemini_v3.json
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data import save_as_clean_text
from lip import LLMSubgroupJudge, collect_llm_responses, save_responses


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Physical_Core_Speed_Sequences")
DEFAULT_REPORT_PATH = os.path.join(HERE, "Damage_Propagation_Modeling.pdf")
DEFAULT_TARGET_DESC_PATH = os.path.join(HERE, "target_description.txt")
DEFAULT_OUT = os.path.join(HERE, "llm_subgroup_queries.json")


def _upload_source_files(client, source_files: list[str]) -> list:
    """Upload each source's y-values as a plain-text file."""
    print(f"\n--- Uploading {len(source_files)} source files to Gemini ---",
          flush=True)
    objs = []
    tmp_dir = os.path.join(HERE, "_tmp_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        for idx, f in enumerate(source_files):
            df = pd.read_csv(f)
            y = df["Physical_core_speed_rpm"].values
            tmp = os.path.join(tmp_dir,
                               f"source_{idx}_{os.path.basename(f)}.txt")
            save_as_clean_text(y, tmp)
            uploaded = client.files.upload(
                file=tmp, config={"mime_type": "text/plain"})
            objs.append(uploaded)
            os.remove(tmp)
            if (idx + 1) % 20 == 0 or idx == len(source_files) - 1:
                print(f"  uploaded {idx+1}/{len(source_files)}", flush=True)
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
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
    ap.add_argument("--api-key", type=str, default=None,
                    help="Gemini API key (default: env GEMINI_API_KEY)")
    ap.add_argument("--model", type=str, default="gemini-3-flash-preview",
                    help="Gemini model name")
    ap.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH,
                    help="path to the physics manual PDF")
    ap.add_argument("--target-desc-path", type=str,
                    default=DEFAULT_TARGET_DESC_PATH,
                    help="path to a file containing the target description")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--subgroup-min", type=int, default=3)
    ap.add_argument("--subgroup-max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-parallel", type=int, default=10,
                    help="concurrent in-flight queries (>=1). publish default "
                         "is 10; the judge's 429 retry handles per-minute TPM.")
    ap.add_argument("--sleep-between", type=float, default=0.0,
                    help="ignored when --n-parallel > 1")
    ap.add_argument("--abort-after", type=int, default=3,
                    help="abort sequential collection after this many "
                         "consecutive API failures (0 to disable; "
                         "ignored in parallel mode)")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--keep-uploads", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Pass --api-key or set GEMINI_API_KEY.")

    if not os.path.exists(args.target_desc_path):
        raise RuntimeError(
            f"Missing target description file: {args.target_desc_path}")
    with open(args.target_desc_path, "r", encoding="utf-8") as f:
        target_desc = f.read()

    source_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    K = len(source_files)
    if K == 0:
        raise RuntimeError(f"no CSVs found in {DATA_DIR}")

    print(f"Source pool: {K} machines")
    print(f"Plan: {args.n_queries} queries, size {args.subgroup_min}-{args.subgroup_max}")
    print(f"Output: {args.out}")
    print(f"Target description:\n{target_desc.strip()}\n")

    from google import genai
    client = genai.Client(api_key=api_key)
    source_file_objs = _upload_source_files(client, source_files)
    try:
        judge = LLMSubgroupJudge(api_key=api_key,
                                  manual_path=args.report_path,
                                  model_name=args.model)

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

    if os.path.exists(args.out):
        bak = args.out + f".bak.{int(time.time())}"
        os.replace(args.out, bak)
        print(f"\nBacked up existing output -> {bak}")

    source_basenames = [os.path.basename(f) for f in source_files]
    save_responses(args.out, target_desc=target_desc, responses=responses,
                   source_basenames=source_basenames,
                   meta={"model": args.model, "n_queries": args.n_queries,
                         "subgroup_size_range": [args.subgroup_min, args.subgroup_max],
                         "seed": args.seed, "n_parallel": args.n_parallel})
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
