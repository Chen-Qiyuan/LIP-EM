"""
LIP elicitation pipeline against the Gemini API.

For each of N queries: pre-sample a subgroup of 3-4 candidate sources,
build a prompt containing the Hopper technical report + target spec +
the first M rows of each candidate's CSV, send to Gemini with a
structured-output schema, parse the {matching, reasoning} JSON, and
append to results.json. Saves after every query so the run is
resumable.

Setup:
    pip install google-genai pandas
    export GEMINI_API_KEY=your_key_here   # or set --api-key-env

Run:
    python lip_query_gemini.py                       # full 50-query run
    python lip_query_gemini.py --n-queries 5         # smoke test
    python lip_query_gemini.py --resume              # continue partial run
    python lip_query_gemini.py --dry-run             # preview prompt size, no API call
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

# Imported lazily inside main() so --dry-run works without the SDK installed.


# ============================================================================
# Subgroup sampling
# ============================================================================

def sample_subgroups(n_queries: int, n_sources: int = 10, seed: int = 42,
                     ratio_size3: float = 0.6) -> list[tuple[int, ...]]:
    """Generate n_queries distinct subgroups; mix of size 3 and size 4."""
    rng = random.Random(seed)
    n_3 = int(round(n_queries * ratio_size3))
    n_4 = n_queries - n_3
    pool_3 = list(combinations(range(1, n_sources + 1), 3))
    pool_4 = list(combinations(range(1, n_sources + 1), 4))
    rng.shuffle(pool_3)
    rng.shuffle(pool_4)
    if len(pool_3) < n_3 or len(pool_4) < n_4:
        sys.exit(f"Not enough distinct subgroups: need {n_3}/{n_4}, "
                 f"have {len(pool_3)}/{len(pool_4)}")
    chosen: list[tuple[int, ...]] = pool_3[:n_3] + pool_4[:n_4]
    rng.shuffle(chosen)
    return chosen


# ============================================================================
# CSV preparation
# ============================================================================

def load_csv_head(path: Path, n_rows: int) -> str:
    """Return first n_rows of the CSV (with header) as a CSV string."""
    df = pd.read_csv(path, nrows=n_rows)
    return df.to_csv(index=False)


# ============================================================================
# Prompt assembly
# ============================================================================

INSTRUCTIONS = """\
You are participating in a "language-induced prior" elicitation experiment for a robotic locomotion task.

# Task

You are given a SUBGROUP of {N} candidate sources drawn from a larger pool. For this query, decide which one (if any) of the {N} candidates is dynamically consistent with the TARGET environment described below. Be conservative — if none of the {N} candidates clearly matches the target, return null.

# Inputs you receive

1. TECHNICAL REPORT — a description of the Hopper simulation environment and the telemetry format (below).
2. TARGET — the target environment specification (below).
3. DATA — for each of the {N} candidates, the first portion of its replay-buffer CSV.

# This query's subgroup

The candidates you must choose between are: {subgroup_list}.

# Output

Return your answer as a JSON object with two fields:

- `matching`: one of {subgroup_list}, or null if none of them matches.
- `reasoning`: a short (1–4 sentence) summary of the analysis that led to your decision, including any quantitative evidence you found in the data.

Do not include any source ID outside this query's subgroup in `matching`.
"""


def build_prompt(subgroup: tuple[int, ...], technical_report: str,
                 target: str, csv_data: dict[int, str], n_rows: int) -> str:
    subgroup_list = ", ".join(f'"source_{s:02d}"' for s in subgroup)
    instr = INSTRUCTIONS.format(N=len(subgroup), subgroup_list=subgroup_list)
    blocks = [
        instr,
        f"# TECHNICAL REPORT\n\n{technical_report.strip()}",
        f"# TARGET\n\n{target.strip()}",
        f"# DATA\n\nFor each candidate below, you have the first {n_rows:,} rows of its replay-buffer CSV. Rows within an episode are consecutive in time (state on row k+1 results from applying row k's action); the data preserves complete episodes.",
    ]
    for s in subgroup:
        blocks.append(
            f"## source_{s:02d}.csv (first {n_rows:,} rows)\n\n"
            f"```csv\n{csv_data[s]}```"
        )
    return "\n\n".join(blocks)


# ============================================================================
# Gemini call
# ============================================================================

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "matching": {
            "type": "STRING",
            "nullable": True,
            "description": "The source_XX from the subgroup that matches the target, or null.",
        },
        "reasoning": {
            "type": "STRING",
            "description": "1-4 sentence explanation of how you arrived at the decision.",
        },
    },
    "required": ["matching", "reasoning"],
}


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _retry_delay_from_error(err: Exception) -> float | None:
    """Extract a retry-after seconds value from a Gemini 429/503 error message."""
    msg = str(err)
    m = _RETRY_DELAY_RE.search(msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_response_text(resp) -> str:
    """Try multiple paths to extract text from a Gemini response object,
    falling back to a structured diagnostic if none of them yield text."""
    # 1. The convenience accessor
    text = getattr(resp, "text", None)
    if text:
        return text
    # 2. Walk candidates -> content -> parts
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        finish = getattr(cand, "finish_reason", None)
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            chunks = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            if chunks:
                return "".join(chunks)
        # No parts — surface the finish reason as diagnostic JSON
        if finish:
            return json.dumps({
                "matching": None,
                "reasoning": f"NO_TEXT (finish_reason={finish})",
            })
    return json.dumps({"matching": None, "reasoning": "NO_TEXT (empty response)"})


def query_gemini(client, model: str, prompt: str, max_retries: int = 5,
                  timeout_s: float = 900.0) -> str:
    """Send prompt to Gemini with structured-output schema, return raw JSON text.

    Retries with exponential backoff. For 429 (rate limit) errors, honors the
    server's suggested retry delay instead of using the exponential schedule.
    Does not retry on 400 (input too large or malformed) — that's a permanent error.

    timeout_s: per-request HTTP timeout (default 15 min). The SDK has no default
    timeout, so without this a stalled stream can hang forever and deadlock the
    worker pool.
    """
    from google.genai import types  # local import

    config = types.GenerateContentConfig(
        temperature=0.0,
        # Gemini 2.5 Pro spends thinking tokens out of this same budget.
        # 65536 is the hard cap; lower and Q sometimes finish_reason=MAX_TOKENS
        # before it emits the JSON answer.
        max_output_tokens=65536,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        # Per-request HTTP timeout in MILLISECONDS (Google SDK convention).
        # Without this, hung connections deadlock worker threads forever.
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return _extract_response_text(resp)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # 400 = bad request (e.g. input too large) — no point retrying
            if "400" in msg and "INVALID_ARGUMENT" in msg:
                raise
            if attempt < max_retries - 1:
                # Honor server's suggested retry delay if present
                server_delay = _retry_delay_from_error(e)
                wait = (server_delay + 1.0) if server_delay else (2 ** attempt + random.random())
                print(f"  retry {attempt + 1}/{max_retries} in {wait:.1f}s: "
                      f"{msg[:200]}", file=sys.stderr)
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


# ============================================================================
# Response parsing
# ============================================================================

def parse_response(text: str) -> dict:
    """Parse Gemini's JSON response. Falls back to regex if straight json.loads fails."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"matching": None, "reasoning": "[parse failed: " + text[:200] + "]"}


def normalize_matching(matching, subgroup_str: list[str]) -> str | None:
    """Coerce an out-of-subgroup or null/'null' matching to None."""
    if matching is None:
        return None
    if isinstance(matching, str):
        if matching.lower() in ("null", "none", ""):
            return None
        if matching in subgroup_str:
            return matching
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cc-dir", type=Path, default=Path("elicitation"),
                   help="Directory with technical_report.md, target.md, data/")
    p.add_argument("--results", type=Path,
                   default=Path("elicitation/results.json"),
                   help="Output results file (saved after every query)")
    p.add_argument("--model", type=str, default="gemini-3-flash-preview",
                   help="Gemini model name (e.g. gemini-3-flash-preview, "
                        "gemini-3-pro-preview, gemini-2.5-flash). Run "
                        "`client.models.list()` to see the canonical names.")
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--n-rows-per-source", type=int, default=2500,
                   help="Rows per source CSV to include in the prompt. "
                        "~2500 keeps 4-source queries below Gemini's 1M-token "
                        "input limit (CSV ratio is ~1.6 chars/token).")
    p.add_argument("--fire-interval", type=float, default=0.0,
                   help="Seconds between consecutive query submissions. "
                        "With --max-concurrent>1, throttles the submission "
                        "rate (e.g. 120 = fire every 2 minutes).")
    p.add_argument("--max-concurrent", type=int, default=1,
                   help="Max number of queries in flight at once. Set to "
                        "3 to fire 3 in parallel under TPM headroom.")
    p.add_argument("--n-sources", type=int, default=10,
                   help="Total candidate sources in the pool")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for subgroup sampling (reproducible)")
    p.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY",
                   help="Env var holding the Gemini API key")
    p.add_argument("--resume", action="store_true",
                   help="Continue from where results.json left off")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts but do not call the API")
    args = p.parse_args()

    cc_dir: Path = args.cc_dir
    data_dir = cc_dir / "data"
    tr_path = cc_dir / "technical_report.md"
    tgt_path = cc_dir / "target.md"
    for f in (data_dir, tr_path, tgt_path):
        if not f.exists():
            sys.exit(f"Missing required path: {f}")
    technical_report = tr_path.read_text(encoding="utf-8")
    target = tgt_path.read_text(encoding="utf-8")

    # Pre-load subsampled CSVs once (indices 1..n_sources)
    print(f"Loading first {args.n_rows_per_source:,} rows per source from {data_dir}")
    csv_data: dict[int, str] = {}
    for s in range(1, args.n_sources + 1):
        path = data_dir / f"source_{s:02d}.csv"
        if not path.exists():
            sys.exit(f"Missing source CSV: {path}")
        csv_data[s] = load_csv_head(path, args.n_rows_per_source)
        print(f"  source_{s:02d}: {len(csv_data[s]):,} chars")

    # Pre-sample all subgroups (deterministic, repeatable across runs)
    subgroups = sample_subgroups(args.n_queries, n_sources=args.n_sources,
                                  seed=args.seed)
    print(f"Sampled {len(subgroups)} distinct subgroups (seed={args.seed})")

    # Resume state
    out_path: Path = args.results
    queries: list[dict] = []
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        queries = existing.get("queries", [])
        # Safety: ensure each saved entry's subgroup matches what we'd
        # have sampled for that query_id.
        for q in queries:
            qid = q["query_id"]
            saved_sg = tuple(int(s.split("_")[1]) for s in q["subgroup"])
            if saved_sg != subgroups[qid]:
                sys.exit(f"Subgroup mismatch at query_id={qid}: "
                         f"saved {saved_sg} vs sampled {subgroups[qid]}. "
                         f"Use a different --results file or --seed.")
        print(f"Resuming with {len(queries)}/{args.n_queries} already done")

    if args.dry_run:
        print("\n[dry-run] Building first prompt for inspection:")
        sg = subgroups[completed]
        prompt = build_prompt(sg, technical_report, target,
                               csv_data, args.n_rows_per_source)
        print(f"  subgroup: {[f'source_{s:02d}' for s in sg]}")
        print(f"  prompt size: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens)")
        print(f"  preview (first 1000 chars):\n{'-' * 60}\n{prompt[:1000]}\n{'-' * 60}")
        return

    # Init Gemini
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"Set ${args.api_key_env} (e.g. export {args.api_key_env}=...)")
    from google import genai
    client = genai.Client(api_key=api_key)
    print(f"Initialized Gemini client (model={args.model})")

    # Determine which query_ids still need to run
    done_ids = {q["query_id"] for q in queries}
    pending_ids = [i for i in range(args.n_queries) if i not in done_ids]
    if not pending_ids:
        print("All queries already complete.")
        return
    print(f"Running {len(pending_ids)} pending queries: "
          f"max_concurrent={args.max_concurrent}, "
          f"fire_interval={args.fire_interval}s")

    save_lock = threading.Lock()

    def run_one(qid: int) -> dict:
        """Build prompt for query qid, send to Gemini, return a result dict."""
        sg = subgroups[qid]
        sg_str = [f"source_{s:02d}" for s in sg]
        prompt = build_prompt(sg, technical_report, target, csv_data,
                               args.n_rows_per_source)
        approx_tokens = len(prompt) // 4
        print(f"  [submit q{qid:02d}] subgroup={sg_str}, "
              f"prompt={len(prompt):,} chars (~{approx_tokens:,} tokens)",
              flush=True)
        try:
            resp_text = query_gemini(client, args.model, prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [error  q{qid:02d}] {e}", file=sys.stderr, flush=True)
            resp_text = json.dumps({"matching": None,
                                     "reasoning": f"API_ERROR: {e}"})
        parsed = parse_response(resp_text)
        matching_raw = parsed.get("matching")
        matching = normalize_matching(matching_raw, sg_str)
        if matching_raw is not None and matching is None and matching_raw != "null":
            print(f"  [warn   q{qid:02d}] invalid matching {matching_raw!r}, "
                   f"coerced to None", flush=True)
        return {
            "query_id": qid,
            "subgroup": sg_str,
            "matching": [matching] if matching else [],
            "reasoning": parsed.get("reasoning", ""),
            "raw_response": resp_text[:4000],
        }

    def on_complete(fut):
        try:
            result = fut.result()
        except Exception as e:  # noqa: BLE001
            print(f"  [thread error] {e}", file=sys.stderr, flush=True)
            return
        with save_lock:
            queries.append(result)
            queries.sort(key=lambda q: q["query_id"])
            out = {"n_completed": len(queries), "queries": queries}
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  [done   q{result['query_id']:02d}] matching={result['matching']}",
              flush=True)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.max_concurrent)) as executor:
        futures = []
        for idx, qid in enumerate(pending_ids):
            fut = executor.submit(run_one, qid)
            fut.add_done_callback(on_complete)
            futures.append(fut)
            if idx < len(pending_ids) - 1 and args.fire_interval > 0:
                time.sleep(args.fire_interval)
        # Drain remaining futures (the on_complete callback handles saving)
        concurrent.futures.wait(futures)

    dt = time.time() - t0
    print(f"\nFinished {len(pending_ids)} queries in {dt / 60:.1f} min. "
          f"Results: {out_path}")


if __name__ == "__main__":
    main()
