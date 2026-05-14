"""
Language-Induced Prior (LIP) for the C-MAPSS case study.

Two halves:
  * **Elicitation** — `LLMSubgroupJudge` + `collect_llm_responses` issue
    subgroup queries to a Gemini judge and collect (S, C, reasoning) triples.
    `save_responses` / `load_responses` round-trip the JSON.
  * **Fit** — `fit_lip` solves paper eq:LIP_opt: a regularized conditional
    logit with an outside option (eq:lip_choice). Returns pi = sigmoid(alpha).

The conditional-logit-with-outside-option likelihood (paper §3.1):

    P(k | S) = exp(α_k) / (exp(α_0) + Σ_{j ∈ S} exp(α_j))   for k ∈ S
    P(0 | S) = exp(α_0) / (exp(α_0) + Σ_{j ∈ S} exp(α_j))    (null option)

requires |C| ≤ 1 per query. The current prompt enforces this; legacy
multi-pick records can be decomposed via `multipick_to_singlepick`.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Sequence

import numpy as np
import torch


# ============================================================================
# LLM Subgroup Judge (Gemini)
# ============================================================================

# Default prompt — C-MAPSS-specific (HPC core speed degradation). Override
# with `prompt_template` at __init__ for other domains. The template is a
# str.format()-compatible string with placeholders:
#   {n_labels}, {labels_str}, {target_env_desc}
DEFAULT_PROMPT_TEMPLATE = """
You are an expert aerospace prognostic engineer. I have attached a NASA Technical Report detailing engine damage propagation.

[TASK]
1. Read the attached NASA Report to understand the physics of the turbofan system.
2. Analyze the 'Target Description' below.
3. Examine the {n_labels} attached Source Dataset files, labeled {labels_str}. Each file contains sensor 9 sequences formatted as (Cycle, High Pressure Compressor Physical Core Speed).
4. Pick exactly ONE of:
   - the single label whose source dataset BEST matches the physical trajectory implied by the Target Description, even if other sources also match — commit to the one you judge most relevant; or
   - an empty list, if none of the sources is a credible match.
5. NEVER return more than one label. The point is to extract within-subgroup preference ranking — when multiple sources all plausibly match, you must still pick the single best one.

[TARGET DESCRIPTION]
"{target_env_desc}"

Respond strictly in JSON with two fields:
  - "selected": one label from {labels_str} (as a single-element list), or an empty list.
  - "reasoning": 1-3 sentences citing physical evidence in the data (e.g. trajectory shape, monotonic trends, terminal value) that drove your decision.

Example formats:
{{
    "selected": ["A"],
    "reasoning": "Source A's RPM declines monotonically with an accelerating drop near end-of-life, consistent with HPC erosion under continuous abrasive ingestion."
}}
or
{{
    "selected": [],
    "reasoning": "None of the sources show the rapid late-life RPM drop that would be expected from severe HPC aerodynamic degradation."
}}
"""


class LLMSubgroupJudge:
    """Subgroup judge backed by the Google Gemini API.

    Uploads the context manual once at construction. On each query, the
    LLM is shown a subgroup of source datasets (3-5 typically) and asked
    to pick the single best match (or none). Returns
    `(chosen_indices, reasoning)` or `None` on persistent API failure.
    """

    # Default prompt — C-MAPSS-specific (HPC core speed degradation). Override
    # with `prompt_template` at __init__ for other domains. The template is a
    # str.format()-compatible string with placeholders:
    #   {n_labels}, {labels_str}, {target_env_desc}
    # See `DEFAULT_PROMPT_TEMPLATE` below for the format.
    def __init__(self, api_key: str, manual_path: str, model_name: str,
                 prompt_template: str | None = None):
        # Imported lazily so JSON-only callers don't need google-genai.
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        print(f"--- [LLM Judge] Uploading context file: {manual_path} ---")
        self.report_file = self.client.files.upload(file=manual_path)
        while getattr(self.report_file, "state", None) == "PROCESSING":
            time.sleep(0.5)
            self.report_file = self.client.files.get(name=self.report_file.name)
        print(f"--- [LLM Judge] Context ready ---")

    def _construct_prompt(self, target_env_desc: str,
                          labels: Sequence[str]) -> str:
        labels_str = ", ".join(labels)
        return self.prompt_template.format(
            n_labels=len(labels), labels_str=labels_str,
            target_env_desc=target_env_desc)


    def query(self, target_env_desc: str, file_objs: Sequence,
              tpm_window_seconds: float = 65.0,
              max_retries: int = 1) -> tuple[list[int], str] | None:
        """One query. Returns (chosen_indices, reasoning) or None on failure.

        Retry policy is TPM-aware (Gemini's bottleneck is per-minute tokens,
        not requests). On 429, sleep `tpm_window_seconds` so the rolling
        window has time to age out at least one prior call's worth of tokens.
        """
        labels = [chr(ord("A") + i) for i in range(len(file_objs))]
        prompt = self._construct_prompt(target_env_desc, labels)

        for attempt in range(max_retries + 1):
            response = None
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, self.report_file] + list(file_objs),
                    config=self._genai.types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.0),
                )
                decision = json.loads(response.text)
                if isinstance(decision, list):
                    decision = (decision[0]
                                if decision and isinstance(decision[0], dict)
                                else {})
                selected = decision.get("selected", [])
                if not isinstance(selected, list):
                    selected = []
                reasoning = decision.get("reasoning", "")
                if not isinstance(reasoning, str):
                    reasoning = str(reasoning) if reasoning is not None else ""
                chosen = []
                for lbl in selected:
                    lbl_up = str(lbl).upper().strip()
                    if lbl_up in labels:
                        chosen.append(labels.index(lbl_up))
                chosen = sorted(set(chosen))
                # Enforce single-choice (paper eq:lip_choice requires |C| <= 1).
                if len(chosen) > 1:
                    print(f"   [warn] LLM returned {len(chosen)} picks "
                          f"({chosen}); keeping first only", flush=True)
                    chosen = chosen[:1]
                return chosen, reasoning
            except Exception as e:
                err_str = str(e)
                is_quota = ("429" in err_str
                            or "RESOURCE_EXHAUSTED" in err_str)
                is_unavailable = ("503" in err_str
                                  or "UNAVAILABLE" in err_str
                                  or "DEADLINE_EXCEEDED" in err_str)
                last = (attempt == max_retries)
                if (is_quota or is_unavailable) and not last:
                    delay = tpm_window_seconds if is_quota else 5.0
                    reason = "quota" if is_quota else "unavailable"
                    print(f"   [retry] {reason}; sleeping {delay:.0f}s",
                          flush=True)
                    time.sleep(delay)
                    continue
                print(f"   [Error] {err_str[:120]}", flush=True)
                return None
        return None


# ============================================================================
# Collection orchestration (sequential or parallel)
# ============================================================================

def collect_llm_responses(judge: LLMSubgroupJudge,
                          source_file_objs: Sequence,
                          target_desc: str,
                          n_queries: int = 200,
                          subgroup_size_range: tuple[int, int] = (3, 5),
                          seed: int = 42,
                          sleep_between: float = 0.0,
                          abort_after_consecutive_failures: int = 3,
                          n_parallel: int = 10,
                          ) -> list[tuple[list[int], list[int], str]]:
    """Generate up to `n_queries` (S, C, reasoning) triples.

    `n_parallel` controls concurrency:
      * `1` -> sequential (paced by `sleep_between`)
      * `>1` -> ThreadPoolExecutor; the judge's 429 retry handles TPM.

    Subgroup samples are deterministic via `seed`, regardless of order.

    The circuit breaker (`abort_after_consecutive_failures`) is sequential-mode
    only — consecutive failure ordering is ill-defined under concurrent calls.
    """
    K = len(source_file_objs)
    sz_lo, sz_hi = subgroup_size_range
    sz_hi = min(sz_hi, K)
    sz_lo = min(sz_lo, sz_hi)

    # Pre-sample subgroups so order doesn't matter.
    rng = np.random.default_rng(seed)
    plans = []
    for q_idx in range(n_queries):
        sz = int(rng.integers(sz_lo, sz_hi + 1))
        S = sorted(rng.choice(K, size=sz, replace=False).tolist())
        plans.append((q_idx, sz, S))

    if n_parallel <= 1:
        return _collect_sequential(judge, source_file_objs, target_desc,
                                    plans, n_queries, sleep_between,
                                    abort_after_consecutive_failures)
    return _collect_parallel(judge, source_file_objs, target_desc,
                              plans, n_queries, n_parallel)


def _format_query_log(q_idx, n_queries, sz, S, C, reasoning):
    rsn = reasoning.replace("\n", " ").strip()
    if len(rsn) > 100:
        rsn = rsn[:97] + "..."
    return (f"  Query {q_idx+1:>3d}/{n_queries}: |S|={sz}  S={S}  ->  C={C}"
            f"  | {rsn}")


def _collect_sequential(judge, source_file_objs, target_desc, plans,
                         n_queries, sleep_between,
                         abort_after_consecutive_failures):
    responses, n_failed, consecutive = [], 0, 0
    for q_idx, sz, S in plans:
        sub_files = [source_file_objs[i] for i in S]
        result = judge.query(target_desc, sub_files)
        if result is None:
            n_failed += 1
            consecutive += 1
            print(f"  Query {q_idx+1:>3d}/{n_queries}: API FAILED", flush=True)
            if (abort_after_consecutive_failures > 0
                    and consecutive >= abort_after_consecutive_failures):
                print(f"\n  CIRCUIT BREAKER: {consecutive} consecutive failures",
                      flush=True)
                return responses
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue
        consecutive = 0
        chosen_local, reasoning = result
        C = sorted([S[i] for i in chosen_local])
        responses.append((S, C, reasoning))
        print(_format_query_log(q_idx, n_queries, sz, S, C, reasoning),
              flush=True)
        if sleep_between > 0:
            time.sleep(sleep_between)
    if n_failed > 0:
        print(f"\n  {n_failed}/{n_queries} queries failed; saved {len(responses)}",
              flush=True)
    return responses


def _collect_parallel(judge, source_file_objs, target_desc, plans,
                       n_queries, n_parallel):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(q_idx, sz, S):
        sub_files = [source_file_objs[i] for i in S]
        return q_idx, sz, S, judge.query(target_desc, sub_files)

    by_qidx, n_failed = {}, 0
    print(f"  ({n_queries} queries on {n_parallel} parallel workers)",
          flush=True)
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futures = [ex.submit(_one, *p) for p in plans]
        for fut in as_completed(futures):
            q_idx, sz, S, result = fut.result()
            if result is None:
                n_failed += 1
                print(f"  Query {q_idx+1:>3d}/{n_queries}: FAILED", flush=True)
                continue
            chosen_local, reasoning = result
            C = sorted([S[i] for i in chosen_local])
            by_qidx[q_idx] = (S, C, reasoning)
            print(_format_query_log(q_idx, n_queries, sz, S, C, reasoning),
                  flush=True)

    responses = [by_qidx[q] for q in sorted(by_qidx)]
    if n_failed > 0:
        print(f"\n  {n_failed}/{n_queries} failed; saved {len(responses)}",
              flush=True)
    return responses


# ============================================================================
# Save / Load JSON
# ============================================================================

def save_responses(path: str, target_desc: str,
                   responses: Sequence,
                   source_basenames: Sequence[str] | None = None,
                   meta: dict | None = None) -> None:
    """Save responses to JSON. Accepts both (S, C) and (S, C, reasoning) tuples.

    Layout:
        {
          "target_desc": "...",
          "responses": [{"S": [...], "C": [...], "reasoning": "..."}, ...],
          "source_basenames": [...],   # optional
          "meta": {...}                # optional
        }
    """
    rec_list = []
    for r in responses:
        if len(r) == 3:
            s, c, reasoning = r
            rec = {"S": list(s), "C": list(c), "reasoning": str(reasoning)}
        else:
            s, c = r
            rec = {"S": list(s), "C": list(c)}
        rec_list.append(rec)
    payload = {"target_desc": target_desc, "responses": rec_list}
    if source_basenames is not None:
        payload["source_basenames"] = list(source_basenames)
    if meta is not None:
        payload["meta"] = dict(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_responses(path: str,
                   target_basename: str | None = None
                   ) -> list[tuple[list[int], list[int]]]:
    """Load (S, C) tuples from JSON, ignoring the optional `reasoning` field.

    If `target_basename` is given and the JSON includes `source_basenames`,
    drop subgroups containing that target and re-index the remaining indices
    to the K-1 source set. (This is how a single global LIP collection can
    serve multiple per-target experiments.)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    responses = [(list(r["S"]), list(r["C"])) for r in data["responses"]]
    if target_basename is not None:
        basenames = data.get("source_basenames")
        if basenames is None:
            raise ValueError(
                f"target_basename given but {path} has no `source_basenames`")
        try:
            target_idx = list(basenames).index(target_basename)
        except ValueError:
            raise ValueError(
                f"target {target_basename!r} not found in `source_basenames`")
        responses = _filter_for_target(responses, target_idx)
    return responses


def _filter_for_target(responses, target_idx) -> list[tuple[list[int], list[int]]]:
    """Drop subgroups containing `target_idx`, re-index the rest."""
    out = []
    for S, C in responses:
        if target_idx in S:
            continue
        S_new = [(s if s < target_idx else s - 1) for s in S]
        C_new = [(c if c < target_idx else c - 1) for c in C]
        out.append((sorted(S_new), sorted(C_new)))
    return out


# ============================================================================
# Likelihood: paper eq:lip_choice (conditional logit with outside option)
# ============================================================================

def _lse(alpha: torch.Tensor, alpha_0: torch.Tensor,
         indices: Sequence[int]) -> torch.Tensor:
    """log(exp(alpha_0) + sum_{i in indices} exp(alpha[i]))."""
    if len(indices) == 0:
        return alpha_0.squeeze()
    idx = torch.as_tensor(list(indices), dtype=torch.long, device=alpha.device)
    vals = torch.cat([alpha_0, alpha[idx]])
    return torch.logsumexp(vals, dim=0)


def log_p_response(alpha: torch.Tensor, alpha_0: torch.Tensor,
                   S: Sequence[int], C: Sequence[int]) -> torch.Tensor:
    """Paper eq:lip_choice — conditional logit with outside option.

        P(k | S) = exp(α_k) / (exp(α_0) + Σ_{j ∈ S} exp(α_j))   [k ∈ S]
        P(0 | S) = exp(α_0) / (exp(α_0) + Σ_{j ∈ S} exp(α_j))   [k = 0, null]

    Requires |C| ≤ 1. For multi-pick legacy data, decompose first via
    `multipick_to_singlepick`.
    """
    if len(C) > 1:
        raise ValueError(
            f"|C| <= 1 required by eq:lip_choice; got |C|={len(C)}. "
            f"Preprocess multi-pick data with `multipick_to_singlepick`.")
    Z = _lse(alpha, alpha_0, S)
    if len(C) == 0:
        return alpha_0.squeeze() - Z
    return alpha[int(C[0])] - Z


def multipick_to_singlepick(responses: Sequence[tuple[Sequence[int], Sequence[int]]]
                            ) -> list[tuple[list[int], list[int]]]:
    """Convert a multi-pick collection to single-pick votes.

    Each `(S, C)` with `|C| >= 1` is expanded into `|C|` records, one per
    chosen engine. Genuinely empty `(S, [])` records pass through.
    """
    out = []
    for S, C in responses:
        S_l = list(S)
        if not C:
            out.append((S_l, []))
        else:
            for c in C:
                out.append((S_l, [int(c)]))
    return out


# ============================================================================
# LIP fit (paper eq:LIP_opt)
# ============================================================================

def fit_lip(responses: Sequence[tuple[Sequence[int], Sequence[int]]],
            K: int,
            p0: float = 0.01,
            eps: float = 0.1,
            n_iter: int = 200,
            verbose: bool = False) -> np.ndarray:
    """Solve paper eq:LIP_opt. Returns pi = sigmoid(alpha) of shape (K,).

    Minimizes the regularized negative log-likelihood:

        -Σ_m log P(k_m | S_m)  +  eps Σ_k (α_k - log(p_0/(1-p_0)))²

    where the per-query likelihood is paper eq:lip_choice (conditional logit
    with outside option). The L2 anchor pulls each α_k toward log(p_0/(1-p_0))
    so that, with no LLM data, pi defaults to a uniform p_0 prior. α_0 (the
    null option's worth) is left unregularized.

    `eps=0.1` was tuned for K=99, ~200 single-choice queries on the C-MAPSS
    target description. For other dataset sizes the "right" eps scales
    roughly with `n_queries / K` (votes per source).
    """
    anchor = math.log(p0 / (1.0 - p0))
    alpha = torch.full((K,), anchor, dtype=torch.float64, requires_grad=True)
    alpha_0 = torch.tensor([anchor], dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.LBFGS([alpha, alpha_0], lr=0.5,
                                  max_iter=n_iter, history_size=50,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        nll = torch.zeros((), dtype=torch.float64)
        for S, C in responses:
            nll = nll - log_p_response(alpha, alpha_0, S, C)
        reg = eps * ((alpha - anchor) ** 2).sum()
        loss = nll + reg
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)

    if verbose:
        with torch.no_grad():
            pi = torch.sigmoid(alpha)
            print(f"[lip] final loss = {float(final_loss):.4f}")
            print(f"[lip] alpha_0   = {float(alpha_0):+.4f}")
            print(f"[lip] pi max = {pi.max():.3f}, mean = {pi.mean():.3f}")

    return torch.sigmoid(alpha).detach().numpy()
