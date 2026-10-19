"""Fit fold-specific LIPs for one DS03 flight class from cached LLM choices.

For every target engine of the class, the subgroup queries that contain the
target are dropped, the remaining choices are mapped to that fold's 14
sources, and paper eq:LIP_opt (conditional logit with an outside option and
an L2 anchor at logit(p0)) is solved. The output JSON holds
`priors_by_target[target_engine][source_engine]`, which `prognostics.py`
reads.

Inputs (one directory per flight class, see `llm_responses/`):
  source_mapping_scoring_only.json   anonymous file name -> engine id, class
  <elicitation>/results.json         30 subgroup choices with reasoning
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import common


P0 = 0.01
EPSILON = 0.1
MAX_ITERATIONS = 200
ELICITATIONS = {
    "claude_agent": (
        "Claude Code (claude-opus-4-7, effort max): one blind coding-agent "
        "session that inspects all 15 anonymous sources and answers the 30 "
        "subgroup queries"
    ),
    "claude_independent": (
        "Claude Code (claude-opus-4-7, effort max): one independent request "
        "per subgroup query showing only that subgroup's sources"
    ),
    "gemini": (
        "gemini-3-flash-preview at temperature 0: one independent request "
        "per subgroup query showing only that subgroup's sources"
    ),
}


def fit_lip(
    observations: list[tuple[list[int], int | None]],
    source_count: int,
) -> list[float]:
    """Solve paper eq:LIP_opt; returns pi = sigmoid(alpha).

    Minimizes -sum_m log P(k_m | S_m) + eps sum_k (alpha_k - logit(p0))^2
    with the null utility alpha_0 unpenalized. This is the objective of
    `lip.lip.fit_lip`; here it is solved with scipy's L-BFGS-B, which
    reproduces the released priors exactly.
    """
    anchor = math.log(P0 / (1.0 - P0))
    start = np.full(source_count + 1, anchor, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = parameters[:source_count]
        alpha_null = parameters[source_count]
        total = EPSILON * float(np.sum((alpha - anchor) ** 2))
        gradient = np.zeros_like(parameters)
        gradient[:source_count] = 2.0 * EPSILON * (alpha - anchor)
        for subgroup, selected in observations:
            utilities = np.concatenate(([alpha_null], alpha[subgroup]))
            maximum = float(np.max(utilities))
            exponentials = np.exp(utilities - maximum)
            probabilities = exponentials / exponentials.sum()
            log_partition = maximum + math.log(float(exponentials.sum()))

            if selected is None:
                total -= alpha_null - log_partition
                gradient[source_count] -= 1.0
            else:
                total -= alpha[selected] - log_partition
                gradient[selected] -= 1.0

            gradient[source_count] += probabilities[0]
            for offset, source in enumerate(subgroup):
                gradient[source] += probabilities[offset + 1]
        return float(total), gradient

    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": MAX_ITERATIONS,
            "maxls": 50,
            "ftol": 1e-12,
            "gtol": 1e-9,
        },
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"LIP optimizer returned nonfinite values: {result}")
    if not result.success and result.status != 1:
        raise RuntimeError(f"LIP optimizer failed: {result.message}")
    return expit(result.x[:source_count]).tolist()


def fit_elicitation(
    elicitation: str,
    result_path: Path,
    mapping: list[dict[str, str]],
    target_class: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(result_path.read_text())
    alias_to_source = {row["source_file"]: row for row in mapping}
    choices = payload["subgroup_choices"]

    # Scoring only: was the choice a matching-class source (or `none` when
    # the subgroup held no matching source)?
    correct = []
    for choice in choices:
        relevant = {
            alias
            for alias in choice["candidates"]
            if alias_to_source[alias]["flight_class_scoring_only"]
            == target_class
        }
        correct.append(
            choice["choice"] in relevant if relevant
            else choice["choice"] == "none"
        )

    targets = [
        row for row in mapping
        if row["flight_class_scoring_only"] == target_class
    ]
    priors_by_target: dict[str, dict[str, float]] = {}
    folds: dict[str, Any] = {}
    for target in targets:
        sources = [
            row for row in mapping if row["engine_id"] != target["engine_id"]
        ]
        engine_to_index = {
            row["engine_id"]: index for index, row in enumerate(sources)
        }
        observations = []
        for choice in choices:
            if target["source_file"] in choice["candidates"]:
                continue
            subgroup = [
                engine_to_index[alias_to_source[alias]["engine_id"]]
                for alias in choice["candidates"]
            ]
            selected = (
                None
                if choice["choice"] == "none"
                else engine_to_index[
                    alias_to_source[choice["choice"]]["engine_id"]
                ]
            )
            observations.append((subgroup, selected))
        values = fit_lip(observations, len(sources))
        prior = {
            source["engine_id"]: float(value)
            for source, value in zip(sources, values)
        }
        priors_by_target[target["engine_id"]] = prior
        is_matching = {
            source["engine_id"]: (
                source["flight_class_scoring_only"] == target_class
            )
            for source in sources
        }
        matching = [prior[e] for e, match in is_matching.items() if match]
        nonmatching = [
            prior[e] for e, match in is_matching.items() if not match
        ]
        folds[target["engine_id"]] = {
            "queries_after_target_exclusion": len(observations),
            "matching_mean_prior": float(np.mean(matching)),
            "nonmatching_mean_prior": float(np.mean(nonmatching)),
            "source_classification_auc": common.pairwise_auc(
                matching, nonmatching
            ),
        }
    prior_payload = {
        "protocol": "blind subgroup conditional-logit LIP",
        "elicitation": ELICITATIONS[elicitation],
        "target_class": target_class,
        "p0": P0,
        "epsilon": EPSILON,
        "optimizer": "scipy L-BFGS-B",
        "optimizer_max_iterations": MAX_ITERATIONS,
        "queries_used": len(choices),
        "priors_by_target": priors_by_target,
    }
    summary = {
        "elicitation": elicitation,
        "queries": len(choices),
        "choice_accuracy_scoring_only": float(np.mean(correct)),
        "relevant_sources_reported": payload["relevant_sources"],
        "confidence": payload["confidence"],
        "folds": folds,
    }
    return prior_payload, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="llm_responses/<class> directory",
    )
    parser.add_argument(
        "--target-class", choices=common.FLIGHT_CLASSES, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    common.prepare_output(arguments.output)
    mapping = json.loads(
        (arguments.responses / "source_mapping_scoring_only.json").read_text()
    )
    combined = {}
    for elicitation in ELICITATIONS:
        result_path = arguments.responses / elicitation / "results.json"
        if not result_path.exists():
            continue
        priors, summary = fit_elicitation(
            elicitation, result_path, mapping, arguments.target_class
        )
        (arguments.output / f"{elicitation}_lip_priors.json").write_text(
            json.dumps(priors, indent=2)
        )
        combined[elicitation] = summary
    (arguments.output / "summary.json").write_text(
        json.dumps(combined, indent=2)
    )
    print(
        json.dumps(
            {
                name: {
                    "queries": summary["queries"],
                    "choice_accuracy_scoring_only": summary[
                        "choice_accuracy_scoring_only"
                    ],
                }
                for name, summary in combined.items()
            }
        )
    )


if __name__ == "__main__":
    main()
