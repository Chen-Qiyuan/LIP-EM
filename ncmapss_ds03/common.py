"""Shared data, spline, health-index, and metric code for N-CMAPSS DS03."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_FEATURES = HERE / "data" / "cycle_features.csv"
FLIGHT_CLASSES = ("short", "medium", "long")
PROGNOSTIC_FRACTIONS = tuple(index / 10.0 for index in range(1, 10))
BASELINE_FLIGHTS = 5
LOG_OFFSET = 0.05
PATH_X_MAX = 100.0
PATH_KNOTS = tuple(np.linspace(0.0, PATH_X_MAX, 7)[1:-1])
PATH_MODEL = "natural_spline"
UNIFORM_PRIOR_VALUE = 0.01
METHODS = (
    "target_only",
    "global_pool",
    "uniform_em",
    "llm_lip_em",
)
FUSION_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path: Path) -> None:
    """Create an output directory, refusing to overwrite earlier results."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to reuse nonempty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def load_engines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    engines: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        sensor_columns = [
            column
            for column in reader.fieldnames or []
            if column.startswith("adjusted_")
        ]
        for row in reader:
            engine = engines.setdefault(
                row["engine_id"],
                {
                    "domain_id": row["engine_id"],
                    "flight_class_name": row["flight_class_name"],
                    "flight_class": int(row["flight_class"]),
                    "split": row["split"],
                    "unit": int(row["unit"]),
                    "cycle": [],
                    "rul": [],
                    "observations": [],
                    "sensors": [],
                },
            )
            engine["cycle"].append(float(row["cycle"]))
            engine["rul"].append(float(row["rul"]))
            engine["observations"].append(float(row["observations"]))
            engine["sensors"].append(
                [float(row[column]) for column in sensor_columns]
            )
    result = list(engines.values())
    for engine in result:
        engine["cycle"] = np.asarray(engine["cycle"])
        engine["rul"] = np.asarray(engine["rul"])
        engine["observations"] = np.asarray(engine["observations"])
        engine["sensors"] = np.asarray(engine["sensors"])
    return result, sensor_columns


def natural_spline_basis(cycles: np.ndarray) -> np.ndarray:
    """Five-dimensional natural-cubic spline on unnormalized raw cycles."""
    knots = np.asarray(PATH_KNOTS, dtype=float)
    boundary = knots[-1]

    def truncated(index: int) -> np.ndarray:
        denominator = max(boundary - knots[index], 1e-12)
        return (
            np.maximum(cycles - knots[index], 0.0) ** 3
            - np.maximum(cycles - boundary, 0.0) ** 3
        ) / denominator

    columns = [np.ones_like(cycles), cycles]
    last = truncated(len(knots) - 2)
    for index in range(len(knots) - 2):
        columns.append(truncated(index) - last)
    return np.column_stack(columns)


def fold_copy(engines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow per-fold copies so fold-specific health paths never leak."""
    return [dict(engine) for engine in engines]


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one."""
    ordered = np.sort(values)[::-1]
    partial = np.cumsum(ordered)
    candidates = np.flatnonzero(
        ordered * np.arange(1, len(ordered) + 1) > partial - 1.0
    )
    if not len(candidates):
        return np.full_like(values, 1.0 / len(values))
    index = int(candidates[-1])
    threshold = (partial[index] - 1.0) / (index + 1)
    return np.maximum(values - threshold, 0.0)


def fusion_terms(
    development: list[dict[str, Any]],
    magnitudes: dict[str, np.ndarray],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    count = len(development)
    monotonicity: list[tuple[np.ndarray, np.ndarray]] = []
    for engine in development:
        values = magnitudes[engine["domain_id"]]
        differences = values[:-1] - values[1:]
        epoch_weights = np.arange(1, len(values), dtype=float)
        epoch_weights /= epoch_weights.sum() * count
        monotonicity.append((differences, epoch_weights))

    terminal = np.vstack(
        [magnitudes[engine["domain_id"]][-1] for engine in development]
    )
    centering = np.eye(count) - np.ones((count, count)) / count
    terminal_quadratic = (
        terminal.T @ centering @ terminal / max(count - 1, 1)
    )
    return monotonicity, terminal_quadratic


def fusion_objective_gradient(
    weights: np.ndarray,
    tradeoff: float,
    monotonicity: list[tuple[np.ndarray, np.ndarray]],
    terminal_quadratic: np.ndarray,
    smoothing: float = 1e-4,
) -> tuple[float, np.ndarray]:
    objective = 0.0
    gradient = np.zeros_like(weights)
    for differences, epoch_weights in monotonicity:
        signed_steps = differences @ weights
        root = np.sqrt(signed_steps**2 + smoothing**2)
        hinge = 0.5 * (signed_steps + root)
        derivative = 0.5 * (1.0 + signed_steps / root)
        objective += (1.0 - tradeoff) * float(epoch_weights @ hinge)
        gradient += (1.0 - tradeoff) * (
            differences.T @ (epoch_weights * derivative)
        )
    objective += tradeoff * float(
        weights @ terminal_quadratic @ weights
    )
    gradient += 2.0 * tradeoff * terminal_quadratic @ weights
    return objective, gradient


def optimize_fusion_weights(
    sensor_count: int,
    tradeoff: float,
    monotonicity: list[tuple[np.ndarray, np.ndarray]],
    terminal_quadratic: np.ndarray,
    maximum_iterations: int = 20_000,
) -> tuple[np.ndarray, int, float]:
    weights = np.full(sensor_count, 1.0 / sensor_count)
    objective = math.inf
    for iteration in range(maximum_iterations):
        objective, gradient = fusion_objective_gradient(
            weights,
            tradeoff,
            monotonicity,
            terminal_quadratic,
        )
        step = 1.0
        while step >= 1e-12:
            candidate = project_simplex(weights - step * gradient)
            candidate_objective, _ = fusion_objective_gradient(
                candidate,
                tradeoff,
                monotonicity,
                terminal_quadratic,
            )
            sufficient_decrease = objective - 1e-4 * float(
                gradient @ (weights - candidate)
            )
            if candidate_objective <= sufficient_decrease:
                break
            step *= 0.5
        if np.max(np.abs(candidate - weights)) < 1e-10:
            return candidate, iteration + 1, candidate_objective
        weights = candidate
    return weights, maximum_iterations, objective


def fusion_metrics(
    weights: np.ndarray,
    development: list[dict[str, Any]],
    magnitudes: dict[str, np.ndarray],
    monotonicity: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    violation = sum(
        float(epoch_weights @ np.maximum(differences @ weights, 0.0))
        for differences, epoch_weights in monotonicity
    )
    terminal = np.asarray(
        [
            float(magnitudes[engine["domain_id"]][-1] @ weights)
            for engine in development
        ]
    )
    correlations = []
    for engine in development:
        health = magnitudes[engine["domain_id"]] @ weights
        correlations.append(
            float(np.corrcoef(np.arange(len(health)), health)[0, 1])
        )
    return {
        "weighted_monotonicity_violation": violation,
        "terminal_mean": float(terminal.mean()),
        "terminal_variance": float(terminal.var(ddof=1)),
        "terminal_cv": float(terminal.std(ddof=1) / terminal.mean()),
        "mean_time_correlation": float(np.mean(correlations)),
        "minimum_time_correlation": float(np.min(correlations)),
    }


def select_fusion_weights(
    sensor_count: int,
    development: list[dict[str, Any]],
    magnitudes: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Select the source-only efficient-frontier point nearest the utopia."""
    monotonicity, terminal_quadratic = fusion_terms(
        development, magnitudes
    )
    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
    for tradeoff in FUSION_GRID:
        weights, iterations, objective = optimize_fusion_weights(
            sensor_count,
            tradeoff,
            monotonicity,
            terminal_quadratic,
        )
        row: dict[str, Any] = {
            "tradeoff": tradeoff,
            "optimizer_iterations": iterations,
            "smoothed_objective": objective,
            **fusion_metrics(
                weights, development, magnitudes, monotonicity
            ),
        }
        candidates.append((weights, row))

    violations = np.asarray(
        [row["weighted_monotonicity_violation"] for _, row in candidates]
    )
    variances = np.asarray(
        [row["terminal_variance"] for _, row in candidates]
    )

    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        if span <= 0:
            return np.zeros_like(values)
        return (values - values.min()) / span

    distances = np.sqrt(normalize(violations) ** 2 + normalize(variances) ** 2)
    selected_index = int(distances.argmin())
    sensitivity = [
        {
            **row,
            "normalized_utopia_distance": float(distances[index]),
            "selected": int(index == selected_index),
        }
        for index, (_, row) in enumerate(candidates)
    ]
    return (
        candidates[selected_index][0],
        float(candidates[selected_index][1]["tradeoff"]),
        sensitivity,
    )


def add_health_paths(
    engines: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    sensor_count: int,
) -> dict[str, Any]:
    """Attach the source-only composite health index and spline design.

    Sensor scales and fusion weights are estimated from the source engines
    only (Liu, Gebraeel and Shi, 2013, data-level fusion).
    """
    source_rows = np.vstack([engine["sensors"] for engine in sources])
    scale = source_rows.std(axis=0)
    scale[scale < 1e-8] = 1.0
    magnitudes = {
        engine["domain_id"]: np.abs(engine["sensors"] / scale)
        for engine in engines
    }
    weights, tradeoff, sensitivity = select_fusion_weights(
        sensor_count, sources, magnitudes
    )
    for engine in engines:
        health = magnitudes[engine["domain_id"]] @ weights
        engine["health"] = health
        engine["path_design"] = natural_spline_basis(engine["cycle"])
        engine["path_outcome"] = np.log(health + LOG_OFFSET)
        engine["path_sensor_outcome"] = engine["sensors"] / scale
    return {
        "source_only": True,
        "sensor_scale": scale.tolist(),
        "fusion_weights": weights.tolist(),
        "fusion_tradeoff": tradeoff,
        "fusion_sensitivity": sensitivity,
        "path_model": PATH_MODEL,
    }


def matching(engine: dict[str, Any], target_class: str) -> bool:
    return engine["flight_class_name"] == target_class


def append_weights(
    rows: list[dict[str, Any]],
    cutoff: float,
    target: dict[str, Any],
    sources: list[dict[str, Any]],
    weights: dict[str, np.ndarray],
    iterations: dict[str, int],
    llm_prior: np.ndarray,
) -> None:
    for method, values in weights.items():
        for source_index, (source, value) in enumerate(zip(sources, values)):
            rows.append(
                {
                    "task": "prognostics",
                    "cutoff": cutoff,
                    "target_engine": target["domain_id"],
                    "source_engine": source["domain_id"],
                    "source_class_scoring_only": source[
                        "flight_class_name"
                    ],
                    "matching_source_scoring_only": int(
                        matching(source, target["flight_class_name"])
                    ),
                    "method": method,
                    "initial_prior": (
                        UNIFORM_PRIOR_VALUE
                        if method == "uniform_em"
                        else float(llm_prior[source_index])
                    ),
                    "final_weight": float(value),
                    "iterations": iterations[method],
                }
            )


def prefix_length(engine: dict[str, Any], fraction: float) -> int:
    return max(
        BASELINE_FLIGHTS + 2,
        min(
            len(engine["cycle"]),
            int(math.ceil(fraction * len(engine["cycle"]))),
        ),
    )


def nasa_score(error: float) -> float:
    if error >= 0:
        return math.exp(error / 10.0) - 1.0
    return math.exp(-error / 13.0) - 1.0


def summarize_predictions(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fraction in PROGNOSTIC_FRACTIONS:
        result[str(fraction)] = {}
        for method in METHODS:
            local = [
                row
                for row in rows
                if row["observed_fraction"] == fraction
                and row["method"] == method
            ]
            targets = sorted({row["target_engine"] for row in local})
            engine_rmse = []
            for target in targets:
                engine_rows = [
                    row for row in local if row["target_engine"] == target
                ]
                engine_rmse.append(
                    math.sqrt(
                        float(
                            np.mean(
                                [row["squared_error"] for row in engine_rows]
                            )
                        )
                    )
                )
            result[str(fraction)][method] = {
                "mean_engine_rmse": float(np.mean(engine_rmse)),
                "sd_engine_rmse": float(np.std(engine_rmse, ddof=1)),
                "pooled_rmse": math.sqrt(
                    float(np.mean([row["squared_error"] for row in local]))
                ),
                "target_engines": len(targets),
                "predictions": len(local),
                "rul_cap_hit_rate": float(
                    np.mean([row["rul_cap_hit"] for row in local])
                ),
            }
    return result


def pairwise_auc(positive: list[float], negative: list[float]) -> float:
    comparisons = [
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positive
        for right in negative
    ]
    return float(np.mean(comparisons))


def summarize_selection(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"prognostics": {}}
    for cutoff in PROGNOSTIC_FRACTIONS:
        result["prognostics"][str(cutoff)] = {}
        for method in ("uniform_em", "llm_lip_em"):
            local = [
                row
                for row in rows
                if row["cutoff"] == cutoff and row["method"] == method
            ]
            relevant = [
                row["final_weight"]
                for row in local
                if row["matching_source_scoring_only"]
            ]
            irrelevant = [
                row["final_weight"]
                for row in local
                if not row["matching_source_scoring_only"]
            ]
            result["prognostics"][str(cutoff)][method] = {
                "matching_mean_weight": float(np.mean(relevant)),
                "nonmatching_mean_weight": float(np.mean(irrelevant)),
                "source_classification_auc": pairwise_auc(
                    relevant, irrelevant
                ),
                "matching_selected_above_half": int(
                    np.sum(np.asarray(relevant) > 0.5)
                ),
                "matching_opportunities": len(relevant),
                "nonmatching_selected_above_half": int(
                    np.sum(np.asarray(irrelevant) > 0.5)
                ),
                "nonmatching_opportunities": len(irrelevant),
            }
    return result
