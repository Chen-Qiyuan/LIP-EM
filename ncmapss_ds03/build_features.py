"""Build leakage-free, flight-level features for N-CMAPSS DS03.

DS03 contains one fault mechanism and three dataset-provided flight classes:
short-, medium-, and long-haul. Each individual engine is a source/target
domain downstream; its flight class is retained only as the target prompt
label and for evaluating whether the LIP recovers matching engines.

For each engine, the first five completed flights define a healthy,
operating-condition response surface for each measured sensor. Every flight
is then represented by the mean percentage residual from that response
surface while preserving DS03's flight-class metadata and official dev/test
engine split.

The raw file `N-CMAPSS_DS03-012.h5` (3.7 GB) is not shipped. Pass it with
`--data`, or set NCMAPSS_DATA_DIR to the directory that holds it. The output
of this script is shipped as `data/cycle_features.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import common


RAW_FILE = "N-CMAPSS_DS03-012.h5"
CLASS_NAMES = {1: "short", 2: "medium", 3: "long"}
BASELINE_FLIGHTS = common.BASELINE_FLIGHTS


def default_data_path() -> Path:
    directory = os.environ.get("NCMAPSS_DATA_DIR")
    if directory:
        return Path(directory) / RAW_FILE
    return common.HERE / "raw_data" / RAW_FILE


def decode(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ]


def contiguous_unit_slices(
    unit_column: np.ndarray,
) -> list[tuple[int, slice]]:
    starts = np.r_[0, np.flatnonzero(unit_column[1:] != unit_column[:-1]) + 1]
    ends = np.r_[starts[1:], len(unit_column)]
    return [
        (int(unit_column[start]), slice(int(start), int(end)))
        for start, end in zip(starts, ends)
    ]


def quadratic_operating_design(
    operating: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_scale: np.ndarray,
) -> np.ndarray:
    standardized = (operating - baseline_mean) / baseline_scale
    columns = [np.ones(len(standardized))]
    columns.extend(standardized[:, column] for column in range(4))
    columns.extend(standardized[:, column] ** 2 for column in range(4))
    columns.extend(
        standardized[:, first] * standardized[:, second]
        for first in range(4)
        for second in range(first + 1, 4)
    )
    return np.column_stack(columns)


def aggregate_unit(
    cycle: np.ndarray,
    operating: np.ndarray,
    sensors: np.ndarray,
    rul: np.ndarray,
    sensor_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    integer_cycle = cycle.astype(int)
    cycles = np.unique(integer_cycle)
    baseline_cycles = cycles[:BASELINE_FLIGHTS]
    baseline_mask = np.isin(integer_cycle, baseline_cycles)

    baseline_operating = operating[baseline_mask]
    baseline_sensors = sensors[baseline_mask]
    operating_mean = baseline_operating.mean(axis=0)
    operating_scale = baseline_operating.std(axis=0)
    operating_scale[operating_scale < 1e-8] = 1.0
    baseline_design = quadratic_operating_design(
        baseline_operating, operating_mean, operating_scale
    )
    healthy_coefficients = np.linalg.lstsq(
        baseline_design, baseline_sensors, rcond=None
    )[0]
    healthy_prediction = (
        quadratic_operating_design(operating, operating_mean, operating_scale)
        @ healthy_coefficients
    )
    scale = np.median(np.abs(baseline_sensors), axis=0)
    scale = np.maximum(scale, 1e-12)
    unique_cycle, cycle_inverse, cycle_counts = np.unique(
        integer_cycle, return_inverse=True, return_counts=True
    )
    residuals = 100.0 * (sensors - healthy_prediction) / scale
    adjusted = np.zeros((len(unique_cycle), sensors.shape[1]))
    for column in range(sensors.shape[1]):
        adjusted[:, column] = (
            np.bincount(cycle_inverse, weights=residuals[:, column])
            / cycle_counts
        )

    label_by_cycle = {}
    for local_cycle in unique_cycle:
        labels = np.unique(rul[integer_cycle == local_cycle])
        if len(labels) != 1:
            raise RuntimeError(
                f"RUL is not constant within cycle {local_cycle}: {labels}"
            )
        label_by_cycle[int(local_cycle)] = float(labels[0])
    terminal_cycle = int(cycles.max())
    label_error = max(
        abs(label_by_cycle[int(local_cycle)] - (terminal_cycle - local_cycle))
        for local_cycle in unique_cycle
    )
    if label_error > 1e-8:
        raise RuntimeError(f"unexpected RUL definition error {label_error}")

    rows = []
    for row_index, local_cycle in enumerate(unique_cycle):
        row: dict[str, Any] = {
            "cycle": int(local_cycle),
            "rul": label_by_cycle[int(local_cycle)],
            "terminal_cycle_scoring_only": terminal_cycle,
            "observations": int(cycle_counts[row_index]),
        }
        for column, sensor in enumerate(sensor_names):
            row[f"adjusted_{sensor}"] = float(adjusted[row_index, column])
        rows.append(row)
    diagnostics = {
        "terminal_cycle": terminal_cycle,
        "retained_cycles": len(rows),
        "raw_rows": len(cycle),
        "healthy_model_rows": int(baseline_mask.sum()),
        "healthy_model_columns": int(baseline_design.shape[1]),
    }
    return rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.data.exists():
        raise FileNotFoundError(
            f"{arguments.data} not found; pass --data or set NCMAPSS_DATA_DIR"
        )
    common.prepare_output(arguments.output)

    output_rows: list[dict] = []
    diagnostics: list[dict] = []
    with h5py.File(arguments.data, "r") as handle:
        sensor_names = decode(handle["X_s_var"])
        auxiliary_names = decode(handle["A_var"])
        if auxiliary_names[:3] != ["unit", "cycle", "Fc"]:
            raise RuntimeError(
                f"unexpected DS03 auxiliary schema: {auxiliary_names}"
            )

        for split in ("dev", "test"):
            auxiliary = handle[f"A_{split}"]
            unit_column = auxiliary[:, 0].astype(int)
            for unit, unit_slice in contiguous_unit_slices(unit_column):
                local_auxiliary = auxiliary[unit_slice, :3]
                flight_classes = np.unique(local_auxiliary[:, 2].astype(int))
                if len(flight_classes) != 1:
                    raise RuntimeError(
                        f"unit {unit} has flight classes {flight_classes}"
                    )
                flight_class = int(flight_classes[0])
                rows, local_diagnostics = aggregate_unit(
                    local_auxiliary[:, 1],
                    handle[f"W_{split}"][unit_slice],
                    handle[f"X_s_{split}"][unit_slice],
                    np.asarray(handle[f"Y_{split}"][unit_slice]).reshape(-1),
                    sensor_names,
                )
                engine_id = f"DS03-{split}-U{unit:02d}"
                for row in rows:
                    output_rows.append(
                        {
                            "engine_id": engine_id,
                            "flight_class_name": CLASS_NAMES[flight_class],
                            "flight_class": flight_class,
                            "split": split,
                            "unit": unit,
                            **row,
                        }
                    )
                diagnostics.append(
                    {
                        "engine_id": engine_id,
                        "flight_class_name": CLASS_NAMES[flight_class],
                        "flight_class": flight_class,
                        "split": split,
                        "unit": unit,
                        **local_diagnostics,
                    }
                )
                print(
                    f"{engine_id} [{CLASS_NAMES[flight_class]}]: "
                    f"{local_diagnostics['retained_cycles']} flights, "
                    f"{local_diagnostics['raw_rows']:,} raw rows",
                    flush=True,
                )

    if not output_rows:
        raise RuntimeError("DS03 extraction produced no flight-level rows")
    output_path = arguments.output / "cycle_features.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    unit_counts = Counter(
        (row["split"], row["flight_class_name"]) for row in diagnostics
    )
    metadata = {
        "dataset": RAW_FILE,
        "raw_file_bytes": arguments.data.stat().st_size,
        "source_domain_definition": "individual engine",
        "flight_class_role": "target prompt and recovery scoring only",
        "class_names": CLASS_NAMES,
        "baseline_flights": BASELINE_FLIGHTS,
        "operating_adjustment": (
            "per-engine quadratic response surface fit only on the first "
            "five completed flights"
        ),
        "measured_sensors": sensor_names,
        "engines": diagnostics,
        "unit_counts": {
            f"{split}_{domain}": count
            for (split, domain), count in sorted(unit_counts.items())
        },
        "flight_rows": len(output_rows),
        "leakage_checks": {
            "domain_is_machine": True,
            "normalized_life_used": False,
            "future_sensor_values_used_in_features": False,
            "rul_equals_terminal_cycle_minus_cycle": True,
        },
    }
    (arguments.output / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps({"output": output_path.name, "flight_rows": len(output_rows)}))


if __name__ == "__main__":
    main()
