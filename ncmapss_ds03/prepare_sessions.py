"""Prepare a blind DS03 source-relevance session for one flight class.

Writes two directories:

* `--sessions/claude_agent/`: the blind working directory given to the LLMs
  (15 anonymous `source_XX.csv` engine histories, the target description,
  the 30 required subgroup queries, task/schema files, a validator and
  `run_claude.sh` for the Claude Code agent session). `query_gemini.py` and
  `query_claude.py` read their inputs from here.
* `--scoring/`: the hidden anonymous-file -> engine mapping (used only to fit
  fold-specific priors and to score recovery), the queries, and the protocol.

Source aliases and subgroups depend only on `--seed`, so all three flight
classes share them; only the target description differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np

import common


QUERIES = 30
SEED = 42

TARGET_REGIMES = {
    "short": "short-haul",
    "medium": "medium-haul",
    "long": "long-haul",
}

TASK_TEMPLATE = """# Blind DS03 machine-source relevance task

Act as an aerospace prognostics coding agent. The dataset-provided target
description is:

> The target aircraft belongs to the {haul_label} flight class.

The target has too little degradation history to fit a reliable
remaining-useful-life model. Determine which of the 15 anonymous complete
machine histories would be most informative to borrow. Source filename
numbers are random, and no source flight-class labels are supplied.

Requirements:

1. Inspect every `source_*.csv`.
2. Decide what evidence in the supplied histories bears on relevance. Write
   any diagnostic code only under `work/`.
3. Rank all 15 sources from most to least informative for the described target.
4. Answer every query in `required_subgroups.json`. Select exactly one best
   source when a credible relevant candidate is present; select `none` when
   the subgroup contains no credible relevant source.
5. Do not use the number of recorded flights as a proxy for short/medium/long
   duration.
6. Write `results.json` following `RESULT_SCHEMA.md`, write a technical
   `analysis_report.md`, and run `python3 validate_result.py results.json`.

Blind restrictions: use only files inside this directory. Do not inspect
parent/sibling directories, mappings, prior outputs, or the internet. Do not
infer meaning from anonymous filename numbers.
"""

DATA_README = """# Data

Each `source_XX.csv` is one complete N-CMAPSS DS03 engine trajectory with one
row per completed flight.

- `cycle`: chronological flight index.
- `flight_duration_hours`: number of simulator samples in that flight divided
  by 3600.
- `adjusted_*`: condition-adjusted measured-sensor residuals. Each engine's
  first five flights define its healthy operating-response baseline.

The source filenames are randomly assigned. Flight-class labels, original
machine IDs, split tags, and terminal RUL labels are deliberately absent.
"""

RESULT_SCHEMA = """# Required result schema

Write `results.json` as:

```json
{
  "ranked_sources": ["source_XX.csv", "... all 15 exactly once ..."],
  "relevant_sources": ["source_XX.csv"],
  "reasoning": "Concise quantitative justification.",
  "confidence": "high, medium, or low",
  "subgroup_choices": [
    {
      "query_id": 0,
      "candidates": ["source_XX.csv", "source_YY.csv"],
      "choice": "source_XX.csv",
      "reasoning": "One concise reason."
    }
  ]
}
```

There must be one `subgroup_choices` entry for every query, in query order.
Each `candidates` array must exactly match the corresponding required query.
`choice` must be one listed candidate or the string `none`.
"""

RULES = """# Blind coding-agent session

Read `TASK.md`, `DATA_README.md`, and `RESULT_SCHEMA.md` before acting.
Complete the investigation autonomously.

Never read outside this directory, inspect parent or sibling directories, use
internet access, or infer meaning from anonymous filename numbers. Create
scripts and intermediate artifacts only under `work/`. The only permitted
top-level outputs are `results.json` and `analysis_report.md`.
"""

VALIDATOR = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.json")
payload = json.loads(path.read_text())
required = json.loads(Path("required_subgroups.json").read_text())["queries"]
sources = sorted(p.name for p in Path(".").glob("source_*.csv"))

assert sorted(payload["ranked_sources"]) == sources
assert len(payload["ranked_sources"]) == len(sources)
assert set(payload["relevant_sources"]) <= set(sources)
assert payload["confidence"] in {"high", "medium", "low"}
assert str(payload["reasoning"]).strip()
choices = payload["subgroup_choices"]
assert len(choices) == len(required)
for position, (answer, query) in enumerate(zip(choices, required)):
    assert answer["query_id"] == query["query_id"] == position
    assert answer["candidates"] == query["candidates"]
    assert answer["choice"] in query["candidates"] + ["none"]
    assert str(answer["reasoning"]).strip()
print("validation passed")
'''

CLAUDE_RUNNER = """#!/usr/bin/env bash
set -euo pipefail
session_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model="${LIP_CLAUDE_MODEL:-claude-opus-4-7}"
cd "$session_dir"
exec claude \\
    --print \\
    --model "$model" \\
    --effort max \\
    --permission-mode bypassPermissions \\
    --disallowed-tools "WebSearch,WebFetch" \\
    --no-session-persistence \\
    "Read TASK.md and complete the blind investigation autonomously. Obey CLAUDE.md, write results.json and analysis_report.md, and run the validator."
"""


def load_rows(path: Path) -> OrderedDict[str, list[dict[str, str]]]:
    engines: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            engines.setdefault(row["engine_id"], []).append(row)
    return engines


def write_source(path: Path, rows: list[dict[str, str]]) -> None:
    sensor_columns = [
        name for name in rows[0] if name.startswith("adjusted_")
    ]
    fields = ["cycle", "flight_duration_hours", *sensor_columns]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "cycle": row["cycle"],
                    "flight_duration_hours": (
                        float(row["observations"]) / 3600.0
                    ),
                    **{name: row[name] for name in sensor_columns},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=common.DEFAULT_FEATURES
    )
    parser.add_argument(
        "--target-class", choices=common.FLIGHT_CLASSES, required=True
    )
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    haul_label = TARGET_REGIMES[arguments.target_class]

    engines = load_rows(arguments.features)
    if len(engines) != 15:
        raise RuntimeError(f"expected 15 DS03 engines, found {len(engines)}")
    if arguments.queries < 1:
        raise ValueError("--queries must be positive")
    directory = arguments.sessions / "claude_agent"
    common.prepare_output(directory)
    common.prepare_output(arguments.scoring)

    engine_ids = list(engines)
    aliases = [f"source_{index:02d}.csv" for index in range(1, 16)]
    random.Random(arguments.seed).shuffle(aliases)
    engine_to_alias = dict(zip(engine_ids, aliases))

    rng = np.random.default_rng(arguments.seed)
    queries = []
    for query_id in range(arguments.queries):
        size = int(rng.integers(3, 6))
        selected = rng.choice(aliases, size=size, replace=False).tolist()
        queries.append({"query_id": query_id, "candidates": selected})

    mapping = [
        {
            "source_file": engine_to_alias[engine_id],
            "engine_id": engine_id,
            "flight_class_scoring_only": rows[0]["flight_class_name"],
        }
        for engine_id, rows in engines.items()
    ]
    subgroups = json.dumps(
        {"seed": arguments.seed, "queries": queries}, indent=2
    )
    (arguments.scoring / "source_mapping_scoring_only.json").write_text(
        json.dumps(mapping, indent=2)
    )
    (arguments.scoring / "required_subgroups.json").write_text(subgroups)
    (arguments.scoring / "protocol.json").write_text(
        json.dumps(
            {
                "target_class": arguments.target_class,
                "target_description": (
                    f"The target aircraft belongs to the {haul_label} "
                    "flight class."
                ),
                "query_count": arguments.queries,
                "random_seed": arguments.seed,
                "query_seed": arguments.seed,
                "queries_supplied_directly": True,
                "source_labels_shown_to_agent": False,
            },
            indent=2,
        )
    )

    (directory / "work").mkdir()
    for engine_id, rows in engines.items():
        write_source(directory / engine_to_alias[engine_id], rows)
    (directory / "TASK.md").write_text(
        TASK_TEMPLATE.format(haul_label=haul_label)
    )
    (directory / "TARGET_DESCRIPTION.txt").write_text(
        f"The target aircraft belongs to the {haul_label} flight class.\n"
    )
    (directory / "DATA_README.md").write_text(DATA_README)
    (directory / "RESULT_SCHEMA.md").write_text(RESULT_SCHEMA)
    (directory / "required_subgroups.json").write_text(subgroups)
    (directory / "CLAUDE.md").write_text(RULES)
    (directory / "validate_result.py").write_text(VALIDATOR)
    runner = directory / "run_claude.sh"
    runner.write_text(CLAUDE_RUNNER)
    runner.chmod(0o755)
    print(
        json.dumps(
            {
                "session": str(directory),
                "sources": len(engines),
                "queries": len(queries),
                "target_class": arguments.target_class,
            }
        )
    )


if __name__ == "__main__":
    main()
