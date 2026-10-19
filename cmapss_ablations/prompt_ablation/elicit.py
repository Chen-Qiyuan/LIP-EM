"""Prepare and collect a blind Claude Code session for a new target description.

This is how the response files under responses/ were produced; it is only
needed to elicit a new description. `prepare` writes a self-contained session
folder: the 100 FD001 trajectories under anonymous names (source_001.csv, ...,
in the source order of the response files), the NASA report, the 200 subgroups
of the cached LIP-C and the task files. Run Claude Code inside that folder
(we used Claude Opus 4.7 at maximum effort, one fresh session per
description). `collect` then converts the session's results.json into a
response file that run.py can use (add it to conditions.py).

Usage:
    python elicit.py prepare --description "This aircraft is ..." --session sessions/mine
    (cd sessions/mine && claude)      # the agent writes results.json
    python elicit.py collect --session sessions/mine --model claude-opus-4-7 \\
        --out responses/opus/mine.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from conditions import CMAPSS_DATA


REFERENCE = CMAPSS_DATA / "llm_subgroup_queries_claude.json"
SEQ_DIR = CMAPSS_DATA / "Physical_Core_Speed_Sequences"
REPORT = CMAPSS_DATA / "Damage_Propagation_Modeling.pdf"

CLAUDE_MD = """
# Blind aerospace-prognostics session

Read `TASK.md` and `RESULT_SCHEMA.md` completely before acting. Complete the
investigation autonomously.

Never read outside this directory, inspect parent or sibling directories, use
the internet, or infer meaning from anonymous filename or query numbers. Write
scripts and intermediate artifacts only under `work/`. The only permitted
top-level outputs are `results.json` and `analysis_report.md`.
"""

TASK_MD = """
# Blind C-MAPSS source-relevance task

Act as an expert aerospace prognostics coding agent. Read
`Damage_Propagation_Modeling.pdf` for the C-MAPSS damage-propagation context.
Each anonymous `source_*.csv` contains one complete sensor-9 trajectory,
formatted as flight cycle and high-pressure-compressor physical core speed.

The only auxiliary information about the target is:

> {description}

For every subgroup in `required_subgroups.json`, select either:

- the single source whose physical trajectory best matches the trajectory
  implied by the target description; or
- `none` when no source in that subgroup is a credible match.

Requirements:

1. Inspect and quantitatively characterize all 100 anonymous trajectories.
2. You may write diagnostic code only under `work/`.
3. Answer all 200 subgroups in their given order. Never select more than one
   source for a subgroup.
4. Base every choice only on the supplied target description, report, and
   trajectory data. Filename and query numbers carry no information.
5. Do not inspect files outside this directory and do not use the internet.
6. Write `results.json` exactly as specified by `RESULT_SCHEMA.md`, write a
   concise `analysis_report.md`, and run
   `python3 validate_result.py results.json`.
"""

RESULT_SCHEMA_MD = """
# Required result schema

Write `results.json` as:

```json
{
  "subgroup_choices": [
    {
      "query_id": 0,
      "candidates": ["source_001.csv", "source_002.csv"],
      "choice": "source_001.csv",
      "reasoning": "One concise trajectory-based reason."
    }
  ]
}
```

There must be exactly one entry for every required query, in query order. Each
`candidates` array must exactly match the corresponding required query.
`choice` must be one listed candidate or the string `none`. Do not add global
rankings, relevance sets, confidence fields, or alternative choices.
"""

VALIDATE_PY = """
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.json")
payload = json.loads(path.read_text())
assert set(payload) == {"subgroup_choices"}
required = json.loads(Path("required_subgroups.json").read_text())["queries"]
choices = payload["subgroup_choices"]
assert len(choices) == len(required) == 200
for position, (answer, query) in enumerate(zip(choices, required)):
    assert set(answer) == {"query_id", "candidates", "choice", "reasoning"}
    assert answer["query_id"] == query["query_id"] == position
    assert answer["candidates"] == query["candidates"]
    assert answer["choice"] in query["candidates"] + ["none"]
    assert str(answer["reasoning"]).strip()
print("validation passed")
"""


def source_name(index: int) -> str:
    return f"source_{index + 1:03d}.csv"


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def prepare(description: str, session: Path) -> None:
    reference = json.loads(REFERENCE.read_text())
    session.mkdir(parents=True, exist_ok=False)
    (session / "work").mkdir()
    for i, basename in enumerate(reference["source_basenames"]):
        shutil.copy2(SEQ_DIR / basename, session / source_name(i))
    shutil.copy2(REPORT, session / REPORT.name)
    queries = [{"query_id": q, "candidates": [source_name(i) for i in r["S"]]}
               for q, r in enumerate(reference["responses"])]
    write_text(session / "required_subgroups.json",
               json.dumps({"queries": queries}, indent=2))
    write_text(session / "CLAUDE.md", CLAUDE_MD)
    write_text(session / "TASK.md", TASK_MD.format(description=description))
    write_text(session / "RESULT_SCHEMA.md", RESULT_SCHEMA_MD)
    write_text(session / "validate_result.py", VALIDATE_PY)
    write_text(session / "description.txt", description)
    print(f"Prepared {session}; run Claude Code inside it.")


def collect(session: Path, model: str, out: Path) -> None:
    reference = json.loads(REFERENCE.read_text())
    required = json.loads((session / "required_subgroups.json").read_text())["queries"]
    choices = json.loads((session / "results.json").read_text())["subgroup_choices"]
    assert len(choices) == len(required) == len(reference["responses"])
    responses = []
    for q, (answer, query) in enumerate(zip(choices, required)):
        assert answer["query_id"] == query["query_id"] == q
        assert answer["candidates"] == query["candidates"]
        S = [int(name[7:10]) - 1 for name in query["candidates"]]
        C = [] if answer["choice"] == "none" else [int(answer["choice"][7:10]) - 1]
        assert set(C) <= set(S)
        responses.append({"S": S, "C": C, "reasoning": str(answer["reasoning"])})
    payload = {
        "target_desc": (session / "description.txt").read_text().strip(),
        "responses": responses,
        "source_basenames": reference["source_basenames"],
        "meta": {"model": model, "protocol": "blind_agentic_single_choice",
                 "n_queries": len(responses), "subgroup_size_range": [3, 5],
                 "seed": 42, "condition": out.stem,
                 "reference_query_file": "cmapss/data/" + REFERENCE.name},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text(out, json.dumps(payload, indent=2))
    print(f"Wrote {out}: {sum(bool(r['C']) for r in responses)} selections, "
          f"{sum(not r['C'] for r in responses)} nulls")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--description", required=True,
                   help="the target description shown to the agent")
    p.add_argument("--session", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--session", type=Path, required=True)
    c.add_argument("--model", required=True, help="recorded in the meta field")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.description, args.session)
    else:
        collect(args.session, args.model, args.out)


if __name__ == "__main__":
    main()
