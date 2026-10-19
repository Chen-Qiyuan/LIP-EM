"""Run independent blind DS03 subgroup queries with Claude Code.

Claude receives exactly the per-query prompt built for Gemini
(`query_gemini.build_query_prompt`): the target description, the neutral
data README, and only the three to five anonymous source histories in that
query. Each query runs in a fresh temporary working directory with no
conversational state from any other query. Requires the `claude` CLI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from query_gemini import (
    build_query_prompt,
    derived_source_summary,
    load_session,
    parse_json_text,
    prompt_sha256,
    validate_answer,
)


DEFAULT_MODEL = "claude-opus-4-7"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "query_id": {"type": "integer"},
        "choice": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["query_id", "choice", "reasoning"],
    "additionalProperties": False,
}


def extract_answer(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = parse_json_text(stdout)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and payload.get("type") == "result":
        metadata = {
            key: payload.get(key)
            for key in (
                "subtype",
                "duration_ms",
                "duration_api_ms",
                "num_turns",
                "usage",
                "modelUsage",
                "permission_denials",
            )
            if key in payload
        }
        answer = payload.get("structured_output", payload.get("result"))
        if isinstance(answer, str):
            answer = parse_json_text(answer)
    else:
        answer = payload
    if isinstance(answer, list) and len(answer) == 1 and isinstance(
        answer[0], dict
    ):
        answer = answer[0]
    if not isinstance(answer, dict):
        raise ValueError("Claude returned a non-object JSON value")
    return answer, metadata


def run_query(
    session: Path,
    output: Path,
    query: dict[str, Any],
    model: str,
    effort: str,
    timeout: float,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = output / f"query_{query['query_id']:03d}.json"
    if checkpoint.exists() and resume:
        payload = json.loads(checkpoint.read_text())
        if payload["model"] != model:
            raise ValueError(f"{checkpoint} was produced by a different model")
        answer = payload["answer"]
        validate_answer(answer, query)
        return answer, payload

    prompt = build_query_prompt(session, query)
    with tempfile.TemporaryDirectory(
        prefix=f"claude-query-{query['query_id']:03d}-"
    ) as temporary:
        work = Path(temporary)
        for filename in query["candidates"]:
            shutil.copy2(session / filename, work / filename)
        command = [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(ANSWER_SCHEMA, separators=(",", ":")),
            "--model",
            model,
            "--effort",
            effort,
            "--permission-mode",
            "bypassPermissions",
            "--disallowed-tools",
            "WebSearch,WebFetch",
            "--no-session-persistence",
            "--safe-mode",
            prompt,
        ]
        completed = subprocess.run(
            command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Claude query {query['query_id']} failed with exit code "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )
    answer, run_metadata = extract_answer(completed.stdout)
    validate_answer(answer, query)
    payload = {
        "model": model,
        "effort": effort,
        "query_id": query["query_id"],
        "answer": answer,
        "run_metadata": run_metadata,
        "prompt_characters": len(prompt),
        "prompt_sha256": prompt_sha256(prompt),
    }
    checkpoint.write_text(json.dumps(payload, indent=2))
    return answer, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="the claude_agent session written by prepare_sessions.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="max",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if (
        arguments.output.exists()
        and any(arguments.output.iterdir())
        and not arguments.resume
    ):
        raise FileExistsError(
            "refusing to reuse a nonempty output directory; choose a fresh "
            "--output or pass --resume explicitly"
        )
    queries, sources = load_session(arguments.session)
    arguments.output.mkdir(parents=True, exist_ok=True)

    records = [
        run_query(
            arguments.session,
            arguments.output,
            query,
            arguments.model,
            arguments.effort,
            arguments.timeout,
            arguments.resume,
        )
        for query in queries
    ]
    answers = [record[0] for record in records]
    checkpoints = [record[1] for record in records]
    ranking, relevant = derived_source_summary(sources, queries, answers)
    result = {
        "ranked_sources": ranking,
        "relevant_sources": relevant,
        "reasoning": (
            "Ranking is derived after collection from independent subgroup "
            "choices; it was never requested from or shown to Claude."
        ),
        "confidence": "medium",
        "subgroup_choices": [
            {
                "query_id": query["query_id"],
                "candidates": query["candidates"],
                "choice": answer["choice"],
                "reasoning": answer["reasoning"],
            }
            for query, answer in zip(queries, answers)
        ],
    }
    (arguments.output / "results.json").write_text(json.dumps(result, indent=2))
    metadata = {
        "model": arguments.model,
        "effort": arguments.effort,
        "protocol": "independent blind subgroup choice",
        "queries": len(queries),
        "source_labels_shown": False,
        "other_sources_shown": False,
        "other_queries_shown": False,
        "global_ranking_requested": False,
        "conversation_persisted": False,
        "prompt_sha256": [checkpoint["prompt_sha256"] for checkpoint in checkpoints],
        "run_metadata": [checkpoint["run_metadata"] for checkpoint in checkpoints],
    }
    (arguments.output / "query_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "model": arguments.model,
                "queries": len(queries),
            }
        )
    )


if __name__ == "__main__":
    main()
