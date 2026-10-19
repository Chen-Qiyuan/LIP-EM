"""Run independent blind DS03 subgroup queries with the Gemini API.

Each request contains the dataset-provided target description, the neutral
data README, and only the three to five anonymous source histories of that
query. Hidden source labels, mappings, RUL results, other queries, and other
models' outputs are never included.

The API key is read from the GEMINI_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODEL = "gemini-3-flash-preview"


def api_request(
    url: str,
    key: str,
    payload: dict[str, Any],
    timeout: float = 900.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    if not texts:
        finish = candidates[0].get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned no text; finish={finish}")
    return "".join(texts).strip()


def parse_json_text(value: str) -> Any:
    clean = value.strip()
    if clean.startswith("```"):
        clean = clean[clean.find("\n") + 1 :]
        if clean.endswith("```"):
            clean = clean[:-3]
    return json.loads(clean.strip())


def call_model(
    key: str,
    model: str,
    prompt: str,
    max_retries: int,
) -> tuple[Any, dict[str, Any]]:
    url = API_ROOT + f"{urllib.parse.quote(model, safe='')}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        retry_delay = min(60.0, 2.0**attempt)
        try:
            response = api_request(url, key, payload)
            return (
                parse_json_text(extract_text(response)),
                response.get("usageMetadata", {}),
            )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"Gemini HTTP {error.code}: {detail[:1000]}"
            )
            if error.code not in {429, 500, 502, 503, 504}:
                break
            if error.code == 429:
                match = re.search(r"Please retry in ([0-9.]+)s", detail)
                if match:
                    retry_delay = float(match.group(1)) + 1.0
        except Exception as error:
            last_error = error
        time.sleep(retry_delay)
    raise RuntimeError("Gemini request failed or returned invalid JSON") from (
        last_error
    )


def build_query_prompt(session: Path, query: dict[str, Any]) -> str:
    """The exact per-query prompt (also used for Claude independent queries)."""
    target = (session / "TARGET_DESCRIPTION.txt").read_text().strip()
    readme = (session / "DATA_README.md").read_text()
    candidates = query["candidates"]
    blocks = [
        (
            "You are an expert aerospace prognostic engineer. Decide which "
            "anonymous source machine would be most informative for adapting "
            "a remaining-useful-life prognostics model to the described "
            "target. Use only the material in this request. Do not infer "
            "meaning from filename numbers or use the number of recorded "
            "flights as a proxy for a flight class.\n\n"
            f"[TARGET DESCRIPTION]\n{target}\n\n"
            "[CHOICE TASK]\n"
            f"Query ID: {query['query_id']}\n"
            f"Candidates: {json.dumps(candidates)}\n"
            "Select exactly one listed candidate if a credible relevant "
            "source is present; otherwise select `none`. Return exactly one "
            "JSON object with keys `query_id`, `choice`, and `reasoning`. "
            "`choice` must be a listed filename or `none`; `reasoning` must "
            "be a concise quantitative explanation. Do not add Markdown or "
            "text outside the JSON."
        ),
        "# DATA README\n" + readme,
    ]
    for filename in candidates:
        source = session / filename
        blocks.append(f"# FILE {filename}\n```csv\n{source.read_text()}\n```")
    return "\n\n".join(blocks)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def validate_answer(answer: dict[str, Any], query: dict[str, Any]) -> None:
    if answer.get("query_id") != query["query_id"]:
        raise ValueError(
            f"query ID mismatch: expected {query['query_id']}, "
            f"got {answer.get('query_id')}"
        )
    if answer.get("choice") not in query["candidates"] + ["none"]:
        raise ValueError(
            f"query {query['query_id']} returned invalid choice "
            f"{answer.get('choice')}"
        )
    if not str(answer.get("reasoning", "")).strip():
        raise ValueError(f"query {query['query_id']} returned no reasoning")


def run_query(
    session: Path,
    output: Path,
    query: dict[str, Any],
    key: str,
    model: str,
    max_retries: int,
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
    answer, usage = call_model(key, model, prompt, max_retries)
    if isinstance(answer, list) and len(answer) == 1 and isinstance(
        answer[0], dict
    ):
        answer = answer[0]
    if not isinstance(answer, dict):
        raise ValueError(
            f"query {query['query_id']} returned a non-object JSON value"
        )
    validate_answer(answer, query)
    payload = {
        "model": model,
        "query_id": query["query_id"],
        "answer": answer,
        "usage_metadata": usage,
        "prompt_characters": len(prompt),
        "prompt_sha256": prompt_sha256(prompt),
    }
    checkpoint.write_text(json.dumps(payload, indent=2))
    return answer, payload


def copy_blind_inputs(source: Path, destination: Path) -> None:
    for name in (
        "TASK.md",
        "TARGET_DESCRIPTION.txt",
        "DATA_README.md",
        "RESULT_SCHEMA.md",
        "required_subgroups.json",
        "validate_result.py",
    ):
        shutil.copy2(source / name, destination / name)
    for path in source.glob("source_*.csv"):
        shutil.copy2(path, destination / path.name)


def derived_source_summary(
    sources: list[str],
    queries: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Descriptive ranking by selection rate; the LIP fit never uses it."""
    appearances = {source: 0 for source in sources}
    selections = {source: 0 for source in sources}
    for query, answer in zip(queries, answers):
        for source in query["candidates"]:
            appearances[source] += 1
        if answer["choice"] != "none":
            selections[answer["choice"]] += 1
    ranking = sorted(
        sources,
        key=lambda source: (
            -selections[source] / max(appearances[source], 1),
            -selections[source],
            source,
        ),
    )
    relevant = [source for source in ranking if selections[source] > 0]
    return ranking, relevant


def load_session(session: Path) -> tuple[list[dict[str, Any]], list[str]]:
    queries = json.loads((session / "required_subgroups.json").read_text())[
        "queries"
    ]
    sources = sorted(path.name for path in session.glob("source_*.csv"))
    if len(sources) != 15:
        raise RuntimeError(f"expected 15 anonymous sources, found {len(sources)}")
    return queries, sources


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
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("set the GEMINI_API_KEY environment variable")
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
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
    copy_blind_inputs(arguments.session, arguments.output)

    def execute(query: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return run_query(
            arguments.session,
            arguments.output,
            query,
            key,
            arguments.model,
            arguments.max_retries,
            arguments.resume,
        )

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        records = list(pool.map(execute, queries))
    answers = [record[0] for record in records]
    checkpoints = [record[1] for record in records]

    ranking, relevant = derived_source_summary(sources, queries, answers)
    result = {
        "ranked_sources": ranking,
        "relevant_sources": relevant,
        "reasoning": (
            "Ranking is a descriptive summary derived from independent "
            "subgroup choices; LIP fitting uses the choices directly."
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
        "temperature": 0,
        "protocol": "independent blind subgroup choice",
        "queries": len(queries),
        "subgroup_size_range": [
            min(len(query["candidates"]) for query in queries),
            max(len(query["candidates"]) for query in queries),
        ],
        "workers": arguments.workers,
        "source_labels_shown": False,
        "other_queries_shown": False,
        "prompt_sha256": [checkpoint["prompt_sha256"] for checkpoint in checkpoints],
        "usage_metadata": [
            checkpoint["usage_metadata"] for checkpoint in checkpoints
        ],
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
