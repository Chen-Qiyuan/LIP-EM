# Claude Code local-agent elicitation (Hopper, LIP-C)

This folder is the Claude Opus 4.7 local agent's session for the Hopper LIP-C prior (paper appendix "Using a Local Agent for Relevance Assessment"). The session ran on 2026-05-04.

| File | What it is |
|---|---|
| `PROMPT.md` | The task given to the agent. It also read `../technical_report.md` and `../target.md` and the ten anonymized `data/source_NN.csv` files (full replay buffers). The source-to-gravity mapping was not in its working directory. |
| `estimate_gravity.py`, `verify_gravity.py`, `verify_gravity_v2.py`, `verify_gravity_v3.py`, `verify_gravity_v4.py`, `final_check.py` | The six gravity estimators the agent wrote and compared, in the order it wrote them (paper methods 1–6). `final_check.py` is the long-flight ballistic fit the agent kept. |
| `run_queries.py` | The script the agent used to answer the 50 subgroup queries. |
| `../claude_responses.json` | The agent's 50 answers, its original `results.json` (see below). |

## How the agent answered

The agent estimated the gravity of each source once with `final_check.py`, which gives `source_10` ≈ 9.10 m/s² and `source_08` ≈ 9.50 m/s². It then answered every query with the same rule, hard-coded in `run_queries.py`: pick `source_10` if it is in the subgroup, otherwise pick none. It judged `source_08` closer to Earth (9.81) than to Venus (8.87).

The subgroups are 50 distinct 3- or 4-source subsets, drawn from a seed taken from `os.urandom`. As a result, 18 queries select `source_10` and 32 select none. The true map is source_10 → g = 9 and source_08 → g = 10, so the agent picked the source closest to the target's 8.87 m/s².

## About `claude_responses.json`

`../claude_responses.json` is the agent's own output file (`results.json` from its working directory), byte for byte, including its Windows line endings. It records the seed (`730438886`) behind the 50 subgroups.

The `reasoning` strings are the per-query summaries that `run_queries.py` writes. They are not free text from the model.

Refitting it reproduces the shipped prior:

```bash
# from hopper/
python src/fit_lip.py --responses elicitation/claude_responses.json --compare lips/lip_claude.json
#   max |pi - pi[lip_claude.json]| = 3.60e-07
```

## Re-running the agent's estimators

The scripts read `data/source_NN.csv` next to them. The command below rebuilds, byte for byte, the full (not truncated) CSVs the agent read, from the replay buffers with the checksums listed in `../../README.md`:

```bash
# from hopper/, with the replay buffers in data/sources_replay/ (not hosted, see hopper/README.md, section 2).
# The source-to-gravity map goes to the default elicitation/_truth.json (gitignored), not next to the agent's files.
python src/setup_elicitation.py --seed 42 --max-rows 0 --out-dir elicitation/claude_agent/data
cd elicitation/claude_agent && python final_check.py
```

`final_check.py` then prints the per-source medians that `run_queries.py` hard-codes (for example, `source_10` 9.100 and `source_08` 9.503).
