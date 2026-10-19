# Source-Identification Elicitation Task

You will perform a "language-induced prior" elicitation experiment for
the Hopper robotic locomotion task. The goal is to test whether you
can identify, from raw telemetry data alone, which source out of 10
candidates is dynamically consistent with a target physical
environment.

## Inputs in your working directory

- `technical_report.md` — describes the Hopper simulation
  environment and the telemetry format.
- `target.md` — gives the target environment's physical parameter.
- `data/` — 10 CSV files: `source_01.csv`, `source_02.csv`, …,
  `source_10.csv`. Each is the data buffer of one candidate
  environment. Source IDs are anonymized labels; the only signal
  about each source is its telemetry data.

## Your task

Perform **50 independent multiple-choice queries**. For each query
you must:

1. Choose a **subgroup** of 3 or 4 distinct sources from the 10
   candidates. Across the 50 queries the subgroups must all be
   different combinations (sample without replacement at the subgroup
   level — i.e., do not repeat the same combination, but individual
   sources can appear in many subgroups). Use a fresh random seed
   (e.g., from your environment) so that the 50 subgroups are not
   predictable.
2. Read the corresponding CSVs from `data/`. You may use any tool you
   like — pandas, numpy, raw `csv` reader, your own Python script,
   etc. — to extract the information you need. You may also write
   helper scripts and persist them.
3. Read `technical_report.md` to understand how to interpret the
   telemetry, then read `target.md` to learn the target's physical
   parameter.
4. For each candidate in the subgroup, decide whether it is the best match
   for the target environment. Be conservative, if you think none of them matches, just choose none.

5. Record this query's outcome.

## Output

Append each query's result to `results.json` in your working
directory. Save **after every query** so progress is preserved if the
session is interrupted. The file format:

```json
{
  "n_completed": 7,
  "queries": [
    {
      "query_id": 0,
      "subgroup": ["source_03", "source_09", "source_14"],
      "matching": ["source_09"],
      "reasoning": "Per-candidate notes here..."
    },
    ...
  ]
}
```

`subgroup` and `matching` are lists of anonymized source IDs (e.g.
`"source_09"`). `matching` is a (possibly empty) subset of `subgroup`.
`reasoning` is your free-form explanation per query — include
what evidence in the data led to your decision.

## Practical guidance

- Be honest: if you cannot make a confident judgment from a
  candidate's data, treat it as non-matching.
- Do not infer anything from the source IDs themselves — they are
  arbitrary anonymous labels. The only basis for your decision is
  the telemetry data.

When all 50 queries are complete, your `results.json` should have
`n_completed: 50` and 50 entries in `queries`.
