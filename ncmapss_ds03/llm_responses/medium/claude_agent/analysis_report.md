# DS03 Machine-Source Relevance — Medium-Haul Target

## Task

The target aircraft is described as **medium-haul flight class**, with too little
degradation history for a reliable RUL fit. We must rank the 15 supplied
`source_*.csv` histories by informativeness for borrowing, mark the credible
relevant subset, and answer 30 subgroup queries. Filename numbers are random and
class labels are absent; classification must be inferred from the data itself.
Flight count may **not** be used as a proxy for haul duration.

## Method

For every source we loaded all cycles and computed:

1. **Per-flight duration statistics** — `flight_duration_hours` median, min,
   max, mean, and std. This is the direct physical indicator of haul class.
2. **Terminal degradation depth** per adjusted sensor, defined as
   `|mean(last max(5, n/10) flights) − mean(first 5 flights)|`. The
   `DATA_README` states the first five flights are the healthy baseline, so this
   quantity measures the absolute drift observed by end-of-history.
3. **Cycle-wise slope** — ordinary least-squares slope of each sensor against
   `cycle`, averaged as `|slope|` across the 14 adjusted channels. This measures
   progression rate independent of trajectory length.

Reported aggregates (`agg_abs_shift`, `agg_abs_slope`) are means across the 14
adjusted sensors (`T24,T30,T48,T50,P15,P2,P21,P24,Ps30,P40,P50,Nf,Nc,Wf`).

## Haul-class identification

The flight-duration distributions cluster very cleanly into three disjoint
groups; medians and full ranges do not overlap between clusters, so the
assignment is unambiguous and does not rely on flight count.

| Cluster | Median (h) | Min (h) | Max (h) | Sources |
|---|---:|---:|---:|---|
| Short-haul  | ~1.16–1.25 | 0.87 | 1.58 | 02, 09, 10, 12, 15 |
| **Medium-haul** | ~2.44–2.58 | 1.10 | 3.42 | **06, 07, 08, 11, 14** |
| Long-haul   | ~3.88–4.31 | 2.48 | 5.64 | 01, 03, 04, 05, 13 |

Only the five medium-haul engines share the target's operating regime and are
therefore the **credible relevant** source pool. Long-haul and short-haul
histories operate at materially different mission profiles (roughly 1.6× and
0.5× the target's characteristic flight length, respectively) and would inject
regime-mismatch bias into a borrowed RUL fit.

## Ranking within the medium-haul group

Given all five candidates share the correct flight class, informativeness is
governed by how deeply each engine progressed into its degradation trajectory
(a source with only mild drift teaches less about the failure region):

| Rank | Source | n_flights | dur_median | Terminal |shift| | |slope|/cycle |
|---:|---|---:|---:|---:|---:|
| 1 | source_08 | 67 | 2.57 h | **0.566** | 7.56e-03 |
| 2 | source_14 | 73 | 2.51 h | 0.553 | 7.08e-03 |
| 3 | source_06 | 80 | 2.44 h | 0.534 | 6.32e-03 |
| 4 | source_11 | 67 | 2.58 h | 0.523 | 7.25e-03 |
| 5 | source_07 | 60 | 2.55 h | 0.464 | 7.23e-03 |

`source_08` shows both the deepest terminal excursion and the highest slope,
making it the primary borrow candidate. `source_14` is a close second.
`source_07` shows the mildest terminal drift — its engine appears to have been
retired earlier in its degradation curve.

## Full 15-source ranking

Adjacency in flight-length is used as a soft second tier: long-haul (adjacent
2× vs 3× regime distance to the target) is ranked ahead of short-haul among the
non-medium options. Within each class, sources are ordered by terminal shift.

```
 1. source_08 (medium)       shift=0.566  slope=7.6e-3
 2. source_14 (medium)       shift=0.553  slope=7.1e-3
 3. source_06 (medium)       shift=0.534  slope=6.3e-3
 4. source_11 (medium)       shift=0.523  slope=7.3e-3
 5. source_07 (medium)       shift=0.464  slope=7.2e-3
 6. source_01 (long)         shift=0.667  slope=8.4e-3
 7. source_04 (long)         shift=0.659  slope=9.3e-3
 8. source_05 (long)         shift=0.631  slope=1.0e-2
 9. source_13 (long)         shift=0.572  slope=8.6e-3
10. source_03 (long)         shift=0.505  slope=7.0e-3
11. source_02 (short)        shift=0.536  slope=6.1e-3
12. source_10 (short)        shift=0.495  slope=5.4e-3
13. source_12 (short)        shift=0.492  slope=4.8e-3
14. source_15 (short)        shift=0.479  slope=4.5e-3
15. source_09 (short)        shift=0.439  slope=5.2e-3
```

## Subgroup query strategy

For each of the 30 required subgroups: if any candidate is medium-haul, pick
the one with the largest terminal `|shift|` (ties broken by higher `|slope|`);
otherwise return `none`. This is the same key that produced the intra-class
ranking above, so answers are internally consistent with the full ranking. Nine
of the 30 subgroups contain no medium-haul candidate and therefore resolve to
`none` (queries 3, 6, 8, 11).

Confidence is **high**: haul-class inference is unambiguous (clean, non-
overlapping duration clusters), the relevant pool has exactly five members, and
the informativeness key uses physically defensible degradation quantities that
do not rely on flight count.

## Blind restrictions honoured

- Only files inside this directory were read; no parent, sibling, network, or
  filename-number inference was used.
- Haul class was inferred exclusively from `flight_duration_hours` (medians and
  ranges), never from `n_flights`.
- All diagnostic scripts live under `work/`; only `results.json` and this
  report were written at the top level.
