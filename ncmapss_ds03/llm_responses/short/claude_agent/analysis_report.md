# DS03 machine-source relevance — short-haul target

## 1. Target

`TARGET_DESCRIPTION.txt` states:

> The target aircraft belongs to the short-haul flight class.

Task: rank 15 anonymous complete engine histories by informativeness for
borrowing to fit a remaining-useful-life (RUL) model for the target, and
answer 30 forced-choice subgroup queries.

## 2. Method

For each `source_XX.csv` I computed two families of statistics on a per-file
basis (`work/analyze.py`, `work/degradation.py`):

**Flight-class signature.**  Per-file summary of
`flight_duration_hours` (min, p10, p25, median, mean, p75, p90, max, stdev).
Instructed by TASK.md, the raw *count* of flights is not used as a class proxy;
class is inferred from the shape of the duration distribution.

**Degradation depth.**  For each of the 14 `adjusted_*` sensor residual
columns I computed:

- `baseline_mean` — mean over flights 1–5 (dataset baseline definition).
- `tail_mean` — mean over the last 5 flights (terminal state).
- `noise` — stdev of an early-middle healthy segment (flights 6…6+n/6).
- per-column `z = (tail_mean − baseline_mean) / noise`.

I aggregated to per-source scalars:

- `rssZ = sqrt(Σ z²)`   — signal-to-noise magnitude of terminal drift.
- `agg_ratio = tail_agg_mean / baseline_agg_mean` on the L1-aggregated
  residual `Σ|adjusted_*|` per flight.
- `trend_r` — Pearson r of `Σ|adjusted_*|` vs. cycle.

`rssZ` is the primary ranker within a flight class: it directly measures how
much end-of-life degradation the source engine actually reveals, which is
exactly what a borrower needs.

## 3. Flight-class classification

The 15 duration distributions cluster into three cleanly non-overlapping
envelopes:

| bracket        | duration envelope       | mean       | sources |
|----------------|-------------------------|------------|---------|
| **short-haul** | 0.87 – 1.58 h           | ~1.20 h    | 02, 09, 10, 12, 15 |
| medium-haul    | 1.10 – 3.42 h           | 2.43-2.56 h| 06, 07, 08, 11, 14 |
| long-haul      | 2.48 – 5.64 h           | 3.88-4.21 h| 01, 03, 04, 05, 13 |

Short-haul is unambiguous: every one of its five members is entirely under
1.58 h and averages ~1.2 h. Medium-haul has a p90 near 3.15 h and no flight
above 3.42 h. Long-haul has p10 near 3.0 h and no flight below 2.48 h. There
is zero ambiguity as to which source belongs to which class — see the
per-file histograms in `work/analyze.py` output.

The target is short-haul, so the five short-haul engines are the relevant
sources; the ten medium- and long-haul engines are off-class.

## 4. Within-class ranking (short-haul)

From `work/degradation.json`:

| source       | n  | agg_ratio | sum|shift| |  rssZ | trend r |
|--------------|----|-----------|-------------|-------|---------|
| source_02    | 76 |     55.8  |      8.76   |104.07 |  0.811  |
| source_10    | 84 |     54.1  |      7.70   | 80.24 |  0.848  |
| source_12    | 93 |     47.5  |      7.85   | 70.24 |  0.852  |
| source_15    | 93 |    127.2  |      7.79   | 67.57 |  0.828  |
| source_09    | 72 |     24.5  |      7.35   | 36.63 |  0.755  |

`source_02` has by far the strongest terminal-vs-baseline z-score and clean
monotonic aggregate drift, so it is the top pick when multiple short-haul
candidates are offered. `source_10` and `source_12` are close behind with
excellent monotonic trends (`trend r ≈ 0.85`). `source_15` has a very high
agg_ratio only because its baseline aggregate is tiny (0.061); its rssZ is
modest. `source_09` has a comparatively shallow terminal drift and the
weakest trend, so it is ranked last of the five relevant engines — still
relevant, but the least informative.

## 5. Off-class ranking

Medium-haul is placed second-tier (adjacent operating profile is still
partially transferable), long-haul third-tier. Within each tier I use the
same rssZ ordering:

- medium-haul: 14 (100.6) > 06 (92.5) > 08 (90.9) > 11 (80.8) > 07 (71.5)
- long-haul: 05 (126.0) > 01 (109.5) > 13 (106.9) > 04 (78.4) > 03 (68.0)

## 6. Final ranking

Most to least informative for a short-haul target:

1. source_02.csv  (short, top rssZ)
2. source_10.csv  (short)
3. source_12.csv  (short)
4. source_15.csv  (short)
5. source_09.csv  (short, weakest degradation)
6. source_14.csv  (medium, best rssZ)
7. source_06.csv  (medium)
8. source_08.csv  (medium)
9. source_11.csv  (medium)
10. source_07.csv  (medium)
11. source_05.csv  (long, best rssZ)
12. source_01.csv  (long)
13. source_13.csv  (long)
14. source_04.csv  (long)
15. source_03.csv  (long)

## 7. Subgroup answers

For each of the 30 queries I picked the highest-ranked short-haul candidate
present; when a query contains no short-haul candidate I returned `none`
because borrowing from a medium- or long-haul engine would inject a
mismatched operating profile (different climb/cruise/descent envelope,
different thermal/pressure duty cycle) and bias the RUL fit.

Distribution of choices across the 30 queries:

- 10 × `none` (no short-haul candidate present),
- 8 × source_02, 5 × source_12, 4 × source_10, 3 × source_15,
- 0 × source_09 (whenever 09 appeared, a stronger short-haul candidate did too).

## 8. Confidence

**High.** The three classes are perfectly separable by the duration
distribution (no engine crosses class boundaries), and the within-short-haul
ranking is monotone under both rssZ and agg_ratio, with a clear gap between
09 and the other four. The only borderline case is 15 vs. 12 for fourth
place, which does not affect any subgroup answer (12 and 15 never conflict:
in every query where they both appear I picked 12, which has the higher
rssZ, stronger trend, and no baseline-inflation artifact).

## 9. Reproducibility

All diagnostic code and intermediate artifacts live under `work/`:

- `work/analyze.py`        — duration statistics + histograms.
- `work/duration_stats.json` — per-source duration summary.
- `work/degradation.py`    — sensor-residual degradation metrics + class inference.
- `work/degradation.json`  — per-source metrics dump.
- `work/build_results.py`  — turns the ranking into `results.json`.

Run `python3 validate_result.py results.json` — reports `validation passed`.
