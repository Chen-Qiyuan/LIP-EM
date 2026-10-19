# LIP-EM on N-CMAPSS DS03 (short-, medium- and long-haul fleets)

Code, cached LLM choices and results for the N-CMAPSS DS03 direct-RUL experiment in the appendix. This folder reproduces:

| Artifact | Script | Output |
|---|---|---|
| Per-fleet RUL RMSE tables (long-, medium- and short-haul, 5 target engines each), paired LIP vs Uniform tests | `prognostics.py`, `bootstrap.py`, `analysis.py` | `results/tables.md` (first section), `results/bootstrap_*/` |
| Long-haul table with the Claude LIP from independent queries (paper `tab:ds03`) | same | `results/tables.md`, `results/bootstrap_long_claude_independent/` |
| EM diagnostics (runs that stay at the prior, iteration counts) | `analysis.py` | `results/tables.md` (second section) |
| Pooled secondary analysis over all 15 targets (10–30% observed-life band, sign-flip test) | `analysis.py` | `results/tables.md` (third section), `results/analysis.json` |

## Setup

N-CMAPSS DS03 has 15 engines with one fault mode: five short-haul, five medium-haul and five long-haul (the dataset's flight classes). Each engine is one domain. For one flight class at a time, each of its five engines is the target and the other 14 engines are anonymous sources, and four of them are in the target's class. At each cutoff (10%, 20%, …, 90% of the target's life observed) the model sees only the target's sensor prefix in raw flight cycles (never normalised by engine life) and predicts the remaining useful life (RUL).

- **Backbone.** One multiple-output Gaussian regression of the 14 operating-adjusted sensor residuals on a five-dimensional natural cubic spline in the raw cycle (knots 16.67, 33.33, 50, 66.67, 83.33). Each source engine has one relevance weight shared by all 14 sensors.
- **RUL.** Following the data-level fusion protocol of Liu, Gebraeel and Shi (2013), the predicted sensor paths are mapped to a composite health index fitted on the sources only, and the predicted RUL is the first time the index crosses the mean terminal health of the sources (capped at the longest source life).
- **LIP.** The only target text given to the LLM is the dataset label, e.g. "The target aircraft belongs to the long-haul flight class." Source labels are never shown. The LLM answers 30 subgroup queries of 3–5 anonymous sources (seed 42). The LIP is fitted with p₀ = 0.01 and λ_LIP = 0.1 (paper eq. LIP_opt), separately for each target with the queries that contain the target dropped. Three elicitations are used:
  - `claude_agent`: one blind Claude Code session (claude-opus-4-7, effort max) that inspects all 15 anonymous sources and answers the 30 queries.
  - `claude_independent` (long-haul only): Claude Code answering each query in a fresh, independent request that shows only that subgroup's sources.
  - `gemini`: gemini-3-flash-preview at temperature 0, one independent request per query.
- **Methods.** Claude LIP-EM (LIP-C in `tab:ds03`), Gemini LIP-EM (LIP-G), Uniform EM (π_k = 0.01), Global Pool (Pooled) and Target-Only (Target Only) (ridge λ = 10⁴ with an unpenalised intercept, as for C-MAPSS).
- **EM (paper Algorithm 1).** Exact Gaussian E- and M-steps, τ = 10⁻³, ν = 0.05, maximum-likelihood estimates. Tempering β_k(t) = (1 − e^{−νt})/ε_k with ε_k² = Tr(H₀⁺ H_k) for the pseudo-inverse H₀⁺ (q·Tr for the q = 14 outputs). The empirical Bayes null uses the normalised weights (1 − w_j)/Σ_{i≠k}(1 − w_i). EM stops when ‖w^(t) − w^(t−1)‖_∞ ≤ 10⁻³ for 5 consecutive iterations, counted only from t ≥ t₀ = ⌈ln 2/ν⌉ = 14, with a cap of 1000 iterations. All of this is in `em.py` and is the only code path.
- **Uncertainty.** Per fleet, the five target engines are resampled (B = 20000, seed 42). The pooled analysis resamples engines within each fleet (stratified, B = 20000, seed 42) and adds an exact paired sign-flip test over all 2¹⁵ swaps of the LIP and Uniform errors (two-sided).

## Layout

| Path | Content |
|---|---|
| `common.py` | Feature loading, spline basis, source-only health index, metrics |
| `em.py` | LIP-EM for the shared-design multi-output Gaussian model (final method) |
| `prognostics.py` | Leave-one-engine-out prognostics for one flight class and one LIP |
| `bootstrap.py` | Engine bootstrap for one fleet (Claude run + Gemini run) |
| `analysis.py` | `tables.md` and `analysis.json`: per-fleet tables, EM diagnostics, pooled analysis |
| `run_all.sh` | Everything above, from the cached LLM choices to the tables |
| `fit_lip_priors.py` | Fold-specific LIPs from cached choices |
| `build_features.py` | Flight-level features from the raw DS03 HDF5 file |
| `prepare_sessions.py` | Blind LLM inputs (anonymous sources, 30 queries, Claude Code agent session) |
| `query_gemini.py`, `query_claude.py` | Independent per-query elicitation (Gemini API, Claude Code CLI) |
| `data/cycle_features.csv` | Flight-level features of all 15 engines (output of `build_features.py`, 0.4 MB) |
| `data/feature_metadata.json` | Feature provenance and leakage checks |
| `llm_responses/<class>/` | Cached LLM choices: `<elicitation>/results.json` (30 choices with reasoning), the source mapping (scoring only), the queries and the protocol. `query_metadata.json` holds the model settings and the SHA-256 of every prompt, and `claude_agent/analysis_report.md` is the agent's own report |
| `results/` | Shipped outputs of `./run_all.sh results` (see below) |

`results/` contains `lip_priors/<class>/` (fitted LIPs), one directory per prognostics run `<class>_<elicitation>/` (`prognostics_predictions.csv`, `source_weights.csv` with the final EM weights and iteration counts, `summary.json`, `report.md`), the stdout log `<class>_<elicitation>.log` of each prognostics run, one `bootstrap_<class>_<claude elicitation>/` per fleet, and `tables.md` / `analysis.json`.

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.14.4 (numpy 2.5.1, scipy 1.18.0, h5py 3.16.0) and Python 3.12 (numpy 2.5.3, scipy 1.18.1). `h5py` is needed only by `build_features.py`. The code runs on one CPU core.

## Reproduce the tables (no data download, no API key)

From this folder:

```bash
./run_all.sh                 # writes reproduced/; about 6 min on one CPU core
diff -r results reproduced   # no output with the package versions in requirements.txt
```

`run_all.sh` fits the LIPs from `llm_responses/` (seconds), runs the seven prognostics runs (about 50 s each), the four fleet bootstraps (a few seconds each) and `analysis.py`. Use `PYTHON=/path/to/python ./run_all.sh OUT` to choose the interpreter and output directory. Every step except `analysis.py` refuses to write into a nonempty directory. The individual steps are:

```bash
python fit_lip_priors.py --responses llm_responses/long --target-class long --output OUT/lip_priors/long
python prognostics.py --target-class long --lip-priors OUT/lip_priors/long/claude_independent_lip_priors.json \
    --lip-label "Claude LIP (30 independent queries)" --output OUT/long_claude_independent
python bootstrap.py --target-class long --claude OUT/long_claude_independent --gemini OUT/long_gemini \
    --output OUT/bootstrap_long_claude_independent
python analysis.py --results OUT
```

### Expected output

With numpy 2.5.1 / scipy 1.18.0 the rerun is bit-identical to `results/`. With Python 3.12, numpy 2.5.3 and scipy 1.18.1 only the diagnostic column `predicted_crossing_log_health` changes (by at most 10⁻¹³), and predictions, weights, iterations and all tables are identical. The main numbers follow (RUL RMSE in cycles). Differences are RMSE(LIP) − RMSE(Uniform) with 95% bootstrap CI, and `results/tables.md` has all cutoffs.

- **Long-haul fleet, independent-query protocol** (Claude LIP from independent queries). RMSE (SE) at 10% observed life: Claude 5.92 (1.04), Gemini 5.79 (1.10), Uniform 8.23 (2.10), Global Pool 32.73 (2.55), Target-Only 32.73 (2.55). The early LIP gain is not significant (Claude −2.31 [−5.73, +2.23], Gemini −2.44 [−5.80, +2.17]). Uniform EM is significantly better at 60% and 70% (Claude +1.40 [+0.64, +2.24] and +1.30 [+0.23, +2.02], Gemini +1.68 [+0.35, +2.84] and +1.70 [+0.38, +2.75]).
- **Long-haul with the Claude coding-agent LIP:** −1.11 [−3.12, +2.42] at 10%. **Medium-haul:** Claude −0.20 [−1.53, +1.76], Gemini −0.83 [−2.78, +1.89] at 10%. **Short-haul:** Claude −1.97 [−3.36, +0.11], Gemini −1.43 [−2.01, −0.74] at 10%. From 50% observed life on, most differences are positive (Uniform EM better) or close to zero.
- **Pooled 15 targets, 10–30% band** (secondary, with Claude = coding-agent LIP in every fleet): Claude −2.32 [−3.79, −0.31] (sign-flip p = 0.119), Gemini −2.13 [−3.32, −0.59] (p = 0.059). Both bootstrap intervals exclude 0, but neither sign-flip test is significant. The per-engine contributions in `tables.md` show that most of the gain comes from three short-haul engines (DS03-dev-U05, DS03-dev-U09, DS03-test-U12).
- **EM diagnostics.** At 10% observed life, and at 20% in the long- and medium-haul fleets, no EM run, LIP or Uniform, ends at its prior (max_k |w_k − π_k| < 0.02). From 30% to 60%, most runs end at the prior after the minimum 19 iterations (t₀ = 14 plus 5 patience iterations). Overall 260 of 630 EM runs stop at 19 iterations and none reaches the 1000-iteration cap.

## Rebuild the features from the raw data (optional)

The raw file is not shipped (`N-CMAPSS_DS03-012.h5`, 3,693,395,776 bytes, SHA-256 `f67cc4bd0cf927f09eb8e0198bd62c777e1c6213177cfc46d81cc43bba52c333`). Download the N-CMAPSS data set (Arias Chao et al., 2021) from the NASA Prognostics Center of Excellence data repository, https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/, item 17 "Turbofan Engine Degradation Simulation-2" (at the time of writing the archive is https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip), and unpack it. Only `N-CMAPSS_DS03-012.h5` is needed. Check it with `sha256sum`. Then

```bash
python build_features.py --data /path/to/N-CMAPSS_DS03-012.h5 --output rebuilt_features
# or: export NCMAPSS_DATA_DIR=/path/to/dir_with_h5; python build_features.py --output rebuilt_features
```

This takes 4–10 s and under 1 GB of memory. With numpy 2.5.1 and h5py 3.16.0, `rebuilt_features/cycle_features.csv` is byte-identical to `data/cycle_features.csv`. With numpy 2.4.4 the values differ by at most 7 × 10⁻¹³. Pass `--features rebuilt_features/cycle_features.csv` to `prognostics.py` (and `prepare_sessions.py`) to use them.

## Re-elicit the LIPs (optional, with LLM calls)

`results/` uses the cached choices in `llm_responses/`. To collect new ones for a class (here `long`):

```bash
python prepare_sessions.py --target-class long --sessions sessions/long --scoring sessions/long/scoring
sessions/long/claude_agent/run_claude.sh                     # Claude Code agent session -> results.json
python query_claude.py --session sessions/long/claude_agent --output sessions/long/claude_independent
GEMINI_API_KEY=... python query_gemini.py --session sessions/long/claude_agent --output sessions/long/gemini
```

Then copy `sessions/long/scoring/*.json` and the three `results.json` into a directory laid out like `llm_responses/long/` and run `fit_lip_priors.py` on it. `prepare_sessions.py` reproduces the shipped queries, source mapping and blind inputs exactly, and the per-query prompts built by `query_gemini.build_query_prompt` match the SHA-256 hashes stored in `llm_responses/*/*/query_metadata.json`. The Claude Code agent run (`run_claude.sh`) needs the `claude` CLI, and the Gemini script reads the key only from the `GEMINI_API_KEY` environment variable. `run_claude.sh` and `query_claude.py` start Claude Code with `--permission-mode bypassPermissions` (web tools disabled), so the agent runs shell commands and writes files without asking. Run them in a container or VM. LLM outputs are not deterministic, so new choices, and hence new tables, will differ from the shipped ones.

## Notes

- `fit_lip_priors.py` solves the same objective as the shared `../lip/lip.py:fit_lip` but with scipy's L-BFGS-B, which reproduces the priors used for the paper exactly (the torch L-BFGS in `lip.py` agrees to within 6 × 10⁻⁶).
- Target RUL and future target sensor values are never used for fitting. Health-index weights, the failure threshold and the RUL cap are estimated from the source engines only. Flight-class labels of sources are used only to score source recovery.
