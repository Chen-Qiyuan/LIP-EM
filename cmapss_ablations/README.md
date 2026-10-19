# Ablations and controls for the C-MAPSS case study

Code for four appendix studies of the paper. The three C-MAPSS studies import the released C-MAPSS code in `../cmapss` (`experiment.py`, `em.py`, `dataset.py`) and the shared LIP code in `../lip`, so they use exactly the targets, spline basis, LIPs and EM of Table 1. The non-Gaussian study is self-contained but takes its stopping rule from `../cmapss/em.py`.

| Study | Paper artifact | Folder | Runtime |
|---|---|---|---|
| Null-likelihood sweep | App. "Sensitivity to the Null Likelihood (C-MAPSS)", Table `tab:null_sweep` | `null_sensitivity/` | ~40 s |
| Shape-matched shuffled LIPs | App. "Shuffled LIP Control (C-MAPSS)", Table `tab:shuffled` | `shuffled_lip/` | ~2.5 min + ~2 min |
| Exact Bernoulli and Poisson likelihoods, prior corruption | App. "Non-Gaussian Likelihoods and Controlled Prior Corruption", Tables `tab:nongauss`, `tab:corruption` | `nongaussian_prior/` | ~5 min |
| Target-description (prompt) ablation | App. "Sensitivity to the Target Description (C-MAPSS)", Table `tab:prompt` | `prompt_ablation/` | ~2.5 min |

The scripts work with RUL levels, while the paper labels each cutoff by its observed life, 1 − RUL. So 90/70/50% RUL are the paper's 10/30/50% observed life, and 90–40% RUL is its 10–60%. The LaTeX rows printed by `analyze.py` use the paper's labels.

Each folder has a `run.py`, which runs the experiment and writes `results/`, and an `analyze.py`, which reads `results/` and prints the paper's table (also as LaTeX rows) and the numbers quoted in the text. The shipped `results/` are the outputs of the paper's runs, so every `analyze.py` works without running anything first. No LLM is called and no API key is needed.

## Setup

```
pip install -r requirements.txt
```

The versions are those of `../cmapss/requirements.txt` (Python 3.14.4, CPU only). The CPU build of torch is enough. Run every command from the study's folder inside the full repository: the scripts find `../../cmapss` and `../../lip` relative to their own location. Runtimes were measured on a desktop CPU with `OMP_NUM_THREADS=4`. `replicates.csv.gz` is written with a fixed gzip timestamp, so reruns are byte-identical.

## Method settings

The C-MAPSS studies use the defaults of `../cmapss/em.py` and `../cmapss/experiment.py` (see `../cmapss/README.md`). EM is Algorithm 1 with exact M-step, τ = 1e-3, empirical Bayes null with normalised weights (1 − w_j)/Σ_{i≠k}(1 − w_i), tempering β_k(t) = (1 − e^{−νt})/ε_k with ε_k² = Tr(H₀⁺ H_k) for the pseudo-inverse H₀⁺, ν = 0.05, stopping when ‖w⁽ᵗ⁾ − w⁽ᵗ⁻¹⁾‖_∞ ≤ 1e-3 for 5 consecutive iterations counted from t₀ = ⌈ln 2/ν⌉ = 14, at most 1000 iterations, and maximum-likelihood θ̂. The LIP uses p₀ = 0.01 and λ_LIP = 0.1, Pooled and Target-Only use ridge λ = 1e4, and the 10 target engines are those of Table 1.

## `null_sensitivity/`: null-likelihood sweep

The null enters EM only through p(D_k | c_k = 0). Multiplying it by e^c per observation maps log LR_k to log LR_k − N_k c. `run.py` sweeps c ∈ {−10, −3, −1, −0.3, 0, 0.3, 1, 3, 10} nats per observation at 90/70/50% RUL for LIP-C, LIP-G and Uniform EM, and also replaces the EB null by the parametric pooled null (one Gaussian GLM fitted to the 99 sources pooled). With the EB null and c = 0 the EM loop is exactly `em.em_train`, and `--check` asserts this (bit-identical fits).

```
python run.py            # 900 EM runs, ~40 s -> results/runs.json
python run.py --check    # same, and compares c = 0 with em.em_train
python analyze.py        # Table tab:null_sweep and the numbers in the text
```

`results/runs.json` holds one record per (target, RUL, method, null, c) with the RMSE, final weights summary, iterations, the per-observation log-LR percentiles at the last E-step and the median 1/ε_k, plus the Pooled and Target-Only RMSEs. Expected output: the table of the paper (c = 0 reproduces Table 1). LIP-EM is below Pooled and Target-Only in 54 of 54 cells and below Uniform EM in 52 of 54, with LIP RMSE in [13.2, 35.7] over all null levels and the pooled null. At c = 0 the per-observation log-LR ranges from −8.26 (lowest 5th percentile) to 2.43 (maximum). The median 1/ε_k is 1.8e-2 (90%), 4.8e-5 (70%) and 1.1e-2 (50%). The final weights leave the prior (max_k |w_k − π_k| ≥ 0.02) in 273 of 300 runs at 90% RUL, 65 of 300 at 70% (engines 9, 26, 27, 48 and 51) and 239 of 300 at 50%.

## `shuffled_lip/`: shape-matched shuffled LIPs

For each target the fitted LIP-C and LIP-G are permuted over the 99 source identities (the same permutation for both, drawn with `np.random.default_rng(seed + 10000 * target)`), which keeps the multiset of prior values and destroys only their assignment to sources. Each shuffled prior is run through the C-MAPSS EM. `batched_em.py` runs `em.em_train` vectorised over all priors of one (target, cutoff), using the per-source quantities and settings of `em.py`. For every (target, cutoff), `run.py` also runs `em.em_train` on the unshuffled LIP-C, LIP-G and Uniform priors and asserts that the results agree (max |Δw| 7e-9, identical iteration counts).

```
python run.py                                                  # 100 shuffles, seed 42, all 9 cutoffs, ~2.5 min -> results/shuffles100/
python run.py --shuffles 500 --seed 314159 --rul 0.9,0.7,0.5   # 500 shuffles, ~2 min -> results/shuffles500/
python analyze.py                                              # tables, permutation tests and diagnostics for both
```

`runs.csv` has one row per (target, RUL, method, shuffle) with the RMSE, final-weight summaries and iterations, and `settings.json` records the settings and the check against `em.em_train`. Expected output: Table `tab:shuffled` (shuffled entries average the permutations within each engine, then mean (SE) across engines). From 90% to 40% RUL both real LIPs beat all 100 shuffles (p = 1/101), and all 500 at 90/70/50% (p = 1/501). Of the 100 shuffles, 99–100 are worse than Uniform EM at 90% and 50%, and 75–80 at 70%. At 30%, 43 and 2 shuffles beat LIP-C and LIP-G. At 10% every shuffle beats LIP-C.

## `nongaussian_prior/`: exact Bernoulli and Poisson likelihoods

A synthetic study with K = 20 sources of N_k = 80 observations, five of them relevant (target parameter θ₀ drawn from {0.2, 0.8} or {2, 4}) and 15 at the null parameter (0.5 or 3). EM is Algorithm 2 with d = 1 and τ = 0: exact likelihood-ratio E-step against the true null model, exact weighted-MLE M-step, ε_k² = N_k/N₀. Priors: Uniform (p₀ = 0.01), a hand-specified aligned prior, a random permutation of it (shuffled), an oracle prior, and the oracle with 20–100% of its high-prior assignments swapped onto irrelevant sources. This study uses ν = 0.05, tolerance 1e-8, at most 500 iterations and the same t₀ = ⌈ln 2/ν⌉ = 14 (taken from `../cmapss/em.py`). 1000 replications per (family, N₀), N₀ ∈ {2, 4, 8, 16, 32, 64}, seed 42.

```
python run.py        # 132,000 rows (108,000 EM fits), ~5 min -> results/replicates.csv.gz, results/summary.csv, results/settings.json
python analyze.py    # Tables tab:nongauss and tab:corruption, source AUC, paired contrasts
```

`replicates.csv.gz` has one row per (family, N₀, replication, method) with the estimate, the excess test NLL (KL from the true distribution), the source AUC and Brier score of the final weights and the iterations. Expected output (x 10³): at N₀ = 2 the aligned prior gives 1.5 (Bernoulli) and 1.3 (Poisson) vs 6.5 and 15.5 for Uniform EM and 44.7 and 41.0 for the shuffled prior. The source AUC is 1.00 (aligned) vs 0.85 and 0.82 (shuffled). The excess NLL grows with the corruption share up to 189.9 and 166.5 at 100% (Pooled: 113.4 and 96.7), and is 13.7 and 19.1 at N₀ = 64. 12 of the 108,000 EM fits reach the 500-iteration cap.

## `prompt_ablation/`: target-description ablation

Each description was answered by a fresh, blind Claude Code agent session (Claude Opus 4.7, maximum effort) on the same 200 subgroups, in the same order, as the cached LIP-C of Table 1 (`../cmapss/data/llm_subgroup_queries_claude.json`). Only the target description differs. `run.py` fits the LIP of each response file for each target (dropping the queries that contain the target, as for Table 1) and runs the C-MAPSS EM at all nine cutoffs, next to Uniform EM, Pooled and Target-Only.

```
python run.py                    # 19 LIPs + Uniform, ~2.5 min -> results/runs.json, results/priors.json
python analyze.py                # Table tab:prompt, every condition, headline numbers
python analyze.py --per-cutoff   # also mean (SE) at every cutoff
```

`results/runs.json` has one record per (condition, target, RUL) with the RMSE and EM diagnostics, and `results/priors.json` holds the fitted LIPs (used for the Spearman correlations). The all-cutoff (90–40%) metric averages each target's RMSE over the nine (six) cutoffs and then takes the mean (SE) over targets. Differences are paired, with a 95% bootstrap over target engines (20,000 resamples, `default_rng(20260802)`). Expected output: Table `tab:prompt`. The four informative alternatives (`opus/no_diagnostics` to `opus/inspection_wording`) improve on Uniform EM by 5.84–6.85 (all cutoffs) and 8.79–10.24 (90–40% RUL), with every interval excluding 0, and are correlated with the cached LIP-C at ρ = 0.88–0.93. The no-information control returns the null option for all 200 queries and matches Uniform EM (per-run RMSE differences ≤ 1.6e-4). The cached LIP-C minus the no-information control is −5.81 [−8.73, −2.77].

**Response files** (`responses/`, 18 files, 1.4 MB). The key is the path under `responses/` without `.json`. `opus/` was elicited with Claude Opus 4.7, `cli_default/` with the default Opus model of the Claude Code command-line interface (meta field `claude-opus-cli-session`). Each file has the fields of the other LIP response files (`target_desc`, `responses` with `S`, `C`, `reasoning`, `source_basenames`, `meta`).

| Key | Label | Target description |
|---|---|---|
| `cached` | Original (cached LIP-C of Table 1) | This aircraft is operating in a desert environment. Routine visual inspections confirm severe aerodynamic degradation of the high-pressure compressor, likely due to continuously ingesting fine, dry abrasives at high velocities. |
| `opus/original` | Original (re-elicited) | (the original description) |
| `opus/no_diagnostics` | No-diagnostics | This aircraft is operating in a desert environment. Routine visual inspections confirm severe aerodynamic degradation of the high-pressure compressor. |
| `opus/desert_only` | Desert only | This aircraft is operating in a desert environment. |
| `opus/no_desert` | No-desert wording | Inspection records identify severe aerodynamic degradation of the aircraft's high-pressure compressor, consistent with sustained ingestion of fine, dry airborne abrasives at high velocity. |
| `opus/inspection_wording` | Alt. inspection wording | Inspection findings for this desert-operated aircraft show severe aerodynamic degradation of the high-pressure compressor, consistent with prolonged ingestion of fine, dry abrasive dust at high velocity. |
| `opus/no_information` | No-information control | Maintenance records are available for the target aircraft, but they contain no reliable information about the condition or trajectory of the high-pressure compressor. |
| `opus/contradictory` | Contradictory direction | Controller logs indicate that progressively lower physical core speed is required to maintain commanded operation as the high-pressure compressor degrades. |
| `opus/no_desert_findings` | No-desert wording (findings) | Inspection findings for this aircraft show severe aerodynamic degradation of the high-pressure compressor, consistent with prolonged ingestion of fine, dry abrasive dust at high velocity. |
| `opus/no_desert_routine` | No-desert wording (routine) | Routine inspections of this aircraft show severe aerodynamic degradation of the high-pressure compressor, consistent with prolonged high-velocity ingestion of fine, dry abrasive particles. |
| `opus/desert_operations` | Paraphrase: desert operations | During desert operations, the aircraft continuously draws fine, dry abrasive particles into its high-pressure compressor at high speed. Routine inspections show severe resulting aerodynamic degradation of the compressor. |
| `opus/maintenance_records` | Paraphrase: maintenance records | Maintenance records describe severe high-pressure-compressor erosion after repeated exposure to airborne sand and dust during operation. |
| `opus/maintenance_records_detailed` | Paraphrase: maintenance records, long | Maintenance records describe severe aerodynamic degradation of the high-pressure compressor following sustained exposure to airborne sand and dust during operation, likely caused by continuous high-velocity ingestion of fine, dry abrasive particles. |
| `opus/mechanism_direct` | Mechanism-direct | Controller logs indicate that progressively greater physical core speed is required to maintain commanded operation as the high-pressure compressor degrades. |
| `cli_default/{original, maintenance_records, mechanism_direct, no_information, contradictory}` | CLI: … | as the Opus files of the same name |

Table `tab:prompt` shows `cached` and `opus/no_diagnostics` to `opus/no_information`. `analyze.py` also prints `opus/original` and `opus/contradictory`, and the other files are additional checks.

**Eliciting a new description (optional).** This step runs a Claude Code agent, not an API call from this code. `elicit.py prepare` writes a self-contained session folder with the 100 trajectories under anonymous names, the NASA report, the 200 subgroups and the task files (`CLAUDE.md`, `TASK.md`, `RESULT_SCHEMA.md`, `validate_result.py`) exactly as used for the shipped responses. Start Claude Code in that folder, and the agent writes `results.json`. `elicit.py collect` converts it into a response file. Add the file to `conditions.py` to include it in `run.py`. LLM answers vary between sessions (re-eliciting the original description gives ρ = 0.894 with the cached LIP-C).

```
python elicit.py prepare --description "This aircraft is ..." --session sessions/mine
cd sessions/mine && claude          # the agent answers the 200 subgroups
cd ../.. && python elicit.py collect --session sessions/mine --model claude-opus-4-7 --out responses/opus/mine.json
```

## Data

The C-MAPSS studies read `../cmapss/data/` (the per-engine core-speed CSVs, the NASA report and the cached LIP-C and LIP-G responses), which ships with the repository. See `../cmapss/README.md` for how to rebuild it from the raw NASA files. The prompt ablation also reads `prompt_ablation/responses/`. The non-Gaussian study is synthetic and needs no data.

## Files

| | |
|---|---|
| `null_sensitivity/run.py`, `analyze.py` | null sweep, with results in `results/runs.json` |
| `shuffled_lip/run.py`, `analyze.py` | shuffled-LIP control, with results in `results/shuffles{100,500}/{runs.csv,settings.json}` |
| `shuffled_lip/batched_em.py` | `em.em_train` vectorised over priors |
| `nongaussian_prior/run.py`, `analyze.py` | Bernoulli/Poisson study, with results in `results/{replicates.csv.gz,summary.csv,settings.json}` |
| `prompt_ablation/run.py`, `analyze.py` | prompt ablation, with results in `results/{runs.json,priors.json}` |
| `prompt_ablation/conditions.py` | condition keys, labels and response files |
| `prompt_ablation/elicit.py` | prepares and collects a blind agent session for a new description |
| `prompt_ablation/responses/` | the 18 cached response files |
