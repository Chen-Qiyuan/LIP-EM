# Gaussian estimation (Figure 3)

Reproduces the 1-d and 2-d Gaussian estimation panels of Sec. 5.1 (Figure 3, "Gaussian estimation",
shipped as `results/1dgaussian.pdf` and `results/2dgaussian.pdf`), with the settings listed in the appendix
"Gaussian Estimation".

## Layout

| Path | Content |
|---|---|
| `gaussian.py` | Data, LIP elicitation (`query`), LIP fit, EM, baselines and the figure (`plot`) |
| `gaussian_em.py` | LIP-EM, paper Algorithm 1, for 1-d `(mu, sigma^2)` and 2-d `(mu, Sigma)` Gaussians |
| `data/technical_report.txt` | One-line context file uploaded to the LLM judge |
| `data/lip_responses_{1,2}d.json` | Cached Gemini responses (subgroup `S`, choice `C`, reasoning) used for the paper |
| `results/{1,2}dgaussian.pdf` | The paper figures |
| `results/{1,2}dgaussian.json` | Fitted LIP `pi`, for each method the mean, covariance and KL(truth ‖ estimate), and for the two EM methods the weights and iteration count |

The LIP code is shared with the other experiments and lives in `../lip/lip.py`.

## Setup

```bash
pip install -r requirements.txt     # Python 3.14
```

No data download is needed: all samples are synthetic and drawn from a fixed seed.

## Reproduce the figures (cached LLM responses, no API key)

Run from this folder:

```bash
python gaussian.py plot --dim 1     # -> results/1dgaussian.pdf, results/1dgaussian.json
python gaussian.py plot --dim 2     # -> results/2dgaussian.pdf, results/2dgaussian.json
```

Each command takes about 2 s on a CPU. It fits the LIP to `data/lip_responses_{dim}d.json`,
runs Uniform EM and LIP-EM, computes the Target Only and Pooled baselines, prints a summary and
writes the figure and the JSON. Use `--responses FILE` to fit another response file and
`--outdir DIR` to write elsewhere. With the package versions in `requirements.txt` the figures
render pixel-identical to the ones in the paper. Other matplotlib versions may differ slightly
in rendering.

Expected output (KL is KL(truth ‖ estimate), and the weights are for sources 0, 1, 2, where source 0 is the relevant one):

| | LIP prior `pi` | LIP-EM weights | LIP-EM iters | Uniform EM weights | Uniform EM iters | KL: LIP-EM / Uniform EM / Target Only / Pooled |
|---|---|---|---|---|---|---|
| 1-d | 0.150, 0.010, 0.0006 | 0.501, 0.0022, 6e-18 | 73 | 5e-11, 5e-20, 8e-66 | 19 | 0.020 / 0.969 / 0.969 / 0.157 |
| 2-d | 0.150, 0.010, 0.0006 | 0.991, 0.0005, 2e-12 | 37 | 2e-8, 3e-8, 7e-10 | 19 | 0.030 / 1.582 / 1.582 / 0.269 |

## Query the LLM again (optional)

```bash
export GEMINI_API_KEY=...            # your own key; it is read from the environment only
python gaussian.py query --dim 1     # 3 pairwise subgroup queries -> data/lip_responses_1d.json
python gaussian.py query --dim 2
```

Each command uploads the context file and the three source samples (4 decimals, one row per
point) and asks `gemini-3-flash-preview` (temperature 0) about every pair of sources. Every
attachment is preceded by a text label (`LLMSubgroupJudge(label_attachments=True)`), because the
model cannot otherwise tell bare columns of numbers apart from the report. A failed query aborts
without saving. The command overwrites the cached file, so pass `--responses FILE` to keep it.
LLM answers can change over time, so a new query may give a different `pi`.

## Setup details

* Data (seed 42 set with `np.random.seed`, then the target draw, then the sources in order):
  * 1-d: target N(2, 5²), and sources N(2, 5²) (relevant), N(0, 5²) and N(−5, 5²).
    Description: "The target domain has a larger mean."
  * 2-d: target N((55, 120), Σ₀) with Σ₀ = [[20, 5], [5, 20]], and sources N((55, 120), Σ₀) (relevant),
    N((51, 120), Σ₀) and N((49, 120), Σ₀). Description: "The target domain has a larger x mean."
  * N₀ = 4 target points, N_k = 200 points per source.
* LIP: `fit_lip` (conditional logit with an outside option, paper eq. lip_choice / LIP_opt) with
  p₀ = 0.01 and λ_LIP = 0.01. Uniform EM uses π_k = p₀ for every source.
* EM (paper Algorithm 1): the mean and (co)variance are estimated, and θ̂ are MLEs (covariance divides by n).
  The relevant likelihood is eq. rel_prox with τ = 10⁻³ and the Hessians frozen at the source MLEs.
  The empirical Bayes null is eq. eb-null with normalised weights (1 − w_j)/Σ_{i≠k}(1 − w_i).
  The M-step is closed-form, and the tempering is β_k(t) = (1 − e^{−νt})/ε_k with ν = 0.05 and
  ε_k² = Tr(H₀⁺ H_k) for the pseudo-inverse H₀⁺.
* Stopping: ‖w^(t) − w^(t−1)‖_∞ ≤ 10⁻³ for 5 consecutive iterations, counted only from
  t ≥ t₀ = ⌈ln 2 / ν⌉ = 14, with at most 1000 iterations.
* Baselines: Target Only is the MLE on the 4 target points, and Pooled is the MLE on the target and
  all source points.
