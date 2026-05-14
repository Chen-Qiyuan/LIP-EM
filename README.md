# LIP-aided EM correction — supplementary code

Code and trained artifacts for two case studies in the LIP-aided EM correction paper (anonymized for double-blind review):

| Folder | Case study | Backbone |
|---|---|---|
| [`Hopper_Gravity/`](Hopper_Gravity/) | Offline-RL negative-transfer correction on Hopper-v5 with custom gravity | NN dynamics + IQL |
| [`CMAPSS/`](CMAPSS/) | Remaining-Useful-Life prediction on the NASA C-MAPSS turbofan dataset | Natural-cubic-spline GLM |

Each subdirectory has its own `README.md` with full reproduction instructions, a pinned `requirements.txt`, the code, the LLM-elicitation context, the cached LLM responses, the trained model checkpoints, and the headline results table.

## Data

The Hopper offline RL datasets (10 source replay buffers + 1 target replay buffer, 1.15 GB total HDF5) are **not in this git repo** because each file exceeds GitHub's 100 MB per-file limit. They will be hosted separately via an anonymous deposit (e.g. Zenodo) and linked from the camera-ready version. For now, reviewers can either:

1. Regenerate the datasets from scratch using `Hopper_Gravity/src/collect_target_sac.py` for the target and analogous SAC runs at integer gravities g=1..10 for the sources (~5 hours total), or
2. Skip data-dependent stages and verify against the bundled IQL outputs in `Hopper_Gravity/results/iql/` — `Hopper_Gravity/src/aggregate_results.py` rebuilds the headline table directly from those JSONs.

The C-MAPSS `Physical_Core_Speed_Sequences/` CSV files are small (~1 MB total) and are committed.

## License
Released under the same license as the eventual paper — TBD; placeholder MIT for the review period.
