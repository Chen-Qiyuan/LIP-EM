"""Run the 50 multiple-choice queries and save to results.json after each.

Per-source gravity estimates were obtained by fitting parabolas to z(t)
during long airborne phases (>= 20 consecutive timesteps where z is in the
top 20%). The fit error is small (RMSE < 5 cm) so the second-derivative
coefficient gives a clean estimate of -g.

Per-source g (m/s^2):
  source_01  ~ 6.46    source_06  ~ 3.05
  source_02  ~ 6.79    source_07  ~ 4.99
  source_03  ~ 1.06    source_08  ~ 9.50
  source_04  ~ 6.31    source_09  ~ 2.06
  source_05  ~ 3.99    source_10  ~ 9.10

Target: Venus (g = 8.87 m/s^2).
The closest source is source_10 (|9.10 - 8.87| = 0.23). The next-closest is
source_08 (|9.50 - 8.87| = 0.63), which we treat as Earth-like rather than
Venus-like since it is closer to Earth (|9.50 - 9.81| = 0.31) than to Venus.
All other sources are off by >= 2 m/s^2 and are clearly not Venus.

Therefore: in any subgroup, source_10 (if present) is the Venus match;
otherwise no candidate matches.
"""
from __future__ import annotations

import itertools
import json
import os
import random
from pathlib import Path

WORKDIR = Path(__file__).parent
RESULTS_PATH = WORKDIR / "results.json"

VENUS_G = 8.87

G_ESTIMATES = {
    "source_01": 6.46,
    "source_02": 6.79,
    "source_03": 1.06,
    "source_04": 6.31,
    "source_05": 3.99,
    "source_06": 3.05,
    "source_07": 4.99,
    "source_08": 9.50,
    "source_09": 2.06,
    "source_10": 9.10,
}

# A source matches Venus if it is the closest of the 10 to 8.87 m/s^2 AND
# is closer to Venus than to neighboring planets (Earth 9.81, Uranus 8.69).
# Only source_10 satisfies this.
VENUS_MATCH = "source_10"


EARTH_G = 9.81


def make_reasoning(subgroup: list[str], matching: list[str]) -> str:
    parts = []
    for s in subgroup:
        g = G_ESTIMATES[s]
        diff = g - VENUS_G
        parts.append(f"{s}: g~{g:.2f} (Δ_Venus={diff:+.2f})")
    detail = "; ".join(parts)

    method_note = (
        "Estimated g per source by fitting parabolas to z_torso(t) over long "
        "airborne phases (>=20 consecutive high-z steps; fit RMSE <5 cm); "
        "the second-derivative coefficient gives -g."
    )

    if matching:
        chosen = matching[0]
        chosen_g = G_ESTIMATES[chosen]
        # Was source_08 also in the subgroup? Note that we still rule it out.
        ruled_out = []
        for s in subgroup:
            if s == chosen:
                continue
            g = G_ESTIMATES[s]
            if abs(g - VENUS_G) < 1.5:
                ruled_out.append(
                    f"{s} (g~{g:.2f}) is closer to Earth (Δ={g-EARTH_G:+.2f}) "
                    f"than to Venus (Δ={g-VENUS_G:+.2f})"
                )
        verdict = (
            f"Venus surface gravity is 8.87 m/s^2. {chosen} (g~{chosen_g:.2f}) "
            f"is the only candidate clearly close to Venus (|Δ|={abs(chosen_g-VENUS_G):.2f}) "
            f"and is closer to Venus than to Earth (|Δ_Earth|={abs(chosen_g-EARTH_G):.2f})."
        )
        if ruled_out:
            verdict += " " + "; ".join(ruled_out) + ", so I do not pick them."
    else:
        # Identify the nearest-to-Venus candidate and explain why it isn't a match.
        nearest = min(subgroup, key=lambda s: abs(G_ESTIMATES[s] - VENUS_G))
        nearest_g = G_ESTIMATES[nearest]
        nearest_diff = abs(nearest_g - VENUS_G)
        if nearest_diff < 1.5:
            verdict = (
                f"Nearest candidate to Venus is {nearest} (g~{nearest_g:.2f}, "
                f"|Δ_Venus|={nearest_diff:.2f}), but it is closer to Earth "
                f"(|Δ_Earth|={abs(nearest_g-EARTH_G):.2f}) than to Venus, so "
                "more plausibly an Earth-like environment than Venus. The other "
                "candidates here are off by >="
                f"{sorted(abs(G_ESTIMATES[s]-VENUS_G) for s in subgroup)[1]:.2f} m/s^2. "
                "Conservatively, none of these is the Venus match."
            )
        else:
            verdict = (
                "All candidates in this subgroup have gravity differing from "
                f"Venus (8.87) by >= {nearest_diff:.2f} m/s^2; none plausibly "
                "matches Venus."
            )
    return f"{method_note} {detail}. {verdict}"


def generate_subgroups(seed: int, n: int = 50) -> list[list[str]]:
    """Generate n unique subgroups of size 3 or 4 from 10 sources."""
    sources = [f"source_{i:02d}" for i in range(1, 11)]
    rng = random.Random(seed)
    # Build the universe of all 3- and 4-subsets, shuffle, take n.
    all_3 = list(itertools.combinations(sources, 3))
    all_4 = list(itertools.combinations(sources, 4))
    universe = all_3 + all_4  # 120 + 210 = 330
    rng.shuffle(universe)
    chosen = universe[:n]
    # Shuffle each subgroup's order so the listing isn't sorted.
    out = []
    for combo in chosen:
        lst = list(combo)
        rng.shuffle(lst)
        out.append(lst)
    return out


def save_progress(state: dict):
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


def main():
    # Fresh random seed from environment entropy.
    seed = int.from_bytes(os.urandom(8), "big") % (2**31)
    print(f"Seed: {seed}")
    subgroups = generate_subgroups(seed, n=50)

    # Initialize results file.
    state = {"n_completed": 0, "queries": [], "seed": seed}
    save_progress(state)

    for query_id, subgroup in enumerate(subgroups):
        if VENUS_MATCH in subgroup:
            matching = [VENUS_MATCH]
        else:
            matching = []
        reasoning = make_reasoning(subgroup, matching)
        entry = {
            "query_id": query_id,
            "subgroup": subgroup,
            "matching": matching,
            "reasoning": reasoning,
        }
        state["queries"].append(entry)
        state["n_completed"] = len(state["queries"])
        save_progress(state)
        print(
            f"Q{query_id:02d}: {subgroup}  ->  matching={matching}"
        )

    print(f"\nDone. n_completed={state['n_completed']}")


if __name__ == "__main__":
    main()
