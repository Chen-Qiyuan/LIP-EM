"""Target descriptions of the C-MAPSS prompt ablation and their cached responses.

Every response file holds 200 single-choice answers to the same subgroups (in
the same order) as the cached LIP-C of Table 1; only the target description
(`target_desc` in the file) differs. Files under responses/opus/ were elicited
with Claude Opus 4.7, those under responses/cli_default/ with the default Opus
model of the Claude Code command-line interface (see elicit.py).
"""
from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
CMAPSS_DATA = HERE.parents[1] / "cmapss" / "data"
RESPONSES = HERE / "responses"

CACHED = "cached"
# key -> label; the response file of a key is responses/<key>.json.
LABELS = {
    # Printed by analyze.py. Table tab:prompt shows cached and opus/no_diagnostics to
    # opus/no_information, and opus/original and opus/contradictory are additional checks.
    "opus/original": "Original (re-elicited)",
    "opus/no_diagnostics": "No-diagnostics",
    "opus/desert_only": "Desert only",
    "opus/no_desert": "No-desert wording",
    "opus/inspection_wording": "Alt. inspection wording",
    "opus/no_information": "No-information control",
    "opus/contradictory": "Contradictory direction",
    # Further Opus 4.7 descriptions (additional checks, not reported in the paper).
    "opus/no_desert_findings": "No-desert wording (findings)",
    "opus/no_desert_routine": "No-desert wording (routine)",
    "opus/desert_operations": "Paraphrase: desert operations",
    "opus/maintenance_records": "Paraphrase: maintenance records",
    "opus/maintenance_records_detailed": "Paraphrase: maintenance records, long",
    "opus/mechanism_direct": "Mechanism-direct",
    # The same protocol with the default Opus model of the Claude Code CLI.
    "cli_default/original": "CLI: original",
    "cli_default/maintenance_records": "CLI: maintenance records",
    "cli_default/mechanism_direct": "CLI: mechanism-direct",
    "cli_default/no_information": "CLI: no information",
    "cli_default/contradictory": "CLI: contradictory direction",
}
# key -> (label, response file); the first entry is the reference.
CONDITIONS = {CACHED: ("Original (cached LIP-C of Table 1)",
                       CMAPSS_DATA / "llm_subgroup_queries_claude.json")}
CONDITIONS.update({key: (label, RESPONSES / f"{key}.json")
                   for key, label in LABELS.items()})

TABLE_KEYS = [CACHED, "opus/original", "opus/no_diagnostics",
              "opus/desert_only", "opus/no_desert", "opus/inspection_wording",
              "opus/no_information", "opus/contradictory"]
# The four informative alternative descriptions of Table tab:prompt.
INFORMATIVE_KEYS = TABLE_KEYS[2:6]
