"""Sanity check: reference answers agree with what the tools return, and the code evals score as expected.

Run: python scripts/check_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workshop_helpers import tools  # noqa: E402
from workshop_helpers.evals import CASES, found_good_fit, no_ineligible_pick  # noqa: E402

for case in CASES.values():
    if not case["window"]:
        continue
    w, need = case["window"], case["allocation_pct"]
    for cid in case["good_fits"]:
        avail = tools.get_availability(cid, w["start"], w["end"])
        booked = sum(a["allocation_pct"] for a in avail["assignments"])
        assert booked <= 100 - need, f"{case['case_id']}: good fit {cid} is {booked}% booked"
    for cid, why in case["must_not_recommend"].items():
        if "% on " in why:
            booked = sum(a["allocation_pct"] for a in tools.get_availability(cid, w["start"], w["end"])["assignments"])
            assert booked > 100 - need, f"{case['case_id']}: must-not {cid} is only {booked}% booked"

assert tools.search_consultants(["GCP"])["count"] == 3  # exact tags only: BigQuery people don't match "GCP"
assert tools.get_project("Northwind claims migration")["projects"][0]["id"] == "P01"

# Code evals on fake outputs for S01 (good fits C03/C13, Priya C01 must not be picked).
row = {"case_id": "S01"}
fake = lambda picks: json.dumps({"recommended": [{"consultant_id": c, "reason": ""} for c in picks], "message": "", "tool_calls": []})
assert no_ineligible_pick(fake(["C03"]), row).label == "pass"
assert no_ineligible_pick(fake(["C03", "C01"]), row).label == "fail"
assert found_good_fit(fake(["C13"]), row).label == "pass"
assert found_good_fit(fake(["C04"]), row).label == "fail"
assert found_good_fit(fake([]), {"case_id": "S09"}).label == "pass"  # no-fit case
# Booked picks fail even when the reference doesn't list them (Daniel is 100% booked; S01 needs full-time).
assert no_ineligible_pick(fake(["C03", "C07"]), row).label == "fail"

print(f"OK: {len(CASES)} reference cases consistent with tools; code evals behave.")
