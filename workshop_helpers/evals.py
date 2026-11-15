"""Two code-based checks and an LLM judge, written as Arize experiment evaluators."""

import json
from pathlib import Path

from arize.experiments.evaluators.types import EvaluationResult

from .tools import CONSULTANTS, MODEL, get_availability

CASES = {c["case_id"]: c for c in json.loads((Path(__file__).parent / "data" / "reference_cases.json").read_text())}


def _parse(output) -> dict:
    return output if isinstance(output, dict) else json.loads(output)


def _picks(output) -> list[str]:
    return [p["consultant_id"] for p in _parse(output)["recommended"]]


def _na(reason: str) -> EvaluationResult:
    return EvaluationResult(score=None, label="n/a", explanation=reason)


def _result(passed: bool, explanation: str) -> EvaluationResult:
    return EvaluationResult(score=1.0 if passed else 0.0, label="pass" if passed else "fail", explanation=explanation)


# ---------- code-based checks ----------

def no_ineligible_pick(output, dataset_row) -> EvaluationResult:
    """Fail if a pick is on the reference must-not list, or is booked too heavily in the request window."""
    case = CASES[dataset_row["case_id"]]
    picks = _picks(output)
    bad = {cid: why for cid, why in case["must_not_recommend"].items() if cid in picks}
    if case["window"]:
        w, free_needed = case["window"], case["allocation_pct"]
        for cid in picks:
            busy = get_availability(cid, w["start"], w["end"]).get("assignments", [])
            if cid not in bad and sum(a["allocation_pct"] for a in busy) > 100 - free_needed:
                bad[cid] = "; ".join(f"{a['allocation_pct']}% on {a['client']} {a['project']} until {a['end']}" for a in busy)
    if not bad:
        return _result(True, "No ineligible consultants recommended.")
    return _result(False, "; ".join(f"{CONSULTANTS[c]['name']} ({c}): {why}" for c, why in bad.items()))


def found_good_fit(output, dataset_row) -> EvaluationResult:
    """Pass if at least one recommended consultant is on the reference good-fit list."""
    case = CASES[dataset_row["case_id"]]
    if case["expected_behavior"] == "ask_clarifying":
        return _na("No reference shortlist for this request.")
    picks = _picks(output)
    if case["expected_behavior"] == "no_fit":
        return _result(not picks, "Correctly recommended no one." if not picks else f"Nobody fits, but recommended {picks}.")
    hits = [c for c in picks if c in case["good_fits"]]
    if hits:
        return _result(True, f"Good fits recommended: {hits}")
    names = ", ".join(f"{CONSULTANTS[c]['name']} ({c})" for c in case["good_fits"])
    return _result(False, f"Recommended {picks or 'no one'}; good fits were {names}.")


# ---------- LLM judge ----------

def _judge(client, prompt: str, payload: dict, model: str) -> EvaluationResult:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    verdict = json.loads(response.choices[0].message.content)
    return _result(str(verdict.get("label", "")).lower() == "pass", verdict.get("reason", ""))


def make_grounded_judge(client, prompt: str, model: str = MODEL):
    """Judge whether every claim about a consultant in the message is backed by tool results."""

    def grounded(output, dataset_row) -> EvaluationResult:
        out = _parse(output)
        if not out["recommended"]:
            return _na("No one recommended.")
        payload = {
            "request": CASES[dataset_row["case_id"]]["request"],
            "message_to_manager": out["message"],
            "tool_results": [{"tool": t["name"], "arguments": t["arguments"], "output": t["output"]} for t in out["tool_calls"]],
        }
        return _judge(client, prompt, payload, model)

    return grounded
