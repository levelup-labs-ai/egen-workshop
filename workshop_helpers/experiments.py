"""Thin wrappers over the Arize SDK for datasets and experiments."""

import json
from datetime import datetime

import pandas as pd

from .evals import CASES
from .tools import CONSULTANTS

DATASET_NAME = "bench-reference-requests"


def _names(ids) -> str:
    return ", ".join(f"{CONSULTANTS[c]['name']} ({c})" for c in ids) or "-"


def reference_frame() -> pd.DataFrame:
    """One row per reference case, with the reference answer spelled out for annotators."""
    return pd.DataFrame([
        {
            "case_id": c["case_id"],
            "requested_by": c["requested_by"],
            "request": c["request"],
            "good_fits": _names(c["good_fits"]),
            "must_not_recommend": "\n".join(f"{CONSULTANTS[k]['name']} ({k}): {v}" for k, v in c["must_not_recommend"].items()) or "-",
            "caveats": "\n".join(f"{CONSULTANTS[k]['name']} ({k}): {'; '.join(v)}" for k, v in c["caveats"].items()) or "-",
        }
        for c in CASES.values()
    ])


def ensure_dataset(arize, space_id: str, name: str = DATASET_NAME) -> str:
    """Create the reference dataset in Arize once; reuse it on later runs."""
    existing = next((d for d in arize.datasets.list(space=space_id, name=name).datasets if d.name == name), None)
    if existing:
        return existing.id
    return arize.datasets.create(name=name, space=space_id, examples=reference_frame()).id


def _case_ids(arize, dataset_id: str) -> dict[str, str]:
    examples = arize.datasets.list_examples(dataset=dataset_id, all=True).examples
    return {e.id: e.additional_properties["case_id"] for e in examples}


def run_experiment(arize, space_id: str, dataset_id: str, name: str, task, evaluators=None, concurrency: int = 4):
    """Run an Arize experiment and return (experiment, results). Results get a case_id column."""
    experiment, results = arize.experiments.run(
        name=f"{name}-{datetime.now():%Y%m%d%H%M}",
        dataset=dataset_id,
        space=space_id,
        task=task,
        evaluators=evaluators or [],
        concurrency=concurrency,
    )
    results.insert(0, "case_id", results["example_id"].map(_case_ids(arize, dataset_id)))
    return experiment, results


def list_experiments(arize, dataset_id: str) -> list:
    """Experiments on the dataset, newest first."""
    # First page only (50 experiments); pass a cursor if a dataset ever collects more.
    return sorted(arize.experiments.list(dataset=dataset_id).experiments, key=lambda e: e.created_at, reverse=True)


def load_outputs(arize, experiment_id: str, dataset_id: str) -> pd.DataFrame:
    """case_id and the agent's output for each row of a past experiment."""
    ids = _case_ids(arize, dataset_id)
    runs = [r for r in arize.experiments.list_runs(experiment=experiment_id, all=True).experiment_runs if r.output]
    return pd.DataFrame({"case_id": [ids[r.example_id] for r in runs], "output": [r.output for r in runs]})


def outputs_by_case(results: pd.DataFrame) -> dict[str, str]:
    """Map case_id -> agent output, so we can score saved outputs without re-running the agent."""
    return dict(zip(results["case_id"], results["output"]))


def score_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Pass rate per evaluator, ignoring n/a rows."""
    rows = []
    for col in [c for c in results.columns if c.startswith("eval.") and c.endswith(".label")]:
        labels = results[col][results[col] != "n/a"]
        rows.append({"metric": col.split(".")[1], "scored": len(labels), "passed": int((labels == "pass").sum()),
                     "pass_rate": round((labels == "pass").mean(), 2) if len(labels) else None})
    return pd.DataFrame(rows)


def show_output(output) -> None:
    """Print one agent output in a readable form."""
    out = output if isinstance(output, dict) else json.loads(output)
    print("RECOMMENDED:")
    for p in out["recommended"] or [{"consultant_id": "-", "reason": "(no one)"}]:
        name = CONSULTANTS.get(p["consultant_id"], {}).get("name", "")
        print(f"  {p['consultant_id']} {name}: {p['reason']}")
    print("\nMESSAGE TO MANAGER:\n" + out["message"])
    print(f"\nTOOL CALLS ({len(out['tool_calls'])}):")
    for t in out["tool_calls"]:
        print(f"  {t['name']}({json.dumps(t['arguments'])})")
