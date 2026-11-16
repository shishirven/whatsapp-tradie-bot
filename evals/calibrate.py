"""Compare the LLM judge with human labels.

    python -m evals.calibrate
    python -m evals.calibrate --results evals/results/latest.json

Labels live in evals/human_labels.yaml. A case with pass: null (or a TODO note
and no boolean) is unlabelled. Unlabelled cases are not disagreements.
Agreement is the share of labelled cases whose judge verdict matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evals.harness import LABELS_PATH

_TRUE = {"true", "yes", "pass"}
_FALSE = {"false", "no", "fail"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare judge verdicts with human labels.")
    parser.add_argument(
        "--results",
        default="evals/results/latest.json",
        help="Eval JSON that includes judge verdicts.",
    )
    parser.add_argument(
        "--labels",
        default=str(LABELS_PATH),
        help="Human label file.",
    )
    args = parser.parse_args(argv)
    results_path = Path(args.results)
    labels_path = Path(args.labels)
    if not results_path.is_file():
        print(f"No results at {results_path}. Run python -m evals.run first.")
        return 1
    report = json.loads(results_path.read_text(encoding="utf-8"))
    labels = load_labels(labels_path)
    summary = compare(report, labels)
    print_calibration(summary)
    if summary["labelled"] == 0:
        return 0
    if summary["missing_judge"] or summary["disagreements"]:
        return 1
    return 0


def load_labels(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = []
    for item in data.get("cases") or []:
        labels.append(
            {
                "id": int(item["id"]),
                "pass": item.get("pass"),
                "note": "" if item.get("note") is None else str(item.get("note")),
            }
        )
    return labels


def compare(report: dict, labels: list[dict]) -> dict:
    by_id = {int(case["id"]): case for case in report.get("cases") or []}
    labelled = []
    unlabelled = []
    for item in labels:
        human = _human_pass(item)
        if human is None:
            unlabelled.append(item["id"])
            continue
        labelled.append((item, human))
    agreements = []
    disagreements = []
    missing_judge = []
    for item, human in labelled:
        case = by_id.get(item["id"])
        verdict = _judge_pass(case)
        row = {
            "id": item["id"],
            "human": human,
            "judge": verdict,
            "note": item.get("note") or "",
            "judge_reason": _judge_reason(case),
            "unstable": bool(case and case.get("unstable")),
        }
        if verdict is None:
            missing_judge.append(row)
            continue
        if verdict == human:
            agreements.append(row)
        else:
            disagreements.append(row)
    compared = len(agreements) + len(disagreements)
    agreement = round(100 * len(agreements) / compared, 1) if compared else None
    return {
        "labelled": len(labelled),
        "unlabelled": unlabelled,
        "compared": compared,
        "agreed": len(agreements),
        "agreement_pct": agreement,
        "disagreements": disagreements,
        "missing_judge": missing_judge,
    }


def print_calibration(summary: dict) -> None:
    print(f"Human labels: {summary['labelled']} labelled, {len(summary['unlabelled'])} unlabelled")
    if summary["agreement_pct"] is None:
        print("Agreement: n/a")
    else:
        print(
            f"Agreement: {summary['agreement_pct']}% "
            f"({summary['agreed']}/{summary['compared']})"
        )
    if summary["disagreements"]:
        print("Disagreements:")
        for row in summary["disagreements"]:
            human = "pass" if row["human"] else "fail"
            judge = "pass" if row["judge"] else "fail"
            unstable = " unstable" if row["unstable"] else ""
            note = f" — {row['note']}" if row["note"] and row["note"].strip().upper() != "TODO" else ""
            reason = f" — judge: {row['judge_reason']}" if row["judge_reason"] else ""
            print(f"- {row['id']}: human {human}, judge {judge}{unstable}{note}{reason}")
    else:
        print("Disagreements: none")
    if summary["missing_judge"]:
        ids = ", ".join(str(row["id"]) for row in summary["missing_judge"])
        print(f"No judge verdict for labelled cases: {ids}")
    if summary["unlabelled"]:
        print("Unlabelled: " + ", ".join(str(item) for item in summary["unlabelled"]))


def _human_pass(item: dict) -> bool | None:
    value = _as_bool(item.get("pass"))
    if value is None:
        return None
    return value


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE:
            return True
        if token in _FALSE:
            return False
    return None


def _judge_pass(case: dict | None) -> bool | None:
    if not case:
        return None
    judge = case.get("judge")
    if not isinstance(judge, dict):
        return None
    return _as_bool(judge.get("pass"))


def _judge_reason(case: dict | None) -> str:
    if not case:
        return ""
    judge = case.get("judge")
    if not isinstance(judge, dict):
        return ""
    return str(judge.get("reason") or "")


if __name__ == "__main__":
    raise SystemExit(main())
