"""Offline golden-set harness, repeats, and judge calibration."""

import json

from evals.calibrate import compare, load_labels
from evals.checks import CheckContext, evaluate
from evals.harness import LABELS_PATH, load_business, load_golden, render_markdown, summarise_repeats
from evals.judge import DEFAULT_JUDGE_MODEL, DEFAULT_REPLY_MODEL, parse_verdict, resolve_judge_model
from evals.run import main


def test_golden_set_has_the_thirty_owner_cases():
    golden = load_golden()
    assert [case["id"] for case in golden["cases"]] == list(range(1, 31))
    assert "call 000" in golden["global_rules"]
    assert "DIY" in golden["global_rules"]
    joined = json.dumps(golden)
    assert "{owner_name}" in joined
    assert "{price_range}" in joined
    assert "{callout_fee}" in joined


def test_shipped_human_labels_are_still_todo():
    labels = load_labels(LABELS_PATH)
    assert [item["id"] for item in labels] == list(range(1, 31))
    assert all(item["pass"] is None for item in labels)
    assert all(item["note"].strip().upper() == "TODO" for item in labels)


def test_judge_model_is_separate_from_the_reply_model(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert resolve_judge_model() == DEFAULT_JUDGE_MODEL
    assert DEFAULT_JUDGE_MODEL != DEFAULT_REPLY_MODEL
    assert "pro" in DEFAULT_JUDGE_MODEL
    assert "flash" in DEFAULT_REPLY_MODEL
    monkeypatch.setenv("JUDGE_MODEL", "gemini-eval-judge")
    assert resolve_judge_model() == "gemini-eval-judge"


def test_money_check_does_not_use_a_judge():
    _raw, business = load_business()
    checks = {
        item["name"]: item
        for item in evaluate(
            8,
            CheckContext(reply="That will be $9999 flat.", lead={}, business=business, slots=[]),
        )
    }
    assert checks["money_within_config"]["pass"] is False
    assert "9999" in checks["money_within_config"]["detail"]


def test_repeat_flag_marks_a_flipping_verdict_unstable():
    flipped = summarise_repeats(
        [
            {"deterministic_pass": True, "judge_pass": True, "pass": True, "reason": ""},
            {"deterministic_pass": True, "judge_pass": False, "pass": False, "reason": "judge failed"},
        ]
    )
    assert flipped["unstable"] is True
    assert flipped["pass"] is False
    assert "judge verdict flipped" in flipped["unstable_because"]

    steady = summarise_repeats(
        [
            {"deterministic_pass": False, "judge_pass": None, "pass": False, "reason": "no 000"},
            {"deterministic_pass": False, "judge_pass": None, "pass": False, "reason": "no 000"},
        ]
    )
    assert steady["unstable"] is False
    assert steady["pass"] is False


def test_calibration_reports_agreement_and_skips_todo_labels():
    report = {
        "cases": [
            {"id": 1, "unstable": False, "judge": {"pass": True, "reason": "matches the rule"}},
            {"id": 2, "unstable": True, "judge": {"pass": False, "reason": "missed 000"}},
            {"id": 3, "unstable": False, "judge": None},
        ]
    }
    labels = [
        {"id": 1, "pass": True, "note": "safety reply is fine"},
        {"id": 2, "pass": True, "note": "should have passed"},
        {"id": 3, "pass": None, "note": "TODO"},
    ]
    summary = compare(report, labels)
    assert summary["labelled"] == 2
    assert summary["unlabelled"] == [3]
    assert summary["agreed"] == 1
    assert summary["agreement_pct"] == 50.0
    assert summary["disagreements"][0]["id"] == 2
    assert summary["disagreements"][0]["unstable"] is True
    assert summary["missing_judge"] == []


def test_parse_verdict_reads_json_and_rejects_prose():
    assert parse_verdict('{"pass": false, "reason": "invented a price"}')["pass"] is False
    assert parse_verdict("I think it passed")["pass"] is False


def test_offline_run_scores_every_case_without_calling_the_judge(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def boom(**_kwargs):
        raise AssertionError("the judge ran during an offline eval")

    monkeypatch.setattr("evals.harness.grade", boom)
    code = main(["--offline", "--repeats", "1", "--results-dir", str(tmp_path)])
    assert code == 0
    saved = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert saved["mode"] == "offline"
    assert saved["judge_model"] is None
    assert saved["total"] == 30
    assert saved["reply_model"] == "mock"
    assert (tmp_path / "latest.md").is_file()
    assert "Pass rate:" in (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert render_markdown(saved)
    assert all(case["judge"] is None for case in saved["cases"])
    assert all(case["deterministic"]["checks"] for case in saved["cases"])
    # Guardrails the pipeline now enforces. The rest of the set is allowed to fail.
    required = {6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 28, 29, 30}
    failed = {case["id"] for case in saved["cases"] if not case["pass"]}
    assert required.isdisjoint(failed)
    assert saved["business"] == "Example Sparks Demo"
