"""The regression set runs offline in CI: every conversation must pass."""

import pytest

from evals.regression import CATEGORIES, CHECKS, is_scripted, load_business, load_cases, run_case

CASES = load_cases()


def test_every_category_has_cases_and_every_case_has_a_reason():
    assert {case["category"] for case in CASES} == set(CATEGORIES)
    assert all(case["why"].strip() for case in CASES)
    assert len(CASES) >= 50


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_regression_case_offline(case):
    result = run_case(case, mode="offline", business=load_business())
    failures = [
        f"turn {turn['turn']} {turn['said']!r}: {failure} | reply: {turn['reply'][:160]}"
        for turn in result["turns"]
        for failure in turn["failures"]
    ]
    assert result["passed"], "\n".join(failures)


def test_unknown_checks_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "cases:\n  - id: X1\n    category: wording\n    why: test\n    turns:\n      - say: hi\n        expect: [made_up_check]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown check"):
        load_cases(bad)


def test_replayed_cases_are_marked():
    assert any(is_scripted(case) for case in CASES)
    assert "stores_nothing" in CHECKS


def test_live_sample_is_ten_cases_across_categories_and_repeatable():
    from evals.regression import LIVE_SAMPLE, sample_live

    first = sample_live(CASES, LIVE_SAMPLE, seed=7)
    assert LIVE_SAMPLE == 10 and len(first) == 10
    assert not any(is_scripted(case) for case in first)  # replayed cases never call a model
    live_categories = {case["category"] for case in CASES if not is_scripted(case)}
    assert {case["category"] for case in first} == live_categories  # every behaviour checked
    assert [case["id"] for case in first] == [case["id"] for case in sample_live(CASES, 10, seed=7)]
    assert len(sample_live(CASES, 0, seed=7)) == sum(not is_scripted(case) for case in CASES)


def test_live_run_only_calls_the_model_for_the_sample(monkeypatch, tmp_path):
    import evals.regression as regression

    calls = []

    def fake_run_case(case, *, mode, **kwargs):
        calls.append((case["id"], mode))
        return {"passed": True, "fallback": False, "turns": []}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(regression, "run_case", fake_run_case)
    report = regression.run(results_dir=tmp_path, seed=3)
    live = [case_id for case_id, mode in calls if mode == "live"]
    offline = [case_id for case_id, mode in calls if mode == "offline"]
    assert len(live) == 10 and sorted(live) == report["live_sample"]
    assert len(offline) == len(CASES)  # offline still covers everything
    assert report["seed"] == 3
    assert "10 sampled, seed 3" in (tmp_path / "regression_latest.md").read_text()
