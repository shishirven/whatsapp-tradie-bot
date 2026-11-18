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
