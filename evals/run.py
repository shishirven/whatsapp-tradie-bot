"""Score the golden set.

Offline (deterministic checks only, mock qualifier):

    python -m evals.run --offline

Full run (real Gemini replies plus the judge) when GEMINI_API_KEY is set:

    python -m evals.run
    python -m evals.run --repeats 3

The judge model is JUDGE_MODEL. It is separate from GEMINI_MODEL, which is
the reply model. Hard-rule checks never call the judge.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from evals.harness import print_table, run_eval


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Run the tradie-bot golden set.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Deterministic checks only. Do not call Gemini.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run each case this many times and flag verdicts that flip.",
    )
    parser.add_argument(
        "--results-dir",
        default="evals/results",
        help="Directory for the JSON and Markdown reports.",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    offline = bool(args.offline) or not os.getenv("GEMINI_API_KEY", "").strip()
    if not args.offline and offline:
        print("GEMINI_API_KEY is not set; running deterministic checks only.")
    report = run_eval(offline=offline, repeats=args.repeats, results_dir=args.results_dir)
    print_table(report)
    print(f"Wrote {args.results_dir}/latest.json and latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
