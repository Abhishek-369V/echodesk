"""
Validates the LLM-judge against real human ratings 
-- as the assignment explicitly requires "evidence of how well your judge agrees with a human,"
not just running the judge and trusting it blindly.

Workflow:
    1. Run eval_harness.py first (produces reports/eval_results/eval_run.csv)
    2. Run this script with --sample-for-rating to pull ~30 random rows into
       a CSV with a blank human_score column
    3. I manually fill in human_score (1-5) for each row, same rubric as the judge
    4. Run this script with --compare to compute agreement (Cohen's kappa + %
       exact match) between your human_score and judge_score

Usage:
    python -m src.eval.check_judge_agreement --sample-for-rating --n 30
    # ... fill in human scores ...
    python -m src.eval.check_judge_agreement --compare
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS = REPO_ROOT / "reports" / "eval_results" / "eval_run.csv"
RATING_FILE = REPO_ROOT / "reports" / "eval_results" / "human_rating_sample.csv"


def sample_for_rating(n: int, seed: int = 42):
    if not EVAL_RESULTS.exists():
        raise FileNotFoundError(f"Run eval_harness.py first — {EVAL_RESULTS} not found.")
    df = pd.read_csv(EVAL_RESULTS)
    sample = df.sample(n=min(n, len(df)), random_state=seed)[
        ["customer_text", "draft_reply", "judge_score", "judge_justification"]
    ].copy()
    sample["human_score"] = ""
    sample.to_csv(RATING_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(sample)} rows to {RATING_FILE}")
    print("Fill in human_score (1-5) for each row using the same rubric as the judge, "
          "then rerun with --compare")


def compare():
    if not RATING_FILE.exists():
        raise FileNotFoundError(f"Run --sample-for-rating first — {RATING_FILE} not found.")
    df = pd.read_csv(RATING_FILE, encoding="utf-8-sig")
    df = df[df["human_score"].notna() & (df["human_score"].astype(str).str.strip() != "")]
    if len(df) == 0:
        raise ValueError("No human_score values filled in yet.")

    df["human_score"] = df["human_score"].astype(int)
    df["judge_score"] = df["judge_score"].astype(int)

    exact_match = (df["human_score"] == df["judge_score"]).mean()
    within_1 = (abs(df["human_score"] - df["judge_score"]) <= 1).mean()
    kappa = cohen_kappa_score(df["human_score"], df["judge_score"], weights="linear")

    print(f"Rated examples: {len(df)}")
    print(f"Exact match: {exact_match:.0%}")
    print(f"Within 1 point: {within_1:.0%}")
    print(f"Cohen's kappa (linear weighted): {kappa:.2f}")
    print("\nGuide: kappa > 0.6 = substantial agreement, 0.4-0.6 = moderate, < 0.4 = weak")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-for-rating", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    if args.sample_for_rating:
        sample_for_rating(args.n)
    elif args.compare:
        compare()
    else:
        parser.error("Pass either --sample-for-rating or --compare")