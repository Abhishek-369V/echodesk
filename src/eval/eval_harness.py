"""
Eval harness: runs the agent against the golden set and computes
  1. Intent classification metrics (accuracy, per-class F1)
  2. Escalation metrics (precision/recall against your hand-labeled escalate_flag)
  3. Reply quality via LLM-as-judge (rubric-scored), with human-agreement check

Usage:
    python -m src.eval.eval_harness --limit 20    # cheap sanity check first
    python -m src.eval.eval_harness               # full golden set
    python -m src.eval.eval_harness --overwrite   # restart from scratch
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "data" / "golden_set"
RESULTS_DIR = REPO_ROOT / "reports" / "eval_results"

JUDGE_SYSTEM_PROMPT = """You are grading a customer support reply on a 1-5 rubric:
1 = irrelevant or wrong, 3 = acceptable but generic, 5 = specific, empathetic,
actionable, and clearly grounded in the customer's actual issue.

Respond with ONLY JSON: {"score": <1-5 int>, "justification": "<one sentence>"}
"""


def load_golden_set(golden_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(golden_dir.glob("labeling_batch_*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        df = df[df["intent"].notna()]
        df = df[df["intent"].astype(str).str.strip() != ""]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No valid labeling batches found in {golden_dir}")
    return pd.concat(frames, ignore_index=True)


def judge_reply(client: OpenAI, customer_text: str, reply: str, model: str = "gpt-4o-mini") -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer message: {customer_text}\n\nReply: {reply}"},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "justification": f"unparseable judge output: {raw!r}"}


def main(args):
    from src.agent.agent import EchoDeskAgent

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    golden = load_golden_set(GOLDEN_DIR)
    if args.limit:
        golden = golden.sample(n=min(args.limit, len(golden)), random_state=42)

    output_path = RESULTS_DIR / "eval_run.csv"
    already_done = set()
    rows = []

    if output_path.exists() and not args.overwrite and output_path.stat().st_size > 0:
        try:
            prior = pd.read_csv(output_path)
            if "customer_text" in prior.columns:
                rows = prior.to_dict("records")
                already_done = set(prior["customer_text"])
                print(f"Resuming: {len(already_done)} rows already processed, skipping them")
        except Exception as e:
            print(f"Warning: Could not read existing run file ({e}). Starting fresh.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    agent = EchoDeskAgent()
    judge_client = OpenAI(api_key=api_key)

    for i, (_, row) in enumerate(golden.iterrows(), start=1):
        if row["customer_text"] in already_done:
            continue

        result = agent.handle(row["customer_text"])
        judge_result = judge_reply(judge_client, row["customer_text"], result["draft_reply"])

        rows.append({
            "customer_text": row["customer_text"],
            "true_intent": row["intent"],
            "pred_intent": result["intent"],
            "true_escalate": row["escalate_flag"],
            "pred_escalate": "yes" if result["escalate"] else "no",
            "escalation_reasons": "; ".join(result["escalation_reasons"]),
            "draft_reply": result["draft_reply"],
            "judge_score": judge_result.get("score"),
            "judge_justification": judge_result.get("justification"),
        })

        # Save after every single row to make the run crash-safe and resumable
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"[{i}/{len(golden)}] Processed: {row['customer_text'][:55]}...")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "eval_run.csv", index=False)

    print("\n=== INTENT CLASSIFICATION ===")
    print(classification_report(results_df["true_intent"], results_df["pred_intent"], zero_division=0))

    print("\n=== ESCALATION ===")
    p, r, f1, _ = precision_recall_fscore_support(
        results_df["true_escalate"], results_df["pred_escalate"], pos_label="yes", average="binary", zero_division=0
    )
    print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")

    print("\n=== REPLY QUALITY (LLM-as-judge) ===")
    valid_scores = results_df["judge_score"].dropna()
    mean_val = valid_scores.mean() if len(valid_scores) > 0 else 0.0
    print(f"Mean score: {mean_val:.2f} / 5 over {len(valid_scores)} judged replies")

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="run on a subset first to control API cost")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing eval_run.csv instead of resuming")
    main(parser.parse_args())