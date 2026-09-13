"""
Fixes a real bug: the original eval_harness.py run never enabled the LLM
escalation fallback (defaults to False), so the reported escalation
precision/recall was pure rule-based, not the hybrid system.

This script recomputes ONLY the escalation decision with the fallback
enabled, reusing existing draft_reply and judge_score (which don't depend
on escalation logic, so no need to re-pay for generation or judging).

Cost: only rows where rules originally found nothing get an LLM fallback
call (~116 of 176 rows in the known run), plus one classify call per row
to get confidence (not saved in the original run — needed for the
confidence-threshold escalation rule).

Usage:
    python -m src.eval.recompute_escalation_hybrid
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eval_results" / "eval_run.csv"
OUTPUT_PATH = REPO_ROOT / "reports" / "eval_results" / "eval_run_hybrid.csv"


def main(args):
    from dotenv import load_dotenv
    load_dotenv()
    from src.classifier.fewshot_classifier import FewShotClassifier
    from src.retrieval.reply_retriever import ReplyRetriever
    from src.escalation.escalation_rules import decide_escalation

    df = pd.read_csv(INPUT_PATH)
    classifier = FewShotClassifier()
    retriever = ReplyRetriever(REPO_ROOT / "data" / "processed" / "retrieval_corpus.csv")

    updated_rows = []
    for i, row in df.iterrows():
        text = row["customer_text"]
        try:
            intent_result = classifier.classify(text)
            confidence = intent_result.get("confidence", 0.0)
            retrieved = retriever.retrieve(text, k=1)
            top_sim = retrieved[0]["similarity"] if retrieved else 0.0

            escalation_result = decide_escalation(
                text, classifier_confidence=confidence, top_retrieval_similarity=top_sim,
                use_llm_fallback=True,
            )
            row["pred_escalate"] = "yes" if escalation_result["escalate"] else "no"
            row["escalation_reasons"] = "; ".join(escalation_result["reasons"])
            row["escalation_method"] = escalation_result["method"]
        except Exception as e:
            row["escalation_method"] = f"ERROR: {e}"
        updated_rows.append(row)
        print(f"[{i+1}/{len(df)}] recomputed escalation for: {text[:55]}...")
        pd.DataFrame(updated_rows).to_csv(OUTPUT_PATH, index=False)

    result_df = pd.DataFrame(updated_rows)
    p, r, f1, _ = precision_recall_fscore_support(
        result_df["true_escalate"], result_df["pred_escalate"], pos_label="yes", average="binary", zero_division=0
    )
    print(f"\n=== HYBRID ESCALATION (rule + LLM fallback) ===")
    print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
    print(f"\nOriginal rule-only was: Precision=0.50, Recall=0.68, F1=0.58")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())