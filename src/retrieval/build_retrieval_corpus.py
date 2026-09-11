"""
Builds the (customer_text, apple_reply_text) pair corpus used for RAG retrieval, 
derived from the CLEANED multi-turn processed data — not the raw Day-0 sample, which still contains broadcast-tweet noise.
"""

import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "apple_threads_full.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "retrieval_corpus.csv"


def main(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Can't find {input_path}. Run stitch_threads.py first.")

    df = pd.read_csv(input_path)
    df = df.sort_values(["conversation_root_id", "created_at"])

    pairs = []
    for conv_id, group in df.groupby("conversation_root_id"):
        group = group.reset_index(drop=True)
        first_customer = group[group["inbound"] == True]
        first_apple = group[group["inbound"] == False]
        if len(first_customer) == 0 or len(first_apple) == 0:
            continue
        pairs.append({
            "conversation_root_id": conv_id,
            "customer_text": first_customer.iloc[0]["text"],
            "apple_reply_text": first_apple.iloc[0]["text"],
        })

    result = pd.DataFrame(pairs)
    result = result.dropna(subset=["customer_text", "apple_reply_text"])
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} pairs -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    main(parser.parse_args())