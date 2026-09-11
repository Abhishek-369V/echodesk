"""
Builds a labeling batch CSV from the stitched thread data -- samples one customer opening message per conversation 
for manual intent/escalation labeling. Automatically excludes conversation IDs already used in prior batches.
"""

import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "apple_threads_full.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "golden_set"


def get_already_used_ids(golden_dir: Path) -> set:
    used = set()
    for f in golden_dir.glob("labeling_batch_*.csv"):
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            used.update(df["conversation_root_id"].tolist())
        except Exception as e:
            print(f"Warning: couldn't read {f} for dedup ({e}), skipping it")
    return used


def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Can't find {input_path}. Run stitch_threads.py first.")

    df = pd.read_csv(input_path)
    inbound = df[df["inbound"] == True].copy()

    # first customer message per conversation = the opening complaint
    first_msgs = (
        inbound.sort_values("created_at")
        .groupby("conversation_root_id")
        .first()
        .reset_index()
    )
    first_msgs = first_msgs[
        first_msgs["text"].str.contains(r"[a-zA-Z]{4,}", regex=True, na=False)
    ]

    used_ids = get_already_used_ids(output_dir)
    remaining = first_msgs[~first_msgs["conversation_root_id"].isin(used_ids)]
    print(f"{len(used_ids)} conversation IDs already used across prior batches, "
          f"{len(remaining)} remaining to sample from")

    n = min(args.n, len(remaining))
    sample = remaining.sample(n=n, random_state=args.seed)

    label_df = sample[["conversation_root_id", "tweet_id", "text"]].rename(
        columns={"text": "customer_text"}
    )
    label_df["intent"] = ""
    label_df["escalate_flag"] = ""
    label_df["notes"] = ""

    output_path = output_dir / f"labeling_{args.batch_name}.csv"
    label_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(label_df)} rows -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-name", default="batch_2")
    main(parser.parse_args())