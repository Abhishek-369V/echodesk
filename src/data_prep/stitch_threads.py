"""
Full multi-turn customer<->brand conversation threads from the raw Kaggle twcs.csv file, 
using the in_response_to_tweet_id chain.
"""

import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "twcs.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "apple_threads_full.csv"


def build_parent_map(df: pd.DataFrame) -> dict:
    return dict(zip(df["tweet_id"], df["in_response_to_tweet_id"]))


def find_root(tid, parent_map, cache, max_hops=15):
    if tid in cache:
        return cache[tid]
    path = []
    cur = tid
    hops = 0
    while hops < max_hops:
        path.append(cur)
        parent = parent_map.get(cur)
        if pd.isna(parent) or parent not in parent_map:
            break
        cur = parent
        hops += 1
    root = cur
    for node in path:
        cache[node] = root
    return root


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Can't find {input_path}. Recheck file path correctly!."
        )

    df = pd.read_csv(input_path, dtype=str)
    parent_map = build_parent_map(df)
    root_cache = {}

    brand_tweets = df[df["author_id"] == args.brand]
    if len(brand_tweets) == 0:
        raise ValueError(f"No tweets found for brand '{args.brand}' -- check spelling/case.")

    sample_n = min(args.sample, len(brand_tweets))
    sampled_brand = brand_tweets.sample(n=sample_n, random_state=42)

    target_roots = set()
    for tid in sampled_brand["tweet_id"]:
        target_roots.add(find_root(tid, parent_map, root_cache))

    all_roots = df["tweet_id"].map(lambda t: find_root(t, parent_map, root_cache))
    mask = all_roots.isin(target_roots)

    result = df[mask].copy()
    result["conversation_root_id"] = all_roots[mask]

    conv_sizes = result.groupby("conversation_root_id").size()
    real_conversations = conv_sizes[conv_sizes <= args.max_thread_size].index
    n_dropped = len(conv_sizes) - len(real_conversations)
    result = result[result["conversation_root_id"].isin(real_conversations)]

    result = result.sort_values(["conversation_root_id", "created_at"]).reset_index(drop=True)
    result.to_csv(output_path, index=False)

    n_conv = result["conversation_root_id"].nunique()
    print(f"Wrote {len(result)} rows across {n_conv} conversations -> {output_path}")
    print(f"Dropped {n_dropped} oversized conversations (likely broadcast-tweet noise, >{args.max_thread_size} tweets)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--brand", default="AppleSupport")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--max-thread-size", type=int, default=20)
    main(parser.parse_args())