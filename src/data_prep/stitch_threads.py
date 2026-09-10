"""
constructs full multi-turn customer<->brand conversation threads from the raw Kaggle twcs.csv file, 
using the in_response_to_tweet_id chain.
"""

import argparse
import pandas as pd


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
    df = pd.read_csv(args.input, dtype=str)
    parent_map = build_parent_map(df)
    root_cache = {}

    brand_tweets = df[df["author_id"] == args.brand]
    sample_n = min(args.sample, len(brand_tweets))
    sampled_brand = brand_tweets.sample(n=sample_n, random_state=42)

    target_roots = set()
    for tid in sampled_brand["tweet_id"]:
        target_roots.add(find_root(tid, parent_map, root_cache))

    all_roots = df["tweet_id"].map(lambda t: find_root(t, parent_map, root_cache))
    mask = all_roots.isin(target_roots)

    result = df[mask].copy()
    result["conversation_root_id"] = all_roots[mask]

    # NOTE: broadcast tweets (e.g. general PSAs from the brand, not a reply to a specific customer) 
    # attract many unrelated replies, which incorrectly merge into one giant "conversation." 
    # Filter these out -- a real support conversation is a bounded back-and-forth, not a public announcement's reply thread.
    conv_sizes = result.groupby("conversation_root_id").size()
    real_conversations = conv_sizes[conv_sizes <= args.max_thread_size].index
    n_dropped = len(conv_sizes) - len(real_conversations)
    result = result[result["conversation_root_id"].isin(real_conversations)]

    result = result.sort_values(["conversation_root_id", "created_at"]).reset_index(drop=True)
    result.to_csv(args.output, index=False)

    n_conv = result["conversation_root_id"].nunique()
    print(f"Wrote {len(result)} rows across {n_conv} conversations -> {args.output}")
    print(f"Dropped {n_dropped} oversized conversations (likely broadcast-tweet noise, >{args.max_thread_size} tweets)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--brand", default="AppleSupport")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=3000,
                         help="number of brand tweets to sample before stitching")
    parser.add_argument("--max-thread-size", type=int, default=20,
                         help="drop conversations larger than this (broadcast-tweet noise)")
    main(parser.parse_args())