# Dataset Recon

**Source:** Kaggle — Customer Support on Twitter (thoughtvector), `twcs.csv`
Total rows: 3,002,523 tweets

## Brand volume (top by outbound tweet count)
| Brand | Outbound tweets |
|---|---|
| AmazonHelp | 169,840 |
| **AppleSupport** | **106,860** |
| Uber_Support | 56,270 |
| SpotifyCares | 43,265 |
| Delta | 42,253 |
| Tesco | 38,573 |
| AmericanAir | 36,764 |

## Brand chosen: AppleSupport
Reasoning:
- Second largest brand by volume — enough data without needing the full firehose
- More varied issue types than AmazonHelp (which skews heavily toward generic
  "where's my order" logistics questions with little intent diversity)
- Well-documented in public work using this dataset — useful for sanity-checking
  the pipeline against known patterns

## Stitching approach
For each AppleSupport outbound tweet, walked back via `in_response_to_tweet_id`
to find the originating customer (inbound) tweet. This gives first-turn
customer→AppleSupport thread pairs (not full multi-turn conversations yet —
that's a Day 1 task if we need full threads for the reply generator's context).

Sampled 8,000 AppleSupport outbound tweets → 7,982 successfully stitched to a
customer root tweet (18 dropped: no resolvable parent or parent was also outbound).

Output: `apple_threads_sample_8k.csv`
Columns: apple_tweet_id, customer_tweet_id, customer_text, apple_reply_text, created_at
