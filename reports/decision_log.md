# Decision Log

Non-obvious calls made during this project, and why.

## Day 0
1. **Brand: AppleSupport, not AmazonHelp.** AmazonHelp has more volume (169K vs 106K
   outbound tweets) but skews heavily toward generic order-status questions — low
   intent diversity. AppleSupport gives a richer intent taxonomy and more interesting
   escalation calls.
2. **First-turn stitching only (for now).** Day 0 stitching captures
   customer→AppleSupport first-reply pairs, not full multi-turn threads. Decided to
   defer full thread reconstruction until we know whether the reply generator
   actually needs multi-turn context, to avoid overbuilding early.
3. **`escalation_repeat` may not be a true 6th intent** — it might be a cross-cutting
   flag instead of a category. Deferred decision to Day 1-2 hand-labeling, where
   we'll see how often it co-occurs with other intents vs. stands alone.

<!-- Add entries as the project progresses -->
