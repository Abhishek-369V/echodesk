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

## Day 2

4. **Baseline classifier fails on minority intents (connectivity, account_security:
   0% recall) — accepted as expected, not fixed.** With only 10-11 labeled examples
   per minority class, a TF-IDF+LogReg model has no signal to learn from. Rather than
   over-invest in fixing the trivial baseline, this is documented as a known limitation
   that directly motivates the few-shot LLM classifier as the real approach — a model
   is expected to generalize from a handful of examples per class in a way a purely
   statistical model cannot.

## Day 3

5. **Escalation logic: hybrid rule+LLM, not pure rule-based.** Measured the
   rule-based escalation engine against all 176 hand-labeled examples before
   trusting it: 68% recall (30/44 true escalations caught) and a 20%
   false-positive rate (26/132 non-escalations wrongly flagged). Keyword
   regex cannot catch sarcasm, indirect repeat-contact phrasing, or subtler
   frustration registers. Rather than over-tuning the regex indefinitely,
   added an LLM fallback that only fires when rules find nothing — keeps the
   cheap/fast/deterministic rule path for obvious cases, while catching
   nuance rules miss, without paying LLM cost on every single message.

6. **Eval harness made resumable/incremental**, saving results after every
   row instead of only at the end. A crash or early stop mid-run (e.g. at
   row 140/176) would otherwise mean re-paying for all 140 already-completed
   API calls on retry.