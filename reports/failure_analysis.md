# Failure Analysis

Based on full eval run: 176 golden-set examples, 0 pipeline errors.

## Headline numbers (and why they're misleading alone)

- Intent classification: 76% accuracy, but this is pulled up by imbalance data where `software_bug` being 56% of the data (99/176). Macro-avg F1 is 0.67, a more honest picture of per-class performance, and much weaker.
- Escalation: 50% precision / 68% recall = over-escalates (60 predicted vs 44 true), meaning cost/friction from unnecessary human handoffs is real.
- Reply quality: 4.09/5 mean judge score, but the judge NEVER used a score of 1 or 2 across all 176 replies (only 3, 4, 5 appeared: 76/9/91). This score is inflated by scale compression, not evidence the replies are uniformly good.
- Judge-vs-human agreement (kappa=0.61, 30-sample) is moderate-to-substantial, not strong enough to fully trust the 4.09 headline on its own.

## Failure mode 1: Taxonomy ambiguity, not a model failure

1. 17/27 `app_subscription` misclassifications were predicted as `software_bug`. 
  On inspection, most of these ARE software bugs (broken shuffle button, app crashing, video load failures) mislabeled as `app_subscription` during hand-labeling, because the taxonomy definition for `app_subscription` ("App Store, Apple Music, subscription/billing issues") is ambiguous about whether "issues
  happening inside an app" counts, versus strictly billing/subscription issues.

- **This points to a taxonomy definition gap, not a labeling error** — the category boundary itself doesn't cleanly separate "billing/subscription problem" from "the app is misbehaving," so both a me(hand-labeling) and my classifier have land on different sides of that line depending on how we read an ambiguous case.

- Fix for a v2: split `app_subscription` into `app_functionality_bug` (folds into software_bug) and `billing_subscription` (the real edge case: being charged without access), and re-label affected rows.

## Failure mode 2: Escalation frustration-signal is over-sensitive to caps

2. False positive example: "EXTREME battery drainage" escalated on the `frustration_signal` rule triggering on the
  single capitalized word "EXTREME" -- this is emphasis, not shouting. The `\b[A-Z]{4,}\b` pattern can't distinguish
  one emphasized word from genuine all-caps shouting ("STOP DOING THIS").

- Fix for v2: only trigger on 2+ capitalized words, or ratio of caps-words to total words above a threshold, not any single instance.

## Failure mode 3: Non-English input silently missed

3. One false-negative escalation was a Dutch-language message. Neither the rule engine nor the LLM fallback (untested 
  at scale on non-English input) flagged it, meaning a non-English-speaking customer's complaint could go un-escalated
  with no visibility into the problem. 
  
- Fix for v2: add a language-detection pre-check; auto-escalate anything not confidently English rather than
attempting classification/reply generation on it.

## Failure mode 4: Escalation over-triggers on frustration without real severity

4. 30 false-positive escalations vs. only 14 false negatives: the system is tuned toward over-escalating, which is
  arguably the safer failure direction for a support agent, but has a real operational cost (more human workload
  than necessary). Documented as an intentional trade-off, not hidden as a

- clean win: precision matters for cost, recall matters for trust; chosen to lean toward recall (68%) at the cost of precision (50%).

## Failure mode 5: LLM escalation fallback made things WORSE, not better

5. Built a hybrid escalation system (rules first, LLM fallback for anything rules missed) specifically to close the
  32% recall gap in the rule-only system. Tested it on the full 176-example golden set:
```
| System                       | Precision | Recall | F1   |
| ---------------------------- | --------- | ------ | ---- |
| Rule-only                    | 0.50      | 0.68   | 0.58 |
| Hybrid (rule + LLM fallback) | 0.30      | 0.98   | 0.46 |
```
- Recall improved (0.68 -> 0.98) but precision collapsed (0.50 -> 0.30): 69 of
116 LLM-fallback calls were false positives. 
- Root cause: the fallback prompt ("look for subtler cues: sarcasm, implied repeat contact, quiet exasperation")
is too permissive -- the model treats almost any mention of a problem or mild annoyance as escalation-worthy, even on a clean first-contact message.

> **Decision: shipped rule-only as the default** (better F1: 0.58 > 0.46). 
The hybrid code path still exists and is available via a flag, but is documented here as an experiment that needs a stricter prompt (e.g. requiring the model to cite a SPECIFIC repeat-contact or severity signal, not just "seems like it might need help") before it's safe to enable by default.

## What would most improve results with more time
1. Re-label `app_subscription` split (highest-leverage fix, cheap to do)
2. Tighten frustration_signal caps-detection to reduce false-positive escalations
3. Add language detection as a pre-filter
4. Expand LLM-judge rubric to force use of the full 1-5 range (e.g. require a specific criticism for any score above 3) rather than trusting it will naturally spread scores