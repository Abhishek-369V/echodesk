# EchoDesk - Report

Brand: AppleSupport | Dataset: Kaggle Customer Support on Twitter | 176 hand-labeled golden examples

## 1. Problem framing

**What "good" means for this brand.** AppleSupport handles high-volume, low - complexity technical triage on a public, time-sensitive channel (Twitter). "Good" here is NOT maximum automation - it's **correctly identifying which messages are safe to auto-handle** (common, well-understood issues with clear historical resolution patterns) **and reliably routing everything else to a human**, because a wrong public reply on Twitter is worse than a slow one. We optimized escalation to favor recall over precision for exactly this reason - missing a true escalation is worse than an unnecessary one.

**What we chose not to build:**
- Full multi-turn conversation modeling. We stitch multi-turn threads (see `stitch_threads.py`) but the classifier/generator operate on the opening customer message only. A production system would need conversation state.
- A fine-tuned classifier. With 176 labeled examples and 5 classes, a from-scratch trained model has no chance against a few-shot LLM prompt, we proved this directly (see Section 2 below!).
- Multi-language support. Explicitly out of scope; flagged as a real gap in failure analysis (non-English messages currently fall through both the rule and LLM escalation checks).
- A fully-tuned hybrid escalation system. We built and tested an LLM-fallback escalation layer, found it made results worse, and shipped the simpler system instead (Section 4 has full detailed view with exact numbers).

## 2. Results vs. baselines

### Intent classification
| System | Accuracy | Notes |
|---|---|---|
| Trivial (majority class) | 49% (n=80 early run) | Always predicts `software_bug` |
| Simple (TF-IDF + Logistic Regression) | 54% (n=80) / weak on minority classes | Classic ML baseline |
| **Our system (few-shot LLM)** | **76%** (n=176, full golden set) | Primary approach |

The TF-IDF baseline was re-validated as data grew (176 examples): minority classes (`connectivity`, `account_security`) still had near-zero recall due to data scarcity... exactly the problem a few-shot LLM approach avoids, since it doesn't need per-class training data.

### Escalation
| System | Precision | Recall | F1 |
|---|---|---|---|
| Trivial (always "no") | 0.00 | 0.00 | 0.00 |
| Trivial (always "yes") | 0.25 | 1.00 | 0.40 |
| Simple (rule-based: repeat-contact + frustration keywords) | 0.50 | 0.68 | **0.58** |
| Hybrid (rule + LLM fallback) - tested, not shipped | 0.30 | 0.98 | 0.46 |

Rule-based beats both trivial baselines and the hybrid attempt. Shipped as default (full reasoning in `reports/failure_analysis.md`, Failure Mode 5).

### Reply quality
Mean LLM-judge score: 4.09/5 (176 replies). Judge-human agreement: Cohen's kappa = 0.61 (moderate-to-substantial) on a 30-example manually-rated sample. See Section 4 for why this number needs a caveat.

## 3. Failure analysis

Full detail in `reports/failure_analysis.md`. Top 5 failure modes:
1. **Taxonomy ambiguity**: most `app_subscription` misclassifications are arguably mislabeled, not misclassified (the classifier called them `software_bug`, correctly, based on content).
2. **Escalation frustration-signal over-triggers on single capitalized words**: (e.g. "EXTREME") needs a stricter multi-word or ratio-based threshold.
3. **Non-English input silently missed**: by both escalation paths.
4. **Escalation favors recall over precision** (30 false positives, 14 false negatives) an intentional trade-off, not a bug, but a real cost.
5. **LLM escalation fallback, when tested, made precision worse** (0.50->0.30) root-caused to an overly permissive fallback prompt.

## 4. What is misleading about my headline numbers?

- **76% intent accuracy** is inflated by class imbalance (`software_bug` is 56% of the data). Macro-F1 (0.67) is the honest number, and it's meaningfully lower.
- **4.09/5 reply quality** looks strong, but the judge never used a score of 1 or 2 across all 176 replies, it physically cannot report a truly bad reply under 3, which caps how much this number can tell you about the worst cases. Given 24% of intents were misclassified, some fraction of "4-5 star" replies were almost certainly responding to the wrong problem, and the judge didn't
catch it.
- **68% escalation recall** looks like "the system catches most true escalations," but it comes at a 20% false-positive rate - cheap to state positively, expensive to run in production without accounting for the extra human score workload from 30 unnecessary escalations out of 176 messages.
- **The hybrid escalation experiment's 0.98 recall**: is the most dangerous number in this entire report if read alone, it sounds like a win, but the matching 0.30 precision means it's mostly noise, not signal. This is reported explicitly instead of cherry-picking the recall number.

## 5. What I'd do next if given one more week

1. Re-label the `app_subscription`/`software_bug` boundary (cheapest, highest-leverage fix identified).
2. Rewrite the LLM escalation fallback prompt to require citing a specific, named signal (not vague "seems like it might need help" reasoning), then re-test against the same 176 examples before considering shipping it.
3. Add a language-detection pre-filter; auto-escalate non-English input rather than attempting classification on it.
4. Force the LLM judge to use the full 1-5 range by requiring a stated criticism for any score above 3; currently too willing to default to "good enough."
5. Expand the golden set specifically for minority intents for balancing class (`connectivity`, `account_security`) both still have the fewest examples and the least reliable per-class numbers.