# Decision Log

Non-obvious calls I made during this project, and why?. 
- Organized by day wise schedule, roughly in the order I made them.

## Schedule

| Day | Focus |
|---|---|
| Day 0 | Dataset recon, brand selection, repo scaffold, first intent taxonomy draft |
| Day 1 | Multi-turn thread stitching, first labeling batch |
| Day 2 | Golden set to 176 examples, baseline classifier, RAG retrieval + generation |
| Day 3 | Escalation logic, agent wiring, eval harness |
| Day 4 | Full eval run, failure analysis, report, hybrid escalation experiment |
| Day 5 | Streamlit demo, README/decision log polish, repro check, submission |

## Day 0 - Picking a brand and getting the taxonomy off the ground

1. **Brand: AppleSupport, not AmazonHelp**: AmazonHelp has more volume (169K vs 106K outbound tweets) but skews heavily toward generic order-status questions, low intent diversity. AppleSupport gives a richer taxonomy and more interesting escalation calls.
2. **First-turn stitching only (for now)**: Day 0 stitching captures customer->AppleSupport first-reply pairs, not full multi-turn threads. I deferred full thread reconstruction until I knew whether the reply generator actually needed multi-turn context, to avoid overbuilding early.
3. **`escalation_repeat` might not be a true 6th intent**: it could be a cross-cutting flag instead of its own category. Deferred the decision to Day 1-2 hand-labeling, where I could see how often it co-occurs with other intents vs. stands alone on its own.

## Day 2 - What a small baseline actually tells you

4. **Baseline classifier fails on minority intents (connectivity, account_security: 0% recall) - accepted as expected, not something to fix**: With only 10-11 labeled examples per minority class, a TF-IDF+LogReg model simply has no signal to learn from. Rather than over-invest in patching the trivial baseline, I documented this as a known limitation that motivates the few-shot LLM classifier as the real approach - a model can generalize from a handful of examples per class in a way a purely statistical one can't.

## Day 3 - Escalation logic and making the eval harness safe to rerun

5. **Escalation logic: tried hybrid rule+LLM, not pure rule-based**: I measured the rule-based escalation engine against all 176 hand-labeled examples before trusting it: 68% recall (30/44 true escalations caught) and a 20% false-positive rate. Keyword regex can't catch sarcasm, indirect repeat-contact phrasing, or subtler frustration. Rather than keep tuning the regex indefinitely, I added an LLM fallback that only fires when the rules find nothing!... keeps the cheap, deterministic rule path for obvious cases while catching some of what rules miss.
6. **Made the eval harness resumable/incremental**: saving results after every row instead of only at the end. A crash or early stop mid-run (say, at row 40 of 176) would otherwise mean re-paying for all 140 already-completed API calls on retry.

## Day 4 - Testing the hybrid, finding it made things worse, and catching my own bug

7. **Shipped rule-only escalation as the default, not the hybrid:** I built and tested the LLM-fallback hybrid on the full golden set: precision collapsed from 0.50 to 0.30 (recall rose to 0.98, but at an unacceptable false-positive cost). The fallback prompt was too permissive. Rather than ship the more "sophisticated" version because it sounded better on paper,
I measured both and kept the simpler one that actually performs better (F1 0.58 vs 0.46). The hybrid code stays in the repo behind a flag, documented as needing prompt tightening before it's usable.
8. **Found that my original eval run had silently used rule-only escalation**: the whole time `EchoDeskAgent()` defaults `use_llm_escalation_fallback` to `False`, and my eval harness never passed the flag. This wasn't an intentional decision, it was a bug I caught by re-reading my own code after getting results back. 
   
- Fixed it two ways: 
 (a) made the harness require the flag explicitly and print which mode is active
 (b) built a separate targeted recompute script rather than re-running (and re-paying for) the full classify+generate+judge pipeline just to fix one flag.

## Day 5 - Closing out

9. **Added a Streamlit demo rather than relying on CLI output alone.** The assignment says reviewers will ask me to explain and modify the code live - a working interactive demo makes that conversation faster and shows the pipeline end-to-end without reading through terminal logs.