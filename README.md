# EchoDesk

An AI support agent for AppleSupport, built on the Kaggle "Customer Support on Twitter" dataset. 

- Given an incoming customer message:
1. it classifies the intent, 
2. drafts a reply grounded in how AppleSupport has historically resolved similar issues,
3. decides whether to auto-handle or escalate to a human, with a stated reason for that decision.

I built this for the Hiver SDE Intern take-home assignment. The full writeup problem framing, results against baselines, failure analysis, and what's misleading about the headline numbers.. is in `reports/report.md`. This README covers setup and reproduction.

## How the work was split

| Day | Focus |
|---|---|
| Day 0 | Dataset recon, brand selection, repo scaffold, first intent taxonomy draft |
| Day 1 | Multi-turn thread stitching, first labeling batch |
| Day 2 | Golden set to 176 examples, baseline classifier, RAG retrieval + generation |
| Day 3 | Escalation logic, agent wiring, eval harness |
| Day 4 | Full eval run, failure analysis, report, hybrid escalation experiment |
| Day 5 | Streamlit demo, polish, repro check, submission |

Full reasoning behind every non-obvious call is in `reports/decision_log.md`.

## Repo structure
```
EchoDesk/
├── data/
│   ├── raw/              # Kaggle source, Day-0 sample, dataset recon notes
│   ├── processed/        # stitched multi-turn threads, retrieval corpus
│   └── golden_set/       # 176 hand-labeled examples + labeling guide
├── notebooks/            # intent taxonomy docs
├── reports/              # report, failure analysis, decision log, eval results
├── src/
│   ├── data_prep/        # thread stitching, labeling batch generation
│   ├── classifier/       # TF-IDF baseline + few-shot LLM classifier
│   ├── retrieval/        # RAG: retrieval + grounded reply generation
│   ├── escalation/       # rule-based + hybrid escalation logic
│   ├── agent/            # wires everything into one EchoDeskAgent()
│   └── eval/             # eval harness, judge-agreement check
└── app.py                # Streamlit UI demo
```

## Setup

1. activate virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Create a `.env` file in the repo root (already in `.gitignore`):
```
OPENAI_API_KEY=sk-your-key-here
```

3. Download the Kaggle dataset ("Customer Support on Twitter", thoughtvector),
extract `twcs.csv`, and place it at `data/raw/twcs.csv`.

## Reproducing the headline results (under 15 minutes)

The golden set and processed data are already in the repo, so you don't need to regenerate them from scratch. To rerun the pipeline end to end on a small subsample:

```bash
# 1. Rebuild the retrieval corpus (fast, no API calls)
python -m src.retrieval.build_retrieval_corpus

# 2. Try the agent on a single message
python -m src.agent.agent --text "my bluetooth keeps disconnecting"

# 3. Run the eval harness on a small sample first (controls API cost)
python -m src.eval.eval_harness --limit 20

# 4. Full golden-set eval (176 examples, rule-only escalation - the shipped default)
python -m src.eval.eval_harness
```

Headline numbers from my own full run: 76% intent accuracy, 0.58 escalation F1 (rule-only), 4.09/5 mean reply-quality judge score. All caveats on these numbers are in `reports/report.md`, Section 4.

To regenerate everything from the raw dataset instead of using what's already in the repo (not required, takes longer):

```bash
python -m src.data_prep.stitch_threads
python -m src.data_prep.build_labeling_batch --batch-name batch_1
# ... hand-label the output, then repeat for more batches
python -m src.classifier.baseline_classifier
```

## Interactive demo

```bash
streamlit run app.py
```

Type a customer message and see intent, draft reply, and escalation decision live. Includes a toggle for the hybrid escalation mode (off by default - see `reports/failure_analysis.md`, Failure Mode 5, for why).

## Key findings, briefly

- Few-shot LLM classification (76% accuracy) clearly beats a TF-IDF+LogReg baseline trained on the same 176 examples, especially on minority intents where there's too little data for a trained model to learn anything.
- I tried adding an LLM fallback to the rule-based escalation logic to catch what keyword rules miss. It made things worse (precision dropped from 0.50 to 0.30), so I shipped the simpler rule-only version instead. Full numbers in `reports/failure_analysis.md`.
- The LLM-as-judge for reply quality only ever scored replies 3 or above across all 176 examples — it never once flagged a genuinely bad reply, which caps how much the 4.09/5 headline number can be trusted on its own.

## Reports

- `reports/eval_results/` — raw eval output CSVs
- `reports/decision_log.md` — every non-obvious decision, in order
- `reports/failure_analysis.md` — top 5 failure modes with real examples
- `reports/report.md` — full writeup (problem framing, baselines quality, failure analysis, misleading-numbers section, next steps)