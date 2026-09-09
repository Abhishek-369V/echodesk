# EchoDesk

An AI-assisted customer support triage system built on the Kaggle "Customer Support
on Twitter" dataset, focused on **AppleSupport** conversations.

Given an incoming customer message, EchoDesk:
1. Classifies intent (6 categories — see `notebooks/intent_taxonomy_draft.md`)
2. Generates a grounded reply using retrieval over historically resolved AppleSupport replies
3. Flags whether the conversation should escalate to a human agent, with a stated reason

This project prioritizes **evaluation rigor over model sophistication** — the goal
is a small, honest, well-measured system with a real golden set and real failure
analysis, not a maximal model.

## Status
🚧 In progress — Day 0 complete (data recon + intent taxonomy draft + scaffold).

## Repo structure
```
EchoDesk/
├── data/
│   ├── raw/              # stitched thread samples, recon notes
│   ├── processed/        # cleaned/labeled data (Day 1+)
│   └── golden_set/       # hand-labeled eval set (Day 1-2)
├── src/
│   ├── data_prep/        # thread stitching, cleaning
│   ├── classifier/       # intent classification
│   ├── retrieval/        # RAG for reply generation
│   ├── escalation/       # escalation rule/classifier
│   └── eval/             # metrics, LLM-as-judge, baselines
├── notebooks/            # exploration, taxonomy drafts
├── reports/              # final writeup, decision log
└── configs/              # config files
```

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Dataset
See `data/raw/DATA_RECON.md` for brand selection reasoning and stitching approach.

## Decision log
See `reports/decision_log.md` (to be filled in as the project progresses).

## Reproducing results
_To be filled in once the eval harness is built — target: full repro in under 15 min._
