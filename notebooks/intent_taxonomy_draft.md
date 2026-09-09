# Intent Taxonomy — Draft v1 (from manual read of 80 AppleSupport threads)

Brand: **AppleSupport**
Source: `data/raw/apple_threads_sample_8k.csv` (7,982 stitched customer→AppleSupport first-turn threads)
Sample reviewed: `data/raw/manual_read_sample_80.csv`

## Observed patterns
- Heavy recurring theme: iOS 11 update complaints (this dataset is from Oct-Nov 2017,
  right after iOS 11 launch — the "autocorrect capital I" bug shows up constantly)
- Battery complaints are extremely common and often tied to a specific iOS version
- Repeat complainers: many customers say "you already told me to do X, it didn't work"
  — this is escalation-worthy signal
- Frustration/profanity is common in the customer's opening tweet — tone alone
  isn't a reliable escalation signal, needs to be combined with "already tried fix" signal

## Locked intents (v1)

| # | Intent | Description | Example |
|---|--------|-------------|---------|
| 1 | software_bug | OS/app glitch, unexpected behavior after update | "autocorrect keeps changing I to L" |
| 2 | battery_hardware | Battery drain, physical device malfunction | "battery dies since iOS 11" |
| 3 | connectivity | Bluetooth, WiFi, calls, notifications not working | "bluetooth won't connect after update" |
| 4 | app_subscription | App Store, Apple Music, subscription/billing issues | "Apple Music says I'm not subscribed" |
| 5 | account_security | iCloud, Find My iPhone, backup, lost/stolen device | "need backup after phone was stolen" |
| 6 | escalation_repeat | Already tried official fix, still broken, repeated contact | "3rd time you've told me this, still broken" |

## Open question to validate during hand-labeling (Day 1-2)
- Is `escalation_repeat` really a separate *intent*, or is it a cross-cutting *flag*
  that can apply on top of intents 1-5? Leaning toward: keep it as a flag, not a 6th
  intent category — will decide once we hand-label real examples and see how often
  it co-occurs vs. stands alone.
