# Labeling Guide — labeling_batch_1.csv

For each row, fill in:

- **intent**: one of `software_bug`, `battery_hardware`, `connectivity`,
  `app_subscription`, `account_security`. (5 categories, NOT 6 — escalation
  moved to its own flag, see below.)
- **escalate_flag**: `yes` or `no`. Mark `yes` if the customer indicates they've
  already tried an official fix and it didn't work, or shows repeated/escalating
  frustration (not just a mildly annoyed tone in the opening message — that's too
  common in this dataset to be a signal on its own).
- **notes**: anything ambiguous — e.g. "could be software_bug or battery_hardware,
  picked battery because that's the customer's actual complaint"

## Rule for ambiguous cases
Pick the intent that matches what the customer would need help with FIRST,
not every issue mentioned. If genuinely 50/50, note it — don't force it.

## Taxonomy decision (resolved after seeing real data)
`escalation_repeat` is a FLAG, not a 6th intent — confirmed by looking at the
sample: repeat-complaint language shows up across battery, software_bug, and
connectivity roughly equally, so it's cross-cutting, not its own category.