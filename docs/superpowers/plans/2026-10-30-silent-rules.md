# Silent-rule feedback experiment

User authorized this experiment on a separate branch. Base: main 69d1ed1. Branch: experiment/silent-rule-feedback. Implementation uses the existing writing-plans/test-driven-development/executing-plans workflow, with a fresh read-only review before publication.

## Design and constraints

Four hidden thresholds (2,3,4,5) categorize symbolic tones 0..6 as left/right. Invariant stimuli 0 and 6 have the same answer under all rules; potentially informative stimuli 2,3,4 differ across candidate rules. The environment changes to a different rule with hazard q each trial, without a switch marker. Feedback is the noisy correct category, delivered only AFTER prediction. Training targets use this same observed feedback, not hidden clean truth; clean labels and hidden rules are evaluation diagnostics only. Four latent rules do not imply 25% answer chance: there are two answer classes, and an input-only marginal baseline can exceed 50%.

Reference: exact known-threshold Bayesian filter with the correct transition/noise model. It receives the same past noisy feedback as the learned model, never the current hidden label before prediction. Its belief is a reference, not a claim that learned adapter indices encode rule IDs.

New routers: projected tanh state/current features, then either their product or their normalized sum, alongside existing features. Identical dimensions and parameter counts. Existing concat routing remains the default for existing experiments. Shared forward-step controls can suppress only historical context (zero its read signal) and skip state updates; persistent state tensors must remain intact. Paired normal/disconnected/restored predictions use the same saved state; the normal trajectory alone continues.

New modules: silent_data.py (task/filter/tokenizer), silent.py (train/eval/resume), silent_suite.py (sequential comparisons). Existing data generators remain untouched. Legacy Config defaults/checkpoints must remain compatible. Freeze pretrained GPT-2 by default, tiny offline encoder available; no new pretrained vision model is required for this categorical rule experiment.

Evaluation: noisy-feedback-trained prediction scored against clean category; accuracy/NLL/Brier grouped invariant/informative, novel/returning regime, and switch age. Exact filter and marginal reference. Feedback all/informative-only/invariant-only/none; randomized switch timing; per-trial CSV includes router weights labeled as adapters, not rule probabilities. Paired read disconnection and restoration, full reset as a different diagnostic. Final reports must say inference-state tracking, not solved catastrophic forgetting. No elapsed-wall-time mechanism or online parameter consolidation in this branch.

## Plan

- [x] Task 1: failing generator/filter tests; implement determinism, leakage separation, Bayes likelihood ratios, invariant feedback and transition timing; run tests.
- [x] Task 2: failing router/intervention tests; implement exact-count interaction/additive routers, binary head option and read-only/disconnected calls; run full regression suite.
- [x] Task 3: failing small train/resume/report tests; implement GPU-capable causal runner, matched streams, checkpoint selection, metrics and sweeps; CPU smoke and exact-filter validation.
- [x] Task 4: write protocol/commands, independent review, fix important findings, final tests, publish branch only.

Review focus: target-before-prediction leakage; feedback generated identically under ablations; hazard at correct trial boundary; interventions preserve state and restoration; resumed checkpoints/old defaults; aggregate comparisons at same settings; router weights not mislabeled as beliefs.

Validation: 62 tests passed; 30-step CPU smoke completed. Independent review found and verified fixes for evaluation CSV overwrites and report metadata, plus corrected the marginal expectation to 80%. Publication targets the separate branch only.
