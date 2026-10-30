# Completed interaction vs additive comparison — seed 7

> Superseded by the [completed three-seed, three-router analysis](FULL_SWEEP_2026-10-30.md). This earlier snapshot is retained for provenance.

**30 October 2026.** Both runs completed 1,000 training steps. Concat has a configuration file but no completed results in this snapshot; seeds 17 and 27 are not yet available. This update supersedes the unfinished additive comparison in the [earlier GPU report](GPU_INTERIM_2026-10-30.md).

**Conclusion:** both router variants learned useful history-dependent decisions. Their primary test accuracies are nearly tied in this seed. We have not established that the multiplicative routing term improves the standard task. There is an exploratory advantage for interaction when invariant feedback is withheld, which needs replication.

## Selected checkpoints, not the last printed line

| Measurement | Interaction router | Additive router |
|---|---:|---:|
| Validation-selected step | 925 | 825 |
| Selected validation NLL | 0.223401 | 0.223339 |
| Selected validation accuracy | 92.22% | 92.35% |
| **Held-out test accuracy, all feedback** | **91.94%** | **91.86%** |
| Correct / total | 5,649 / 6,144 | 5,644 / 6,144 |
| Clean test NLL | 0.229429 | 0.232984 |
| Clean test Brier score | 0.128788 | 0.129809 |
| Test accuracy, rule-dependent trials | 86.38% | 86.24% |
| Test accuracy, rule-invariant trials | 100.00% | 100.00% |
| Test accuracy, returning-rule segments | 91.06% | 91.15% |
| Test accuracy, state read disconnected | 79.92% | 79.92% |
| Maximum restoration error | 0.0 | 0.0 |

The additive run's final printed 90.7% is **step-1,000 validation accuracy**, not the performance of its selected checkpoint. Selection uses minimum validation NLL, so its test report evaluates step 825. Interaction likewise uses step 925 rather than its final step.

Both runs use the same frozen DistilGPT-2 backbone, seed, data, optimization settings, 64-dimensional state and **166,279 trainable parameters**. The audit verifies the configuration match apart from router/output directory, and matches every trial's tone, hidden rule, clean target, noisy outcome and switch metadata between runs.

The exact Bayesian reference scores **92.61%**, and the no-history marginal reference scores **79.92%**, on these same streams. The two learned models therefore improve over state-read disconnection by **12.03** and **11.95 percentage points**, respectively.

## What the five-answer difference means

The checkpoints disagree in correctness on 105 trials: interaction alone is correct on 55, additive alone on 50. The net difference is **+0.081 percentage points** for interaction.

A paired bootstrap over the 64 test episodes, keeping each episode's 96 trials together, gives a 95% percentile interval of **−0.277 to +0.456 percentage points** for interaction minus additive accuracy (10,000 resamples). This does not establish a clear accuracy advantage. It also does not prove formal equivalence: no equivalence margin was specified.

This interval measures test-episode variation conditional on these two selected checkpoints. Both were trained with seed 7; it does not account for training-seed variation. The small NLL/Brier edge for interaction is descriptive, not sufficient evidence of an architectural advantage.

## Feedback ablations reveal a possible difference

| Feedback retained | Interaction accuracy | Additive accuracy | Interaction minus additive |
|---|---:|---:|---:|
| All | 91.94% | 91.86% | +0.08 pp |
| Informative only | 89.71% | 88.07% | **+1.64 pp** |
| Invariant only | 70.82% | 70.59% | +0.23 pp |
| None | 70.59% | 70.38% | +0.21 pp |

The informative-only condition is the most interesting follow-up: the interaction checkpoint answers 101 more trials correctly. Its paired episode-bootstrap interval is **+1.25 to +2.05 points**, conditional on these checkpoints. On rule-dependent trials specifically, the scores are **82.61% vs 79.83%**.

This is an exploratory robustness result among several inspected conditions, not a replicated primary-endpoint win. Withheld feedback is replaced by the `no feedback` marker, which was absent from training. Differences may reflect response to this distribution shift as well as handling of evidence; the ablation does not isolate those explanations. The Bayesian filter retains exactly the same score when invariant feedback is removed, whereas both learned models degrade.

Both active loops still perform below the no-history reference without informative feedback. This confirms that poor missing-evidence fallback was not peculiar to the interaction checkpoint. All invariant-trial accuracies remain 100% across these conditions.

## Mechanistic interpretation and next comparisons

The positive result is **useful recurrent state**, demonstrated for both variants by the within-model read intervention. The intervention removes both state-dependent routing and the additive memory readout, so it does not identify the relative contributions of those pathways.

“Additive router” does not mean a wholly additive model: it retains the nonlinear recurrent update, state readout, softmax routing and low-rank adapter bank. The comparison tests the extra projected product term in the router against a matched projected sum. It does not test whether all multiplicative computation or all recurrence can be removed.

The immediate next evidence is the completed concat result, then the paired seeds 17 and 27. The original frozen language/vision runs remain near chance. These results establish neither a general language-model gain nor protection from catastrophic forgetting. Returning-rule performance can arise from re-inference rather than permanent storage of earlier rules.

## Sources and reproduction

- [Interaction test report](../runs/silent-routing/loop-interaction/seed-7/results.json), [validation](../runs/silent-routing/loop-interaction/seed-7/validation.csv), [trials](../runs/silent-routing/loop-interaction/seed-7/trials.csv).
- [Additive test report](../runs/silent-routing/loop-additive/seed-7/results.json), [validation](../runs/silent-routing/loop-additive/seed-7/validation.csv), [trials](../runs/silent-routing/loop-additive/seed-7/trials.csv).
- [Machine-readable comparison and source hashes](ROUTER_COMPARISON_2026-10-30.json).
- Reproduce with `python results/compare_routers_20261030.py` (NumPy required). The [script](compare_routers_20261030.py) checks reported accuracy/NLL against CSV predictions, verifies matched streams/settings, and computes paired episode intervals without model loading or retraining.

These test results have now been inspected during development; subsequent tuning on their findings should be confirmed on fresh held-out streams.
