# Silent-rule local verification

2026-10-30, CPU, Python 3.12, PyTorch 2.15.0+cu132. No GPU or downloaded pretrained training was run.

- Full suite: **62 passed** (`python -m pytest -q`).
- `git diff --check`: passed.
- Tests cover exact filter transitions and enumerated likelihoods, uninformative surprise, matched router budgets/gradients, preserved state and exact restoration, frozen random GPT-2 integration, causal first predictions, checkpoint round trips, exact interrupted/uninterrupted training equality, old checkpoint defaults, evaluation report preservation, and suite settings.
- Independent read-only review found no remaining blocker after reporting fixes.

Bounded end-to-end smoke command:

```bash
python -m transformerx.silent --device cpu --threads 1 --steps 30 --batch 4 --trials 48 --tbptt 12 --dim 32 --state-dim 16 --layers 2 --rank 4 --eval-every 15 --eval-episodes 8 --test-episodes 8 --out runs/silent-smoke
```

Validation selected step 15. On 384 test trials across eight episodes, all feedback:

| Predictor | Clean accuracy | Clean NLL |
|---|---:|---:|
| Loop, interaction router | 0.7344 | 0.4687 |
| Uniform-rule marginal | 0.7266 | 0.4441 |
| Exact known-model filter | 0.9141 | 0.1933 |
| Loop with read disconnected | 0.7344 | 0.4696 |

These are correlated trials from very few episodes, not confidence estimates or evidence of a loop advantage. The marginal's population expectation is 0.80; this small sample deviates from it. The smoke does not establish useful context dependence. Heavy matched multi-seed comparisons are the intended next test. These inspected test streams are development data, not blind confirmation.
