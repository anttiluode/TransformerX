# Silent-rule experiment

This is a separate outcome-driven rule-switching benchmark, included on main. The existing language/vision benchmarks and default model checkpoint layouts remain intact. This is an experiment in context-dependent computation, not evidence that the architecture prevents catastrophic forgetting.

## Run on your GPU

Install the repository requirements and a CUDA-enabled PyTorch build as described in README.md, then update main:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python -m pytest -q

# A single frozen pretrained GPT-2 experiment (downloads weights on first use).
python -m transformerx.silent --backbone hf --device cuda --amp --steps 1000 --out runs/silent-gpt2

# Primary matched comparison, three paired seeds, sequential jobs.
python -m transformerx.silent_suite --backbone hf --device cuda --amp --steps 1000 --variants loop-interaction loop-additive loop-concat --out runs/silent-routing

# Full small-transformer comparison (7 variants x 3 seeds).
python -m transformerx.silent_suite --backbone tiny --device cuda --steps 500 --out runs/silent-tiny
```

Run a command with `--dry-run` on `silent_suite` to print its jobs first. Suite runs are sequential, so they do not compete for GPU memory. On Windows the same commands work with your environment's Python executable. `tiny` trains its backbone; `hf` freezes pretrained GPT-2 by default. This is a two-class categorization head inside an existing transformer, not yet a general language-generation improvement.

```bash
# Resume: steps is the TOTAL number of episode batches.
python -m transformerx.silent --resume runs/silent-gpt2/latest.pt --steps 1500
python -m transformerx.silent_suite --backbone hf --device cuda --amp --steps 1500 --variants loop-interaction loop-additive loop-concat --out runs/silent-routing --resume

# Re-evaluate selected checkpoint, optionally on a different switch rate.
python -m transformerx.silent --evaluate runs/silent-gpt2/best.pt --hazard 0.10 --out runs/silent-gpt2-faster
```

Each training step processes a batch of 96 trials by default, with two forward passes per trial. `--tbptt 16` truncates gradients and updates weights every 16 trials. Lower `--batch` or `--tbptt` if GPU memory is tight; both controls are also accepted by the suite. Evaluation includes several counterfactual passes; use `--eval-episodes` and `--test-episodes` to control its cost. `--help` lists all options. Downloaded frozen backbone weights are omitted from checkpoints and must remain available locally or from Hugging Face for reload.

## What the agent sees

There are four latent frequency thresholds, 2, 3, 4 and 5. A tone at or above the active threshold requires `right`; otherwise it requires `left`. Each trial independently switches to a uniformly selected different rule with probability 0.05. No switch signal, rule identifier, fixed block length, or previous hidden truth is supplied to the model.

1. Receive `categorize tone three` and predict left/right.
2. Receive `tone three outcome left` as feedback, with a default 5% chance of an incorrect label.
3. Carry the updated state into the next trial.

Training cross-entropy uses the noisy observed outcome. Hidden rule IDs, clean labels, switch ages and return markers are diagnostic fields only. Validation selects the best checkpoint by **clean-label** NLL on separate validation streams; final test streams never select weights. This supervised category-feedback task is more explicit than an animal reward task: it does not simulate action selection and reward-dependent feedback.

Sixty percent of tones are 2, 3 or 4, where answers differ across rules (potentially informative feedback); the rest are 0 or 6, where every rule agrees. A surprising error on an invariant tone carries no rule evidence under this noise model. The exact Bayesian filter test verifies that such feedback leaves rule beliefs unchanged.

Four rules do **not** imply 25% answer accuracy: there are only two answers. The supplied uniform-rule marginal predictor exploits known thresholds and reaches 80% expected clean-answer accuracy with this stimulus distribution. The exact filter also knows thresholds, hazard and feedback noise, making it an informed reference rather than a capacity-matched learner. Its latent-rule accuracy is reported separately from answer accuracy. The filter predicts clean-label probabilities, whereas supervised training learns noisy-feedback probabilities; their clean-label NLL/Brier gap includes this calibration difference and is not purely a memory-performance gap.

## Mechanism and comparisons

The original concatenative router remains the default for the original experiments. The new benchmark defaults to the interaction router. Both new variants include the original state and input features plus a projected feature:

- `interaction`: `tanh(P state) * tanh(Q input)`.
- `additive`: `(tanh(P state) + tanh(Q input)) / sqrt(2)`.

They use identical learned tensors and parameter counts. This isolates the interaction operation, although it does not match activation distributions or establish that multiplication is uniquely necessary. Four adapter experts are learned processing modes; their CSV routing weights are **not** calibrated probabilities of the four hidden rules.

The full suite also includes the original concatenative loop, a GRU controller inside the transformer, recurrent latent memory tokens, a stateless transformer, and a bounded history-window transformer. The GRU is not a standalone GRU; memory tokens are not an implementation of the published RMT architecture. Their parameter/state budgets are reported rather than claimed equal. Matched seeds share stimuli and noise; the interaction/additive pair also shares initialization shapes.

At each test query, the runner compares:

- Normal history access.
- **Read disconnection:** zero the state provided to the controller's read/router path, retaining the original stored state. For the window baseline, omit prior history from this diagnostic read. Static adapter processing remains active.
- **Full reset:** make a fresh-state prediction.
- Restored access to the exact original state, checking the prediction matches exactly.

These are paired, one-query counterfactuals. Only the normal trajectory advances. They do not measure recovery after a prolonged lesion. With no stored history, reset and disconnection may coincide; that is expected. For recurrent controllers the read lesion includes both additive state readout and state-dependent routing, not routing alone.

Four feedback conditions replay the same test streams: all feedback, informative only, invariant only, and none. Withheld feedback is replaced by `no feedback` to retain two events per trial. Masking feedback is an out-of-training-distribution diagnostic. Input observations and autonomous recurrent dynamics can still change state when feedback is withheld; **no learned selective-write or consolidation mechanism is claimed here**.

## Reading results

- `results.json`: selected checkpoint, configuration, parameter/state budgets, clean accuracy/NLL/Brier scores by informative/invariant trials, age within the current rule segment, and first-visit/returning rule segments. A returning segment means its rule occurred earlier in that episode.
- `trials.csv`: chronological predictions, hidden diagnostic labels, switch markers, paired interventions and adapter weights for every test condition. Filter `switched` and subsequent `age` to inspect adaptation curves. The initial segment also starts at age zero.
- `validation.csv`: noisy training loss and clean validation metrics.
- Standalone evaluation writes `evaluation.json` with checkpoint identity and effective settings, plus `evaluation_trials.csv`, preserving the training run reports.
- `best.pt` / `latest.pt`: selected and resumable checkpoints.
- Suite `summary.csv`: per-seed comparisons; mixed training/backbone configurations are rejected.

The hypothesis is supported only if gains repeat across seeds, especially on informative trials after switches, without harming invariant trials. Compare to the matched additive router and informed references, inspect switch-age and returning-rule performance, and check whether read disconnection selectively removes useful context influence. A frozen backbone protects its weights from drift; it does not guarantee correct routing, indefinite memory retention, or preservation of all pretrained behavior. Returning-rule accuracy measures within-stream interference, not a complete catastrophic-forgetting benchmark.

The biological motivation is compartmentalized context/sensory integration and a selective intervention. This implementation does not identify multiplication with literal biological XOR, prove a dendritic mechanism, or implement continuous wall-clock dynamics. The completed three-router, three-seed sweep is recorded in [the full GPU analysis](results/FULL_SWEEP_2026-10-30.md). Useful history dependence repeats across all nine runs, while the extra router features show no clear accuracy advantage over concat. Other memory-architecture comparisons remain unrun in this sweep. Passing software tests alone is not a positive scientific result.
