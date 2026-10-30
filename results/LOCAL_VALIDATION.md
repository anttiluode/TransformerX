# Local validation — 30 October 2026

**Implementation checks pass; useful performance has not been demonstrated.**

- 37 tests pass, including both modality training/checkpoint/resume paths.
- Tiny models: forward/backward in all eight modes, state/history dependence, reset equivalence and adapter/write gradients.
- Real Hugging Face GPT-2 and ViT implementations instantiated with small **random** configs: frozen backbone, trainable internal adapters, hook cleanup.
- An independent read-only reviewer additionally exercised CPU BF16/FP16 paths in every mode and both HF families. This does not verify CUDA kernels or GPU VRAM consumption.
- Regressions cover FP16 gradient-scaler overflow recovery, finishing an interrupted final evaluation, rejecting mixed sweep settings/duplicate seeds, balanced dictionary marginals, and noise-independent scene generation.
- Package editable install and CLI help checked. Windows launcher was inspected, not executed on Windows.

Environment: Python 3.12.14, PyTorch 2.15.0+cu132, Transformers 4.57.6; CUDA hardware unavailable. No downloaded pretrained checkpoint was trained/evaluated here.

## Bounded CPU development runs

One seed (7), 80 training batches of 8 episodes, hidden width 48, state width 32, rank 4, two tiny transformer layers. Best checkpoint selected by validation NLL on 32 episodes, then tested on 64 independent episodes. These are short development runs, **not a tuned comparison or confirmation**.

| Task | Mode | Test accuracy | Test NLL |
|---|---|---:|---:|
| Language | Current observation only | 24.48% | 1.3913 |
| Language | Loop | 22.53% | 1.3914 |
| Vision | Current observation only | 26.04% | 1.3889 |
| Vision | Loop | 26.04% | 1.3883 |

Chance is 25% for the balanced language queries and in expectation for visual directions. Erasing/shuffling the loop did not establish a benefit in these runs. The working forward/backward mechanism is not evidence that either task has been learned. Longer GPU training may help, but success is not promised; negative results should be retained.

Compact settings, phase scores, interventions and budgets: [cpu_development.json](cpu_development.json). Checkpoints and generated runs are not committed. Initial language runs used an unbalanced dictionary partition; those were superseded by the balanced-orbit split and are excluded here. Pixel-noise evaluations were recomputed after separating scene and noise RNGs. Other reported clean-stream scores are unchanged by that correction.

Reproduce each row using:

```bash
python -m transformerx.run --task language --mode loop --dim 48 --state-dim 32 --rank 4 --steps 80 --batch 8 --eval-every 40 --eval-episodes 32 --test-episodes 64 --threads 2 --out runs/reproduce-language-loop
```

Replace task/mode and choose a new output path for other rows. GPU numeric results may differ. A convincing next result must beat meaningful conventional memory controls over multiple seeds, with actual memory/compute budgets reported.
