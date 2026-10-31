# TransformerX

**A continuing state that changes how a transformer processes its next observation.**

Three controlled experiments, plus a persistent Qwen chatbot:

- **Language:** remember changing dictionaries in two contexts, answer after distracting events, and return to an earlier context. Pretrained option: DistilGPT-2.
- **Vision:** observe two moving objects, track direction changes, and answer about either object after both disappear. Pretrained option: ViT.
- **Silent rules:** infer an unannounced changing categorization rule from noisy feedback, with matched interaction/additive routing and state-disconnection controls. Pretrained option: DistilGPT-2.
- **Persistent chat:** local Qwen3-0.6B language generation, saved conversations, editable notes, older-exchange recall and an optional separately trained neural loop. [Launch guide](CHAT.md).

This is an experimental implementation inspired by [RoundAndRound](https://github.com/anttiluode/RoundAndRound) and [LoopAImemory](https://github.com/anttiluode/LoopAImemory). **The completed nine-run GPU sweep shows useful history dependence across three seeds, but no clear accuracy advantage for the extra multiplicative router features.** The language task is supervised classification of textual queries, not a chat model or language-generation benchmark. The visual scenes are synthetic. Pretrained integrations are real model modifications. Completed pretrained GPU results and their limits are summarized below; the original CPU verification used small randomly initialized GPT-2/ViT configurations.

## Try the persistent chatbot

After the [Windows environment setup](#start-on-windows), update your checkout and run:

```powershell
git pull
.\.venv\Scripts\python -m pip install -e ".[chat]"
.\run_chat.bat
```

A local browser chat opens at **http://127.0.0.1:7860** after Qwen loads. Conversations and memory notes survive restarts; the app shows which older exchanges enter each prompt. Sessions can be exported, deleted or resumed. First launch downloads model weights; later launches support `--local-files-only`.

**The default is usable text-memory chat, not a claim that the neural loop makes Qwen smarter.** A Qwen-specific loop-training command and load/reset/toggle controls are included. No trained general-chat controller is bundled; the existing classification checkpoints are incompatible. The classifier results below do not measure chatbot performance. See **[CHAT.md](CHAT.md)** for memory behavior, a short GPU training check and comparisons to run.

## GPU results · 30 October 2026 · complete nine-run sweep

**All three routers learned useful state-dependent decisions across seeds 7, 17 and 27.** Their mean test accuracies are nearly identical. The simpler concat router has the lowest clean test loss in every seed and fewer trainable parameters.

| Router | Seed 7 | Seed 17 | Seed 27 | Mean ± sample SD | Mean test NLL | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Interaction | 91.94% | 92.45% | 92.42% | **92.27% ± 0.28 pp** | 0.22115 | 166,279 |
| Additive | 91.86% | 92.48% | 92.32% | **92.22% ± 0.32 pp** | 0.22292 | 166,279 |
| Concat | 91.81% | 92.48% | 92.40% | **92.23% ± 0.36 pp** | 0.21884 | 159,591 |

Each result uses the checkpoint selected by validation NLL after a 1,000-step run, tested on 6,144 trials in 64 episodes. Routers share test streams within each seed. The same-stream no-history marginal averages **80.46%**; the known-model Bayesian filter averages **92.89%**. SD describes variation across three seeds, not a confidence interval.

![Completed router sweep](results/figures/full-sweep-2026-10-30.svg)

**What held up:** disconnecting state access reduces accuracy in all nine runs, by **9.81–12.96 percentage points**. Invariant-trial accuracy remains 100%, and restoring access to the unchanged state restores the original predictions exactly. This establishes useful state access within these trained models; the intervention disables both routing from state and additive memory readout.

**What did not earn a clear advantage:** interaction beats additive by only **0.049 percentage points** on average and concat by **0.038 points**. Concat uses **4.02% fewer trainable parameters** and has lower test NLL in every seed. The extra projected product/sum features are not justified by the main accuracy comparison. All variants retain the same nonlinear recurrent update; this does not show that nonlinear memory itself is unnecessary.

**What still needs work:** with informative-only feedback, mean accuracy is **90.52% / 89.63% / 89.90%** for interaction/additive/concat, with the ranking changing by seed. With no feedback, the corresponding means are **73.23% / 71.25% / 72.29%**, below the 80.46% marginal reference. Feedback masking introduces an unfamiliar marker, so these are evidence-handling diagnostics with distribution shift, not proof of a specific failure mechanism.

The original frozen dictionary-language and occluded-motion runs remain near chance: **23.96%** and **24.74%**, respectively (25% uniform-answer chance). The silent sweep does not establish a general language-generation gain, prevention of catastrophic forgetting, or an advantage over GRU/token/window memory baselines, which are not included in these nine runs.

See the **[complete analysis and limitations](results/FULL_SWEEP_2026-10-30.md)**, [nine-run CSV](results/FULL_SWEEP_2026-10-30.csv), and [reproducible audit](results/analyze_full_sweep_20261030.py). Earlier [interaction-only](results/GPU_INTERIM_2026-10-30.md) and [seed-7](results/ROUTER_COMPARISON_2026-10-30.md) reports are preserved as historical snapshots.

## Start on Windows

Python 3.12 or 3.13 is recommended. From PowerShell:

```powershell
git clone https://github.com/anttiluode/TransformerX.git
cd TransformerX
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

Install a **CUDA-enabled PyTorch build** using the Windows/Pip/Python command from the [official PyTorch selector](https://pytorch.org/get-started/locally/). Use `.\.venv\Scripts\python -m pip` in place of `pip` in that command. The CUDA wheel must be supported by your NVIDIA driver. Then:

```powershell
.\.venv\Scripts\python -m pip install -e ".[pretrained,test]"
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
.\.venv\Scripts\python -m pytest -q
```

CUDA must print `True` for the GPU presets. Linux/macOS can use `python -m venv .venv`, activate it, and run the same Python modules. Tiny runs work on CPU and require no model downloads.

## First, check both pipelines

These are short wiring checks, not meaningful training:

```powershell
.\.venv\Scripts\python -m transformerx.run --task language --steps 3 --batch 2 --eval-episodes 4 --test-episodes 4 --out runs/smoke-language
.\.venv\Scripts\python -m transformerx.run --task vision --steps 3 --batch 2 --eval-episodes 4 --test-episodes 4 --out runs/smoke-vision
```

## Train the actual pretrained models on your GPU

```powershell
.\.venv\Scripts\python -m transformerx.run --config configs/language_gpu.json
.\.venv\Scripts\python -m transformerx.run --config configs/vision_gpu.json
```

Or run **`run_gpu.bat`** to run language and then vision sequentially. First runs download the public backbone weights from Hugging Face; later runs can use the cache with `--local-files-only`. These presets freeze the pretrained backbone and train the memory controller, operator adapters and four-class output head. Freezing weights does not eliminate the activations needed for gradients through the adapter.

Defaults are conservative: language batch 2, vision batch 1, mixed precision, 64 memory coordinates, four rank-8 operators. **The uploaded language/vision runs report peak PyTorch CUDA allocations of 1.59/3.10 GB; these are not total VRAM requirements.** If memory runs out, lower `--batch` first. Lowering `--tbptt` reduces activation memory but also shortens the learning credit horizon; a horizon shorter than the evidence-to-query delay can prevent the write mechanism from learning. `--tbptt 32` covers these default episodes. Lowering `--image-size` changes the synthetic source image, but pretrained ViT still resizes it to its configured resolution.

For a faster initial mechanism study, train the small local backbones:

```powershell
.\.venv\Scripts\python -m transformerx.run --config configs/language_tiny.json
.\.venv\Scripts\python -m transformerx.run --config configs/vision_tiny.json
```

Tiny backbones are trained from scratch; they are not pretrained-model evidence. The two tracks train separate models, not one multimodal network.

## Run comparisons, not only the proposed model

```powershell
.\.venv\Scripts\python -m transformerx.suite --tasks language vision --backbone tiny --steps 500 --device cuda --out runs/tiny-suite
.\.venv\Scripts\python -m transformerx.suite --tasks language vision --backbone hf --modes none static gru tokens loop loop_no_rotation window --steps 1000 --device cuda --out runs/pretrained-suite
```

The default seeds are 7, 17 and 27. Each run is sequential so GPU memory is not shared between jobs. `--dry-run` writes and prints the run manifest without launching training. Add `--resume` to skip completed runs and continue checkpointed runs. The full pretrained sweep is substantial; start with `--modes none gru loop --seeds 7` to check cost before launching it.

| Mode | Persistent information | Purpose |
|---|---|---|
| `none` | None | Current observation only |
| `static` | None | Same low-rank operator family, routed only from current input |
| `gru` | GRU vector | Same adapter interface, conventional recurrent update |
| `tokens` | Recurrent latent tokens | Cross-attention memory read/write control |
| `loop` | Damped, rotating, gated vector | Proposed continuing-state controller |
| `loop_no_rotation` | Same vector without rotation | Does circulation earn its cost? |
| `loop_hard` | Same vector, one operator chosen | Hard routing ablation; straight-through gradients |
| `window` | Last W encoded observations | Fixed history window with temporal attention |

These share data streams, seeds, pretrained backbone and training schedule, **not exactly equal parameter or compute budgets**. `config.json`, `results.json` and `summary.csv` expose the actual counts. The window stores encoded observations rather than raw frames or a full language KV cache. The token control is RMT-inspired, not a reproduction of published RMT. None is a reproduction of Titans, TTT or Infini-attention.

## Checkpoints, resume and results

Each run folder contains:

- `config.json`: architecture, training settings and parameter/state budget.
- `validation.csv`: progress, validation accuracy/NLL and elapsed time.
- `latest.pt`: resumable training checkpoint, saved at validation intervals.
- `best.pt`: selected **only by validation NLL**.
- `results.json`: independent test, phase scores, longer delays, state reset/shuffle, visual noise, runtime and environment.

```powershell
.\.venv\Scripts\python -m transformerx.run --resume runs/language-gpu/latest.pt --steps 2000
.\.venv\Scripts\python -m transformerx.run --evaluate runs/language-gpu/best.pt --device cuda
.\.venv\Scripts\python -m transformerx.suite --summary-only --out runs/pretrained-suite
```

`--steps` is the total target, not an additional count. Resume restores training settings; incompatible architecture or episode/optimizer settings are rejected. If training finished but final evaluation was interrupted, resuming with the same total finishes the report without extra training. Completed steps since the last validation checkpoint are lost on interruption. Do not reuse an existing run folder without `--resume`. Frozen pretrained checkpoints contain trainable additions rather than duplicated backbone weights; keep the Hugging Face cache available. Standalone evaluation writes `evaluation.json`. Sweep summaries reject duplicate seeds or mixed settings within one model group; use separate folders for different experiments.

State reset removes memory before **every** event. Shuffle rotates memory between streams before scored queries. Shuffle needs batch >=2; it is marked unavailable for the default batch-1 vision preset. Long-gap evaluation uses `3 * training_gap + 2`. Noise evaluation adds pixel noise to all visual frames, including the query marker. Test scores inspected during development must not subsequently be called blind confirmation.

## What the loop actually does

At an internal transformer block, a soft router reads the current representation and preceding memory. It chooses a mixture of small low-rank residual operators and adds a projection of the old memory. The modified representation continues through the remaining transformer blocks. Only then does the controller update the memory for the next event.

The loop update rotates coordinate pairs, damps them, and adds a context-gated write, followed by `tanh`. State is bounded and explicit per stream. It starts at zero for every independent episode. Time advances per observed event; no background wall-clock simulation is running. No later event or query label enters the forward computation.

Adapter weights learn during supervised training and remain fixed during evaluation. This version tests **inference-time state adaptation**, not online weight learning, structural consolidation or protected lifelong memory. Mixing adapter outputs does not preserve separate nonlinear hypothesis trajectories. None of the benchmark results implies a Möbius-specific advantage or a biological mechanism.

Read [the experiment protocol](docs/PROTOCOL.md), [implementation design](docs/design.md), and [local verification record](results/LOCAL_VALIDATION.md). Heavy GPU results should be compared across seeds and budgets before making a performance claim.

## Relevant prior work

- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)
- [Infini-attention](https://arxiv.org/abs/2404.07143)
- [Test-Time Training layers](https://arxiv.org/abs/2407.04620)
- [Titans](https://arxiv.org/abs/2501.00663)

MIT license. Built as a falsifiable next step from RoundAndRound.

## Silent-rule experiment

See [SILENT_RULES.md](SILENT_RULES.md) for outcome-driven silent rule switches, a matched multiplicative/additive router comparison, exact belief-filter references, and non-destructive read-disconnection tests. The original commands above remain available.
