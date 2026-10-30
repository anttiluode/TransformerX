# TransformerX

**A continuing state that changes how a transformer processes its next observation.**

Two experiments, one memory mechanism:

- **Language:** remember changing dictionaries in two contexts, answer after distracting events, and return to an earlier context. Pretrained option: DistilGPT-2.
- **Vision:** observe two moving objects, track direction changes, and answer about either object after both disappear. Pretrained option: ViT.

This is an experimental implementation inspired by [RoundAndRound](https://github.com/anttiluode/RoundAndRound) and [LoopAImemory](https://github.com/anttiluode/LoopAImemory). **No improvement over existing transformers has been established.** The language task is supervised classification of textual queries, not a chat model or language-generation benchmark. The visual scenes are synthetic. Pretrained integrations are real model modifications; local verification uses small randomly initialized GPT-2/ViT configurations, not downloaded pretrained accuracy or GPU performance.

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

Defaults are conservative: language batch 2, vision batch 1, mixed precision, 64 memory coordinates, four rank-8 operators. **VRAM use has not been measured here.** If memory runs out, lower `--batch` first. Lowering `--tbptt` reduces activation memory but also shortens the learning credit horizon; a horizon shorter than the evidence-to-query delay can prevent the write mechanism from learning. `--tbptt 32` covers these default episodes. Lowering `--image-size` changes the synthetic source image, but pretrained ViT still resizes it to its configured resolution.

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

## Silent-rule branch experiment

See [SILENT_RULES.md](SILENT_RULES.md) for outcome-driven silent rule switches, a matched multiplicative/additive router comparison, exact belief-filter references, and non-destructive read-disconnection tests. The original commands above remain available.
