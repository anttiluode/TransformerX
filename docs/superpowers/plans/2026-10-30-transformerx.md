# TransformerX Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, then a fresh whole-project review.

**Goal:** Ship reproducible language and vision experiments for state-controlled transformer adapters, ready for local GPU training.

**Architecture:** Explicit recurrent state and one shared adapter bank integrate with tiny offline transformers and pretrained GPT-2/ViT blocks. Independent procedural streams share a training/evaluation harness and controls.

**Tech Stack:** Python >=3.10, PyTorch >=2.6, NumPy, Pillow; optional Transformers 4.x; pytest.

**Spec:** `docs/design.md`

## Global Constraints
- No language/vision performance advantage claimed from smoke runs.
- Labels and hidden task state cannot enter query observations.
- GPU training runs are left to the user; verify CPU paths and randomly initialized HF integrations locally.
- Record actual parameter and state budgets, random seeds, configs and runtime.

## Review Focus
- Future/target leakage, especially query masks and temporal batching.
- Frozen backbone autograd and hook cleanup; controls must use the same pretrained weights.
- Episode reset, padding, long gaps and state interventions.
- Checkpoint/resume compatibility and validation/test separation.
- Windows commands, absent CUDA, interrupted saves and unsupported model architectures.

### Task 1: Data and split contracts
Files: `transformerx/data.py`, `tests/test_data.py`.
Interface: `make_episode(task, split, seed, gap, image_size)` returns observations, query targets/masks and phase tags; `collate` batches equal-length deterministic episodes.
- [x] Write and run failing tests for determinism, disjoint language mappings, no query answer leakage, vision occlusion and metadata.
- [x] Implement generators, text tokenizer for tiny mode, image rendering and batching.
- [x] Run data tests and commit.

### Task 2: Memory and transformer integrations
Files: `transformerx/memory.py`, `transformerx/model.py`, `tests/test_model.py`.
Interface: model `forward_step(observation, state)` returns logits, updated state and diagnostics; `initial_state(batch, device)` and `detach_state` support streaming.
- [x] Write and run failing tests for history dependence, reset equivalence, causality, gradient flow and all controls.
- [x] Implement low-rank adapter bank, damped rotating state, GRU/tokens/static controls and tiny/HF backbones.
- [x] Run CPU forward/backward tests using tiny GPT-2/ViT configs without network and commit.

### Task 3: Reproducible train/eval and GPU entrypoints
Files: `transformerx/run.py`, `transformerx/suite.py`, `tests/test_training.py`, configs/scripts.
- [x] Write and run failing tests for short training, checkpoint reload/resume and metrics/interventions.
- [x] Implement CLI, AMP, TBPTT, checkpoint selection on validation only, final independent test, delay/noise tests, CSV/JSON exports and sweep launcher.
- [x] Run both CPU smoke tracks and checkpoint evaluation; commit.

### Task 4: Documentation, review and delivery
Files: README, protocol/results notes, CI.
- [x] Write Windows setup, quick runs, full GPU sweeps, resume/evaluate examples and mechanism limits.
- [x] Review with a fresh agent; fix material findings with regression tests.
- [ ] Run complete suite and smoke commands, publish to the requested repository and verify remote files.
