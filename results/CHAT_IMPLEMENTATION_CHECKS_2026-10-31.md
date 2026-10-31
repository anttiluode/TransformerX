# Persistent chat implementation checks — 31 October 2026

This records software verification, not evidence that the loop improves Qwen.

- Full Python suite: **77 tests passed**, including the existing 62 tests and
  15 chat/storage/training tests. CPU, PyTorch 2.15.0+cpu, Transformers 4.57.6.
- Real tiny randomly initialized Qwen3 models: causal prefix invariance,
  prompt-only writes, cached generation, zero-initialized adapter identity,
  cross-turn write gradients, frozen backbone weights and checkpoint round trips.
- Local HTTP/storage: atomic persistence, stale-tab rejection, failed generation,
  reset, toggles, export/delete, input validation and origin restrictions.
- DOM + live HTTP smoke test (`tests/chat_ui_smoke.cjs`, jsdom, simulated model
  output): send, notes, reload, session switching, deletion, inert model HTML and
  controller-unavailable restart. The restart case failed before its fix and
  passed afterward. This is functional DOM testing, not visual browser QA.
- Both CLI help commands run; the built wheel includes `chat_ui.html`.
- Official Qwen3-0.6B tokenizer: all eight supplied training turns fit the
  256-token limit (71, 69, 75, 63, 70, 64, 71, 78 tokens).
- Independent read-only code review found no Critical or Important issues; its
  one restart-toggle finding was fixed and regression-tested.

Pretrained response quality, controller improvement, CUDA memory/performance and
native Windows execution were **not measured here**. No long training job ran.
Chromium download was unavailable in the verification environment, so visual
desktop/mobile inspection is not claimed. The neural loop requires a separately
trained Qwen chat controller; persistent text memory works without one.
