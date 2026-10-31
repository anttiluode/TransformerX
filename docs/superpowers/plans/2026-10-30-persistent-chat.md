# Persistent Qwen Chat Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task.

**Goal:** Ship a local persistent Qwen chatbot and a compatible loop-training path.

**Architecture:** Atomic session storage feeds a bounded prompt builder. A frozen
Qwen causal LM generates replies, optionally modified by a trained causal loop.
A localhost browser app exposes memory controls and session management.

**Tech Stack:** Python, PyTorch, Transformers, standard-library HTTP, HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-10-30-persistent-chat.md`

## Global Constraints

- Default Qwen/Qwen3-0.6B; no random controller enabled by default.
- Local data ignored by git. Keep all existing experiments compatible.
- Token-local routing, fixed state during decode, one prompt-only write per turn.
- Train assistant tokens only with gradients through previous writes.

## Review Focus

- Prompt overflow must retain the current question and valid chat-template boundaries.
- Restart, incompatible checkpoints and disabled loops must not silently reuse wrong state.
- Failed generation and simultaneous tabs must not corrupt or overwrite a conversation.
- Teacher forcing must not leak future tokens through pooled routing or writes.
- Browser requests must remain localhost-only; render user/model content as text.

### Task 1: Durable sessions and bounded prompts
- [x] Add failing tests for persistence, invalid paths, recall, overflow and stale writes.
- [x] Implement `transformerx/chat_store.py` and `chat_context.py`.
- [x] Run storage/context tests and record outcome.

### Task 2: Qwen generation and trainable causal controller
- [x] Test causal prefix invariance, zero-initialized identity, frozen weights,
      cross-turn write gradients, state identity and checkpoint round trips.
- [x] Implement `chat_model.py`, `chat_train.py`; JSONL example is a wiring demo.
- [x] Run real tiny-Qwen integration tests; no pretrained quality claims.

### Task 3: Local app and delivery
- [x] Test service transactions, resets, toggles, HTTP validation and export/delete.
- [x] Implement `chat.py`, packaged browser UI, optional chat dependencies, launcher.
- [x] Document usage, model download, memory semantics, GPU training and limitations.
- [x] Run full tests, CLI/help/package checks, DOM/HTTP smoke test and fresh code review (visual browser download unavailable).
- [ ] Publish source/tests/docs to main using the existing authorization; verify remote.
