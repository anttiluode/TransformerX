# Persistent Qwen chatbot

A local browser chatbot using **Qwen3-0.6B**, with saved conversations, editable
memory notes and retrieval of relevant older exchanges. Close it, restart it and
select the same conversation to continue. Qwen generates through its original
language-model head, rather than the classification head used in the experiments.

**Ready to use:** persistent text memory. **Experimental:** an optional trained
64-value recurrent controller inside Qwen. No pretrained chat controller is
included, and no improvement in chatbot quality has been demonstrated. The
92% silent-rule result does not measure this chatbot.

## Start on Windows

In your existing TransformerX folder and Python environment:

```powershell
git pull
.\.venv\Scripts\python -m pip install -e ".[chat]"
.\run_chat.bat
```

For a new installation, first follow the [environment setup in the README](README.md#start-on-windows),
including the CUDA-enabled PyTorch install. Python 3.12 or 3.13 is recommended.
The launcher opens **http://127.0.0.1:7860** once the model has loaded. The first
launch downloads model weights from Hugging Face; it may take several minutes.
Later, add `--local-files-only` to require cached files and avoid model-download requests.

Equivalent commands, including Linux/macOS:

```bash
python -m pip install -e ".[chat]"
python -m transformerx.chat
python -m transformerx.chat --device cuda --local-files-only
```

The app automatically uses CUDA when available and otherwise uses CPU. CUDA
weights use FP16; CPU weights and the small controller use FP32. Start with the
default 1,536-token prompt budget and 256-token response limit. Actual GPU memory
and speed depend on hardware and context; this implementation has CPU integration
tests, not a measured RTX 3060 chat benchmark. For memory errors, reduce the
prompt budget and response length:

```powershell
.\run_chat.bat --device cuda --context-tokens 768
```

The app is local-only. Keep its terminal open; Ctrl+C stops it. No API key or
hosted chat service is required. There are no tools, code execution, microphone,
web access or file-reading abilities inside the chatbot.

## What persists

| Memory | Behavior |
|---|---|
| Transcript | Full conversation saved locally; survives process restarts |
| Notes | User-editable facts/preferences, specific to each conversation |
| Recent context | Up to four recent exchanges included when they fit |
| Older recall | Up to two older exchanges selected by word overlap with your question |
| Neural state | Optional trained controller's state, saved with its checkpoint identity |

Notes are prioritized, then recent context, then older recall. Everything must
fit the prompt budget. Very long notes may be shortened; a current message that
cannot fit is rejected rather than silently cut. The memory details under each
reply report the included recent/archived turns and any note truncation.

Recall is simple lexical retrieval, not semantic search or guaranteed factual
memory. Qwen may still misread context or invent an answer. Correct important
facts in the notes. Archived assistant text is previous model output, not verified
evidence. The assistant does not automatically extract permanent facts.

**Text memory off** excludes saved notes and all prior exchanges from that reply's
prompt. The transcript is still saved. **Neural loop off** bypasses the adapter
and pauses the saved latent state; skipped turns are not replayed when it is
enabled again. Use **Reset loop** when starting a fresh neural trajectory. Reset
keeps notes/transcript. New conversations start with separate empty memories;
notes do not carry between conversations.

The browser provides save/edit notes, session selection, export, delete, response
length and temperature controls. It displays a waiting timer during generation;
responses currently arrive as complete messages, not streamed tokens. Temperature
0 uses deterministic decoding, useful for comparisons.

Files live in `chat-data/`, ignored by git, as ordinary unencrypted JSON. Export
also includes any saved latent state. Delete removes that conversation's local
file, not previously exported copies. Back up this directory if you want durable
backups. Run one server process per data directory; tabs within that server use
revision checks to prevent overwriting stale conversation state.

## Train an experimental neural loop

The DistilGPT-2 silent-rule checkpoints **cannot** be loaded into Qwen. The chat
controller has its own weights and checkpoint format. Random adapters are never
enabled in the default chat.

Supervised training data uses one episode per JSONL line, with at least two turns:

```json
{"turns":[{"user":"Please use Finnish.","assistant":"Selvä."},{"user":"What is two plus two?","assistant":"Neljä."}]}
```

Start with this **wiring check**, not a long sweep:

```powershell
.\.venv\Scripts\python -m transformerx.chat_train --data examples/chat_training.jsonl --steps 10 --max-turns 2 --max-tokens 256 --device cuda --out runs/qwen-chat-smoke
.\run_chat.bat --controller runs/qwen-chat-smoke/controller.pt
```

Then explicitly enable **Trained neural loop** in the browser. The four included
episodes only exercise the training pipeline. They are too small to establish
useful general memory and can make conversation worse. For meaningful training,
supply varied, corrected multi-turn examples and reserve separate unseen episodes
for evaluation. Exported session JSON is an archive, not automatically a training
file: select/correct responses and convert them to the format above.

Training freezes all Qwen parameters and trains only the controller. Each training
turn receives its current user message, the common system prompt and the preceding
latent state, without earlier text. Assistant tokens provide the loss. Gradients
flow across up to `--max-turns` contiguous turns; longer episodes are sampled as
contiguous windows with fresh initial state. Raising this horizon increases
activation memory. Turns that exceed `--max-tokens` cause an error, not truncation.

The controller is saved at step 1, every 10 steps and the final step, alongside
configuration, data hash and training loss. This is the latest checkpoint, not a
validation-selected best model. Existing output checkpoints are protected: choose
a new `--out` for another run. Optimizer resume is not implemented.

## How the neural path works

The controller uses concat routing, four rank-8 operators and the existing bounded
rotation/write update. It modifies a middle Qwen decoder layer while keeping the
original language-model output head. Routing is **per token**, using that token's
hidden representation and the previous turn's state. This differs from the
whole-observation pooling in the classification experiments and avoids future
answer leakage during teacher forcing.

State stays fixed during the entire response, including cached decoding. Once
generation succeeds, it updates exactly once from the mean modified prompt
representations at the injection layer. Generated answer tokens do not feed this
write. With text memory enabled, that prompt also contains selected older text;
training currently uses current-turn-only prompts, so this combined mode is an
additional distribution shift to evaluate. There is no autonomous wall-clock
rotation and no online weight learning during chat.

A newly initialized controller has zero adapter output and zero state readout,
so it initially leaves Qwen logits unchanged. Loading a different checkpoint
resets incompatible saved state on the next enabled turn and reports that reset.
Changing a model's underlying weights under the same model identifier is not
detected; use a distinct identifier and reset state in that situation.

## Evaluate before claiming improvement

Try the same unseen conversations in separate fresh sessions with temperature 0:

1. Text memory on, loop off: practical baseline.
2. Text memory off, loop off: current-message-only baseline.
3. Text memory off, trained loop on: isolates the neural memory path.
4. Both on: checks whether the combination adds value.

Look for correct preference changes, recovery from corrections, resistance to
irrelevant messages and preservation of ordinary answers. Changing toggles midway
through one trajectory does not give a matched experiment. A retained 64-number
state is not proof of useful memory or a replacement for a factual archive.

Run tests with `python -m pytest -q`. The tests include real tiny randomly
initialized Qwen3 models, frozen-weight checks, causal prefix invariance,
cross-turn write gradients, generation/state checkpointing, persistence, prompt
budgets and local HTTP access. They verify implementation, not pretrained quality.
