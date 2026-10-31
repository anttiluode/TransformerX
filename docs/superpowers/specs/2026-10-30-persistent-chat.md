# Persistent Qwen chat

The user selected a general persistent chatbot and asked to publish it on the
TransformerX GitHub project page. Build a local browser app for Windows/Linux,
defaulting to Qwen/Qwen3-0.6B in non-thinking mode. Preserve its causal language
model head; freeze backbone parameters. No GPU sweep is authorized or needed here.

## Approach

The alternatives discussed were another classification benchmark, a controlled
conversational benchmark, and a general chatbot. The user chose the chatbot.
Use durable transcript storage, editable explicit notes, bounded recent context,
and lexical recall of older exchanges for immediate utility. These are text
memory, not evidence of a learned neural-loop improvement.

Provide an optional trainable 64-coordinate loop inside a Qwen decoder layer,
with concat routing and the existing bounded recurrent update. Existing binary
DistilGPT-2 checkpoints cannot initialize this controller. Never enable random
adapter perturbations in the default chatbot. A separate GPU training command
accepts supervised multi-turn JSONL episodes and saves compatible controllers.

## Behavioral contracts

- Sessions, notes, transcripts and optional latent state persist atomically under
  ignored `chat-data/`; no conversation is published with the code.
- Use a localhost-only browser UI with session selection, new conversation,
  editable notes, text-memory toggle, trained-loop toggle, reset-loop control,
  export and delete. Show which memory mechanisms are active and recalled turns.
- Token budgets retain the latest user message or reject it clearly. Drop older
  context before truncating notes; never silently truncate the current request.
- Neural routing is token-local and causal. The old state is fixed throughout a
  response; update once from the prompt, then commit after successful generation.
  Never pool future response tokens into routing or memory writes.
- A state is tied to a model/controller identity. Incompatible state is reset
  with a visible explanation. Disabling the loop preserves its saved state.
- Failures do not append partial turns or change latent state. Serialize model
  use and reject stale session revisions from multiple tabs.
- Training freezes Qwen, supervises assistant tokens only, and carries gradients
  across short episodes so earlier writes can learn from later responses.
- This app is experimental: no chatbot improvement or forgetting prevention is
  claimed. GPU speed and response quality remain to be measured by the user.

## Implementation choices

Use Python stdlib HTTP/JSON plus a packaged HTML/CSS/JS interface (no Node build
or hosted service). Use PyTorch and Transformers 4.51–4.x. Start with FP32
controller parameters and FP16 backbone on CUDA, FP32 on CPU. Default short
contexts; make no fixed VRAM guarantee. Unit tests use tiny randomly initialized
Qwen models, with no downloads. Keep existing experiments unchanged.
