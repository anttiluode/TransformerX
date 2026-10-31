"""Train a Qwen chat controller on supervised multi-turn JSONL episodes.

This is an experimental training path, not a pretrained conversational memory.
Each turn receives only its own user message and the preceding latent state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .chat_context import SYSTEM, prompt_ids
from .chat_model import LoopCausalLM, load_qwen


def load_episodes(path):
    episodes = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            turns = value.get('turns') if isinstance(value, dict) else None
            if not isinstance(turns, list) or len(turns) < 2:
                raise ValueError(f'Line {number}: supply at least two supervised turns.')
            for turn in turns:
                if not isinstance(turn, dict) or any(not isinstance(turn.get(k), str) or not turn[k].strip()
                                                     for k in ('user', 'assistant')):
                    raise ValueError(f'Line {number}: every turn needs nonempty user and assistant strings.')
            episodes.append(turns)
    if not episodes:
        raise ValueError('Training file contains no episodes.')
    return episodes


def prepare_turn(tokenizer, turn, max_tokens):
    prompt = prompt_ids(tokenizer, [{'role': 'system', 'content': SYSTEM},
                                    {'role': 'user', 'content': turn['user']}])
    target = tokenizer.encode(turn['assistant'] + tokenizer.eos_token, add_special_tokens=False)
    if not target or len(prompt) + len(target) > max_tokens:
        raise ValueError('Training turn exceeds --max-tokens; shorten the example or raise the budget.')
    ids = torch.tensor([prompt + target], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prompt) + target], dtype=torch.long)
    return ids, labels, len(prompt)


def train_step(model, tokenizer, episode, optimizer, *, max_tokens):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    state = model.zero_state()
    losses = []
    for turn in episode:
        ids, labels, length = prepare_turn(tokenizer, turn, max_tokens)
        output, state = model.forward_turn(ids.to(model.device), state, prompt_length=length,
                                           labels=labels.to(model.device))
        losses.append(output.loss)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite training loss; no optimizer step was taken.')
    loss.backward()
    # No detach between turns: a later answer can train earlier memory writes.
    torch.nn.utils.clip_grad_norm_(model.controller.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='JSONL: {"turns": [{"user": ..., "assistant": ...}, ...]}')
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--out', default='runs/qwen-chat')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max-turns', type=int, default=2, help='Credit horizon; longer uses more activation memory')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args(argv)
    if args.steps < 1 or args.max_turns < 2 or args.max_tokens < 16 or not 0 < args.lr <= .1:
        parser.error('Use positive steps, at least two turns, at least 16 tokens, and 0 < lr <= 0.1.')
    episodes = load_episodes(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'controller.pt').exists():
        parser.error('Output already contains controller.pt; choose a new --out to preserve it.')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    backbone, tokenizer = load_qwen(args.model, args.device, args.local_files_only)
    if args.max_tokens > backbone.config.max_position_embeddings:
        parser.error('--max-tokens exceeds the model context limit.')
    # Fail before training if any supplied example cannot fit; do not silently truncate labels.
    for episode in episodes:
        for turn in episode:
            prepare_turn(tokenizer, turn, args.max_tokens)
    model = LoopCausalLM(backbone, model_name=args.model)
    optimizer = torch.optim.AdamW(model.controller.parameters(), lr=args.lr)
    configuration = dict(vars(args), data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                         episodes=len(episodes), trainable_parameters=sum(p.numel() for p in model.controller.parameters()))
    (out / 'config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    print('Experimental controller training. Training loss does not establish chatbot improvement.', flush=True)
    started = time.monotonic()
    with (out / 'training.jsonl').open('w', encoding='utf-8') as log:
        for step in range(1, args.steps + 1):
            episode = rng.choice(episodes)
            offset = rng.randrange(max(1, len(episode) - args.max_turns + 1))
            loss = train_step(model, tokenizer, episode[offset:offset + args.max_turns], optimizer,
                              max_tokens=args.max_tokens)
            record = dict(step=step, loss=loss, elapsed_seconds=time.monotonic() - started)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                model.save_controller(out / 'controller.pt', trained_steps=step)
                print(f'{step}/{args.steps}: assistant-token loss {loss:.4f}; '
                      f'{record["elapsed_seconds"]:.1f}s; saved {out / "controller.pt"}', flush=True)
    print('Use held-out conversations and the loop-off control before claiming a benefit.')


if __name__ == '__main__':
    main()
