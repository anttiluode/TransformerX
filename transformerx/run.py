"""Train, resume and evaluate both tasks. Run: python -m transformerx.run --help."""
import argparse
import csv
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F
from .data import make_episode, batch_episodes
from .model import Config, StreamModel


@dataclass
class TrainOptions:
    steps: int = 500
    batch: int = 8
    lr: float = .001
    seed: int = 7
    gap: int = 2
    noise: float = 0.
    tbptt: int = 32
    eval_every: int = 50
    eval_episodes: int = 64
    test_episodes: int = 128
    device: str = 'auto'
    amp: bool = False
    threads: int = 4
    out: str = 'runs/experiment'


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.')
    return torch.device(name)


def validate_options(opts):
    if min(opts.steps, opts.batch, opts.tbptt, opts.eval_every, opts.eval_episodes,
           opts.test_episodes, opts.threads) < 1 or opts.lr <= 0:
        raise ValueError('Training counts, threads and learning rate must be positive')
    if opts.gap < 0 or opts.noise < 0 or opts.seed < 0:
        raise ValueError('Gap, noise and seed must be nonnegative')


def get_batch(cfg, opts, split, start, count, gap=None, noise=None):
    episodes = [make_episode(cfg.task, split, opts.seed * 10_000_000 + start + i,
                 opts.gap if gap is None else gap, cfg.image_size,
                 opts.noise if noise is None else noise) for i in range(count)]
    return batch_episodes(episodes)


@torch.inference_mode()
def evaluate(model, opts, split='val', episodes=None, intervention='normal', gap=None, noise=None):
    model.eval()
    episodes = episodes or opts.eval_episodes
    if intervention not in ('normal', 'reset', 'shuffle'):
        raise ValueError('Unknown state intervention')
    if intervention == 'shuffle' and (opts.batch < 2 or episodes < 2):
        return {'available': False, 'reason': 'State shuffling requires batch and episode count >= 2'}
    counts = {}
    total_loss = correct = total = confidence_sum = entropy_sum = entropy_count = 0
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    examples = []
    for start in range(0, episodes, opts.batch):
        count = min(opts.batch, episodes - start)
        batch = get_batch(model.cfg, opts, split, start, count, gap, noise)
        state = model.initial_state(count)
        for t, obs in enumerate(batch.observations):
            targets = batch.targets[t].to(device)
            valid = targets >= 0
            if intervention == 'reset':
                state = model.initial_state(count)
            elif intervention == 'shuffle' and valid.any():
                # Deterministic derangement for complete batches; a singleton tail
                # cannot be shuffled and is omitted, recorded by the query count.
                if count > 1:
                    state = {key: value.roll(1, 0) for key, value in state.items()}
                else:
                    valid = torch.zeros_like(valid)
            logits, state, diagnostics = model.forward_step(obs, state)
            if 'route_entropy' in diagnostics:
                entropy_sum += diagnostics['route_entropy'].sum().item()
                entropy_count += count
            if not valid.any():
                continue
            pred = logits.argmax(-1)
            correct_now = pred.eq(targets)
            n = valid.sum().item()
            total += n
            correct += (correct_now & valid).sum().item()
            total_loss += F.cross_entropy(logits[valid].float(), targets[valid], reduction='sum').item()
            confidence_sum += logits[valid].float().softmax(-1).amax(-1).sum().item()
            for i in range(count):
                if valid[i]:
                    phase = batch.phases[t][i]
                    row = counts.setdefault(phase, {'correct': 0, 'queries': 0})
                    row['correct'] += int(correct_now[i]); row['queries'] += 1
                    if len(examples) < 12:
                        examples.append({'episode': start+i, 'event': t, 'phase': phase,
                                         'target': int(targets[i]), 'prediction': int(pred[i])})
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return {'available': True, 'accuracy': correct / max(total, 1),
            'nll': total_loss / max(total, 1), 'queries': total,
            'mean_confidence': confidence_sum / max(total, 1),
            'route_entropy': entropy_sum / entropy_count if entropy_count else None,
            'phases': {k: {**v, 'accuracy': v['correct']/v['queries']} for k, v in counts.items()},
            'seconds': time.perf_counter() - started, 'examples': examples}


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def model_weights(model):
    # Downloaded pretrained weights can be reconstructed; tiny models are saved whole.
    omit_base = model.cfg.backbone == 'hf' and model.cfg.freeze_backbone
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not (omit_base and k.startswith('backbone.'))}


def save_checkpoint(path, model, optimizer, scaler, opts, step, best):
    payload = {'format_version': 1, 'model_config': asdict(model.cfg), 'options': asdict(opts),
               'model': model_weights(model), 'optimizer': optimizer.state_dict(),
               'scaler': scaler.state_dict(), 'step': step, 'best_val_nll': best,
               'rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def restore_weights(model, payload):
    result = model.load_state_dict(payload['model'], strict=False)
    allowed = model.cfg.backbone == 'hf' and model.cfg.freeze_backbone
    missing = [k for k in result.missing_keys if not (allowed and k.startswith('backbone.'))]
    if missing or result.unexpected_keys:
        raise ValueError(f'Checkpoint weight mismatch: {missing}, {result.unexpected_keys}')


def read_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint version')
    return payload


def load_model(path, device='cpu'):
    payload = read_checkpoint(path)
    model = StreamModel(Config(**payload['model_config'])).to(device_for(str(device)))
    restore_weights(model, payload)
    return model.eval(), payload


def optimizer_step(objective, optimizer, scaler, parameters):
    parameters = list(parameters)
    scaler.scale(objective).backward()
    scaler.unscale_(optimizer)
    # FP16 overflow is recoverable: GradScaler records it during unscale, skips
    # this optimizer step, and lowers the scale. Unscaled/BF16 runs remain strict.
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()


def train(cfg, opts, resume=None):
    validate_options(opts)
    device = device_for(opts.device)
    torch.set_num_threads(opts.threads)
    torch.manual_seed(opts.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.reset_peak_memory_stats(device)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (out / 'latest.pt').exists():
        raise ValueError('Output already has a checkpoint; use --resume or a new --out directory')
    model = StreamModel(cfg).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=opts.lr)
    amp = opts.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and amp_dtype == torch.float16)
    first, best = 0, float('inf')
    if resume:
        payload = read_checkpoint(resume)
        if payload['model_config'] != asdict(cfg):
            raise ValueError('Resume model config differs from checkpoint config')
        # Episode ordering and optimization must stay compatible for a genuine resume.
        for field in ('batch', 'seed', 'gap', 'noise', 'tbptt', 'lr', 'amp'):
            if payload['options'][field] != getattr(opts, field):
                raise ValueError(f'Resume options config differs for {field}')
        restore_weights(model, payload)
        optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        first, best = payload['step'], payload['best_val_nll']
        torch.set_rng_state(payload['rng'])
        if device.type == 'cuda' and payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        if opts.steps < first:
            raise ValueError(f'--steps is total steps and cannot precede resumed step {first}')
        # A new output folder must carry the original best checkpoint to select fairly.
        source_best = Path(resume).parent / 'best.pt'
        if source_best.resolve() != (out / 'best.pt').resolve():
            if not source_best.exists():
                raise ValueError('Resume requires best.pt alongside latest.pt')
            import shutil
            shutil.copy2(source_best, out / 'best.pt')
    atomic_json(out / 'config.json', {'model': asdict(cfg), 'training': asdict(opts), 'budget': model.budget()})
    log_path = out / 'validation.csv'
    started = time.perf_counter()
    for step in range(first, opts.steps):
        model.train()
        batch = get_batch(cfg, opts, 'train', step * opts.batch, opts.batch)
        state = model.initial_state(opts.batch)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        train_loss, train_queries = 0., 0
        for t, obs in enumerate(batch.observations):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                logits, state, _ = model.forward_step(obs, state)
                target = batch.targets[t].to(device)
                valid = target >= 0
                if valid.any():
                    loss = F.cross_entropy(logits[valid].float(), target[valid])
                    losses.append(loss)
                    train_loss += loss.detach().item() * int(valid.sum())
                    train_queries += int(valid.sum())
            if (t + 1) % opts.tbptt == 0 or t + 1 == len(batch.observations):
                if losses:
                    objective = torch.stack(losses).mean()
                    if not torch.isfinite(objective):
                        raise FloatingPointError('Nonfinite loss; lower --lr or disable --amp')
                    optimizer_step(objective, optimizer, scaler, model.parameters())
                # Keep values between chunks, truncate their graph before modifying weights again.
                state = model.detach_state(state)
                losses = []
                optimizer.zero_grad(set_to_none=True)
        if (step + 1) % opts.eval_every == 0 or step + 1 == opts.steps:
            val = evaluate(model, opts)
            is_best = val['nll'] < best
            if is_best:
                best = val['nll']
                save_checkpoint(out / 'best.pt', model, optimizer, scaler, opts, step + 1, best)
            save_checkpoint(out / 'latest.pt', model, optimizer, scaler, opts, step + 1, best)
            exists = log_path.exists()
            with log_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['step', 'train_nll', 'val_nll', 'val_accuracy', 'seconds'])
                if not exists:
                    writer.writeheader()
                writer.writerow({'step': step+1, 'train_nll': train_loss/max(train_queries, 1),
                                 'val_nll': val['nll'], 'val_accuracy': val['accuracy'],
                                 'seconds': time.perf_counter()-started})
            print(f'{cfg.task}/{cfg.mode} step {step+1}/{opts.steps} '
                  f'train {train_loss/max(train_queries,1):.4f} val {val["nll"]:.4f} '
                  f'accuracy {val["accuracy"]:.3f}', flush=True)
    # Selection uses validation only. Test and interventions happen after training.
    best_model, best_payload = load_model(out / 'best.pt', str(device))
    test = {name: evaluate(best_model, opts, 'test', opts.test_episodes, intervention=name)
            for name in ('normal', 'reset', 'shuffle')}
    test['long_gap'] = evaluate(best_model, opts, 'test', opts.test_episodes, gap=opts.gap * 3 + 2)
    test['noise'] = (evaluate(best_model, opts, 'test', opts.test_episodes, noise=max(.08, opts.noise * 2.5))
                     if cfg.task == 'vision' else {'available': False, 'reason': 'Pixel noise applies to vision only'})
    report = {'status': 'experiment result; superiority requires matched multi-seed comparisons',
              'completed_steps': opts.steps, 'selected_step': best_payload['step'],
              'model': asdict(cfg), 'training': asdict(opts), 'budget': model.budget(),
              'test': test, 'seconds_this_invocation': time.perf_counter()-started,
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'device': str(device), 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0}
    atomic_json(out / 'results.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='JSON with model and training sections')
    p.add_argument('--resume', help='latest.pt; restores settings unless explicitly overridden')
    p.add_argument('--evaluate', help='Evaluate a checkpoint without training')
    for cls in (Config, TrainOptions):
        for field in fields(cls):
            if field.type is bool:
                p.add_argument('--' + field.name.replace('_', '-'), action=argparse.BooleanOptionalAction,
                               default=argparse.SUPPRESS)
            else:
                p.add_argument('--' + field.name.replace('_', '-'), type=field.type,
                               default=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    config_file, resume, checkpoint = (args.pop(k) for k in ('config', 'resume', 'evaluate'))
    if resume and checkpoint:
        raise ValueError('Choose --resume or --evaluate')
    base = json.loads(Path(config_file).read_text()) if config_file else {'model': {}, 'training': {}}
    if resume or checkpoint:
        saved = read_checkpoint(resume or checkpoint)
        base = {'model': saved['model_config'], 'training': saved['options']}
        base['training']['out'] = str(Path(resume or checkpoint).parent)
    model_names = {f.name for f in fields(Config)}
    cfg = Config(**(base.get('model', {}) | {k: v for k, v in args.items() if k in model_names}))
    opts = TrainOptions(**(base.get('training', {}) | {k: v for k, v in args.items() if k not in model_names}))
    if checkpoint:
        validate_options(opts)
        torch.set_num_threads(opts.threads)
        model, _ = load_model(checkpoint, str(device_for(opts.device)))
        result = evaluate(model, opts, 'test', opts.test_episodes)
        dest = Path(opts.out)
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / 'evaluation.json', result)
        print(json.dumps(result, indent=2))
    else:
        report = train(cfg, opts, resume=resume)
        print(f'Results: {Path(opts.out) / "results.json"}; test accuracy {report["test"]["normal"]["accuracy"]:.3f}')


if __name__ == '__main__':
    main()
