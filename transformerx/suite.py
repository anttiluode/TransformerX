"""Sequential GPU sweeps with paired seeds and CSV summaries; no shell-specific syntax."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from .memory import MODES


def summarize(root):
    groups = {}
    for path in sorted(Path(root).glob('*/results.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        key = tuple(result['model'][k] for k in ('task', 'mode', 'backbone'))
        groups.setdefault(key, []).append(result)
    rows = []
    for (task, mode, backbone), results in sorted(groups.items()):
        seeds = [r['training']['seed'] for r in results]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seed in {task}/{mode}/{backbone}; use separate suite folders')
        def signature(result):
            settings = {k: v for k, v in result['training'].items() if k not in ('seed', 'out')}
            return json.dumps([result['model'], settings, result['budget']], sort_keys=True)
        if len({signature(r) for r in results}) != 1:
            raise ValueError(f'Inconsistent settings in {task}/{mode}/{backbone}; use separate suite folders')
        scores = [r['test']['normal']['accuracy'] for r in results]
        row = {'task': task, 'mode': mode, 'backbone': backbone, 'seeds': len(results),
               'accuracy_mean': statistics.mean(scores),
               'accuracy_sd': statistics.stdev(scores) if len(scores) > 1 else 0.,
               'nll_mean': statistics.mean(r['test']['normal']['nll'] for r in results),
               'trainable_parameters': results[0]['budget']['trainable_parameters'],
               'persistent_scalars_per_stream': results[0]['budget']['persistent_scalars_per_stream'],
               'evaluation_seconds_mean': statistics.mean(r['test']['normal']['seconds'] for r in results)}
        for intervention in ('reset', 'shuffle', 'long_gap', 'noise'):
            values = [r['test'][intervention]['accuracy'] for r in results
                      if r['test'].get(intervention, {}).get('available', False)]
            row[intervention + '_accuracy'] = statistics.mean(values) if values else ''
        rows.append(row)
    if rows:
        with (Path(root) / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', choices=['language', 'vision'], default=['language', 'vision'])
    p.add_argument('--modes', nargs='+', choices=MODES, default=['none', 'static', 'gru', 'tokens', 'loop', 'loop_no_rotation', 'window'])
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 17, 27])
    p.add_argument('--backbone', choices=['tiny', 'hf'], default='tiny')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--out', default='runs/suite')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Skip completed runs; resume interrupted runs')
    args = p.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        print(json.dumps(summarize(root), indent=2)); return
    runs = []
    for task in args.tasks:
        for mode in args.modes:
            for seed in args.seeds:
                out = root / f'{task}-{args.backbone}-{mode}-s{seed}'
                cmd = [sys.executable, '-m', 'transformerx.run', '--task', task, '--mode', mode,
                       '--backbone', args.backbone, '--seed', str(seed), '--steps', str(args.steps),
                       '--device', args.device, '--out', str(out)]
                if args.backbone == 'hf':
                    cmd += ['--batch', str(args.batch or (2 if task == 'language' else 1)), '--amp',
                            '--lr', '0.0003', '--eval-every', '50']
                elif args.batch:
                    cmd += ['--batch', str(args.batch)]
                runs.append({'task': task, 'mode': mode, 'seed': seed, 'out': str(out), 'command': cmd})
    (root / 'suite.json').write_text(json.dumps({'runs': runs}, indent=2) + '\n', encoding='utf-8')
    for run in runs:
        cmd = run['command']
        output = Path(run['out'])
        if args.resume and (output / 'results.json').exists():
            result = json.loads((output / 'results.json').read_text())
            if result['completed_steps'] >= args.steps:
                continue
        if args.resume and (output / 'latest.pt').exists():
            cmd = [sys.executable, '-m', 'transformerx.run', '--resume', str(output / 'latest.pt'),
                   '--steps', str(args.steps), '--device', args.device]
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    if not args.dry_run:
        # An extended/resumed sweep is temporarily heterogeneous until all runs finish.
        summarize(root)


if __name__ == '__main__':
    main()
