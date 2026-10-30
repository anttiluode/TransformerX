import json
import pytest


def test_suite_dry_run_has_both_tasks_paired_seeds(tmp_path):
    from transformerx.suite import main
    main(['--tasks', 'language', 'vision', '--modes', 'none', 'loop', '--seeds', '7', '8',
          '--steps', '2', '--out', str(tmp_path), '--dry-run'])
    manifest = json.loads((tmp_path / 'suite.json').read_text())
    assert len(manifest['runs']) == 8
    assert len({r['out'] for r in manifest['runs']}) == 8
    assert {r['task'] for r in manifest['runs']} == {'language', 'vision'}
    assert all('--steps' in r['command'] for r in manifest['runs'])


def test_summary_reports_actual_budgets_and_seed_count(tmp_path):
    from transformerx.suite import summarize
    for seed in [7, 8]:
        folder = tmp_path / str(seed)
        folder.mkdir()
        (folder / 'results.json').write_text(json.dumps({
            'model': {'task': 'language', 'mode': 'loop', 'backbone': 'tiny'},
            'training': {'seed': seed}, 'budget': {'trainable_parameters': 17,
                'persistent_scalars_per_stream': 16}, 'test': {'normal': {'accuracy': .5,
                'nll': 1., 'seconds': 2.}}, 'peak_cuda_bytes': 0}))
    rows = summarize(tmp_path)
    assert rows[0]['seeds'] == 2
    assert rows[0]['accuracy_mean'] == .5
    assert rows[0]['trainable_parameters'] == 17


def test_summary_rejects_duplicate_seeds_and_inconsistent_training(tmp_path):
    from transformerx.suite import summarize
    result = {'model': {'task': 'language', 'mode': 'loop', 'backbone': 'tiny'},
              'training': {'seed': 7, 'steps': 10, 'out': 'a'},
              'budget': {'trainable_parameters': 17, 'persistent_scalars_per_stream': 16},
              'test': {'normal': {'accuracy': .5, 'nll': 1., 'seconds': 2.}}}
    for name in ['a', 'b']:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='seed'):
        summarize(tmp_path)
    result['training']['seed'] = 8
    result['training']['steps'] = 20
    (tmp_path / 'b' / 'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='settings'):
        summarize(tmp_path)
