import json
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize('task', ['language', 'vision'])
def test_train_checkpoint_eval_resume(tmp_path, task):
    from transformerx.model import Config
    from transformerx.run import TrainOptions, train, load_model, evaluate
    cfg = Config(task=task, dim=32, state_dim=16, rank=4)
    opts = TrainOptions(steps=2, batch=2, eval_every=2, eval_episodes=4,
                        test_episodes=4, tbptt=32, out=str(tmp_path / task))
    report = train(cfg, opts)
    assert report['completed_steps'] == 2
    assert report['test']['normal']['queries'] > 0
    assert set(report['test']) == {'normal', 'reset', 'shuffle', 'long_gap', 'noise'}
    checkpoint = Path(opts.out) / 'latest.pt'
    model, payload = load_model(checkpoint, 'cpu')
    assert payload['step'] == 2
    a = evaluate(model, opts, 'test', episodes=4)
    b = evaluate(model, opts, 'test', episodes=4)
    assert a['accuracy'] == b['accuracy']
    assert a['nll'] == b['nll']
    opts.steps = 3
    resumed = train(cfg, opts, resume=checkpoint)
    assert resumed['completed_steps'] == 3
    assert json.loads((Path(opts.out) / 'results.json').read_text())['completed_steps'] == 3


def test_resume_mismatch_rejected(tmp_path):
    from transformerx.model import Config
    from transformerx.run import TrainOptions, train
    cfg = Config(dim=32, state_dim=16)
    opts = TrainOptions(steps=1, batch=2, eval_episodes=2, test_episodes=2, out=str(tmp_path))
    train(cfg, opts)
    with pytest.raises(ValueError, match='config'):
        train(Config(dim=32, state_dim=16, mode='none'), opts, resume=tmp_path / 'latest.pt')


def test_invalid_run_options(tmp_path):
    from transformerx.model import Config
    from transformerx.run import TrainOptions, train
    with pytest.raises(ValueError):
        train(Config(), TrainOptions(steps=0, out=str(tmp_path)))


def test_fp16_scaler_can_skip_overflow_then_recover():
    from transformerx.run import optimizer_step
    p = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    loss = p * torch.tensor(1e35)  # finite loss, scaled gradient overflows
    optimizer_step(loss, optimizer, scaler, [p])
    assert p.item() == 1.
    assert scaler.get_scale() < 65536.
    optimizer.zero_grad(set_to_none=True)
    optimizer_step(p.square(), optimizer, scaler, [p])
    assert p.item() < 1.


def test_interrupted_final_report_can_resume_without_extra_training(tmp_path):
    from transformerx.model import Config
    from transformerx.run import TrainOptions, train, read_checkpoint
    cfg = Config(dim=32, state_dim=16)
    opts = TrainOptions(steps=1, batch=2, eval_episodes=2, test_episodes=2, out=str(tmp_path))
    train(cfg, opts)
    before = read_checkpoint(tmp_path / 'latest.pt')['model']
    (tmp_path / 'results.json').unlink()
    report = train(cfg, opts, resume=tmp_path / 'latest.pt')
    after = read_checkpoint(tmp_path / 'latest.pt')['model']
    assert report['completed_steps'] == 1
    assert (tmp_path / 'results.json').exists()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_legacy_checkpoint_defaults_can_resume(tmp_path):
    from transformerx.model import Config
    from transformerx.run import TrainOptions, train, read_checkpoint
    cfg=Config(dim=16,state_dim=8,layers=1,rank=2)
    opts=TrainOptions(steps=1,batch=1,eval_episodes=1,test_episodes=1,
                      threads=1,out=str(tmp_path))
    train(cfg,opts)
    path=tmp_path/'latest.pt'
    payload=read_checkpoint(path)
    payload['model_config'].pop('router')
    payload['model_config'].pop('num_classes')
    torch.save(payload,path)
    opts.steps=2
    assert train(cfg,opts,resume=path)['completed_steps']==2
