import pytest
import torch
from transformerx.data import make_episode, batch_episodes

torch.set_num_threads(1)


@pytest.mark.parametrize('task', ['language', 'vision'])
@pytest.mark.parametrize('mode', ['none', 'static', 'gru', 'tokens', 'loop', 'loop_no_rotation', 'loop_hard', 'window'])
def test_modes_forward_backward(task, mode):
    from transformerx.model import Config, StreamModel
    torch.manual_seed(9)
    model = StreamModel(Config(task=task, mode=mode, dim=32, state_dim=16, layers=2, rank=4))
    batch = batch_episodes([make_episode(task, 'train', i) for i in range(2)])
    state = model.initial_state(2)
    for obs in batch.observations[:3]:
        logits, state, diag = model.forward_step(obs, state)
    loss = logits.square().mean()
    loss.backward()
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert model.budget()['trainable_parameters'] > 0
    assert all(not x.requires_grad for x in model.detach_state(state).values() if isinstance(x, torch.Tensor))


def test_history_reset_and_stream_isolation():
    from transformerx.model import Config, StreamModel
    torch.manual_seed(13)
    model = StreamModel(Config(mode='loop', dim=32, state_dim=16, layers=2, rank=4)).eval()
    start = model.initial_state(2)
    obs = ['in workshop cedar word dax means red', 'in workshop cedar word dax means blue']
    _, changed, _ = model.forward_step(obs, start)
    query = ['in workshop cedar translate dax'] * 2
    a, _, _ = model.forward_step(query, changed)
    b, _, _ = model.forward_step(query, model.initial_state(2))
    c, _, _ = model.forward_step(query, model.initial_state(2))
    assert not torch.allclose(a[0], a[1])
    assert not torch.allclose(a, b)
    torch.testing.assert_close(b, c)
    assert torch.count_nonzero(start['memory']) == 0  # caller-owned state not mutated


def test_adapter_changes_internal_computation_and_gets_gradients():
    from transformerx.model import Config, StreamModel
    model = StreamModel(Config(mode='loop', dim=32, state_dim=16, layers=2, rank=4))
    state = model.initial_state(2)
    for _ in range(3):
        logits, state, _ = model.forward_step(['in workshop cedar translate dax'] * 2, state)
    logits.square().sum().backward()
    assert model.controller.up.grad.abs().sum() > 0
    assert model.controller.write.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('task', ['language', 'vision'])
def test_hf_random_backbone_frozen_but_adapter_trainable(task):
    transformers = pytest.importorskip('transformers')
    from transformerx.model import Config, StreamModel
    if task == 'language':
        config = transformers.GPT2Config(vocab_size=40, n_embd=32, n_layer=2, n_head=4, n_positions=64,
                                         resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
        backbone = transformers.GPT2Model(config)
    else:
        config = transformers.ViTConfig(image_size=32, patch_size=8, hidden_size=32,
                    num_hidden_layers=2, num_attention_heads=4, intermediate_size=64)
        backbone = transformers.ViTModel(config)
    model = StreamModel(Config(task=task, backbone='hf', mode='loop', state_dim=16, rank=4),
                        backbone=backbone)
    batch = batch_episodes([make_episode(task, 'train', i) for i in range(2)])
    state = model.initial_state(2)
    for obs in batch.observations[:2]:
        logits, state, _ = model.forward_step(obs, state)
    logits.sum().backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert model.controller.up.grad.abs().sum() > 0
    assert len(model.hook_block._forward_hooks) == 0


def test_invalid_config_fails_early():
    from transformerx.model import Config, StreamModel
    with pytest.raises(ValueError):
        StreamModel(Config(state_dim=15))
    with pytest.raises(ValueError):
        StreamModel(Config(mode='invented'))
