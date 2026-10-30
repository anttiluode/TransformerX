import pytest
import torch
from transformerx.model import Config, StreamModel
from transformerx.silent_data import RuleTokenizer

torch.set_num_threads(1)


def test_router_parameter_matching_and_interaction_gradients():
    models = [StreamModel(Config(dim=32,state_dim=16,rank=4,router=r,num_classes=2),
                          tokenizer=RuleTokenizer()) for r in ('interaction','additive')]
    assert models[0].budget() == models[1].budget()
    for model in models:
        state = model.initial_state(2)
        for text in ['tone two outcome left', 'categorize tone three']:
            logits, state, diag = model.forward_step([text]*2, state)
        logits.square().sum().backward()
        assert model.controller.state_project.weight.grad.abs().sum() > 0
        assert logits.shape == (2,2)
        assert diag['route_weights'].shape == (2,4)


@pytest.mark.parametrize('mode', ['loop','gru','tokens','window'])
def test_read_disconnection_preserves_state_and_restores_exactly(mode):
    model = StreamModel(Config(mode=mode,dim=32,state_dim=16,rank=4,num_classes=2),
                        tokenizer=RuleTokenizer()).eval()
    state = model.initial_state(2)
    for text in ['tone three outcome left', 'tone two outcome right']:
        _, state, _ = model.forward_step([text]*2,state)
    saved = {k:v.clone() for k,v in state.items()}
    query = ['categorize tone three']*2
    a, _, _ = model.forward_step(query,state,update_state=False)
    b, returned, _ = model.forward_step(query,state,context_enabled=False,update_state=False)
    c, _, _ = model.forward_step(query,state,update_state=False)
    assert not torch.allclose(a,b)
    torch.testing.assert_close(a,c,rtol=0,atol=0)
    for key in state:
        torch.testing.assert_close(saved[key],state[key],rtol=0,atol=0)
        torch.testing.assert_close(saved[key],returned[key],rtol=0,atol=0)


def test_legacy_state_dict_unchanged_with_default_config():
    model = StreamModel(Config(dim=32,state_dim=16))
    assert not any('project' in k for k in model.controller.state_dict())
    with pytest.raises(ValueError):
        StreamModel(Config(mode='none',router='interaction'))


def test_frozen_gpt2_interaction_gradient_and_read_restoration():
    transformers = pytest.importorskip('transformers')
    backbone = transformers.GPT2Model(transformers.GPT2Config(
        vocab_size=40,n_embd=32,n_layer=2,n_head=4,n_positions=64,
        resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.))
    model=StreamModel(Config(backbone='hf',state_dim=16,rank=4,
                             num_classes=2,router='interaction'),
                      backbone=backbone,tokenizer=RuleTokenizer())
    state=model.initial_state(2)
    _,state,_=model.forward_step(['tone three outcome left','tone three outcome right'],state)
    obs=['categorize tone three']*2
    logits,_,_=model.forward_step(obs,state,update_state=False)
    logits.square().mean().backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert model.controller.state_project.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        a,_,_=model.forward_step(obs,state,update_state=False)
        off,_,_=model.forward_step(obs,state,context_enabled=False,update_state=False)
        b,_,_=model.forward_step(obs,state,update_state=False)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not torch.allclose(a,off)
