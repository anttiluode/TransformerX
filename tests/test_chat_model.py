import torch
import pytest
transformers = pytest.importorskip('transformers', minversion='4.51.0')
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformerx.chat_model import LoopCausalLM


def tiny():
    torch.manual_seed(19)
    backbone = Qwen3ForCausalLM(Qwen3Config(vocab_size=40, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.0))
    return LoopCausalLM(backbone, model_name='test-qwen', state_dim=8, experts=2, rank=4)


def activate(model):
    with torch.no_grad():
        model.controller.up.normal_(0, .1)
        model.controller.read.weight.normal_(0, .1)


def test_initial_controller_is_exact_identity_and_backbone_stays_frozen():
    model = tiny()
    ids = torch.tensor([[1, 4, 5, 6]])
    state = torch.ones(1, 8)
    plain = model.backbone(ids, use_cache=False).logits
    output, _ = model.forward_turn(ids, state, prompt_length=3)
    torch.testing.assert_close(output.logits, plain, atol=0, rtol=0)
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_future_answer_tokens_cannot_change_prefix_or_prompt_write():
    model = tiny()
    activate(model)
    state = torch.ones(1, 8)
    first, next1 = model.forward_turn(torch.tensor([[1, 4, 5, 6, 7]]), state, prompt_length=3)
    second, next2 = model.forward_turn(torch.tensor([[1, 4, 5, 21, 22]]), state, prompt_length=3)
    torch.testing.assert_close(first.logits[:, :3], second.logits[:, :3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(next1, next2, atol=1e-6, rtol=1e-6)
    assert not torch.equal(first.logits[:, 3:], second.logits[:, 3:])


def test_later_answer_trains_earlier_write_without_backbone_gradients():
    model = tiny()
    activate(model)
    before = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    _, state = model.forward_turn(torch.tensor([[1, 4, 5]]), model.zero_state(), prompt_length=3)
    labels = torch.tensor([[-100, -100, 8, 9]])
    output, _ = model.forward_turn(torch.tensor([[1, 7, 8, 9]]), state, prompt_length=2, labels=labels)
    output.loss.backward()
    assert model.controller.write.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    optimizer = torch.optim.Adam(model.controller.parameters(), lr=.01)
    optimizer.step()
    for n, p in model.backbone.named_parameters():
        assert torch.equal(p, before[n])


def test_cached_generation_commits_one_prompt_write_and_disabled_loop_preserves_state():
    model = tiny()
    activate(model)
    ids = torch.tensor([[1, 4, 5]])
    state = torch.ones(1, 8) * .2
    _, expected_state = model.forward_turn(ids, state, prompt_length=3)
    tokens, new_state = model.generate_turn(ids, state, max_new_tokens=3, temperature=0)
    assert 1 <= tokens.shape[1] <= 3
    torch.testing.assert_close(new_state, expected_state, atol=1e-6, rtol=1e-6)
    _, unchanged = model.generate_turn(ids, state, max_new_tokens=2, temperature=0, enabled=False)
    assert torch.equal(unchanged, state)
    assert not model.block._forward_hooks


def test_checkpoint_rejects_wrong_model_and_untrained_controller(tmp_path):
    model = tiny()
    activate(model)
    path = tmp_path / 'loop.pt'
    model.save_controller(path, trained_steps=3)
    restored = tiny()
    identity = restored.load_controller(path)
    assert len(identity) == 64
    for name, tensor in model.controller.state_dict().items():
        assert torch.equal(tensor, restored.controller.state_dict()[name])
    restored.model_name = 'different-model'
    with pytest.raises(ValueError, match='model'):
        restored.load_controller(path)
    model.save_controller(path, trained_steps=0)
    with pytest.raises(ValueError, match='trained'):
        tiny().load_controller(path)
