import json
import torch
import pytest
from test_chat_model import tiny
from transformerx.chat_train import load_episodes, prepare_turn, train_step


class TinyTokenizer:
    eos_token = '<end>'
    def apply_chat_template(self, messages, **kwargs):
        return [1, 4, 5]
    def encode(self, text, **kwargs):
        return [6, 7, 2]


def test_training_masks_prompt_and_keeps_answer_tokens():
    ids, labels, length = prepare_turn(TinyTokenizer(), {'user': 'Hi', 'assistant': 'Hello'}, 10)
    assert ids.tolist() == [[1, 4, 5, 6, 7, 2]]
    assert labels.tolist() == [[-100, -100, -100, 6, 7, 2]]
    assert length == 3
    with pytest.raises(ValueError, match='exceeds'):
        prepare_turn(TinyTokenizer(), {'user': 'Hi', 'assistant': 'Hello'}, 5)


def test_jsonl_requires_multiturn_supervised_episodes(tmp_path):
    path = tmp_path / 'examples.jsonl'
    path.write_text(json.dumps({'turns': [{'user': 'A', 'assistant': 'B'}]}) + '\n')
    with pytest.raises(ValueError, match='two'):
        load_episodes(path)
    path.write_text(json.dumps({'turns': [{'user': 'A', 'assistant': 'B'},
                                        {'user': 'C', 'assistant': 'D'}]}) + '\n')
    assert len(load_episodes(path)[0]) == 2


def test_training_step_updates_controller_and_checkpoint_can_load(tmp_path):
    model = tiny()
    before = model.controller.up.detach().clone()
    optimizer = torch.optim.AdamW(model.controller.parameters(), lr=.01)
    episode = [{'user': 'A', 'assistant': 'B'}, {'user': 'C', 'assistant': 'D'}]
    loss = train_step(model, TinyTokenizer(), episode, optimizer, max_tokens=16)
    assert 0 < loss < 20
    assert not torch.equal(before, model.controller.up)
    assert all(p.grad is None for p in model.backbone.parameters())
    model.save_controller(tmp_path / 'trained.pt', trained_steps=1)
    assert tiny().load_controller(tmp_path / 'trained.pt')
