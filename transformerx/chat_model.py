"""Causal Qwen language generation with an optional trained recurrent controller.

The classification experiments pool a whole observation to route adapters. That
would leak future answer tokens in teacher-forced chat training. Here routing is
token-local; the historical state stays fixed throughout one response.
"""
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .chat_context import prompt_ids
from .memory import OperatorMemory

FORMAT = 'transformerx-qwen-chat-v1'


def read_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or value.get('format') != FORMAT:
        raise ValueError('Not a TransformerX Qwen chat checkpoint. Classification checkpoints are incompatible.')
    if value.get('trained_steps', 0) < 1:
        raise ValueError('Chat controller has not been trained.')
    return value


class LoopCausalLM(nn.Module):
    def __init__(self, backbone, *, model_name, state_dim=64, experts=4, rank=8, layer=None):
        super().__init__()
        if backbone.config.model_type != 'qwen3':
            raise ValueError('This chat integration currently supports Qwen3 causal language models.')
        if state_dim <= 0 or state_dim % 2 or experts < 1 or rank < 1:
            raise ValueError('State dimension must be positive and even; experts/rank must be positive.')
        self.backbone = backbone.requires_grad_(False).eval()
        self.model_name = model_name
        self.layer = max(0, len(backbone.model.layers) // 2 - 1) if layer is None else layer
        if not 0 <= self.layer < len(backbone.model.layers):
            raise ValueError('Invalid decoder layer')
        self.settings = dict(state_dim=state_dim, experts=experts, rank=rank, layer=self.layer)
        self.controller = OperatorMemory(backbone.config.hidden_size, state_dim, experts, rank,
                                         mode='loop', router='concat').to(device=self.device, dtype=torch.float32)
        # A new controller must initially leave pretrained generation unchanged.
        nn.init.zeros_(self.controller.up)
        nn.init.zeros_(self.controller.read.weight)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def block(self):
        return self.backbone.model.layers[self.layer]

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def zero_state(self):
        return torch.zeros(1, self.settings['state_dim'], device=self.device, dtype=torch.float32)

    def _apply_causal(self, hidden, state):
        c = self.controller
        x = c.norm(hidden.float())
        features = torch.cat((x, state[:, None, :].expand(-1, x.shape[1], -1)), dim=-1)
        probabilities = c.route(features).softmax(-1)
        low = torch.einsum('btd,kdr->btkr', x, c.down)
        delta = torch.einsum('btkr,krd,btk->btd', low, c.up, probabilities)
        delta = delta + c.read(state).unsqueeze(1)
        return hidden + (c.scale.sigmoid() * delta).to(hidden.dtype)

    @contextmanager
    def _intervene(self, state, prompt_length, enabled):
        captured = {}
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = self._apply_causal(hidden, state)
            # First call is prefill. Cached decoding must never replace this write.
            if 'prompt' not in captured:
                if changed.shape[1] < prompt_length:
                    raise ValueError('Prompt capture requires a full prefill.')
                captured['prompt'] = changed[:, :prompt_length].float().mean(1)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        handle = self.block.register_forward_hook(hook) if enabled else None
        try:
            yield captured
        finally:
            if handle is not None:
                handle.remove()

    def forward_turn(self, input_ids, state, *, prompt_length, labels=None, enabled=True):
        if not 0 < prompt_length <= input_ids.shape[1]:
            raise ValueError('Invalid prompt length')
        with self._intervene(state, prompt_length, enabled) as captured:
            output = self.backbone(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                   labels=labels, use_cache=False)
        next_state = self.controller.update(captured['prompt'], state) if enabled else state
        return output, next_state

    @torch.inference_mode()
    def generate_turn(self, input_ids, state, *, max_new_tokens=256, temperature=.7, enabled=True):
        if not 1 <= max_new_tokens <= 2048 or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('Invalid generation settings')
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0, use_cache=True,
                      pad_token_id=self.backbone.config.pad_token_id or self.backbone.config.eos_token_id)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=.8, top_k=20)
        with self._intervene(state, input_ids.shape[1], enabled) as captured:
            output = self.backbone.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                            **kwargs)
        next_state = self.controller.update(captured['prompt'], state) if enabled else state
        return output[:, input_ids.shape[1]:], next_state

    def save_controller(self, path, *, trained_steps):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format=FORMAT, model_name=self.model_name, hidden_size=self.backbone.config.hidden_size,
                       settings=self.settings, trained_steps=trained_steps,
                       controller={k: v.detach().cpu() for k, v in self.controller.state_dict().items()})
        fd, name = tempfile.mkstemp(prefix='.controller-', suffix='.pt', dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, name)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load_controller(self, path):
        value = read_checkpoint(path)
        if value['model_name'] != self.model_name or value['hidden_size'] != self.backbone.config.hidden_size:
            raise ValueError('Controller model does not match the loaded Qwen model.')
        if value['settings'] != self.settings:
            raise ValueError('Controller dimensions or layer do not match.')
        if any(not torch.isfinite(v).all() for v in value['controller'].values()):
            raise ValueError('Controller contains non-finite parameters.')
        self.controller.load_state_dict(value['controller'], strict=True)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_qwen(model_name, device='auto', local_files_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.')
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


class QwenChatEngine:
    def __init__(self, model_name='Qwen/Qwen3-0.6B', *, device='auto', checkpoint=None,
                 local_files_only=False):
        backbone, self.tokenizer = load_qwen(model_name, device, local_files_only)
        settings = read_checkpoint(checkpoint)['settings'] if checkpoint else {}
        self.model = LoopCausalLM(backbone, model_name=model_name, **settings)
        self.identity = self.model.load_controller(checkpoint) if checkpoint else None
        self.model_name = model_name
        self.max_positions = backbone.config.max_position_embeddings
        self.device = str(self.model.device)

    def reply(self, messages, *, neural_state=None, neural_identity=None, loop_enabled=False,
              max_new_tokens=256, temperature=.7):
        if loop_enabled and self.identity is None:
            raise ValueError('Load a trained chat controller with --controller before enabling the neural loop.')
        state = self.model.zero_state()
        notice = ''
        if loop_enabled and neural_state is not None:
            valid = (neural_identity == self.identity and len(neural_state) == state.shape[1]
                     and all(isinstance(x, (int, float)) and math.isfinite(x) for x in neural_state))
            if valid:
                state = torch.tensor([neural_state], device=self.model.device, dtype=torch.float32)
            else:
                notice = 'Saved neural state was reset because its controller identity or dimensions changed.'
        ids = torch.tensor([prompt_ids(self.tokenizer, messages)], device=self.model.device)
        if ids.shape[1] + max_new_tokens > self.max_positions:
            raise ValueError('Prompt plus response exceeds the model context limit.')
        tokens, next_state = self.model.generate_turn(ids, state, max_new_tokens=max_new_tokens,
                                                     temperature=temperature, enabled=loop_enabled)
        answer = self.tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
        if not answer:
            raise ValueError('The model produced an empty response; the conversation was not changed.')
        values = next_state[0].cpu().tolist() if loop_enabled else neural_state
        if values is not None and not all(math.isfinite(x) for x in values):
            raise ValueError('Neural state became non-finite; the conversation was not changed.')
        return dict(answer=answer, neural_state=values,
                    neural_identity=self.identity if loop_enabled else neural_identity,
                    generated_tokens=tokens.shape[1], notice=notice)
