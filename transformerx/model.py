"""One-step transformer integration, with optional frozen GPT-2 or ViT backbones."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .data import TinyTokenizer
from .memory import MODES, OperatorMemory, TokenMemory, masked_mean


@dataclass
class Config:
    task: str = 'language'
    mode: str = 'loop'
    backbone: str = 'tiny'
    model_name: str = ''
    dim: int = 96
    state_dim: int = 64
    layers: int = 2
    experts: int = 4
    rank: int = 8
    memory_tokens: int = 4
    window: int = 8
    image_size: int = 32
    freeze_backbone: bool = True
    local_files_only: bool = False
    router: str = "concat"
    num_classes: int = 4


class TinyBackbone(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.task, self.dim = cfg.task, cfg.dim
        if cfg.task == 'language':
            self.embedding = nn.Embedding(vocab_size, cfg.dim, padding_idx=0)
            self.position = nn.Parameter(torch.randn(128, cfg.dim) * .02)
        else:
            self.patch = nn.Conv2d(3, cfg.dim, kernel_size=8, stride=8)
            count = (cfg.image_size // 8) ** 2
            self.position = nn.Parameter(torch.randn(count, cfg.dim) * .02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(cfg.dim, 4, cfg.dim * 3,
                                  dropout=0., batch_first=True, norm_first=True)
                                    for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x, mask=None):
        if self.task == 'language':
            h = self.embedding(x)
        else:
            h = self.patch(x).flatten(2).transpose(1, 2)
        h = h + self.position[:h.shape[1]]
        # Each observation is a fully received event/frame. Temporal causality is
        # enforced outside this encoder: no future event is passed to this call.
        for block in self.blocks:
            h = block(h, src_key_padding_mask=None if mask is None else ~mask)
        return self.norm(h)


class StreamModel(nn.Module):
    def __init__(self, cfg: Config, backbone=None, tokenizer=None):
        super().__init__()
        if cfg.task not in ('language', 'vision') or cfg.mode not in MODES:
            raise ValueError('Invalid task or memory mode')
        if cfg.backbone not in ('tiny', 'hf'):
            raise ValueError('backbone must be tiny or hf')
        if cfg.state_dim < 2 or cfg.state_dim % 2 or cfg.dim % 4:
            raise ValueError('state_dim must be positive/even; dim divisible by 4')
        if min(cfg.layers, cfg.rank, cfg.experts, cfg.memory_tokens, cfg.window) < 1:
            raise ValueError('Layer, rank, expert, token and window counts must be positive')
        if cfg.image_size < 16 or cfg.image_size % 8:
            raise ValueError('image_size must be >=16 and divisible by 8')
        if cfg.num_classes < 2 or cfg.router not in ('concat', 'interaction', 'additive'):
            raise ValueError('Invalid classifier size or router')
        if cfg.router != 'concat' and cfg.mode in ('none', 'static', 'tokens', 'window'):
            raise ValueError('Interaction/additive routing requires a stateful operator controller')
        self.cfg = cfg
        self.tokenizer = tokenizer or TinyTokenizer()
        self.image_mean, self.image_std = (.5,) * 3, (.5,) * 3
        self.input_size = cfg.image_size
        if cfg.backbone == 'tiny':
            self.backbone = TinyBackbone(cfg, self.tokenizer.vocab_size)
            self.dim = cfg.dim
            block = self.backbone.blocks[max(0, cfg.layers // 2 - 1)]
        else:
            if backbone is None:
                from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
                name = cfg.model_name or ('distilbert/distilgpt2' if cfg.task == 'language'
                                          else 'google/vit-base-patch16-224-in21k')
                self.backbone = AutoModel.from_pretrained(name, local_files_only=cfg.local_files_only)
                if cfg.task == 'language':
                    self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=cfg.local_files_only)
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = 'right'
                else:
                    processor = AutoImageProcessor.from_pretrained(name, local_files_only=cfg.local_files_only)
                    self.image_mean = tuple(processor.image_mean)
                    self.image_std = tuple(processor.image_std)
            else:
                self.backbone = backbone
            config = self.backbone.config
            expected = 'gpt2' if cfg.task == 'language' else 'vit'
            if config.model_type != expected:
                raise ValueError(f'Only {expected} backbones are supported; got {config.model_type}')
            self.dim = config.hidden_size
            if cfg.task == 'language':
                block = self.backbone.h[max(0, len(self.backbone.h) // 2 - 1)]
            else:
                block = self.backbone.encoder.layer[max(0, len(self.backbone.encoder.layer) // 2 - 1)]
                self.input_size = config.image_size
            if cfg.freeze_backbone:
                self.backbone.requires_grad_(False)
        # A plain reference avoids registering the same backbone block twice.
        object.__setattr__(self, 'hook_block', block)
        if cfg.mode in ('none', 'window'):
            self.controller = None
        elif cfg.mode == 'tokens':
            self.controller = TokenMemory(self.dim, cfg.memory_tokens)
        else:
            self.controller = OperatorMemory(self.dim, cfg.state_dim, cfg.experts, cfg.rank, cfg.mode, cfg.router)
        if cfg.mode == 'window':
            self.temporal_position = nn.Parameter(torch.randn(cfg.window, self.dim) * .02)
            self.temporal = nn.TransformerEncoderLayer(self.dim, 4, self.dim * 2,
                                    dropout=0., batch_first=True, norm_first=True)
        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, cfg.num_classes))

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.backbone == 'hf' and self.cfg.freeze_backbone:
            self.backbone.eval()  # disable pretrained dropout, but DO NOT use no_grad
        return self

    def initial_state(self, batch, device=None):
        device = device or next(self.parameters()).device
        if self.cfg.mode == 'tokens':
            return {'memory': torch.zeros(batch, self.cfg.memory_tokens, self.dim, device=device)}
        if self.cfg.mode == 'window':
            return {'history': torch.zeros(batch, 0, self.dim, device=device)}
        if self.cfg.mode in ('none', 'static'):
            return {}
        return {'memory': torch.zeros(batch, self.cfg.state_dim, device=device)}

    @staticmethod
    def detach_state(state):
        return {key: value.detach() for key, value in state.items()}

    def _prepare(self, observation):
        device = next(self.parameters()).device
        if self.cfg.task == 'language':
            if isinstance(self.tokenizer, TinyTokenizer):
                ids, mask = self.tokenizer(observation)
            else:
                encoding = self.tokenizer(observation, padding=True, truncation=False, return_tensors='pt')
                ids, mask = encoding['input_ids'], encoding['attention_mask'].bool()
            limit = 128 if self.cfg.backbone == 'tiny' else self.backbone.config.max_position_embeddings
            if ids.shape[1] > limit:
                raise ValueError(f'Observation exceeds backbone token limit ({limit})')
            return ids.to(device), mask.to(device)
        x = observation.to(device)
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        if self.cfg.backbone == 'hf':
            mean = x.new_tensor(self.image_mean)[None, :, None, None]
            std = x.new_tensor(self.image_std)[None, :, None, None]
            x = (x - mean) / std
        return x, None

    def forward_step(self, observation, state, *, context_enabled=True, update_state=True):
        x, mask = self._prepare(observation)
        diagnostics = {}

        def inject(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            memory = state.get('memory')
            if not context_enabled and memory is not None:
                memory = torch.zeros_like(memory)
            changed, weights = self.controller.apply(hidden, memory, mask)
            if weights is not None:
                diagnostics['route_weights'] = weights.detach()
                diagnostics['route_entropy'] = -(weights * weights.clamp_min(1e-8).log()).sum(-1).detach()
            return (changed,) + output[1:] if isinstance(output, tuple) else changed

        handle = self.hook_block.register_forward_hook(inject) if self.controller is not None else None
        try:
            if self.cfg.backbone == 'tiny':
                hidden = self.backbone(x, mask)
            elif self.cfg.task == 'language':
                hidden = self.backbone(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
            else:
                hidden = self.backbone(pixel_values=x).last_hidden_state
        finally:
            if handle is not None:
                handle.remove()
        pooled = masked_mean(hidden, mask)
        if self.cfg.mode == 'window':
            previous = state['history'] if context_enabled else state['history'][:, :0]
            history = torch.cat((previous, pooled.unsqueeze(1)), 1)[:, -self.cfg.window:]
            z = history + self.temporal_position[-history.shape[1]:]
            pooled = self.temporal(z)[:, -1]
            new_state = {'history': history}
        elif self.cfg.mode == 'tokens':
            new_state = {'memory': self.controller.update(hidden, state['memory'], mask)}
        elif self.cfg.mode not in ('none', 'static'):
            new_state = {'memory': self.controller.update(pooled, state['memory'])}
        else:
            new_state = {}
        logits = self.head(pooled)
        return logits, new_state if update_state else state, diagnostics

    def budget(self):
        mode = self.cfg.mode
        scalars = (self.cfg.memory_tokens * self.dim if mode == 'tokens' else
                   self.cfg.window * self.dim if mode == 'window' else
                   0 if mode in ('none', 'static') else self.cfg.state_dim)
        return {'parameters': sum(p.numel() for p in self.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'persistent_scalars_per_stream': scalars,
                'persistent_bytes_fp32_per_stream': 4 * scalars}
