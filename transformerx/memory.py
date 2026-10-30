"""Bounded memory controllers. All episode state belongs to the caller."""
import math
import torch
from torch import nn

MODES = ('none', 'static', 'gru', 'tokens', 'loop', 'loop_no_rotation', 'loop_hard', 'window')


def masked_mean(h, mask):
    if mask is None:
        return h.mean(1)
    w = mask.to(h.dtype).unsqueeze(-1)
    return (h * w).sum(1) / w.sum(1).clamp_min(1)


class OperatorMemory(nn.Module):
    def __init__(self, dim, state_dim, experts, rank, mode):
        super().__init__()
        self.mode, self.state_dim = mode, state_dim
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Parameter(torch.randn(experts, dim, rank) / math.sqrt(dim))
        self.up = nn.Parameter(torch.randn(experts, rank, dim) * .01)
        self.route = nn.Linear(dim + (0 if mode == 'static' else state_dim), experts)
        self.scale = nn.Parameter(torch.tensor(-2.0))
        if mode != 'static':
            self.read = nn.Linear(state_dim, dim, bias=False)
            if mode == 'gru':
                self.gru = nn.GRUCell(dim, state_dim)
            else:
                self.write = nn.Linear(dim, state_dim)
                self.context = nn.Linear(state_dim, state_dim, bias=False)
                self.decay = nn.Parameter(torch.full((state_dim // 2,), 3.5))
                self.register_buffer('omega', torch.linspace(.02, .2, state_dim // 2))

    def apply(self, h, memory, mask=None):
        x = self.norm(h)
        pooled = masked_mean(x, mask)
        features = pooled if self.mode == 'static' else torch.cat((pooled, memory), -1)
        probs = self.route(features).softmax(-1)
        if self.mode == 'loop_hard':
            hard = torch.nn.functional.one_hot(probs.argmax(-1), probs.shape[-1]).to(probs)
            # Straight-through training; exactly one operator in the forward computation.
            probs = hard + probs - probs.detach()
        low = torch.einsum('btd,kdr->btkr', x, self.down)
        delta = torch.einsum('btkr,krd,bk->btd', low, self.up, probs)
        if self.mode != 'static':
            delta = delta + self.read(memory).unsqueeze(1)
        return h + self.scale.sigmoid() * delta, probs

    def update(self, pooled, memory):
        if self.mode == 'static':
            return memory
        if self.mode == 'gru':
            return self.gru(pooled, memory)
        pairs = memory.reshape(memory.shape[0], -1, 2)
        angle = self.omega if self.mode != 'loop_no_rotation' else self.omega * 0
        c, s = angle.cos(), angle.sin()
        rot = torch.stack((c * pairs[..., 0] - s * pairs[..., 1],
                           s * pairs[..., 0] + c * pairs[..., 1]), -1)
        old = (rot * self.decay.sigmoid()[None, :, None]).flatten(1)
        signal = self.write(pooled)
        # A bounded contextual write; neither a Möbius strip nor online weight training.
        return torch.tanh(old + .2 * signal * (1 + .5 * self.context(memory).tanh()))


class TokenMemory(nn.Module):
    """Recurrent latent tokens with cross-attention read/write; not published RMT."""
    def __init__(self, dim, slots=4):
        super().__init__()
        self.slots, self.dim = slots, dim
        self.queries = nn.Parameter(torch.randn(slots, dim) * .02)
        self.reader = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.writer = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(-2.))

    def apply(self, h, memory, mask=None):
        read, _ = self.reader(h, memory, memory, need_weights=False)
        return h + self.scale.sigmoid() * read, None

    def update(self, hidden, memory, mask=None):
        query = memory + self.queries[None]
        write, _ = self.writer(query, hidden, hidden,
                              key_padding_mask=None if mask is None else ~mask,
                              need_weights=False)
        return self.norm(.8 * memory + write)
