"""Procedural streams. Hidden labels are carried separately from observations."""
from dataclasses import dataclass
from itertools import permutations
import numpy as np
import torch

WORDS = ('dax', 'wug', 'zup', 'kif')
LABELS = ('red', 'blue', 'green', 'gold')
CONTEXTS = ('cedar', 'maple')
DIRECTIONS = ('right', 'left', 'down', 'up')
OFFSETS = {'train': 0, 'val': 1_000_000_007, 'test': 2_000_000_011}


@dataclass
class Episode:
    observations: object
    targets: list[int]
    phases: list[str]


@dataclass
class Batch:
    observations: list
    targets: torch.Tensor  # time, batch; -100 means no scored query
    phases: list[list[str]]


def mapping_pool(split):
    """Disjoint complete dictionaries, not disjoint individual associations."""
    if split not in OFFSETS:
        raise ValueError(f'Unknown split: {split}')
    # Each orbit contains every global cyclic relabeling of a dictionary.
    # Keeping complete orbits balances every word/label marginal in EACH split.
    representatives = [(0,) + p for p in permutations((1, 2, 3))]
    order = np.random.default_rng(413).permutation(6)
    start, end = {'train': (0, 3), 'val': (3, 4), 'test': (4, 6)}[split]
    return [tuple((value + shift) % 4 for value in representatives[i])
            for i in order[start:end] for shift in range(4)]


def _language(rng, split, gap):
    pool = mapping_pool(split)
    indices = rng.choice(len(pool), 2, replace=False)
    dictionaries = [pool[i] for i in indices]
    names = list(rng.permutation(CONTEXTS))
    observations, targets, phases = [], [], []

    def add(text, target, phase):
        observations.append(text)
        targets.append(target)
        phases.append(phase)

    for phase, context in [('first', 0), ('switch', 1), ('return', 0)]:
        if phase != 'return':
            for word in rng.permutation(4):
                add(f'in workshop {names[context]} word {WORDS[word]} means '
                    f'{LABELS[dictionaries[context][word]]}', -100, phase)
        for _ in range(gap):
            add(f'the clock ticks {rng.choice(("quietly", "slowly", "steadily"))}', -100, phase)
        for word in rng.permutation(4):
            add(f'in workshop {names[context]} translate {WORDS[word]}',
                dictionaries[context][word], phase)
    return Episode(observations, targets, phases)


def _vision(rng, gap, size, noise, noise_rng):
    colors = np.array([[1., .1, .1], [.1, .4, 1.]], dtype=np.float32)
    velocities = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    positions = rng.uniform(.3 * size, .7 * size, (2, 2))
    directions = rng.integers(0, 4, size=2)
    speed = rng.uniform(.035, .065) * size
    radius = max(1, size // 16)
    frames, targets, phases = [], [], []

    def frame(visible, query=None):
        img = np.zeros((3, size, size), dtype=np.float32)
        if visible:
            for pos, color in zip(positions, colors):
                x, y = np.rint(pos).astype(int)
                img[:, max(4, y-radius):min(size, y+radius+1),
                    max(0, x-radius):min(size, x+radius+1)] = color[:, None, None]
        if query is not None:
            img[:, :4, :max(4, size // 4)] = colors[query, :, None, None]
        if noise:
            img += noise_rng.normal(0, noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)

    for phase in ('first', 'switch', 'late'):
        if phase == 'switch':
            # Both objects change direction; no hidden target is directly rendered.
            directions = (directions + rng.integers(1, 4, size=2)) % 4
        # Reposition before each visible segment; infer movement from frames, not endpoint.
        positions = rng.uniform(.3 * size, .7 * size, (2, 2))
        for _ in range(4):
            positions += velocities[directions] * speed
            positions[:, 0] %= size
            positions[:, 1] = 5 + (positions[:, 1] - 5) % (size - 6)
            frames.append(frame(True)); targets.append(-100); phases.append(phase)
        for _ in range(gap):
            frames.append(frame(False)); targets.append(-100); phases.append(phase)
        for obj in rng.permutation(2):
            frames.append(frame(False, obj)); targets.append(int(directions[obj])); phases.append(phase)
    return Episode(np.stack(frames), targets, phases)


def make_episode(task, split, seed, gap=2, image_size=32, noise=0.):
    if task not in ('language', 'vision') or split not in OFFSETS:
        raise ValueError('Expected language/vision and train/val/test')
    if gap < 0 or image_size < 16 or noise < 0 or seed < 0:
        raise ValueError('Require gap/noise/seed >= 0 and image_size >= 16')
    rng = np.random.default_rng(np.random.SeedSequence([OFFSETS[split], int(seed)]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([OFFSETS[split], int(seed), 1]))
    return (_language(rng, split, gap) if task == 'language'
            else _vision(rng, gap, image_size, noise, noise_rng))


def batch_episodes(episodes):
    if not episodes or len({len(e.targets) for e in episodes}) != 1:
        raise ValueError('Batch requires nonempty equal-length episodes')
    targets = torch.tensor([e.targets for e in episodes], dtype=torch.long).T.contiguous()
    phases = [[e.phases[t] for e in episodes] for t in range(len(episodes[0].targets))]
    if isinstance(episodes[0].observations, np.ndarray):
        data = torch.from_numpy(np.stack([e.observations for e in episodes], axis=1))
        observations = list(data.unbind(0))
    else:
        observations = [[e.observations[t] for e in episodes] for t in range(len(phases))]
    return Batch(observations, targets, phases)


class TinyTokenizer:
    """Fixed task vocabulary. Byte-free and download-free, never fitted on test data."""
    def __init__(self):
        words = ('in workshop word means translate the clock ticks quietly slowly steadily '
                 + ' '.join(WORDS + LABELS + CONTEXTS)).split()
        self.vocab = {word: i + 2 for i, word in enumerate(sorted(set(words)))}
        self.vocab_size = len(self.vocab) + 2

    def __call__(self, texts):
        rows = [[self.vocab.get(w, 1) for w in text.split()] for text in texts]
        length = max(map(len, rows))
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
        return ids, ids.ne(0)
