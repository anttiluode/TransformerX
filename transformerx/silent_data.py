"""Silent threshold switches and an exact causal belief-filter reference."""
from dataclasses import dataclass
import numpy as np
from .data import OFFSETS, TinyTokenizer

THRESHOLDS = np.array([2, 3, 4, 5])
TONES = ('zero', 'one', 'two', 'three', 'four', 'five', 'six')


def category(tone, rule):
    return int(tone >= THRESHOLDS[rule])


@dataclass(frozen=True)
class Trial:
    query: str
    feedback: str
    outcome: int
    # The following fields are diagnostic only; they never enter the model.
    tone: int
    rule: int
    truth: int
    informative: bool
    switched: bool
    age: int
    returning: bool


def generate(split, seed, trials=96, hazard=.05, feedback_noise=.05, informative_fraction=.6):
    if split not in OFFSETS or seed < 0 or trials < 1:
        raise ValueError('Require known split, nonnegative seed, positive trial count')
    if not 0 <= hazard <= 1 or not 0 <= feedback_noise < .5 or not 0 <= informative_fraction <= 1:
        raise ValueError('Require hazard/fraction in [0,1] and feedback noise in [0,.5)')
    # Separate streams preserve the world when feedback noise is changed.
    rng = np.random.default_rng(np.random.SeedSequence([OFFSETS[split], seed, 17]))
    corruption = np.random.default_rng(np.random.SeedSequence([OFFSETS[split], seed, 29]))
    rule = int(rng.integers(4))
    visited = {rule}
    age, returning = 0, False
    result = []
    for t in range(trials):
        switched = bool(t > 0 and rng.random() < hazard)
        if switched:
            rule = (rule + int(rng.integers(1, 4))) % 4
            returning = rule in visited
            visited.add(rule)
            age = 0
        informative = bool(rng.random() < informative_fraction)
        tone = int(rng.choice((2, 3, 4) if informative else (0, 6)))
        truth = category(tone, rule)
        outcome = truth ^ int(corruption.random() < feedback_noise)
        result.append(Trial(f'categorize tone {TONES[tone]}',
                            f'tone {TONES[tone]} outcome {("left", "right")[outcome]}',
                            outcome, tone, rule, truth, informative, switched, age, returning))
        age += 1
    return result


class RuleTokenizer(TinyTokenizer):
    def __init__(self):
        vocabulary = sorted(set(('categorize tone outcome left right no feedback ' + ' '.join(TONES)).split()))
        self.vocab = {word: i+2 for i, word in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary) + 2


class BayesFilter:
    """Known thresholds/hazard/noise. Call advance before trials AFTER the first."""
    def __init__(self, hazard=.05, noise=.05):
        if not 0 <= hazard <= 1 or not 0 <= noise < .5:
            raise ValueError('Invalid filter hazard or noise')
        self.hazard, self.noise = hazard, noise
        self.belief = np.full(4, .25)

    def advance(self):
        p = self.belief
        self.belief = (1-self.hazard)*p + self.hazard*(1-p)/3

    def predict(self, tone):
        labels = tone >= THRESHOLDS
        # Predict the clean category; feedback likelihood separately includes noise.
        return np.array([self.belief[~labels].sum(), self.belief[labels].sum()])

    def observe(self, tone, outcome):
        labels = (tone >= THRESHOLDS).astype(int)
        likelihood = np.where(labels == outcome, 1-self.noise, self.noise)
        updated = self.belief * likelihood
        if updated.sum() <= 0:
            raise ValueError('Impossible observation under zero-noise reference')
        self.belief = updated / updated.sum()


def feedback_visible(trial, mode):
    if mode not in ('all', 'informative_only', 'invariant_only', 'none'):
        raise ValueError('Unknown feedback condition')
    return mode == 'all' or (mode == 'informative_only' and trial.informative) or (mode == 'invariant_only' and not trial.informative)
