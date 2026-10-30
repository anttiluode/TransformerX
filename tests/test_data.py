import numpy as np
import pytest


def test_language_split_and_query_contract():
    from transformerx.data import make_episode, mapping_pool, LABELS
    pools = [set(mapping_pool(s)) for s in ('train', 'val', 'test')]
    assert not (pools[0] & pools[1] or pools[1] & pools[2] or pools[0] & pools[2])
    e = make_episode('language', 'test', 7, gap=3)
    assert e.observations == make_episode('language', 'test', 7, gap=3).observations
    assert len(e.targets) == len(e.observations) == len(e.phases)
    assert sum(y >= 0 for y in e.targets) == 12
    for text, y in zip(e.observations, e.targets):
        if y >= 0:
            assert 'translate' in text
            assert all(label not in text.split() for label in LABELS)
    assert set(p for p, y in zip(e.phases, e.targets) if y >= 0) == {'first', 'switch', 'return'}


def test_visual_hidden_queries_and_reproducibility():
    from transformerx.data import make_episode
    e = make_episode('vision', 'train', 8, gap=2, image_size=32)
    f = make_episode('vision', 'train', 8, gap=2, image_size=32)
    assert np.array_equal(e.observations, f.observations)
    assert e.observations.shape[1:] == (3, 32, 32)
    assert sum(y >= 0 for y in e.targets) == 6
    assert set(e.targets) <= {-100, 0, 1, 2, 3}
    for frame, target in zip(e.observations, e.targets):
        if target >= 0:
            assert np.count_nonzero(frame[:, 4:]) == 0  # object hidden; only query marker
            assert np.count_nonzero(frame[:, :4]) > 0


@pytest.mark.parametrize('task', ['language', 'vision'])
def test_seed_namespaces_and_validation(task):
    from transformerx.data import make_episode
    e = make_episode(task, 'train', 4)
    f = make_episode(task, 'test', 4)
    assert not np.array_equal(e.observations, f.observations)
    with pytest.raises(ValueError):
        make_episode(task, 'train', 0, gap=-1)


def test_batch_is_time_major_and_tiny_text_padding():
    from transformerx.data import make_episode, TinyTokenizer, batch_episodes
    batch = batch_episodes([make_episode('language', 'train', i) for i in range(2)])
    assert batch.targets.shape[1] == 2
    tokens, mask = TinyTokenizer()(batch.observations[0])
    assert tokens.shape == mask.shape
    assert tokens.shape[0] == 2
    assert (mask.sum(1) > 0).all()


def test_language_each_split_balances_each_word_label():
    from transformerx.data import mapping_pool
    for split in ('train', 'val', 'test'):
        maps = np.array(mapping_pool(split))
        for word in range(4):
            assert len(set(np.bincount(maps[:, word], minlength=4))) == 1


def test_visual_noise_does_not_change_hidden_scene_or_query_schedule():
    from transformerx.data import make_episode
    clean = make_episode('vision', 'test', 11, noise=0.)
    noisy = make_episode('vision', 'test', 11, noise=.08)
    assert clean.targets == noisy.targets
    assert clean.phases == noisy.phases
    assert np.mean(np.abs(clean.observations - noisy.observations)) < .08
