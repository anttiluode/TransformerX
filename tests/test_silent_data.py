import numpy as np
import pytest


def test_task_determinism_and_no_answer_in_query():
    from transformerx.silent_data import generate, RuleTokenizer
    a = generate('test', 12, trials=300)
    b = generate('test', 12, trials=300)
    assert a == b
    assert a != generate('train', 12, trials=300)
    assert any(t.switched for t in a)
    assert any(t.returning for t in a)
    for t in a:
        assert 'left' not in t.query and 'right' not in t.query
        assert ('left' in t.feedback) != ('right' in t.feedback)
        assert t.informative == (t.tone in (2, 3, 4))
    tokenizer = RuleTokenizer()
    ids, _ = tokenizer([a[0].query, a[0].feedback])
    assert not (ids == 1).any()  # all experimental tokens in fixed vocabulary


def test_feedback_noise_does_not_change_hidden_world():
    from transformerx.silent_data import generate
    a = generate('test', 9, trials=100, feedback_noise=0)
    b = generate('test', 9, trials=100, feedback_noise=.45)
    assert [(t.tone, t.rule, t.truth, t.query) for t in a] == [(t.tone, t.rule, t.truth, t.query) for t in b]
    assert any(x.outcome != y.outcome for x, y in zip(a, b))


def test_filter_invariant_evidence_and_known_rule_switch_prior():
    from transformerx.silent_data import BayesFilter
    f = BayesFilter(hazard=.12, noise=.05)
    f.belief = np.array([1., 0, 0, 0])
    f.advance()
    np.testing.assert_allclose(f.belief, [.88,.04,.04,.04])
    before = f.belief.copy()
    f.observe(0, 0)
    np.testing.assert_allclose(before, f.belief)
    f.observe(0, 1)  # surprising, but equally unlikely under every rule
    np.testing.assert_allclose(before, f.belief)


def test_filter_prediction_precedes_feedback_and_updates_likelihood():
    from transformerx.silent_data import BayesFilter
    f = BayesFilter(hazard=0, noise=.1)
    np.testing.assert_allclose(f.predict(3), [.5,.5])
    f.observe(3, 0)
    np.testing.assert_allclose(f.belief, [.05,.05,.45,.45])
    np.testing.assert_allclose(f.predict(3), [.9,.1])


def test_filter_exact_match_to_enumerated_two_trial_joint():
    from transformerx.silent_data import BayesFilter, category
    f = BayesFilter(hazard=.2, noise=.1)
    f.observe(2, 1); f.advance(); f.observe(4, 0)
    joint = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            joint[i,j] = .25 * (.9 if category(2,i)==1 else .1) * (.8 if i==j else .2/3) * (.9 if category(4,j)==0 else .1)
    np.testing.assert_allclose(f.belief, joint.sum(0)/joint.sum())


@pytest.mark.parametrize('kwargs', [{'hazard':-1}, {'hazard':1.1}, {'feedback_noise':.5}, {'trials':0}])
def test_invalid_task_parameters(kwargs):
    from transformerx.silent_data import generate
    with pytest.raises(ValueError):
        generate('train', 1, **kwargs)


def test_marginal_answer_reference_is_eighty_percent_not_rule_chance():
    from transformerx.silent_data import BayesFilter
    f=BayesFilter()
    expected=.4*1 + .6*np.mean([f.predict(tone).max() for tone in (2,3,4)])
    assert expected==pytest.approx(.8)
