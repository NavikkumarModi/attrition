"""Tests for attrition.distillation -- collecting decision traces from an
expensive policy and training a fast DistilledPolicy to approximate it.
Offline, no network (uses MockLLMClient as the "expensive" source, the
same deterministic offline stand-in every other LLM-policy test in this
repo uses).
"""

import numpy as np
import pytest

from attrition import (State, Greedy, ECI, LLMPolicy, MockLLMClient,
                       VOTER_PERSONAS, DecisionTrace, collect_decision_traces,
                       DistilledPolicy, train_distilled_policy,
                       evaluate_agreement)


def _random_states(n, seed=0, n_arms=4):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        v_hat = rng.uniform(0.0, 1.0, size=n_arms)
        p = rng.uniform(0.0, 1.0, size=n_arms)
        e = rng.uniform(0.0, 1.0, size=n_arms)
        t = int(rng.integers(0, 5))
        states.append(State(available=np.arange(n_arms), t=t, horizon=5,
                            v_hat=v_hat, p=p, e=e, delta=0.3, rng=rng))
    return states


def test_collect_decision_traces_records_one_trace_per_state():
    states = _random_states(10, seed=1)
    traces = collect_decision_traces(Greedy(), states)
    assert len(traces) == 10
    for tr, st in zip(traces, states):
        assert isinstance(tr, DecisionTrace)
        assert tr.state is st
        assert tr.chosen_arm == Greedy().select(st)


def test_collect_decision_traces_is_exactly_one_call_per_state():
    calls = {"n": 0}

    class CountingGreedy(Greedy):
        def select(self, state):
            calls["n"] += 1
            return super().select(state)

    states = _random_states(7, seed=2)
    collect_decision_traces(CountingGreedy(), states)
    assert calls["n"] == 7


def test_distilled_policy_achieves_high_training_agreement_on_greedy():
    """Greedy is a simple, near-linear-in-v_hat decision rule -- a
    DistilledPolicy trained on enough Greedy traces should recover it
    almost exactly."""
    states = _random_states(200, seed=3)
    traces = collect_decision_traces(Greedy(), states)
    policy = train_distilled_policy(traces, epochs=300, lr=0.5, seed=0)
    assert policy.training_curve[-1] > 0.95
    assert evaluate_agreement(policy, traces) > 0.95


def test_distilled_policy_generalises_to_held_out_states():
    train_states = _random_states(200, seed=4)
    test_states = _random_states(50, seed=5)
    train_traces = collect_decision_traces(Greedy(), train_states)
    test_traces = collect_decision_traces(Greedy(), test_states)

    policy = train_distilled_policy(train_traces, epochs=300, lr=0.5, seed=0)
    assert evaluate_agreement(policy, test_traces) > 0.85


def test_distilled_policy_is_a_real_policy_and_drops_into_select():
    states = _random_states(100, seed=6)
    traces = collect_decision_traces(ECI(), states)
    policy = train_distilled_policy(traces, epochs=200, seed=0)
    for st in states[:5]:
        arm = policy.select(st)
        assert arm in st.available


def test_training_curve_is_nondecreasing_on_average_and_reports_progress():
    states = _random_states(150, seed=7)
    traces = collect_decision_traces(Greedy(), states)
    policy = train_distilled_policy(traces, epochs=100, lr=0.5, seed=0)
    assert len(policy.training_curve) == 100
    # later-half agreement should not be worse than early-half -- learning,
    # not noise
    early = np.mean(policy.training_curve[:10])
    late = np.mean(policy.training_curve[-10:])
    assert late >= early


def test_evaluate_agreement_empty_traces_is_zero():
    policy = DistilledPolicy(weights=[0, 0, 0, 0, 0, 0])
    assert evaluate_agreement(policy, []) == 0.0


def test_train_distilled_policy_rejects_empty_traces():
    with pytest.raises(ValueError):
        train_distilled_policy([])


# ------------------------------------ distilling a real Level 2 (LLM) policy
def test_distills_llm_policy_decisions_offline():
    """The real target use case: an LLMPolicy (here backed by
    MockLLMClient, the same deterministic offline stand-in used
    throughout this repo's test suite -- a real client would work
    identically, just at real per-call cost) is expensive by nature. A
    DistilledPolicy trained on a modest number of its traces should
    agree with it on most held-out decisions."""
    persona = VOTER_PERSONAS["voter-swing"]
    source = LLMPolicy(persona, client=MockLLMClient(seed=0))

    train_states = _random_states(150, seed=8, n_arms=2)
    test_states = _random_states(40, seed=9, n_arms=2)
    train_traces = collect_decision_traces(source, train_states)
    test_traces = collect_decision_traces(source, test_states)

    policy = train_distilled_policy(train_traces, epochs=300, lr=0.5, seed=0)
    agreement = evaluate_agreement(policy, test_traces)
    assert agreement >= 0.6


def test_distilled_policy_is_much_cheaper_to_call_than_llm_policy():
    """Not a timing assertion (flaky) -- a structural one: DistilledPolicy
    never touches an LLM client at select() time."""
    states = _random_states(20, seed=10, n_arms=2)
    persona = VOTER_PERSONAS["voter-swing"]
    source = LLMPolicy(persona, client=MockLLMClient(seed=0))
    traces = collect_decision_traces(source, states)
    policy = train_distilled_policy(traces, epochs=100, seed=0)
    assert not hasattr(policy, "client")
