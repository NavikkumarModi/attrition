"""Tests for the LLM-driven policy and population simulation layer.

Offline only: everything here uses `MockLLMClient` or plain stub client
objects -- no network access, no API key.
"""

import numpy as np
import pytest

from attrition import (State, Greedy, LLMPolicy, Persona, MockLLMClient,
                       Population, simulate_population,
                       simulate_population_simultaneous, ConsumableBandit,
                       SimultaneousPool, derive_antibiotic_parameters)


def _state(seed=0):
    rng = np.random.default_rng(seed)
    n = 5
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
    return State(available=np.arange(n), t=0, horizon=10, v_hat=v, p=p, e=e,
                delta=0.05)


PERSONA = Persona(name="test", role="tester", description="a test persona",
                  risk_tolerance=0.5)


def test_llm_policy_selects_available_arm():
    policy = LLMPolicy(PERSONA, client=MockLLMClient(seed=0))
    for seed in range(10):
        st = _state(seed)
        arm = policy.select(st)
        assert arm in set(int(a) for a in st.available)


class _GarbageClient:
    def complete(self, system, user):
        return "I'm not sure, maybe try option B?"


def test_llm_policy_falls_back_to_greedy_on_unparseable_response():
    st = _state(0)
    policy = LLMPolicy(PERSONA, client=_GarbageClient())
    assert policy.select(st) == Greedy().select(st)


class _OutOfRangeClient:
    def complete(self, system, user):
        return "CHOICE: 9999"


def test_llm_policy_falls_back_to_greedy_on_out_of_range_choice():
    st = _state(0)
    policy = LLMPolicy(PERSONA, client=_OutOfRangeClient())
    assert policy.select(st) == Greedy().select(st)


def test_llm_policy_logs_every_decision():
    policy = LLMPolicy(PERSONA, client=MockLLMClient(seed=0))
    for seed in range(3):
        policy.select(_state(seed))
    assert len(policy.log) == 3


def _small_population():
    personas = [Persona(name=f"p{i}", role="tester", description="",
                        risk_tolerance=r)
                for i, r in enumerate([0.1, 0.5, 0.9])]
    return Population.from_personas(personas, client=MockLLMClient(seed=0))


def test_simulate_population_turn_taking_runs_to_completion():
    v = np.array([1.0, 0.9, 0.8, 0.6])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    e = np.array([1.0, 0.5, 0.2, 0.0])
    env = ConsumableBandit(v, p, e, delta=0.1, horizon=12, seed=1)
    population = _small_population()
    result = simulate_population(env, population)
    total_pulls = sum(a["pulls"] for a in result["agents"].values())
    assert 0 < total_pulls <= 12
    assert result["system_value"] == pytest.approx(
        sum(a["value"] for a in result["agents"].values()))
    assert len(result["trace"]) == total_pulls


def test_simulate_population_simultaneous_alive_count_nonincreasing():
    v = np.array([1.0, 0.9, 0.8, 0.6])
    p = np.array([0.6, 0.6, 0.6, 0.6])
    e = np.array([1.0, 0.5, 0.2, 0.0])
    pool = SimultaneousPool(v, p, e, delta=0.1, horizon=6, n_agents=3, seed=1)
    population = _small_population()
    result = simulate_population_simultaneous(pool, population, rounds=6)
    alive_by_t = {}
    for row in result["trace"]:
        alive_by_t[row["t"]] = row["alive"]
    ordered = [alive_by_t[t] for t in sorted(alive_by_t)]
    assert all(a >= b for a, b in zip(ordered, ordered[1:]))


def test_derive_antibiotic_parameters_has_dispersed_kappa():
    v, p, e, spectrum = derive_antibiotic_parameters()
    assert len(v) == len(p) == len(e) == len(spectrum)
    kappa = p * e
    assert kappa.std() > 0.0
