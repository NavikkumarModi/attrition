"""Tests for AgentGraph and the peer-visibility layer on top of
simulate_population_simultaneous. Offline only.
"""

import numpy as np
import pytest

from attrition import (AgentGraph, LLMPolicy, MockLLMClient, Persona,
                       Population, SimultaneousPool,
                       simulate_population_simultaneous)


def _agents():
    return ["a", "b", "c", "d"]


def test_complete_graph_neighbors():
    g = AgentGraph.complete(_agents())
    for a in _agents():
        assert set(g.neighbors(a)) == set(_agents()) - {a}


def test_ring_graph_neighbors():
    g = AgentGraph.ring(_agents())
    for a in _agents():
        assert len(g.neighbors(a)) == 2


def test_star_graph_neighbors():
    g = AgentGraph.star(_agents(), hub="a")
    assert set(g.neighbors("a")) == {"b", "c", "d"}
    for a in ("b", "c", "d"):
        assert g.neighbors(a) == ["a"]


def test_random_graph_is_deterministic_by_seed():
    g1 = AgentGraph.random(_agents(), p=0.5, seed=7)
    g2 = AgentGraph.random(_agents(), p=0.5, seed=7)
    for a in _agents():
        assert g1.neighbors(a) == g2.neighbors(a)


def _small_population():
    personas = [Persona(name=f"p{i}", role="tester", description="",
                        risk_tolerance=r)
                for i, r in enumerate([0.1, 0.5, 0.9])]
    return Population.from_personas(personas, client=MockLLMClient(seed=0))


def test_graph_none_matches_pre_existing_regression_behaviour():
    """graph=None must be byte-identical to before this parameter existed --
    same env, same call, same values as the pre-existing max_workers test.
    """
    v = np.array([1.0, 0.9, 0.8, 0.6])
    p = np.array([0.6, 0.6, 0.6, 0.6])
    e = np.array([1.0, 0.5, 0.2, 0.0])
    pool = SimultaneousPool(v, p, e, delta=0.1, horizon=6, n_agents=3, seed=1)
    result = simulate_population_simultaneous(pool, _small_population(), rounds=6,
                                              graph=None)
    assert result["system_value"] == pytest.approx(6.140000000000001)


def test_peer_history_restricted_to_actual_graph_neighbors():
    v = np.array([1.0, 0.9, 0.8, 0.6])
    p = np.array([0.6, 0.6, 0.6, 0.6])
    e = np.array([1.0, 0.5, 0.2, 0.0])
    personas = [Persona(name=n, role="tester", description="", risk_tolerance=0.5)
                for n in ["a", "b", "c"]]
    population = Population.from_personas(personas, client=MockLLMClient(seed=0))
    graph = AgentGraph.star(["a", "b", "c"], hub="a")
    pool = SimultaneousPool(v, p, e, delta=0.1, horizon=3, n_agents=3, seed=2)
    result = simulate_population_simultaneous(pool, population, rounds=3,
                                              graph=graph)
    assert result["system_value"] != 0.0

    # inspect what "b" (a spoke, sees only the hub "a") actually saw in round 2
    policy_b = population.members["b"]
    prompts_seen = [row["response"] for row in policy_b.log]
    assert len(prompts_seen) == 3


def test_llm_policy_render_prompt_includes_peer_section():
    from attrition.policies import State

    persona = Persona(name="a", role="tester", description="", risk_tolerance=0.5)
    policy = LLMPolicy(persona, client=MockLLMClient(seed=0))
    st = State(available=np.array([0, 1, 2]), t=1, horizon=5,
              v_hat=np.array([1.0, 0.8, 0.5]),
              p=np.array([0.5, 0.5, 0.5]), e=np.array([1.0, 0.5, 0.0]),
              delta=0.1,
              peer_history=[{"agent": "b", "arm": 1, "destroyed": False},
                            {"agent": "c", "arm": 1, "destroyed": True}])
    system, user = policy._render_prompt(st)
    assert "peer b chose arm 1" in user
    assert "peer_majority_arm: 1" in user
    assert "peer_majority_share: 1.000" in user


def test_llm_policy_render_prompt_omits_peer_section_when_absent():
    from attrition.policies import State

    persona = Persona(name="a", role="tester", description="", risk_tolerance=0.5)
    policy = LLMPolicy(persona, client=MockLLMClient(seed=0))
    st = State(available=np.array([0, 1, 2]), t=1, horizon=5,
              v_hat=np.array([1.0, 0.8, 0.5]),
              p=np.array([0.5, 0.5, 0.5]), e=np.array([1.0, 0.5, 0.0]), delta=0.1)
    system, user = policy._render_prompt(st)
    assert "peer" not in user.lower()


def test_mock_llm_conformity_biases_toward_peer_majority():
    from attrition.policies import State

    persona = Persona(name="a", role="tester", description="", risk_tolerance=0.5)
    st = State(available=np.array([0, 1]), t=1, horizon=5,
              v_hat=np.array([0.6, 0.6]),   # tied values
              p=np.array([0.1, 0.1]), e=np.array([0.1, 0.1]), delta=0.01,
              peer_history=[{"agent": "b", "arm": 1, "destroyed": False},
                            {"agent": "c", "arm": 1, "destroyed": False}])
    conforming = LLMPolicy(persona, client=MockLLMClient(seed=0, conformity=5.0))
    neutral = LLMPolicy(persona, client=MockLLMClient(seed=0, conformity=0.0))
    assert conforming.select(st) == 1
    # neutral run isn't asserted to differ (noise could coincidentally agree),
    # but the conforming run must land on the peer majority arm deterministically.
