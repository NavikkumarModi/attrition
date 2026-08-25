"""Tests for attrition.election.policy_training -- the from-scratch trained
campaign-contact policy that closes FINDINGS.md#31's "no external checkpoint
to transplant" open item. Offline, no network, no RL-framework dependency
(plain numpy gradient-bandit updates).
"""

import numpy as np
import pytest

from attrition import (ConsumableBandit, Greedy, ECI, compare, run,
                       derive_campaign_parameters, SimultaneousPool,
                       Population, simulate_population_simultaneous)
from attrition.election import GradientBanditPolicy, train_gradient_bandit


def _campaign_env_factory():
    v, p, e, _ = derive_campaign_parameters()
    delta, horizon = 0.35, 9
    return v, p, e, delta, horizon, lambda s: ConsumableBandit(
        v, p, e, delta=delta, horizon=horizon, seed=s)


def test_train_gradient_bandit_returns_a_selectable_policy():
    _, _, _, _, _, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=200, seed=0)
    assert isinstance(policy, GradientBanditPolicy)
    env = env_factory(0)
    st = env.state()
    arm = policy.select(st)
    assert arm in set(int(a) for a in st.available)


def test_trained_policy_preferences_shape_matches_arm_count():
    _, _, _, _, _, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=200, seed=0)
    assert policy.h.shape == (5,)


def test_trained_policy_is_reproducible_given_same_seed():
    _, _, _, _, _, env_factory = _campaign_env_factory()
    p1 = train_gradient_bandit(env_factory, episodes=300, seed=7)
    p2 = train_gradient_bandit(env_factory, episodes=300, seed=7)
    assert np.allclose(p1.h, p2.h)


def test_trained_policy_probs_sum_to_one_over_available_arms():
    _, _, _, _, _, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=200, seed=0)
    probs = policy.probs(np.array([0, 2, 4]))
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0).all()


def test_trained_policy_runs_through_the_standard_run_and_compare_contract():
    """A trained policy is a Policy like any other -- drops into run()/
    compare() unchanged, no special-cased evaluation path."""
    _, _, _, _, _, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=500, seed=0)
    result = run(env_factory(0), policy, seed=0)
    assert "value" in result and "regret" in result

    stats = compare(env_factory, [Greedy(), ECI(), policy], seeds=20)
    assert set(stats) == {"greedy", "eci", policy.name}


def test_trained_policy_converges_toward_greedy_not_eci():
    """The predicted result documented in policy_training.py's module
    docstring: realised per-pull reward already folds the externality into
    a common burden term identical across every available arm in a round,
    so a policy trained to maximise it has no gradient distinguishing a
    cheap-to-the-pool arm from an expensive one -- it should land close to
    greedy's near-zero regret, not ECI's positive regret.
    """
    v, p, e, delta, horizon, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=4000, alpha=0.15, seed=0)
    stats = compare(env_factory, [Greedy(), ECI(), policy], seeds=200)
    trained_regret = stats[policy.name]["regret"]
    eci_regret = stats["eci"]["regret"]
    assert trained_regret < eci_regret / 2


def test_trained_policy_decentralises_through_simultaneous_population():
    """Same policy object, frozen, run as m independent non-coordinating
    copies sharing one SimultaneousPool -- the decentralisation step
    examples/20_campaign_policy_checkpoint.py performs, mirroring
    FINDINGS.md#24.4's AMR checkpoint decentralisation."""
    v, p, e, delta, horizon, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=500, seed=0)
    m = 3
    population = Population({f"campaign-{i}": policy for i in range(m)})
    pool = SimultaneousPool(v, p, e, delta=delta, horizon=horizon,
                            n_agents=m, seed=0)
    result = simulate_population_simultaneous(pool, population, rounds=horizon)
    assert result["system_value"] == pytest.approx(
        sum(a["value"] for a in result["agents"].values()))
    assert sum(a["pulls"] for a in result["agents"].values()) > 0


def test_training_curve_is_recorded_on_the_policy():
    _, _, _, _, _, env_factory = _campaign_env_factory()
    policy = train_gradient_bandit(env_factory, episodes=150, seed=0)
    assert len(policy.training_curve) == 150
