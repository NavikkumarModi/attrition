"""Regression tests: each theorem is a test.

If a change to the library breaks one of these, a theorem has been contradicted.
"""

from functools import lru_cache
from itertools import permutations

import numpy as np
import pytest

from evolving_bandits import (ConsumableBandit, Greedy, ECI, Conservative,
                              SortByE, ThompsonSampling, UCB, run, compare)


def exact_dp(v, p, e, delta, T):
    """Exact optimum by DP over subsets. Ground truth for small instances."""
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    @lru_cache(maxsize=None)
    def G(S, t):
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: R(i, S))
        return R(a, S) + p[a]*G(S-{a}, t+1) + (1-p[a])*G(S, t+1)

    return V(full, 0), G(full, 0)


# ---------------------------------------------------------------- T1
def test_t1_sufficiency_greedy_optimal_when_kappa_constant():
    """kappa = p*e constant => greedy is exactly optimal."""
    rng = np.random.default_rng(0)
    for _ in range(6):
        n = 5
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = 0.4 / p                      # kappa == 0.4 for every arm
        opt, greedy = exact_dp(v, p, e, 0.15, 7)
        assert opt - greedy < 1e-9


def test_t1_necessity_greedy_suboptimal_when_kappa_varies():
    """kappa dispersion => strictly positive gap, equal to m*delta*E."""
    for m in [1, 3, 6]:
        for E in [0.5, 1.5]:
            n, eps, delta = m + 1, 0.01, 1.0
            v = np.array([1.0] + [1.0 - eps] * m)
            p = np.ones(n)
            e = np.array([E] + [0.0] * m)
            opt, greedy = exact_dp(v, p, e, delta, n + 1)
            assert abs((opt - greedy) - m * delta * E) < 1e-9


# ---------------------------------------------------------------- T2b
def test_t2b_sort_by_e_optimal_in_pure_sequencing():
    """p=1, T=n: optimal order is ascending in e, independent of v."""
    rng = np.random.default_rng(9)
    for _ in range(5):
        n = 5
        v = rng.uniform(0.3, 1.3, n)
        e = rng.uniform(0.0, 2.0, n)
        delta = 0.2

        def value(order):
            tot, b = 0.0, 0.0
            for a in order:
                tot += v[a] - b
                b += delta * e[a]
            return tot

        brute = max(value(o) for o in permutations(range(n)))
        rule = value(sorted(range(n), key=lambda i: e[i]))
        assert abs(brute - rule) < 1e-9


# ---------------------------------------------------------------- T3
def test_t3_sample_budget_is_horizon_independent():
    """E[N] saturates at sum_j 1/p_j regardless of T."""
    rng = np.random.default_rng(3)
    p = np.full(8, 0.5)
    predicted = float(np.sum(1 / p))
    for T in [50, 200, 800]:
        lengths = []
        for _ in range(400):
            alive = np.ones(8, dtype=bool)
            t = 0
            while t < T and alive.any():
                idx = np.flatnonzero(alive)
                a = idx[rng.integers(len(idx))]
                if rng.random() < p[a]:
                    alive[a] = False
                t += 1
            lengths.append(t)
        assert abs(np.mean(lengths) - predicted) < 1.0


def test_t3_estimator_variance_floor():
    """SE >= 2*sigma/sqrt(N), attained exactly at a balanced split."""
    sigma, N = 0.3, 20
    floor = 2 * sigma / np.sqrt(N)
    for n_b in range(1, N):
        se = sigma * np.sqrt(1 / n_b + 1 / (N - n_b))
        assert se >= floor - 1e-12
    balanced = sigma * np.sqrt(1 / 10 + 1 / 10)
    assert abs(balanced - floor) < 1e-12


# ---------------------------------------------------------------- T4
def test_t4_zero_regret_unbounded_loss():
    """Greedy holds regret at zero while value loss grows without bound."""
    prev_gap = -1.0
    for m in [2, 4, 8, 12]:
        n, eps, delta, E = m + 1, 0.01, 1.0, 2.0
        v = np.array([1.0] + [1.0 - eps] * m)
        p = np.ones(n)
        e = np.array([E] + [0.0] * m)
        opt, greedy = exact_dp(v, p, e, delta, n + 1)
        gap = opt - greedy
        assert abs(gap - m * delta * E) < 1e-9
        assert gap > prev_gap
        prev_gap = gap
    assert greedy < 0 < opt        # zero-regret policy ends up net negative


def test_t4_greedy_regret_is_identically_zero():
    """Greedy's regret against the best-available benchmark is zero always."""
    env = ConsumableBandit.random(n=15, k_spread=2.0, delta=0.05,
                                  horizon=40, seed=1)
    res = run(env, Greedy(), seed=1)
    assert res["regret"] < 1e-9
    assert res["value"] < float("inf")


# ---------------------------------------------------------------- policies
def test_eci_beats_greedy_under_coupling():
    f = lambda s: ConsumableBandit.random(n=20, k_spread=1.5, delta=0.03,
                                          horizon=60, seed=s)
    r = compare(f, [Greedy(), ECI()], seeds=15)
    assert r["eci"]["value"] > r["greedy"]["value"]


def test_eci_beats_thompson_and_ucb():
    f = lambda s: ConsumableBandit.random(n=20, k_spread=1.5, delta=0.03,
                                          horizon=60, seed=s)
    r = compare(f, [ThompsonSampling(), UCB(), ECI()], seeds=15)
    assert r["eci"]["value"] > r["thompson"]["value"]
    assert r["eci"]["value"] > r["ucb"]["value"]


def test_conservative_needs_no_per_arm_knowledge():
    """The recommended default beats greedy without knowing per-arm e."""
    f = lambda s: ConsumableBandit.random(n=20, k_spread=1.5, delta=0.03,
                                          horizon=60, seed=s)
    r = compare(f, [Greedy(), Conservative(e_bound=2.0)], seeds=15)
    assert r["conservative"]["value"] > r["greedy"]["value"]


def test_lower_regret_can_mean_worse_value():
    """The headline: regret ordering and value ordering can be reversed."""
    f = lambda s: ConsumableBandit.random(n=20, k_spread=1.5, delta=0.03,
                                          horizon=60, seed=s)
    r = compare(f, [Greedy(), ECI()], seeds=15)
    assert r["greedy"]["regret"] < r["eci"]["regret"]      # greedy looks better
    assert r["greedy"]["value"] < r["eci"]["value"]        # greedy is worse


def test_env_reproducible():
    a = run(ConsumableBandit.random(n=10, seed=4), Greedy(), seed=4)
    b = run(ConsumableBandit.random(n=10, seed=4), Greedy(), seed=4)
    assert abs(a["value"] - b["value"]) < 1e-12


def test_cannot_pull_dead_arm():
    env = ConsumableBandit(v=[1.0], p=[1.0], e=[0.0], horizon=5, seed=0)
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)
