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


# ---------------------------------------------------------------- domains
def test_all_domains_run_and_have_kappa_dispersion():
    """Every domain instantiates, and each has non-zero kappa dispersion,
    so by Theorem 1 greedy is suboptimal in all of them."""
    from evolving_bandits import (agent_tools, adaptive_therapy,
                                  platform_trial, design_space)
    for factory in [agent_tools, adaptive_therapy, platform_trial, design_space]:
        env = factory(seed=0)
        assert env.n >= 3
        assert np.std(env.kappa) > 0.1
        res = run(env, Greedy(), seed=0)
        assert res["pulls"] > 0


def test_greedy_zero_regret_in_every_domain():
    """The headline claim holds in all four instantiations."""
    from evolving_bandits import (agent_tools, adaptive_therapy,
                                  platform_trial, design_space)
    for factory in [agent_tools, adaptive_therapy, platform_trial, design_space]:
        res = run(factory(seed=2), Greedy(), seed=2)
        assert res["regret"] < 1e-9


def test_correction_beats_greedy_in_every_domain():
    """And greedy loses in all four despite its perfect regret record."""
    from evolving_bandits import (agent_tools, adaptive_therapy,
                                  platform_trial, design_space)
    for factory in [agent_tools, adaptive_therapy, platform_trial, design_space]:
        g = np.mean([run(factory(seed=s), Greedy(), seed=s)["value"]
                     for s in range(15)])
        i = np.mean([run(factory(seed=s), ECI(), seed=s)["value"]
                     for s in range(15)])
        assert i > g


def test_adaptive_therapy_mtd_is_greedy():
    """In the therapy domain, dose intensity orders v, p and e together,
    so maximum tolerated dose is exactly the greedy policy."""
    from evolving_bandits import adaptive_therapy
    env = adaptive_therapy(seed=0)
    assert np.argmax(env.v) == np.argmax(env.p) == np.argmax(env.e)
    assert np.std(env.kappa) > 0.5


def test_domain_notes_present():
    from evolving_bandits import DOMAIN_NOTES
    for k in ["agent_tools", "adaptive_therapy", "platform_trial",
              "design_space"]:
        assert k in DOMAIN_NOTES and len(DOMAIN_NOTES[k]) > 100


# ---------------------------------------------------------------- robustness
def test_t4_robust_to_coupling_form():
    """Theorem 4 holds under additive, multiplicative, saturating, networked
    and concave coupling: zero regret, positive loss, in all five."""
    from experiments.exp26_coupling_robustness import q2_theorem4, FORMS
    for form in FORMS:
        reg, gap = q2_theorem4(form, inst=5)
        assert reg < 1e-9, f"{form}: greedy regret should be zero"
        assert gap > 0.05, f"{form}: greedy should still lose value"


def test_t1_holds_for_separable_coupling():
    """Theorem 1 holds exactly for additively separable burden forms."""
    from experiments.exp26_coupling_robustness import q1_theorem1
    for form in ["additive", "multiplicative", "networked"]:
        const, vary = q1_theorem1(form, inst=5)
        assert const < 1e-6, f"{form}: constant kappa should give zero gap"
        assert vary > 0.5, f"{form}: varying kappa should break greedy"


def test_t1_degrades_gracefully_for_nonseparable():
    """Under non-separable coupling the gap is small, not catastrophic."""
    from experiments.exp26_coupling_robustness import q1_theorem1
    for form in ["saturating", "concave"]:
        const, vary = q1_theorem1(form, inst=5)
        assert const < 2.0, f"{form}: residual gap should stay small"
        assert vary > const, f"{form}: varying kappa still worse than constant"


def test_t3_oracle_lower_bounds_joint_estimator():
    """Theorem 3's repaired proof: the oracle-others-known estimator has weakly
    smaller variance than the joint estimator, so its variance is a valid lower
    bound. This is what makes the AM-HM floor apply to the real problem."""
    from experiments.exp27_theorem3_repair import simulate, variances
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(20):
        n = int(rng.integers(4, 7))
        T = int(rng.integers(25, 60))
        p = np.clip(rng.uniform(0.15, 0.6, n), 0.05, 1.0)
        e = np.clip(rng.uniform(0.2, 2.0, n), 0.0, None)
        X, y, v, dt, rounds = simulate(n, T, p, e, 0.1, 0.3, rng)
        cands = [i for i in range(n) if 0 < dt[i] < rounds - 1]
        if not cands:
            continue
        target = int(rng.choice(cands))
        n_b, n_a = int(dt[target]), int(rounds - dt[target] - 1)
        if n_b < 1 or n_a < 1:
            continue
        joint, oracle = variances(X, 0.3, n, target)
        if joint is None:
            continue
        floor = 0.3**2 * (1.0/n_b + 1.0/n_a)
        assert joint >= oracle - 1e-9, "joint must be at least as hard as oracle"
        assert oracle >= floor - 1e-6, "oracle must meet the AM-HM floor"
        checked += 1
    assert checked >= 5


def test_t1_sufficiency_burden_is_policy_invariant():
    """The closed sufficiency proof rests on this identity: under constant kappa,
    E[total burden] is the same for every policy. Exact DP, no sampling."""
    from experiments.exp28_sufficiency_proof import exact_expected_burden
    rng = np.random.default_rng(3)
    for T, n, const in [(5, 9, True), (25, 5, True), (5, 9, False)]:
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.25, 1.0, n), 0.05, 1.0)
        e = (0.4 / p) if const else np.clip(rng.uniform(0.0, 2.0, n), 0.0, None)
        rules = [lambda S: max(S, key=lambda i: v[i]),
                 lambda S: min(S, key=lambda i: v[i]),
                 lambda S: max(S, key=lambda i: p[i]),
                 lambda S: max(S, key=lambda i: e[i])]
        vals = np.array([exact_expected_burden(v, p, e, 0.1, T, r) for r in rules])
        spread = float(vals.max() - vals.min())
        if const:
            assert spread < 1e-9, f"constant kappa must give invariant burden"
        else:
            assert spread > 1e-3, "varying kappa must give policy-dependent burden"


def test_converse_private_regret_of_optimal():
    """Theorem 5: the optimal policy's private regret is exactly m*eps."""
    from experiments.exp29_converse_separation import analyse, hot_safe
    for m in [2, 4, 8]:
        for E in [0.5, 1.5]:
            eps = 0.05
            v, p, e, delta, n = hot_safe(m, E, eps=eps)
            r = analyse(v, p, e, delta, n)
            assert abs(r["optimal"][1] - m * eps) < 1e-9
            assert abs((r["Vstar"] - r["greedy"][0]) - m * delta * E) < 1e-9


def test_benchmarks_rank_policies_oppositely():
    """Private regret and system regret order the policies backwards."""
    from experiments.exp29_converse_separation import analyse
    rng = np.random.default_rng(17)
    v = np.sort(rng.uniform(0.4, 1.2, 6))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 1.0, 6), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.2, 6), 0.0, None)
    r = analyse(v, p, e, 0.12, 8)
    assert r["greedy"][1] < r["optimal"][1]          # greedy looks better
    assert r["greedy"][0] < r["optimal"][0]          # greedy is worse


def test_eci_beats_ratio_against_exact_optimum():
    """ECI dominates the scale-free ratio heuristic against truth."""
    from experiments.exp31_eci_vs_ratio import gaps
    rng = np.random.default_rng(61)
    for T in [8, 16]:
        E_, R_ = [], []
        for _ in range(8):
            v = np.sort(rng.uniform(0.4, 1.2, 6))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 1.0, 6), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.5, 6), 0.0, None)
            _, b, c = gaps(v, p, e, 0.12, T)
            E_.append(b); R_.append(c)
        assert np.mean(E_) < np.mean(R_)


def test_kappa_aware_agents_reduce_price_of_anarchy():
    """Multi-agent: ECI agents recover most of the planner's value."""
    from experiments.exp32_multiagent_poa import (planner_value,
                                                  decentralised_value,
                                                  rule_greedy, rule_eci)
    rng = np.random.default_rng(5)
    n, T, delta, m = 6, 4, 0.12, 2
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
    pl = planner_value(v, p, e, delta, T, m)
    dg = decentralised_value(v, p, e, delta, T, m, rule_greedy, seeds=200)
    de = decentralised_value(v, p, e, delta, T, m, rule_eci, seeds=200)
    assert de > dg, "kappa-aware agents must beat greedy agents"
    assert pl / de < pl / dg, "and must have a lower price of anarchy"


# ---------------------------------------------------------------- engines
def test_derived_parameters_have_competitive_release_signature():
    """v, p and e derived from the dynamics all rise with dose intensity."""
    from evolving_bandits import derive_arm_parameters
    v, p, e, doses = derive_arm_parameters(engine_kwargs={"dt": 5.0}, s0_sd=0.26)
    assert np.all(np.diff(v) > 0), "higher dose must give more immediate control"
    assert np.all(np.diff(p) >= 0), "higher dose must be more likely to exhaust"
    assert np.std(p * e) > 0.05, "kappa must be dispersed, else T1 says greedy is fine"


def test_mtd_is_greedy_and_loses_on_derived_parameters():
    """On mechanistically derived parameters, MTD has zero regret and loses."""
    from evolving_bandits import derive_arm_parameters
    from experiments.exp33_mechanistic_therapy import exact
    v, p, e, _ = derive_arm_parameters(engine_kwargs={"dt": 5.0}, s0_sd=0.26)
    vs, vg, vi, rg, ri = exact(v, p, e, 0.30, 8)
    assert rg < 1e-9, "MTD/greedy must record zero private regret"
    assert vg < vs, "and must lose system value"
    assert vi > vg, "the corrected policy must beat it"


def test_adaptive_dosing_extends_time_to_progression():
    """The dynamics reproduce competitive release without any bandit layer."""
    from experiments.exp33_mechanistic_therapy import (time_to_progression, mtd,
                                                       make_adaptive)
    t_mtd, _ = time_to_progression(mtd)
    t_adapt, _ = time_to_progression(make_adaptive(backoff=0.20, floor=0.50))
    assert t_adapt > t_mtd * 1.4, "adaptive dosing must substantially extend TTP"
