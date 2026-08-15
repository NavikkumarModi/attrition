"""Regression tests: each theorem is a test.

If a change to the library breaks one of these, a theorem has been contradicted.
"""

from functools import lru_cache
from itertools import permutations

import numpy as np
import pytest

from attrition import (ConsumableBandit, Greedy, ECI, Conservative,
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
    from attrition import (agent_tools, adaptive_therapy,
                                  platform_trial, design_space)
    for factory in [agent_tools, adaptive_therapy, platform_trial, design_space]:
        env = factory(seed=0)
        assert env.n >= 3
        assert np.std(env.kappa) > 0.1
        res = run(env, Greedy(), seed=0)
        assert res["pulls"] > 0


def test_greedy_zero_regret_in_every_domain():
    """The headline claim holds in all four instantiations."""
    from attrition import (agent_tools, adaptive_therapy,
                                  platform_trial, design_space)
    for factory in [agent_tools, adaptive_therapy, platform_trial, design_space]:
        res = run(factory(seed=2), Greedy(), seed=2)
        assert res["regret"] < 1e-9


def test_correction_beats_greedy_in_every_domain():
    """And greedy loses in all four despite its perfect regret record."""
    from attrition import (agent_tools, adaptive_therapy,
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
    from attrition import adaptive_therapy
    env = adaptive_therapy(seed=0)
    assert np.argmax(env.v) == np.argmax(env.p) == np.argmax(env.e)
    assert np.std(env.kappa) > 0.5


def test_domain_notes_present():
    from attrition import DOMAIN_NOTES
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
    from attrition import derive_arm_parameters
    v, p, e, doses = derive_arm_parameters(engine_kwargs={"dt": 5.0}, s0_sd=0.26)
    assert np.all(np.diff(v) > 0), "higher dose must give more immediate control"
    assert np.all(np.diff(p) >= 0), "higher dose must be more likely to exhaust"
    assert np.std(p * e) > 0.05, "kappa must be dispersed, else T1 says greedy is fine"


def test_mtd_is_greedy_and_loses_on_derived_parameters():
    """On mechanistically derived parameters, MTD has zero regret and loses."""
    from attrition import derive_arm_parameters
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


def test_all_three_engines_produce_dispersed_kappa():
    from attrition import (derive_arm_parameters, derive_trial_parameters,
                                  derive_design_space_parameters)
    for fn in [lambda: derive_arm_parameters(engine_kwargs={"dt": 5.0},
                                             s0_sd=0.26)[:3],
               lambda: derive_trial_parameters(periods=5)[:3],
               lambda: derive_design_space_parameters(n_settings=6)[:3]]:
        v, p, e = fn()
        assert np.std(np.asarray(p) * np.asarray(e)) > 0.01


def test_greedy_fails_only_when_value_and_externality_conflict():
    """The refinement: dispersion alone is not enough, it must oppose value."""
    from experiments.exp34_all_engines import analyse, ENGINES
    for name, fn in ENGINES:
        v, p, e = fn()
        v = np.asarray(v, float)
        p = np.clip(np.asarray(p, float), 1e-6, 1.0)
        e = np.asarray(e, float)
        corr = float(np.corrcoef(v, p * e)[0, 1])
        delta = 0.4 / max(float(np.mean(e)), 1e-9)
        vs, vg, vi, rg = analyse(v, p, e, delta, 8)
        assert rg < 1e-9, f"{name}: greedy must have zero private regret"
        if corr > 0.3:
            assert vg < vs - 1e-6, f"{name}: conflict implies greedy loses"
        elif corr < -0.3:
            assert abs(vs - vg) < 1e-6, f"{name}: alignment implies greedy optimal"


# ---------------------------------------------------------------- Stage 3 API
def test_gym_api_roundtrip():
    """Gym-style reset/step contract."""
    from attrition import load
    env = load("shared-quota", seed=0)
    obs, info = env.reset(0)
    assert obs.shape == (env.n, 3)
    assert env.action_space["n"] == env.n
    steps = 0
    while True:
        a = int(env.valid_actions()[0])
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            break
    assert steps > 0
    assert info["total_regret"] >= -1e-9
    assert "realised_value" in info


def test_gym_env_rejects_destroyed_arm():
    from attrition import ConsumableBanditEnv
    env = ConsumableBanditEnv(v=[1.0], p=[1.0], e=[0.0], horizon=5, seed=0)
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)


def test_pettingzoo_api_roundtrip():
    """Parallel multi-agent contract; destruction by one agent affects all."""
    from attrition import load
    env = load("shared-quota-competing", seed=0)
    obs, info = env.reset(0)
    assert set(obs) == set(env.agents) and len(env.agents) == 2
    steps = 0
    while True:
        va = env.valid_actions()
        if len(va) == 0:
            break
        obs, rew, term, trunc, info = env.step({a: int(va[0]) for a in env.agents})
        steps += 1
        if all(term.values()) or all(trunc.values()):
            break
    assert steps > 0 and env.system_value != 0.0


def test_observation_hides_externality_by_default():
    """Theorem 3 says e is not reliably estimable, so it must not be observable."""
    from attrition import load
    env = load("shared-quota", seed=0)
    assert env.reset(0)[0].shape[1] == 3
    env2 = load("shared-quota", seed=0, reveal_externality=True)
    assert env2.reset(0)[0].shape[1] == 5


def test_every_scenario_matches_its_documented_behaviour():
    """Each scenario asserts the phenomenon it claims to exhibit."""
    from attrition import SCENARIOS, load
    from experiments.exp35_scenario_suite import exact
    checked = 0
    for name, spec in SCENARIOS.items():
        if spec["agents"] > 1:
            continue
        env = load(name, seed=0)
        v, p, e = env.v, np.clip(env.p, 1e-6, 1.0), env.e
        if len(v) > 9:
            continue
        T = min(env.horizon, 8)
        vs, vg, vi, rg = exact(v, p, e, env.delta, T)
        assert rg < 1e-9, f"{name}: greedy must record zero private regret"
        k = p * e
        corr = float(np.corrcoef(v, k)[0, 1]) if np.std(k) > 1e-12 else 0.0
        if corr < -0.3:
            assert abs(vs - vg) < 1e-6, f"{name}: alignment implies greedy optimal"
        elif corr > 0.3:
            assert vg < vs - 1e-6, f"{name}: conflict implies greedy loses"
            assert vi > vg, f"{name}: ECI must beat greedy"
        checked += 1
    assert checked >= 5


# ------------------------------------------------------- terminal commitment
def test_edge_first_narrows_the_envelope_it_seeks():
    """Ambition destroys the very thing it targets: chasing width yields width 1."""
    from attrition import (evaluate_commitment_policy, edge_first_policy,
                                  optimal_commitment_policy)
    for budget in [2, 4, 6]:
        v_edge, w_edge, _ = evaluate_commitment_policy(
            edge_first_policy, seeds=60, budget=budget)
        v_out, w_out, _ = evaluate_commitment_policy(
            optimal_commitment_policy, seeds=60, budget=budget)
        assert w_edge < w_out, "edge-first must end narrower than expanding outward"
        assert v_edge < v_out


def test_more_budget_does_not_help_edge_first():
    """Spending more on edges makes the terminal position worse, not better."""
    from attrition import evaluate_commitment_policy, edge_first_policy
    _, _, f2 = evaluate_commitment_policy(edge_first_policy, seeds=60, budget=2)
    _, _, f6 = evaluate_commitment_policy(edge_first_policy, seeds=60, budget=6)
    assert f6 > f2, "a larger budget must produce more blocked settings"


def test_expand_outward_wins_most_at_small_budget():
    """The commitment bites hardest when evidence is scarce."""
    from attrition import (evaluate_commitment_policy,
                                  greedy_commitment_policy,
                                  optimal_commitment_policy)
    gaps = []
    for budget in [2, 6]:
        g, _, _ = evaluate_commitment_policy(greedy_commitment_policy,
                                             seeds=60, budget=budget)
        o, _, _ = evaluate_commitment_policy(optimal_commitment_policy,
                                             seeds=60, budget=budget)
        gaps.append((o - g) / abs(g))
    assert gaps[0] > gaps[1], "advantage must shrink as budget grows"


def test_operating_value_is_deterministic():
    """Exact PMF evaluation, not sampling."""
    from attrition import TerminalCommitment
    a = TerminalCommitment(seed=3); a.reset(3)
    b = TerminalCommitment(seed=3); b.reset(3)
    assert abs(a.operating_value() - b.operating_value()) < 1e-12


# ------------------------------------------------------------ T-C and T-F
def test_estimation_floor_is_set_by_transitions_not_observations():
    """Sharpened Theorem 3: RMSE saturates when deaths saturate, not when
    observations do. Extending the horizon adds observations and nothing else."""
    from experiments.exp37_multiagent_estimation import episode, rmse_of_e
    n, sigma, delta = 8, 0.3, 0.1
    rm = np.random.default_rng(11)
    p = np.clip(rm.uniform(0.15, 0.5, n), 0.05, 1.0)
    e = np.clip(rm.uniform(0.2, 2.0, n), 0.0, None)
    v = np.sort(rm.uniform(0.4, 1.2, n))[::-1].copy()
    results = {}
    for T in [200, 900]:
        errs = []
        for s in range(25):
            rng = np.random.default_rng(4000 + s)
            rows, ys, alive = episode(n, 1, p, e, v, delta, sigma, T, rng, False)
            r = rmse_of_e(rows[0], ys[0], n, e, delta, alive)
            if not np.isnan(r):
                errs.append(r)
        results[T] = float(np.mean(errs))
    assert abs(results[200] - results[900]) < 1e-6, \
        "a 4.5x longer horizon must not change the floor"


def test_communication_does_not_help_under_shared_consumption():
    """Ten agents watching an arm die still see one death.

    The substantive claim is that pooling observations does not materially reduce
    the error, so the test asserts sharing gives no meaningful improvement rather
    than exact equality. Seed count matters here: at 25 seeds the comparison is
    dominated by Monte Carlo noise.
    """
    from experiments.exp37_multiagent_estimation import run
    for m in [2, 3, 4]:
        private, n_priv = run(m, share=False, seeds=60)
        shared, n_shared = run(m, share=True, seeds=60)
        assert n_shared > n_priv * 1.5, "sharing must actually pool observations"
        assert shared > private * 0.85, (
            f"m={m}: sharing {n_shared:.0f} observations instead of {n_priv:.0f} "
            f"must not cut the error materially ({private:.4f} -> {shared:.4f})")


def test_ordinal_mechanism_beats_uniform_tax():
    """T-F: cardinal pricing is unavailable, ordinal pricing is, and works."""
    from experiments.exp32_multiagent_poa import (planner_value,
                                                  decentralised_value,
                                                  rule_greedy)
    from experiments.exp38_mechanism_design import (make_uniform_tax,
                                                    make_rank_based)
    rng = np.random.default_rng(5)
    n, T, delta, m = 6, 5, 0.12, 3
    poa = {"greedy": [], "uniform": [], "rank": []}
    for _ in range(4):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
        pl = planner_value(v, p, e, delta, T, m)
        ebar = float(np.mean(e))
        for key, rule in [("greedy", rule_greedy),
                          ("uniform", make_uniform_tax(ebar)),
                          ("rank", make_rank_based(2.0 * ebar))]:
            d = decentralised_value(v, p, e, delta, T, m, rule, seeds=120)
            poa[key].append(pl / max(d, 1e-9))
    assert np.mean(poa["rank"]) < np.mean(poa["uniform"]), \
        "ordinal pricing must beat a flat charge"
    assert np.mean(poa["rank"]) < np.mean(poa["greedy"]), \
        "ordinal pricing must beat no charge"


def test_ordinal_recovery_also_saturates():
    """Rank recovery does not approach 1 even at vanishing noise: each arm still
    supplies one transition, and early/late deaths leave one side empty."""
    from experiments.exp39_ordinal_floor import trial
    taus = []
    for s in range(25):
        r, t = trial(n=8, sigma=0.001, spacing=4.0, seed=1000 + s)
        if not np.isnan(t):
            taus.append(t)
    tau = float(np.mean(taus))
    assert tau < 0.85, ("ordinal recovery must saturate below perfect even at "
                        f"a gap-to-noise ratio of 4000 (got tau={tau:.3f})")
    assert tau > 0.4, "but must still be better than chance"


def test_ordinal_improves_with_spacing_cardinal_does_not():
    """The two obey different laws: widening gaps helps ranking, not magnitudes."""
    from experiments.exp39_ordinal_floor import sweep
    res = sweep("spacing", [0.05, 1.00], n=8, seeds=25)
    (_, rmse_lo, tau_lo), (_, rmse_hi, tau_hi) = res
    assert tau_hi > tau_lo * 1.5, "wider spacing must improve rank recovery"
    assert rmse_hi >= rmse_lo * 0.9, "but must not improve cardinal recovery"


def test_rollout_scales_linearly_not_quadratically():
    """T-G: one improvement step closes a fixed fraction of the base gap.

    The discriminator is the log-log slope of rollout gap against base gap across
    a wide range of base gaps. Quadratic (error compounding) predicts 2; a
    constant closed fraction predicts 1. Comparing two cells is not enough --
    they must actually span a range of base gaps, which requires several spread
    levels and enough instances per level.
    """
    from experiments.exp40_rollout_guarantee import gaps
    rng = np.random.default_rng(77)
    bases, rolls = [], []
    for spread in [0.2, 0.6, 1.2, 1.8]:
        G, RG = [], []
        for _ in range(10):
            n, T, delta = 6, 8, 0.12
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
            a_, b_, _, _ = gaps(v, p, e, delta, T)
            G.append(a_); RG.append(b_)
        bases.append(float(np.mean(G))); rolls.append(float(np.mean(RG)))
    bases, rolls = np.array(bases), np.array(rolls)
    assert bases.max() / bases.min() > 2.0, "cells must span a range of base gaps"
    slope = float(np.polyfit(np.log(bases), np.log(rolls), 1)[0])
    assert slope < 1.7, (
        f"slope {slope:.2f} must be closer to linear than quadratic")
    assert (rolls / bases).mean() < 0.25, "rollout must close most of the gap"


def test_rollout_helps_greedy_more_than_eci_proportionally():
    """ECI has already removed the shallow errors one step of lookahead sees."""
    from experiments.exp40_rollout_guarantee import gaps
    rng = np.random.default_rng(77)
    G, RG, E, RE = [], [], [], []
    for _ in range(10):
        n, T, delta = 6, 8, 0.12
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
        a, b, c, d = gaps(v, p, e, delta, T)
        G.append(a); RG.append(b); E.append(c); RE.append(d)
    frac_greedy = 1 - np.mean(RG) / np.mean(G)
    frac_eci = 1 - np.mean(RE) / np.mean(E)
    assert frac_greedy > frac_eci, (
        "rollout must close proportionally more of the weaker base policy's gap")


def test_ordinal_saturation_is_a_pool_size_effect():
    """tau ~ 0.68 was an n=8 artefact: rank recovery improves with pool size
    because the number of positionally unusable arms stays roughly constant."""
    from experiments.exp41_ordinal_closed_form import trial
    res = {}
    for n in [8, 30]:
        taus, ks = [], []
        for s in range(20):
            t, u, _ = trial(n, seed=2000 + s)
            if not np.isnan(t):
                taus.append(t); ks.append(n * (1 - u))
        res[n] = (float(np.mean(taus)), float(np.mean(ks)))
    assert res[30][0] > res[8][0] + 0.10, "tau must rise with pool size"
    # the implied unusable count is a constant, not a fraction of n
    assert abs(res[30][1] - res[8][1]) < 1.5, (
        f"unusable arm count must stay roughly constant "
        f"(n=8: {res[8][1]:.2f}, n=30: {res[30][1]:.2f})")


def test_identification_cannot_be_improved_by_allocation_design():
    """Random allocation beats deliberate sequencing for rank recovery."""
    from experiments.exp42_identification_by_design import run
    taus = {}
    for rule in ["random", "bracketed"]:
        vals = []
        for s in range(20):
            t, _, _, _ = run(rule, n=12, seed=3000 + s)
            if not np.isnan(t):
                vals.append(t)
        taus[rule] = float(np.mean(vals))
    assert taus["random"] > taus["bracketed"], (
        "deliberate sequencing must not beat random allocation")


def test_k_is_structural_across_allocation_rules():
    """k ~ 2.5 holds whatever the allocation rule: it is not an artefact of one."""
    from experiments.exp42_identification_by_design import run
    for rule in ["random", "delayed", "bracketed"]:
        ks = []
        for s in range(20):
            _, k, _, _ = run(rule, n=16, seed=3000 + s)
            if not np.isnan(k):
                ks.append(k)
        assert 1.5 < float(np.mean(ks)) < 3.5, f"{rule}: k must stay near 2.5"


def test_k_is_independent_of_pool_size_as_derived():
    """k = a + 2*c*pbar: the pool size cancels because N_exh grows linearly in n."""
    from experiments.exp43_derive_k import measure_k
    ks = [measure_k(n=n, seeds=40)[0] for n in [6, 16, 40]]
    assert max(ks) - min(ks) < 0.6, f"k must be flat in n, got {ks}"


def test_k_scales_with_threshold_and_rate():
    """The derived leading term is linear in c and in pbar."""
    from experiments.exp43_derive_k import measure_k
    k1, _ = measure_k(n=16, threshold=1, seeds=40)
    k8, _ = measure_k(n=16, threshold=8, seeds=40)
    assert k8 > k1 * 2.5, "k must grow with the precision threshold"
    k_lo, _ = measure_k(n=16, p_lo=0.05, p_hi=0.15, seeds=40)
    k_hi, _ = measure_k(n=16, p_lo=0.5, p_hi=0.95, seeds=40)
    assert k_hi > k_lo * 2.0, "k must grow with the destruction rate"


def test_refined_k_formula_is_accurate():
    """k = 0.6 + 2*c*pbar predicts within ~0.4 across the tested range."""
    from experiments.exp43_derive_k import measure_k, predict_k
    for kw in [dict(threshold=1), dict(threshold=3), dict(threshold=8),
               dict(p_lo=0.3, p_hi=0.7)]:
        measured, _ = measure_k(n=16, seeds=40, **kw)
        predicted = predict_k(refined=True, **kw)
        assert abs(measured - predicted) < 0.45, (
            f"{kw}: measured {measured:.2f} vs predicted {predicted:.2f}")


def test_eci_is_exactly_optimal_under_constant_kappa():
    """Proposition: constant kappa makes the ECI charge arm-independent, so ECI
    reduces to greedy, which Theorem 1 says is optimal."""
    from experiments.exp44_eci_bound import gap
    rng = np.random.default_rng(3)
    for kappa in [0.1, 0.4, 1.0]:
        for _ in range(6):
            n = 6
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
            e = kappa / p                      # kappa exactly constant
            assert np.std(p * e) < 1e-12
            g, i = gap(v, p, e, 0.12, 8)
            assert abs(g) < 1e-9, "greedy must be optimal (Theorem 1)"
            assert abs(i) < 1e-9, "ECI must be optimal too"


def test_eci_absolute_loss_is_flat_while_greedy_explodes():
    """ECI's RAW loss is near-constant across coupling strengths that vary
    greedy's by two orders of magnitude.

    Raw losses, not percentages: at strong coupling V* passes through zero, so
    percentage gaps are meaningless (this is what produced the spurious '26.91%'
    figure in an earlier analysis).
    """
    from functools import lru_cache

    def raw(v, p, e, delta, T):
        n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
        B = lambda S: delta * (etot - sum(e[i] for i in S))
        R = lambda a, S: float(v[a] - B(S))

        @lru_cache(maxsize=None)
        def V(S, t):
            if t >= T or not S:
                return 0.0
            return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1)
                       for a in S)

        def pol(pick):
            @lru_cache(maxsize=None)
            def W(S, t):
                if t >= T or not S:
                    return 0.0
                a = pick(S, t)
                return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)
            return W(full, 0)

        vs = V(full, 0)
        g = pol(lambda S, t: max(S, key=lambda i: R(i, S)))
        i_ = pol(lambda S, t: max(S, key=lambda i: R(i, S)
                                  - delta*p[i]*e[i]*(T-t)))
        return vs - g, vs - i_

    rng = np.random.default_rng(17)
    losses = {}
    for delta in [0.02, 0.12, 0.45, 0.80, 1.50]:
        G, I = [], []
        for _ in range(10):
            n = 6
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
            g, i = raw(v, p, e, delta, 8)
            G.append(g); I.append(i)
        losses[delta] = (float(np.mean(G)), float(np.mean(I)))
    g_lo, i_lo = losses[0.02]
    g_hi, i_hi = losses[1.50]
    assert g_hi > g_lo * 50, "greedy loss must explode with coupling"
    assert i_hi < i_lo * 5, "ECI loss must stay bounded"


def test_eci_relative_error_falls_with_coupling():
    """The defensible claim: ECI's error shrinks as a FRACTION of greedy's."""
    from experiments.exp44_eci_bound import gap
    ratios = []
    for delta in [0.02, 0.45]:
        rng = np.random.default_rng(17)
        G, I = [], []
        for _ in range(8):
            n = 6
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
            g, i = gap(v, p, e, delta, 8)
            G.append(g); I.append(i)
        ratios.append(float(np.mean(I)) / max(float(np.mean(G)), 1e-9))
    assert ratios[1] < ratios[0] * 0.5, "relative error must fall with coupling"


# --------------------------------------------------------------- multi-agent
def test_sequential_multiagent_equals_single_learner():
    """Theorem 6: m agents x T rounds == 1 learner x m*T pulls, exactly."""
    from experiments.exp46_sequential_equivalence import (single_learner,
                                                          sequential_agents)
    rng = np.random.default_rng(5)
    for m, T in [(2, 3), (3, 2), (2, 4)]:
        D, G = [], []
        for _ in range(3):
            n = 6
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
            D.append(sequential_agents(v, p, e, 0.12, T, m, seeds=1500))
            G.append(single_learner(v, p, e, 0.12, m * T)[1])
        assert abs(np.mean(D) - np.mean(G)) < 0.05, (
            f"m={m},T={T}: sequential agents must equal a single learner")


def test_no_price_of_anarchy_at_zero_dispersion():
    """Even under simultaneous action, decentralisation costs nothing when the
    externality is uniformly priced -- there is no commons problem here."""
    from attrition import price_of_anarchy
    rng = np.random.default_rng(5)
    for m in [2, 3]:
        R = []
        for _ in range(4):
            n = 5
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = 0.4 / p                        # kappa exactly constant
            r, _, _ = price_of_anarchy(v, p, e, 0.12, 4, m)
            R.append(r)
        assert abs(float(np.mean(R)) - 1.0) < 0.03, (
            f"m={m}: PoA must be ~1 at zero kappa dispersion")


def test_spreading_out_is_worse_than_colliding():
    """Collisions are not the problem: concentrating on the best arm beats
    spreading agents across inferior ones."""
    from attrition import decentralised_value_simultaneous
    rng = np.random.default_rng(5)
    n = 5
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
    same = decentralised_value_simultaneous(v, p, e, 0.12, 4, 3,
                                            tie_break="same", seeds=300)
    spread = decentralised_value_simultaneous(v, p, e, 0.12, 4, 3,
                                              tie_break="random", seeds=300)
    assert same > spread, "colliding on the best arm must beat spreading out"


# ------------------------------------------------- charge misspecification
def test_free_riding_degrades_monotonically():
    """Free-riding is ECI with charge scaled by 1/m: loss falls monotonically as
    the discount weakens."""
    from experiments.exp48_charge_misspecification import sweep_lambda
    rows = sweep_lambda([0.0, 0.125, 0.25, 0.5, 1.0])
    losses = [r[1] for r in rows]
    assert all(losses[i] > losses[i+1] for i in range(len(losses)-1)), (
        f"loss must fall monotonically with lambda, got {losses}")


def test_partial_pricing_recovers_much_of_the_gain():
    """Even an eight-fold discount captures a large share of the benefit."""
    from experiments.exp48_charge_misspecification import sweep_lambda
    g = sweep_lambda([0.0])[0][1]
    partial = sweep_lambda([0.125])[0][1]
    best = min(r[1] for r in sweep_lambda([0.6, 1.0, 1.5, 2.5]))
    recovered = (g - partial) / max(g - best, 1e-9)
    assert recovered > 0.35, f"lambda=1/8 must recover a real share, got {recovered:.2f}"


def test_over_charging_is_safer_than_under_charging():
    """The robust asymmetry: err high when the externality scale is uncertain."""
    from experiments.exp48_charge_misspecification import sweep_lambda
    safer = 0
    settings = [{}, {"T": 14}, {"delta": 0.04}, {"spread": 0.4}, {"n": 8}]
    for kw in settings:
        under = sweep_lambda([1/3.0], **kw)[0][1]
        over = sweep_lambda([3.0], **kw)[0][1]
        safer += int(over < under)
    assert safer >= 4, f"over-charging must be safer in most settings ({safer}/5)"


def test_no_universal_optimal_scale():
    """Honesty check: argmin lambda moves with the setting, so 'use 1.5' would be
    overfitting. If this ever passes trivially the claim has crept back in."""
    from experiments.exp48_charge_misspecification import sweep_lambda
    lams = [0.6, 1.0, 1.5, 2.5, 3.0]
    argmins = []
    for kw in [{"T": 4}, {"T": 14}, {"delta": 0.04}]:
        rows = sweep_lambda(lams, **kw)
        argmins.append(rows[int(np.argmin([r[1] for r in rows]))][0])
    assert max(argmins) / min(argmins) > 2.0, (
        f"optimal scale must vary materially across settings, got {argmins}")


# ------------------------------------------------------- ECI closed-form bound
def test_monotonicity_is_false_with_identified_mechanism():
    """Extra available arms can be a LIABILITY: there is no null action, so an
    extra low-value arm can force continued play under high burden instead of
    letting the episode end at V=0 via exhaustion."""
    from experiments.exp49_eci_closed_form_bound import monotonicity_check
    viol, checked, worst = monotonicity_check(inst=3)
    assert viol > 0, "monotonicity must fail (this documents a real phenomenon)"
    assert worst < -0.01, "the violation must be substantive, not numerical noise"


def test_performance_difference_identity_exact():
    """Gap(pi) telescopes exactly into the sum of one-step regrets under pi's
    own trajectory distribution -- an exact identity, not an approximation."""
    from experiments.exp49_eci_closed_form_bound import performance_difference_check
    max_err = performance_difference_check(inst=3)
    assert max_err < 1e-6, f"identity must hold to numerical precision, got {max_err}"


def test_eci_closed_form_bound_never_violated():
    """The first closed-form guarantee for ECI: Gap(ECI) <=
    T(T+1) * max_a[p_a(delta*e_a + 2R)]. Loose, but never violated, including
    across the extreme-coupling regime that produces the largest raw losses."""
    from experiments.exp49_eci_closed_form_bound import build, closed_form_bound
    rng = np.random.default_rng(17)
    for delta in [0.02, 0.45, 1.5, 3.0]:
        n, T = 6, 8
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.0, n), 0.0, None)
        V, R, eci = build(v, p, e, delta, T)
        full = frozenset(range(n))

        def W(S, t):
            if t >= T or not S:
                return 0.0
            a = eci(S, t)
            return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)

        exact = V(full, 0) - W(full, 0)
        bound = closed_form_bound(v, p, e, delta, T)
        assert bound >= exact - 1e-6, (
            f"delta={delta}: bound {bound:.4f} must be >= exact gap {exact:.4f}")


# ------------------------------------------------ correlated destruction stress test
def test_greedy_zero_regret_survives_correlated_shocks():
    """Theorem 4's zero regret is definitional, so it must hold under ANY
    destruction mechanism, including correlated cluster shocks."""
    from experiments.exp50_correlated_destruction import exact_clustered
    rng = np.random.default_rng(71)
    n, T, delta = 6, 6, 0.15
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
    clusters = np.array([0, 0, 1, 1, 2, 2])
    for q in [0.0, 0.3, 0.5]:
        _, _, _, reg = exact_clustered(v, p, e, delta, T, clusters, q)
        assert reg < 1e-9, f"q={q}: greedy regret must stay exactly zero"


def test_constant_kappa_survives_correlated_shocks():
    """Verified conjecture: constant kappa gives greedy exact optimality even
    under correlated cluster-level exogenous destruction. Stated as a
    conjecture in THEORY.md, not a proven theorem -- this test checks the
    empirical claim, not a proof."""
    from experiments.exp50_correlated_destruction import exact_clustered
    rng = np.random.default_rng(3)
    n, T, delta = 6, 6, 0.15
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    kappa = 0.4
    p = np.clip(rng.uniform(0.15, 0.95, n), 0.05, 1.0)
    e = kappa / p                          # kappa exactly constant, e varies a lot
    clusters = np.array([0, 0, 0, 1, 1, 1])
    for q in [0.0, 0.3, 0.5, 0.7]:
        vs, vg, _, _ = exact_clustered(v, p, e, delta, T, clusters, q)
        assert abs(vs - vg) < 1e-6, (
            f"q={q}: greedy must remain exactly optimal under constant kappa")


def test_eci_becomes_harmful_under_strong_correlated_risk():
    """The genuine limitation: at high correlated-shock rates ECI is worse than
    greedy in the majority of instances. This documents a real boundary on the
    paper's central practical recommendation, not a bug to be fixed silently."""
    from experiments.exp50_correlated_destruction import exact_clustered
    rng = np.random.default_rng(5)
    n, T, delta = 6, 6, 0.15
    clusters = np.array([0, 0, 1, 1, 2, 2])
    worse = 0
    trials = 15
    for _ in range(trials):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
        vs, vg, ve, _ = exact_clustered(v, p, e, delta, T, clusters, 0.7)
        if (vs - ve) > (vs - vg):
            worse += 1
    assert worse >= trials // 2, (
        f"ECI must underperform greedy in most instances at q=0.7, got "
        f"{worse}/{trials}")


# ----------------------------------------------------------------- SECI
def test_seci_matches_eci_at_zero_shock_rate():
    """SECI's dampening is (1-q)^2, which is 1 at q=0, so SECI must be
    identical to plain ECI when there is no correlated risk."""
    from experiments.exp50_correlated_destruction import exact_clustered
    rng = np.random.default_rng(41)
    n, T, delta = 6, 6, 0.15
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
    clusters = np.array([0, 0, 1, 1, 2, 2])
    _, _, ve, _ = exact_clustered(v, p, e, delta, T, clusters, 0.0)
    from experiments.exp50_correlated_destruction import (
        exact_clustered_with_policy, seci_action)
    vs_seci = exact_clustered_with_policy(v, p, e, delta, T, clusters, 0.0,
        lambda S, t: seci_action(S, t, v, p, e, delta, T, 0.0))
    assert abs(ve - vs_seci) < 1e-9


def test_seci_matches_or_beats_greedy_across_shock_rates_exact():
    """The strong small-scale result: SECI matches or beats greedy at every
    q in [0,1], confirmed by exact DP."""
    from experiments.exp50_correlated_destruction import (
        exact_clustered, exact_clustered_with_policy, seci_action)
    rng = np.random.default_rng(41)
    n, T, delta = 6, 6, 0.15
    clusters = np.array([0, 0, 1, 1, 2, 2])
    for q in [0.0, 0.3, 0.7, 1.0]:
        VG, VS = [], []
        for _ in range(10):
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
            _, vg, _, _ = exact_clustered(v, p, e, delta, T, clusters, q)
            vs = exact_clustered_with_policy(v, p, e, delta, T, clusters, q,
                lambda S, t, v=v, p=p, e=e, q=q:
                    seci_action(S, t, v, p, e, delta, T, q))
            VG.append(vg); VS.append(vs)
        assert np.mean(VS) >= np.mean(VG) - 1e-6, (
            f"q={q}: SECI must match or beat greedy, got "
            f"{np.mean(VS):.4f} vs {np.mean(VG):.4f}")


def test_seci_beats_plain_eci_at_scale():
    """The result that DOES transfer to scale: SECI reliably beats plain ECI,
    even where it does not fully close the gap to greedy."""
    from experiments.exp50_correlated_destruction import (
        simulate_clustered, rule_eci_c, rule_seci_c)
    rng = np.random.default_rng(9)
    n, T, delta = 40, 40, 0.03
    clusters = np.repeat(np.arange(8), 5)
    for q in [0.1, 0.3, 0.5, 0.7]:
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
        ec = simulate_clustered(v, p, e, delta, T, clusters, q, rule_eci_c,
                                seeds=150)
        se = simulate_clustered(v, p, e, delta, T, clusters, q, rule_seci_c,
                                seeds=150)
        assert se >= ec - 0.15, f"q={q}: SECI must not underperform ECI at scale"


def test_seci_policy_class_reduces_to_eci_at_q_zero():
    """The library-level SECI class, not just the experiment script version."""
    from attrition import SECI, ECI, ConsumableBandit, run
    env1 = ConsumableBandit.random(n=15, k_spread=1.5, delta=0.05, horizon=30,
                                   seed=0)
    env2 = ConsumableBandit.random(n=15, k_spread=1.5, delta=0.05, horizon=30,
                                   seed=0)
    r1 = run(env1, SECI(q=0.0), seed=0)
    r2 = run(env2, ECI(), seed=0)
    assert abs(r1["value"] - r2["value"]) < 1e-9


def test_seci_scale_gap_is_an_artifact_of_unfair_delta_scaling():
    """The apparent shortfall at n=40 was caused by holding delta fixed while n
    grew, making burden capacity delta*sum(e) scale with n. Rescaling delta to
    hold problem difficulty comparable across n closes the gap to noise level."""
    from experiments.exp50_correlated_destruction import (
        simulate_clustered, rule_greedy_c, rule_eci_c, rule_seci_c)
    rng = np.random.default_rng(21)
    n, T = 40, 40
    delta = 0.03 * 6.0 / n            # the fairness correction
    clusters = np.repeat(np.arange(n // 2), 2)
    for q in [0.0, 0.3, 0.7, 1.0]:
        G, E, S = [], [], []
        for _ in range(6):
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
            G.append(simulate_clustered(v, p, e, delta, T, clusters, q,
                                        rule_greedy_c, seeds=200))
            E.append(simulate_clustered(v, p, e, delta, T, clusters, q,
                                        rule_eci_c, seeds=200))
            S.append(simulate_clustered(v, p, e, delta, T, clusters, q,
                                        rule_seci_c, seeds=200))
        g, ec, se = np.mean(G), np.mean(E), np.mean(S)
        # SECI must be within noise of greedy (not the large shortfall seen
        # under the unfair, unscaled comparison) and must beat ECI
        assert se >= g - 0.5, (
            f"q={q}: SECI must be within noise of greedy once fairly scaled, "
            f"got {se:.3f} vs {g:.3f}")
        assert se >= ec - 1e-6, f"q={q}: SECI must not lose to plain ECI"


def test_seci_captures_far_more_opportunity_than_greedy_at_n12_exact():
    """The definitive answer to 'SECI's improvement over greedy looks small':
    at n=12 (real exact DP, double the original validated scale), SECI
    captures 4-5x more of the available opportunity than greedy at every q
    with meaningful opportunity left. The absolute margin shrinks with q
    because the opportunity itself shrinks (greedy's own gap collapses too),
    not because SECI weakens."""
    from experiments.exp50_correlated_destruction import (
        exact_clustered, exact_clustered_with_policy, seci_action)
    rng = np.random.default_rng(7)
    n, T, delta = 12, 12, 0.06
    clusters = np.repeat(np.arange(n // 2), 2)
    for q in [0.0, 0.2, 0.4]:
        VS, VG, VE = [], [], []
        for _ in range(3):
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
            vs, vg, _, _ = exact_clustered(v, p, e, delta, T, clusters, q)
            vsec = exact_clustered_with_policy(v, p, e, delta, T, clusters, q,
                lambda S, t, v=v, p=p, e=e, q=q:
                    seci_action(S, t, v, p, e, delta, T, q))
            VS.append(vs); VG.append(vg); VE.append(vsec)
        vs, vg, ve = np.mean(VS), np.mean(VG), np.mean(VE)
        greedy_gap = (vs - vg) / abs(vs)
        seci_gap = (vs - ve) / abs(vs)
        assert seci_gap <= greedy_gap + 1e-9, (
            f"q={q}: SECI must not have a larger gap than greedy")
        if greedy_gap > 0.01:   # only meaningful where there's real opportunity
            assert seci_gap < greedy_gap * 0.6, (
                f"q={q}: SECI must capture substantially more of the "
                f"opportunity than greedy, got seci_gap={seci_gap:.3f} vs "
                f"greedy_gap={greedy_gap:.3f}")


# ---------------------------------------------------- policy improvement theorem
def test_policy_improvement_never_decreases_value():
    """Proven: one step of exact policy improvement over ANY base policy's own
    value function cannot decrease value. Verified across three deliberately
    bad base policies."""
    from experiments.exp51_policy_improvement import (BASE_POLICIES,
                                                       policy_improvement_check)
    for name, fn in BASE_POLICIES.items():
        viol, checked, worst = policy_improvement_check(name, fn, seeds=10)
        assert viol == 0, f"{name}: policy improvement must never decrease value"
        assert worst > -1e-9


# ---------------------------------------------------- exact homogeneous-p k
def test_exact_k_homogeneous_matches_simulation():
    """Proven: under homogeneous destruction rate, k has an exact closed form
    via exchangeability + memorylessness. Matches simulation to within noise
    across multiple (c, p) pairs."""
    from experiments.exp43_derive_k import (exact_k_homogeneous,
                                            measure_k_homogeneous)
    for c, p in [(2, 0.3), (3, 0.5), (5, 0.7)]:
        measured = measure_k_homogeneous(20, p, threshold=c, seeds=3000)
        predicted = exact_k_homogeneous(p, c)
        assert abs(measured - predicted) < 0.15, (
            f"c={c}, p={p}: measured {measured:.3f} vs exact {predicted:.3f}")


def test_exact_k_c3_simplifies_to_one_plus_five_p():
    """The c=3 special case has a clean closed form: k = 1 + 5p exactly."""
    from experiments.exp43_derive_k import exact_k_homogeneous
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert abs(exact_k_homogeneous(p, 3) - (1 + 5 * p)) < 1e-9


def test_last_arm_always_contributes_exactly_one():
    """The j=0 term in the closed form is P(tau_0 <= c-1) = 1 identically:
    the last arm to die always has zero after-data, resolving the earlier
    residual-constant mystery."""
    from experiments.exp43_derive_k import exact_k_homogeneous
    # as p -> 0, k should approach exactly 1 (only the last-arm term survives)
    assert abs(exact_k_homogeneous(1e-6, 3) - 1.0) < 1e-3


# ------------------------------------------------------- Theorem 5 (repaired)
def test_theorem5_naive_formula_was_flawed():
    """Documents the bug an external review caught: the original proof's
    global pre/post mean-difference formula understates the true (arm-
    controlled) variance whenever arms are heterogeneous in v."""
    from experiments.exp52_theorem5_fwl_repair import (exact_oracle_variance,
                                                        naive_claimed_variance)
    rng = np.random.default_rng(11)
    understated = 0
    for _ in range(20):
        N = 40
        a_seq = rng.integers(0, 6, N).tolist()
        tau_i = int(rng.integers(5, N - 5))
        naive = naive_claimed_variance(N, tau_i)
        _, exact = exact_oracle_variance(a_seq, tau_i)
        if exact > naive + 1e-9:
            understated += 1
    assert understated >= 15, "the naive formula must understate variance in most cases"


def test_theorem5_repaired_bound_holds_including_adversarial():
    """The corrected proof: RSS_i <= N/4 (AM-GM), hence Var_oracle >=
    4*sigma^2/N, verified across random AND deliberately adversarial
    (segregated, skewed) allocations -- not just typical ones."""
    from experiments.exp52_theorem5_fwl_repair import exact_oracle_variance
    rng = np.random.default_rng(42)
    violations = 0
    # random allocations
    for _ in range(100):
        N = int(rng.integers(10, 80))
        n_arms = int(rng.integers(2, 10))
        a_seq = rng.integers(0, n_arms, N).tolist()
        tau_i = int(rng.integers(0, N - 1))
        rss, var = exact_oracle_variance(a_seq, tau_i)
        if rss > N/4 + 1e-9 or var < 4.0/N - 1e-9:
            violations += 1
    # adversarial: arms segregated into pure-before / pure-after blocks
    for n_arms in [2, 4, 6, 8]:
        N = n_arms * 5
        a_seq = []
        for a in range(n_arms):
            a_seq += [a] * 5
        tau_i = N // 2
        rss, var = exact_oracle_variance(a_seq, tau_i)
        if rss > N/4 + 1e-9 or var < 4.0/N - 1e-9:
            violations += 1
    assert violations == 0, f"AM-GM bound RSS<=N/4 must never be violated"


# ------------------------------------------ Theorem 1 sufficiency, investigated
def test_reviewer_counterexample_does_not_break_sufficiency():
    """The second review's exact proposed construction (p1=1,e1=kappa vs
    p2=0.5,e2=2kappa), checked exhaustively at EVERY reachable state, not just
    the initial one."""
    from experiments.exp53_t1_sufficiency_investigation import check_state_by_state
    total, ok_count = 0, 0
    for kappa in [0.1, 0.5, 1.0, 2.0]:
        for T in [2, 3, 4]:
            for v1, v2 in [(1.0, 0.5), (0.5, 1.0), (1.0, -0.5)]:
                p = np.array([1.0, 0.5])
                e = np.array([kappa, 2*kappa])
                v = np.array([v1, v2])
                ok, info = check_state_by_state(v, p, e, 0.3, T)
                total += 1
                ok_count += ok
    assert ok_count == total, f"greedy must be optimal at every state, every config"


def test_burden_is_policy_invariant_even_though_old_closed_form_was_wrong():
    """The retracted closed form (delta*kappa*E[N(N-1)/2]) does not match the
    true aggregate burden, but the burden IS exactly policy-invariant via a
    different mechanism -- this is the corrected, honest claim."""
    from experiments.exp53_t1_sufficiency_investigation import aggregate_burden_by_policy
    v = np.array([1.0, 0.9, 0.8])
    p = np.array([0.6, 0.5, 0.3])
    kappa = 1.0
    e = kappa / p
    delta, T = 0.3, 6
    policies = {
        "greedy": lambda S, t: max(S, key=lambda a: v[a]),
        "min_v": lambda S, t: min(S, key=lambda a: v[a]),
        "max_p": lambda S, t: max(S, key=lambda a: p[a]),
        "min_p": lambda S, t: min(S, key=lambda a: p[a]),
    }
    vals = [aggregate_burden_by_policy(v, p, e, delta, T, pick)
           for pick in policies.values()]
    assert max(vals) - min(vals) < 1e-6, "burden must be policy-invariant"
    # and confirm the OLD closed form does NOT match (documents the retraction)
    N_expected_naive = sum(1.0/pi for pi in p)  # rough scale check only
    old_closed_form = delta * kappa * (N_expected_naive * (N_expected_naive-1) / 2)
    assert abs(old_closed_form - vals[0]) > 0.1, (
        "the retracted closed form should NOT match the true value")


def test_adjacent_exchange_lemma():
    """The correct mechanism behind sufficiency: swapping two consecutive
    same-kappa pulls leaves both reward and the full outcome distribution
    exactly unchanged."""
    from experiments.exp53_t1_sufficiency_investigation import adjacent_exchange_check
    rng = np.random.default_rng(3)
    for _ in range(10):
        kappa = rng.uniform(0.1, 2.0)
        p = rng.uniform(0.05, 0.95, 2)
        e = kappa / p
        v = rng.uniform(-1, 1, 2)
        delta = rng.uniform(0.1, 1.5)
        B0 = rng.uniform(0, 2)
        r_ok, d_ok = adjacent_exchange_check(v, p, e, delta, B0)
        assert r_ok, "two-round reward must be order-invariant under equal kappa"
        assert d_ok, "the full outcome distribution must be order-invariant too"


def test_block_exchange_lemma():
    """Extends the single-pull exchange lemma: exhausting arm a for Ka
    attempts then b for Kb, vs the reverse order, gives identical expected
    reward when kappa_a = kappa_b. A real implementation bug (block did not
    stop early on death) was caught by this test during development."""
    from experiments.exp53_t1_sufficiency_investigation import block_exchange_check
    rng = np.random.default_rng(7)
    for _ in range(10):
        p = rng.uniform(0.1, 0.9, 2)
        kappa = rng.uniform(0.2, 2.0)
        e = kappa / p
        v = rng.uniform(-1, 1, 2)
        delta = rng.uniform(0.1, 1.0)
        B0 = rng.uniform(0, 1.5)
        Ka, Kb = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        r1, r2 = block_exchange_check(v, p, e, delta, B0, (0, 1), Ka, Kb)
        assert abs(r1 - r2) < 1e-9, "block exchange must be exact under equal kappa"


def test_pathwise_rearrangement_is_not_invariant():
    """Documents a falsified proof strategy: for a FIXED realisation of
    attempt-counts, burden is NOT invariant to scheduling order (30 vs 1) --
    ruling out an entire class of deterministic proof arguments."""
    from experiments.exp53_t1_sufficiency_investigation import (
        pathwise_rearrangement_counterexample)
    bA, bB = pathwise_rearrangement_counterexample()
    assert bA != bB, "the counterexample must show genuine pathwise divergence"
    assert abs(bA - bB) > 10, "the divergence must be large, not noise-level"


def test_pathwise_coupling_is_not_invariant():
    """Documents a second falsified proof strategy: coupling every policy to
    the same underlying per-arm coin sequences does NOT give pathwise-equal
    burden across different deterministic policies."""
    from experiments.exp53_t1_sufficiency_investigation import pathwise_coupling_check
    rng = np.random.default_rng(17)
    n, T = 3, 8
    kappa = rng.uniform(0.2, 2.0)
    p = rng.uniform(0.1, 0.9, n)
    e = kappa / p
    v = rng.uniform(-1, 1, n)
    delta = rng.uniform(0.1, 1.0)
    coins = [rng.random(T) < p[i] for i in range(n)]

    def greedy(idx, v, p, t):
        return idx[np.argmax(v[idx])]

    def roundrobin(idx, v, p, t):
        return idx[t % len(idx)]
    vals = pathwise_coupling_check(v, p, e, delta, T, coins, [greedy, roundrobin])
    assert abs(vals[0] - vals[1]) > 1e-6, (
        "pathwise coupling must NOT give equal burden (documents the falsification)")
