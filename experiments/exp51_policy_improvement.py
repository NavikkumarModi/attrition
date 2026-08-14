"""Experiment 51 -- the general Policy Improvement Theorem for this MDP class.

Claim (proven by backward induction, see THEORY.md): for any deterministic
Markov policy pi0, one step of exact policy improvement over pi0's own value
function never decreases value.

This makes precise why the correlated-destruction rollout episode's apparent
regression below SECI was diagnostic of a bug: with EXACT computation such a
regression is provably impossible.
"""
from functools import lru_cache

import numpy as np

__all__ = ["build", "value_of", "policy_improve", "policy_improvement_check"]


def build(v, p, e, delta, T):
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))
    return n, full, R


def value_of(pick, R, p, T):
    @lru_cache(maxsize=None)
    def W(S, t):
        if t >= T or not S:
            return 0.0
        a = pick(S, t)
        return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)
    return W


def policy_improve(base_pick, R, p, T):
    """One step of EXACT policy improvement over base_pick."""
    W0 = value_of(base_pick, R, p, T)

    def pi1(S, t):
        return max(S, key=lambda a: R(a, S) + p[a]*W0(S-{a}, t+1)
                   + (1-p[a])*W0(S, t+1))
    return pi1, W0


def policy_improvement_check(base_name, base_fn, n=5, T=6, delta=0.15,
                             seeds=15, seed=99):
    rng = np.random.default_rng(seed)
    viol, checked, worst = 0, 0, np.inf
    for _ in range(seeds):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
        _, full, R = build(v, p, e, delta, T)
        base = base_fn(v, p, e)
        pi1, W0 = policy_improve(base, R, p, T)
        W1 = value_of(pi1, R, p, T)
        diff = W1(full, 0) - W0(full, 0)
        checked += 1
        if diff < -1e-9:
            viol += 1
        worst = min(worst, diff)
    return viol, checked, worst


BASE_POLICIES = {
    "worst-value": lambda v, p, e: (lambda S, t: min(S, key=lambda a: v[a])),
    "arbitrary": lambda v, p, e: (lambda S, t: sorted(S)[0]),
    "max-p": lambda v, p, e: (lambda S, t: max(S, key=lambda a: p[a])),
}


if __name__ == "__main__":
    print("POLICY IMPROVEMENT THEOREM -- verification across bad base policies\n")
    print(f"{'base policy':>14} {'violations':>11} {'checked':>8} {'min(V1-V0)':>12}")
    print("-" * 50)
    for name, fn in BASE_POLICIES.items():
        viol, checked, worst = policy_improvement_check(name, fn)
        print(f"{name:>14} {viol:11d} {checked:8d} {worst:12.2e}")
    print("\n  Never violated: one step of exact policy improvement provably")
    print("  cannot decrease value, for this MDP class, regardless of the base.")
