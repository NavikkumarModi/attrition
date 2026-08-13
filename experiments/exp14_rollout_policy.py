r"""Experiment 14 -- a policy that beats the index.

The index I(a,t) = v_a - delta*kappa_a*(T-t) captures 52-99% of the optimality
gap. The remainder comes from the option-loss term the index ignores, for which
exp13 showed no closed form exists.

Standard fix: policy improvement. Use the index policy as a base policy pi_0,
and act greedily with respect to ITS value function:

    Q(a) = r(a,S) + p_a * V_{pi_0}(S\{a}, t+1) + (1-p_a) * V_{pi_0}(S, t+1)
    pi_1(S,t) = argmax_a Q(a)

The policy improvement theorem guarantees V_{pi_1} >= V_{pi_0} pointwise. The
option-loss term is captured implicitly -- V_{pi_0}(S) - V_{pi_0}(S\{a}) is
exactly the marginal value of holding arm a under the base policy, which is
what the index was missing.

V_{pi_0} is computed exactly by memoised recursion (cheap: the base policy is
deterministic, so no maximisation is needed inside).

Policies compared: greedy, index, rollout(index), rollout(greedy), exact DP.
"""

from functools import lru_cache
import numpy as np


def make_problem(n, T, delta, spread, rng):
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
    return v, p, e


def evaluate(v, p, e, delta, T):
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))

    def B(S):
        return delta * (etot - sum(e[i] for i in S))

    def R(a, S):
        return float(v[a] - B(S))

    # ---- exact optimum
    @lru_cache(maxsize=None)
    def Vstar(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*Vstar(S-{a}, t+1) + (1-p[a])*Vstar(S, t+1)
                   for a in S)

    # ---- base policies
    def pick_greedy(S, t):
        return max(S, key=lambda i: R(i, S))

    def pick_index(S, t):
        return max(S, key=lambda i: R(i, S) - delta*p[i]*e[i]*(T-t))

    def make_value(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S:
                return 0.0
            a = pick(S, t)
            return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)
        return W

    Wg, Wi = make_value(pick_greedy), make_value(pick_index)

    # ---- rollout: one step of policy improvement over a base value function
    def make_rollout(W):
        def pick(S, t):
            return max(S, key=lambda a: R(a, S)
                       + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1))
        return pick

    Wrg = make_value(make_rollout(Wg))
    Wri = make_value(make_rollout(Wi))

    return (Vstar(full, 0), Wg(full, 0), Wi(full, 0),
            Wrg(full, 0), Wri(full, 0))


if __name__ == "__main__":
    rng = np.random.default_rng(97)
    N, T, DELTA, INST = 6, 8, 0.12, 15
    print("gap to exact optimum (%), lower is better\n")
    print(f"{'std(k)':>7} {'greedy':>9} {'index':>9} {'roll(grd)':>10} "
          f"{'roll(idx)':>10}")
    print("-" * 50)
    for spread in [0.1, 0.3, 0.6, 1.0, 1.5]:
        acc = {k: [] for k in ["g", "i", "rg", "ri"]}
        sds = []
        for _ in range(INST):
            v, p, e = make_problem(N, T, DELTA, spread, rng)
            vs, wg, wi, wrg, wri = evaluate(v, p, e, DELTA, T)
            acc["g"].append((vs-wg)/abs(vs)*100)
            acc["i"].append((vs-wi)/abs(vs)*100)
            acc["rg"].append((vs-wrg)/abs(vs)*100)
            acc["ri"].append((vs-wri)/abs(vs)*100)
            sds.append(float(np.std(p*e)))
        print(f"{np.mean(sds):7.3f} {np.mean(acc['g']):8.3f}% "
              f"{np.mean(acc['i']):8.3f}% {np.mean(acc['rg']):9.3f}% "
              f"{np.mean(acc['ri']):9.3f}%")
