"""Exact optimal policy for the evolving-action-set bandit, by dynamic
programming over subsets of available arms.

For small n this settles the benchmark question exactly, with no Whittle
relaxation and no approximation error.

State: (frozenset of available arm indices, rounds remaining).
Transition on pulling arm a, in the no-arrival case:
    with prob p            -> S \\ {a}     (endogenous death)
    with prob 1-p          -> S            (survives)
plus independent exogenous death of every arm with prob mu.

V_t(S) = max_{a in S} [ v_a + E[ V_{t+1}(S') ] ]
"""

from functools import lru_cache
from itertools import combinations
import numpy as np


def exact_optimal(values, p, T, mu=0.0):
    """Exact V_0(full set) and the optimal first action, no arrivals."""
    n = len(values)
    full = frozenset(range(n))

    def exo_outcomes(S):
        """All post-exogenous-churn subsets with probabilities."""
        if mu == 0.0:
            return [(S, 1.0)]
        out = []
        S = list(S)
        for k in range(len(S) + 1):
            for survivors in combinations(S, k):
                surv = frozenset(survivors)
                prob = (1 - mu) ** k * mu ** (len(S) - k)
                if prob > 0:
                    out.append((surv, prob))
        return out

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        best = -np.inf
        for a in S:
            # immediate reward
            val = values[a]
            # endogenous outcome
            after_endo = [(S - {a}, p), (S, 1 - p)]
            cont = 0.0
            for S1, pr1 in after_endo:
                if pr1 == 0:
                    continue
                for S2, pr2 in exo_outcomes(S1):
                    cont += pr1 * pr2 * V(S2, t + 1)
            total = val + cont
            if total > best:
                best = total
        return best

    v_opt = V(full, 0)

    # optimal first action
    best_a, best_val = None, -np.inf
    for a in full:
        cont = 0.0
        for S1, pr1 in [(full - {a}, p), (full, 1 - p)]:
            if pr1 == 0:
                continue
            for S2, pr2 in exo_outcomes(S1):
                cont += pr1 * pr2 * V(S2, 1)
        tot = values[a] + cont
        if tot > best_val:
            best_val, best_a = tot, a
    return v_opt, best_a


def greedy_value(values, p, T, mu=0.0, n_mc=20000, seed=0):
    """Monte-Carlo value of the myopic greedy policy."""
    rng = np.random.default_rng(seed)
    n = len(values)
    totals = np.zeros(n_mc)
    for m in range(n_mc):
        alive = np.ones(n, dtype=bool)
        tot = 0.0
        for t in range(T):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            a = idx[np.argmax(values[idx])]
            tot += values[a]
            if rng.random() < p:
                alive[a] = False
            if mu > 0:
                for i in np.flatnonzero(alive):
                    if rng.random() < mu:
                        alive[i] = False
        totals[m] = tot
    return float(totals.mean()), float(totals.std() / np.sqrt(n_mc))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'n':>3} {'T':>3} {'p':>5} {'mu':>5} {'V*':>9} {'greedy':>9} "
          f"{'gap':>8} {'greedy=opt?':>12}")
    print("-" * 62)
    for trial in range(12):
        n = int(rng.integers(4, 8))
        T = int(rng.integers(3, 10))
        p = float(rng.choice([0.2, 0.5, 0.8, 1.0]))
        mu = float(rng.choice([0.0, 0.05]))
        values = np.sort(rng.uniform(0, 1, size=n))[::-1].copy()

        v_opt, a_star = exact_optimal(values, p, T, mu)
        v_greedy, se = greedy_value(values, p, T, mu, seed=trial)
        gap = v_opt - v_greedy
        # greedy picks arm 0 (values sorted descending)
        same = "yes" if a_star == 0 else f"NO (a*={a_star})"
        print(f"{n:3d} {T:3d} {p:5.2f} {mu:5.2f} {v_opt:9.4f} {v_greedy:9.4f} "
              f"{gap:8.4f} {same:>12}")
