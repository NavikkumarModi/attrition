"""Experiment 32 -- Stage 1: multi-agent shared pools, and the price of anarchy.

Several decision-makers draw on ONE consumable pool. Two orchestrators burning the
same API quota; two sponsors competing for the same eligible patients; two brand
teams consuming shared payer goodwill.

Each agent collects its own reward but every destruction raises the burden for
EVERYONE. So restraint is a public good: an agent that declines a high-kappa arm
pays the cost alone and shares the benefit. Classic commons structure, with a
bandit inside.

Definitions.
    system value W = sum over agents of their collected reward
    central planner   -- one controller choosing all m pulls per round to
                         maximise W (exact DP over subsets x rounds)
    decentralised     -- each agent independently applies its own policy to the
                         shared pool, seeing only its own reward
    price of anarchy  PoA = W(planner) / W(decentralised)

The measurement of interest: PoA under consumption, as a function of the number of
agents and the externality dispersion. Nothing in the bandit literature reports it,
because no existing model has an action set that the decision-makers consume.

Turn order within a round is round-robin, so agents see the pool as it stands when
they act -- an agent later in the order inherits the damage done earlier in the
same round.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np


# ------------------------------------------------------------------ planner
def planner_value(v, p, e, delta, T, m):
    """Exact optimum for a central controller making m pulls per round.

    State is the surviving set; within a round the controller picks m distinct
    arms (or all remaining if fewer), and each destroys independently.
    """
    n = len(v)
    etot = float(np.sum(e))

    def burden(S):
        return delta * (etot - sum(e[i] for i in S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        k = min(m, len(S))
        best = -np.inf
        for combo in combinations(sorted(S), k):
            b = burden(S)
            immediate = sum(v[a] - b for a in combo)
            # enumerate destruction outcomes over the chosen arms
            cont = 0.0
            for mask in range(1 << k):
                prob, dead = 1.0, []
                for j, a in enumerate(combo):
                    if mask >> j & 1:
                        prob *= p[a]; dead.append(a)
                    else:
                        prob *= (1 - p[a])
                if prob > 0:
                    cont += prob * V(S - frozenset(dead), t + 1)
            tot = immediate + cont
            if tot > best:
                best = tot
        return best

    return V(frozenset(range(n)), 0)


# ------------------------------------------------------------ decentralised
def decentralised_value(v, p, e, delta, T, m, rule, seeds=400):
    """m independent agents, round-robin within each round, Monte Carlo."""
    n = len(v)
    etot = float(np.sum(e))
    totals = []
    for s in range(seeds):
        rng = np.random.default_rng(9000 + s)
        alive = np.ones(n, dtype=bool)
        tot = 0.0
        for t in range(T):
            if not alive.any():
                break
            for _ in range(m):
                idx = np.flatnonzero(alive)
                if idx.size == 0:
                    break
                b = delta * float(e[~alive].sum())
                a = rule(idx, v, p, e, delta, T, t, b)
                tot += v[a] - b
                if rng.random() < p[a]:
                    alive[a] = False
        totals.append(tot)
    return float(np.mean(totals))


def rule_greedy(idx, v, p, e, delta, T, t, b):
    return int(idx[np.argmax(v[idx])])


def rule_eci(idx, v, p, e, delta, T, t, b):
    return int(idx[np.argmax(v[idx] - delta * p[idx] * e[idx] * (T - t))])


def rule_eci_selfish(idx, v, p, e, delta, T, t, b):
    """An agent that prices only the share of damage it expects to bear itself.

    With m agents each taking roughly 1/m of future pulls, a selfish agent
    discounts the externality by 1/m -- the free-riding response.
    """
    return int(idx[np.argmax(v[idx] - delta * p[idx] * e[idx] * (T - t) / 3.0)])


if __name__ == "__main__":
    print("PRICE OF ANARCHY UNDER CONSUMPTION\n")
    print("W(planner) / W(decentralised); higher means decentralisation costs more\n")
    rng = np.random.default_rng(5)
    print(f"{'agents':>7} {'std(k)':>7} {'planner':>9} {'dec-greedy':>11} "
          f"{'dec-ECI':>9} {'PoA greedy':>11} {'PoA ECI':>9}")
    print("-" * 68)
    for m in [1, 2, 3]:
        for spread in [0.4, 1.2]:
            P, DG, DE, SD = [], [], [], []
            for _ in range(6):
                n, T, delta = 6, 5, 0.12
                v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
                p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
                e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
                P.append(planner_value(v, p, e, delta, T, m))
                DG.append(decentralised_value(v, p, e, delta, T, m, rule_greedy))
                DE.append(decentralised_value(v, p, e, delta, T, m, rule_eci))
                SD.append(float(np.std(p * e)))
            pl, dg, de = np.mean(P), np.mean(DG), np.mean(DE)
            print(f"{m:7d} {np.mean(SD):7.3f} {pl:9.3f} {dg:11.3f} {de:9.3f} "
                  f"{pl/max(dg,1e-9):11.3f} {pl/max(de,1e-9):9.3f}")

    print("\n\nFREE-RIDING: does discounting the externality by one's own share hurt?\n")
    print(f"{'agents':>7} {'social ECI':>12} {'selfish ECI':>13} {'loss':>9}")
    print("-" * 44)
    rng2 = np.random.default_rng(11)
    for m in [2, 3]:
        S_, F_ = [], []
        for _ in range(8):
            n, T, delta = 6, 6, 0.12
            v = np.sort(rng2.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng2.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = np.clip(1.0 + rng2.normal(0, 1.2, n), 0.0, None)
            S_.append(decentralised_value(v, p, e, delta, T, m, rule_eci))
            F_.append(decentralised_value(v, p, e, delta, T, m,
                                          rule_eci_selfish))
        s_, f_ = np.mean(S_), np.mean(F_)
        print(f"{m:7d} {s_:12.3f} {f_:13.3f} {100*(s_-f_)/abs(s_):8.1f}%")
    print("\nAgents that price only their own share of the damage destroy system")
    print("value -- the commons structure, with a bandit inside.")
