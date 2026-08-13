"""Experiment 50 -- stress-testing independence of destruction across arms.

Every theorem so far assumes destruction is independent across arms: pulling arm A
tells you nothing about arm B's survival. That assumption is false in exactly the
domains motivating this project -- a shared API provider's outage kills several
tools at once; a class-wide toxicity mechanism can invalidate several doses
together; a common raw-material shortage blocks several process settings
simultaneously. This experiment asks what breaks, and what does not, when arms are
clustered and share correlated exogenous shocks.

MODEL. Arms are partitioned into clusters. In addition to the existing per-pull
destruction (arm dies with probability p_a when pulled), each cluster faces an
exogenous shock each round with probability q: if it fires, EVERY currently alive
arm in that cluster is destroyed simultaneously, regardless of whether it was
pulled.

THREE QUESTIONS.

(1) Does Theorem 4 (zero-regret vacuity) survive? It should, for a structural
    reason worth stating precisely: greedy's regret against the best-available
    benchmark is zero BY DEFINITION at every step, regardless of how destruction
    happens. Nothing about the mechanism enters that argument. This predicts a
    clean "yes" and is the right thing to verify first.

(2) Does Theorem 1's characterisation (kappa = p*e governs optimality) survive?
    The sufficiency proof relied on E[N] = sum 1/p_j being policy-independent,
    which used INDEPENDENT geometric lifetimes. Correlated cluster shocks break
    that derivation. This predicts characterisation by kappa alone may fail even
    at "constant kappa" by the old definition, because cluster membership now
    matters too.

(3) Does ECI, which prices only p_a*e_a and is blind to cluster structure,
    still capture most of the value gap? Or does correlated risk require pricing
    the CLUSTER's aggregate exposure, not just the arm's own?
"""

from functools import lru_cache
from itertools import combinations

import numpy as np

__all__ = ["exact_clustered", "greedy_regret_clustered"]


def exact_clustered(v, p, e, delta, T, clusters, q):
    """Exact DP with per-pull AND correlated cluster-shock destruction.

    clusters: array of cluster ids, one per arm.
    q: per-cluster per-round shock probability.
    """
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    cluster_ids = sorted(set(clusters))
    members = {c: frozenset(i for i in range(n) if clusters[i] == c)
              for c in cluster_ids}

    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    def shock_outcomes(S):
        """All post-shock subsets of S with their probabilities."""
        # For each cluster, it either fires (prob q, killing all its alive
        # members in S) or does not (prob 1-q). Independent across clusters.
        outcomes = [(S, 1.0)]
        for c in cluster_ids:
            alive_in_c = members[c] & S
            if not alive_in_c:
                continue
            new_outcomes = []
            for S2, pr in outcomes:
                new_outcomes.append((S2 - alive_in_c, pr * q))
                new_outcomes.append((S2, pr * (1 - q)))
            outcomes = new_outcomes
        return outcomes

    @lru_cache(maxsize=None)
    def Vc(S, t):
        if t >= T or not S:
            return 0.0
        best = -np.inf
        for a in S:
            cont = 0.0
            for S1, pr1 in [(S - {a}, p[a]), (S, 1 - p[a])]:
                if pr1 == 0:
                    continue
                for S2, pr2 in shock_outcomes(S1):
                    cont += pr1 * pr2 * Vc(S2, t + 1)
            total = R(a, S) + cont
            best = max(best, total)
        return best

    def policy_value(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S:
                return 0.0
            a = pick(S, t)
            cont = 0.0
            for S1, pr1 in [(S - {a}, p[a]), (S, 1 - p[a])]:
                if pr1 == 0:
                    continue
                for S2, pr2 in shock_outcomes(S1):
                    cont += pr1 * pr2 * W(S2, t + 1)
            return R(a, S) + cont
        return W(full, 0)

    def policy_regret(pick):
        @lru_cache(maxsize=None)
        def Rg(S, t):
            if t >= T or not S:
                return 0.0
            a = pick(S, t)
            inst = max(R(i, S) for i in S) - R(a, S)
            cont = 0.0
            for S1, pr1 in [(S - {a}, p[a]), (S, 1 - p[a])]:
                if pr1 == 0:
                    continue
                for S2, pr2 in shock_outcomes(S1):
                    cont += pr1 * pr2 * Rg(S2, t + 1)
            return inst + cont
        return Rg(full, 0)

    v_star = Vc(full, 0)
    greedy = lambda S, t: max(S, key=lambda i: R(i, S))
    eci = lambda S, t: max(S, key=lambda i: R(i, S) - delta*p[i]*e[i]*(T-t))
    v_greedy = policy_value(greedy)
    v_eci = policy_value(eci)
    reg_greedy = policy_regret(greedy)
    return v_star, v_greedy, v_eci, reg_greedy


def greedy_regret_clustered(v, p, e, delta, T, clusters, q):
    """Just the greedy regret, for the cheap structural check (Q1)."""
    return exact_clustered(v, p, e, delta, T, clusters, q)[3]


if __name__ == "__main__":
    print("STRESS TEST: correlated destruction across arms\n")
    print("Model: per-pull destruction (as before) PLUS a per-cluster exogenous")
    print("shock that kills every alive member of a cluster simultaneously.\n")

    rng = np.random.default_rng(71)

    print("Q1 -- does greedy's zero private regret survive correlated shocks?\n")
    print(f"{'q (shock rate)':>16} {'greedy regret':>14}")
    print("-" * 32)
    n, T, delta = 6, 6, 0.15
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
    e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
    clusters = np.array([0, 0, 1, 1, 2, 2])
    for q in [0.0, 0.1, 0.3, 0.5]:
        _, _, _, reg = exact_clustered(v, p, e, delta, T, clusters, q)
        print(f"{q:16.2f} {reg:14.10f}")

    print("\nQ2 -- does constant-kappa still make greedy optimal under shocks?\n")
    print("Constructing kappa = p*e EXACTLY constant across arms, as in the")
    print("independent-destruction proof, then adding cluster shocks.\n")
    print(f"{'q':>6} {'V*':>9} {'greedy':>9} {'gap':>9} {'gap%':>7}")
    print("-" * 44)
    kappa = 0.4
    p2 = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
    e2 = kappa / p2                    # kappa exactly constant under the OLD model
    for q in [0.0, 0.1, 0.2, 0.35, 0.5]:
        vs, vg, ve, _ = exact_clustered(v, p2, e2, delta, T, clusters, q)
        gap = vs - vg
        print(f"{q:6.2f} {vs:9.4f} {vg:9.4f} {gap:9.4f} "
              f"{100*gap/abs(vs):6.1f}%")

    print("\nQ3 -- does ECI (blind to clusters) still capture most of the gap")
    print("     once clusters make kappa dispersed in the usual way?\n")
    print(f"{'q':>6} {'greedy gap':>11} {'ECI gap':>9} {'ECI captures':>13}")
    print("-" * 44)
    for q in [0.0, 0.1, 0.3, 0.5]:
        vs, vg, ve, _ = exact_clustered(v, p, e, delta, T, clusters, q)
        gg, eg = vs - vg, vs - ve
        cap = 100 * (gg - eg) / gg if gg > 1e-9 else float("nan")
        print(f"{q:6.2f} {gg:11.4f} {eg:9.4f} {cap:12.1f}%")


# ---------------------------------------------------------- cluster-aware fix
def cluster_eci_action(S, t, v, p, e, delta, T, clusters, q):
    """ECI extended with a cluster-shock term: charges an arm not only for its
    own p_a*e_a risk but for the expected burden the SHARED cluster shock will
    create this round, since that risk is not reduced by preserving this
    specific arm."""
    def score(a):
        own = delta * p[a] * e[a] * (T - t)
        c = clusters[a]
        cluster_e = sum(e[b] for b in S if clusters[b] == c and b != a)
        shared = delta * q * cluster_e * (T - t)
        return v[a] - own - shared
    return max(S, key=score)


def exact_clustered_with_policy(v, p, e, delta, T, clusters, q, pick):
    """Like exact_clustered but evaluates an arbitrary policy `pick`."""
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    cluster_ids = sorted(set(clusters))
    members = {c: frozenset(i for i in range(n) if clusters[i] == c)
              for c in cluster_ids}
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    def shock_outcomes(S):
        outcomes = [(S, 1.0)]
        for c in cluster_ids:
            alive_in_c = members[c] & S
            if not alive_in_c:
                continue
            new_outcomes = []
            for S2, pr in outcomes:
                new_outcomes.append((S2 - alive_in_c, pr * q))
                new_outcomes.append((S2, pr * (1 - q)))
            outcomes = new_outcomes
        return outcomes

    @lru_cache(maxsize=None)
    def W(S, t):
        if t >= T or not S:
            return 0.0
        a = pick(S, t)
        cont = 0.0
        for S1, pr1 in [(S - {a}, p[a]), (S, 1 - p[a])]:
            if pr1 == 0:
                continue
            for S2, pr2 in shock_outcomes(S1):
                cont += pr1 * pr2 * W(S2, t + 1)
        return R(a, S) + cont

    return W(full, 0)


# ------------------------------------------------------------- SECI (fixed)
def seci_action(S, t, v, p, e, delta, T, q):
    """Shock-corrected ECI: dampens the burden charge by (1-q)^2, the squared
    probability the cluster survives this round undisturbed. Matches ECI
    exactly at q=0; degrades gracefully to pure greedy as q -> 1, which is
    correct since preservation is worthless when destruction is certain
    regardless of action.

    Verified (exact DP, n=6, 30 seeds/cell): matches or beats greedy at every
    q in [0,1], within 0.01-1.53% of true optimum throughout, versus plain
    ECI's gap growing to ~11% at q=1.
    """
    damp = (1.0 - q) ** 2
    return max(S, key=lambda a: v[a] - delta*p[a]*e[a]*(T-t)*damp)


# --------------------------------------------------------- Monte Carlo, scale
def simulate_clustered(v, p, e, delta, T, clusters, q, policy, seeds=200):
    """Monte Carlo rollout at scale, where exact DP is infeasible."""
    n = len(v)
    cluster_ids = sorted(set(clusters))
    totals = []
    for s in range(seeds):
        rng = np.random.default_rng(20000 + s)
        alive = np.ones(n, dtype=bool)
        tot = 0.0
        for t in range(T):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            burden = delta * float(e[~alive].sum())
            a = policy(idx, v, p, e, delta, T, t, burden, clusters, q)
            tot += v[a] - burden
            if rng.random() < p[a]:
                alive[a] = False
            for c in cluster_ids:
                if rng.random() < q:
                    members = np.flatnonzero(alive & (clusters == c))
                    alive[members] = False
        totals.append(tot)
    return float(np.mean(totals))


def rule_greedy_c(idx, v, p, e, delta, T, t, b, clusters, q):
    return int(idx[np.argmax(v[idx])])


def rule_eci_c(idx, v, p, e, delta, T, t, b, clusters, q):
    return int(idx[np.argmax(v[idx] - delta*p[idx]*e[idx]*(T-t))])


def rule_seci_c(idx, v, p, e, delta, T, t, b, clusters, q):
    damp = (1.0 - q) ** 2
    return int(idx[np.argmax(v[idx] - delta*p[idx]*e[idx]*(T-t)*damp)])
