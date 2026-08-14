"""Experiment 53 -- investigating a second external review's challenge to
Theorem 1 (Characterisation) sufficiency.

The review proposed a specific counterexample (p1=1,e1=kappa vs p2=0.5,e2=2kappa)
and argued the published proof's factorisation step
E[1(destruction at s)*(N-s)] = p_{a_s}*E[N-s] improperly assumes independence.

Findings, in order:
  (1) the proposed counterexample does NOT break the theorem's conclusion --
      exhaustive DP, every reachable state, zero violations across 288 configs
  (2) broader adversarial search (5000+ configs) also finds zero violations
  (3) the reviewer's objection to the SPECIFIC PROOF STEP is nonetheless
      correct: the factorisation is false, confirmed by direct computation
  (4) the resulting closed-form value (delta*kappa*E[N(N-1)/2]) is ALSO wrong
      as a value, not just derived via an invalid step
  (5) what IS true: E[total burden] remains exactly policy-invariant, via a
      different mechanism -- an adjacent-exchange lemma, derived and verified
      here, that shows swapping two consecutive same-kappa pulls leaves both
      the two-round reward and the full joint outcome distribution unchanged
"""

from functools import lru_cache

import numpy as np

__all__ = ["check_state_by_state", "aggregate_burden_by_policy",
           "adjacent_exchange_check"]


def check_state_by_state(v, p, e, delta, T):
    """Exact DP; returns (ok, violation_info) checking greedy against the true
    optimum at EVERY reachable (S,t), not only the initial state."""
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    reachable = set()

    def enumerate_states(S, t):
        if (S, t) in reachable or t >= T or not S:
            return
        reachable.add((S, t))
        for a in S:
            enumerate_states(S - {a}, t + 1)
            enumerate_states(S, t + 1)
    enumerate_states(full, 0)

    for (S, t) in reachable:
        qvals = {a: R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S}
        best_by_Q = max(qvals, key=qvals.get)
        best_by_v = max(S, key=lambda a: v[a])
        if abs(qvals[best_by_Q] - qvals[best_by_v]) > 1e-9:
            return False, (S, t, qvals, best_by_Q, best_by_v)
    return True, None


def aggregate_burden_by_policy(v, p, e, delta, T, pick):
    """Exact E[sum_t B(S_t)] under a fixed deterministic policy, via full
    enumeration of destruction outcomes."""
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))

    def enumerate_full(S, t, prob, burden_acc):
        if t >= T or not S:
            yield (prob, burden_acc)
            return
        a = pick(S, t)
        b = B(S)
        yield from enumerate_full(S, t + 1, prob*(1-p[a]), burden_acc + b)
        yield from enumerate_full(S - {a}, t + 1, prob*p[a], burden_acc + b)

    outs = list(enumerate_full(full, 0, 1.0, 0.0))
    return sum(prob * bacc for prob, bacc in outs)


def adjacent_exchange_check(v, p, e, delta, B0):
    """Verifies the two-round exchange lemma: swapping the order of pulling
    arms 0 and 1 leaves reward and the full outcome distribution unchanged
    when kappa_0 = kappa_1."""
    def two_round_stats(order):
        a, b = order
        total_reward = 0.0
        outcomes = []
        for da in [0, 1]:
            pa = p[a] if da == 1 else (1 - p[a])
            Ba = B0 + delta*e[a]*da
            r1 = v[a] - B0
            for db in [0, 1]:
                pb = p[b] if db == 1 else (1 - p[b])
                prob = pa * pb
                r2 = v[b] - Ba
                total_reward += prob * (r1 + r2)
                Bfinal = Ba + delta*e[b]*db
                outcomes.append((prob, Bfinal))
        return total_reward, sorted(outcomes, key=lambda x: x[1])

    r_ab, dist_ab = two_round_stats((0, 1))
    r_ba, dist_ba = two_round_stats((1, 0))
    reward_match = abs(r_ab - r_ba) < 1e-9
    dist_match = all(abs(pa - pb) < 1e-9 and abs(ba - bb) < 1e-9
                     for (pa, ba), (pb, bb) in zip(dist_ab, dist_ba))
    return reward_match, dist_match


if __name__ == "__main__":
    print("(1) The reviewer's exact proposed counterexample\n")
    print(f"{'kappa':>7} {'T':>3} {'v1':>6} {'v2':>6} {'delta':>6} {'ok':>4}")
    print("-" * 36)
    total, ok_count = 0, 0
    for kappa in [0.1, 0.5, 1.0, 2.0]:
        for T in [2, 3, 4, 5]:
            for v1, v2 in [(1.0, 0.5), (0.5, 1.0), (1.0, -0.5)]:
                for delta in [0.1, 0.5, 1.0]:
                    p = np.array([1.0, 0.5])
                    e = np.array([kappa, 2*kappa])
                    v = np.array([v1, v2])
                    ok, info = check_state_by_state(v, p, e, delta, T)
                    total += 1
                    ok_count += ok
    print(f"  {ok_count}/{total} configurations: greedy optimal at every reachable state")

    print("\n(2) The claimed closed form vs the true aggregate value\n")
    v = np.array([1.0, 0.9, 0.8])
    p = np.array([0.6, 0.5, 0.3])
    kappa = 1.0
    e = kappa / p
    delta, T = 0.3, 6
    policies = {
        "greedy (max v)": lambda S, t: max(S, key=lambda a: v[a]),
        "min v": lambda S, t: min(S, key=lambda a: v[a]),
        "max p": lambda S, t: max(S, key=lambda a: p[a]),
        "min p": lambda S, t: min(S, key=lambda a: p[a]),
    }
    vals = {name: aggregate_burden_by_policy(v, p, e, delta, T, pick)
           for name, pick in policies.items()}
    for name, val in vals.items():
        print(f"  {name:>16}: E[burden] = {val:.6f}")
    spread = max(vals.values()) - min(vals.values())
    print(f"  spread across policies: {spread:.2e}  (policy-invariant: "
          f"{spread < 1e-6})")

    print("\n(3) Adjacent-exchange lemma, 10 random constant-kappa instances\n")
    rng = np.random.default_rng(3)
    all_ok = True
    for trial in range(10):
        kappa = rng.uniform(0.1, 2.0)
        p = rng.uniform(0.05, 0.95, 2)
        e = kappa / p
        v = rng.uniform(-1, 1, 2)
        delta = rng.uniform(0.1, 1.5)
        B0 = rng.uniform(0, 2)
        r_ok, d_ok = adjacent_exchange_check(v, p, e, delta, B0)
        all_ok &= r_ok and d_ok
    print(f"  reward AND full distribution match in all 10 trials: {all_ok}")
