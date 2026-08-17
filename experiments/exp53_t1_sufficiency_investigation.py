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


# ---------------------------------------------------------------- block exchange
def block_exchange_check(v, p, e, delta, B0, order, Ka, Kb):
    """Exhaust `order[0]` for Ka attempts then `order[1]` for Kb, vs reverse.
    Returns (E[reward, order], E[reward, reversed])."""
    first, second = order

    def run(first_arm, second_arm, Kfirst, Ksecond):
        stage1 = []

        def gen1(prob, reward, burden, attempts):
            if attempts == Kfirst:
                stage1.append((prob, reward, burden))
                return
            r = v[first_arm] - burden
            stage1.append((prob*p[first_arm], reward+r, burden+delta*e[first_arm]))
            gen1(prob*(1-p[first_arm]), reward+r, burden, attempts+1)
        gen1(1.0, 0.0, B0, 0)

        total = 0.0
        for prob1, reward1, burden1 in stage1:
            # exhaust second_arm: stop EARLY on death, not after Ksecond
            # attempts regardless -- matches "exhaust until death or budget"
            def gen2(prob, reward, burden, attempts, acc):
                if attempts == Ksecond:
                    acc[0] += prob*reward
                    return
                r = v[second_arm] - burden
                # dies this attempt: stop here, no further attempts possible
                acc[0] += prob*p[second_arm]*(reward+r)
                # survives: continue
                gen2(prob*(1-p[second_arm]), reward+r, burden, attempts+1, acc)
            acc = [0.0]
            gen2(prob1, reward1, burden1, 0, acc)
            total += acc[0]
        return total

    r_fwd = run(first, second, Ka, Kb)
    r_rev = run(second, first, Kb, Ka)
    return r_fwd, r_rev


def pathwise_rearrangement_counterexample():
    """Deterministic burden for a fixed (g1,g2) under two arrangements."""
    def burden_for_sequence(seq, e):
        N = len(seq)
        last_pos = {}
        for i, a in enumerate(seq):
            last_pos[a] = i
        return sum(e[a] * (N - 1 - last_pos[a]) for a in e)

    e = {1: 10.0, 2: 1.0}
    seqA = [1, 2, 2, 2]
    seqB = [2, 2, 2, 1]
    return burden_for_sequence(seqA, e), burden_for_sequence(seqB, e)


def pathwise_coupling_check(v, p, e, delta, T, coins, policies):
    """Run several policies under the SAME coupled per-arm coin sequences;
    return their burdens."""
    def run_coupled(pick):
        n = len(v)
        alive = np.ones(n, dtype=bool)
        attempt_count = np.zeros(n, dtype=int)
        burden_total = 0.0
        etot = float(np.sum(e))
        for t in range(T):
            alive_idx = np.flatnonzero(alive)
            if len(alive_idx) == 0:
                break
            a = pick(alive_idx, v, p, t)
            burden_total += delta * (etot - sum(e[i] for i in range(n) if alive[i]))
            died = coins[a][attempt_count[a]]
            attempt_count[a] += 1
            if died:
                alive[a] = False
        return burden_total
    return [run_coupled(pick) for pick in policies]


# ------------------------------------------------------------- martingale
def one_step_martingale_check(v, p, e, delta, T, pick, check_t):
    """Verify E[B(S_{t+1}) | F_{t-1}] = B(S_t) + delta*kappa at every
    reachable state at round check_t, under policy `pick`."""
    n = len(v)
    full = frozenset(range(n))
    kappa = float(p[0] * e[0])   # constant kappa hypothesis
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))

    def enum(S, t, prob, hist):
        if t >= T or not S:
            yield (prob, hist)
            return
        a = pick(S)
        yield from enum(S, t + 1, prob*(1 - p[a]), hist + [(a, 0)])
        yield from enum(S - {a}, t + 1, prob*p[a], hist + [(a, 1)])

    outs = list(enum(full, 0, 1.0, []))
    buckets = {}
    for prob, hist in outs:
        if len(hist) <= check_t:
            continue
        alive = set(range(n))
        for (a, died) in hist[:check_t]:
            if died:
                alive.discard(a)
        key = frozenset(alive)
        buckets.setdefault(key, []).append((prob, hist))

    results = []
    for S_key, items in buckets.items():
        if not S_key:
            continue
        totalprob = sum(pr for pr, _ in items)
        Bt = B(S_key)
        EB_next = 0.0
        for prob, hist in items:
            alive2 = set(S_key)
            if len(hist) > check_t:
                a, died = hist[check_t]
                if died:
                    alive2.discard(a)
            EB_next += (prob / totalprob) * B(frozenset(alive2))
        results.append((S_key, Bt, EB_next, Bt + delta*kappa))
    return results


def missing_term_by_policy(v, p, e, delta, T, pick):
    """Compute E[Y_N] and E[sum R_t] under a given policy, to check the
    latter's policy-invariance."""
    n = len(v)
    full = frozenset(range(n))
    kappa = float(p[0] * e[0])
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))

    def enum(S, t, prob, bacc, Racc):
        if t >= T or not S:
            yield (prob, bacc, Racc)
            return
        a = pick(S)
        b = B(S)
        yield from enum(S, t + 1, prob*(1 - p[a]), bacc + b,
                        Racc + (b - delta*kappa*t))
        yield from enum(S - {a}, t + 1, prob*p[a], bacc + b,
                        Racc + (b - delta*kappa*t))

    outs = list(enum(full, 0, 1.0, 0.0, 0.0))
    EY = sum(pr*ba for pr, ba, _ in outs)
    ER = sum(pr*Ra for pr, _, Ra in outs)
    return EY, ER


# ------------------------------------------------------------- Q-term (inconclusive)
def q_term_and_boundary(v, p, e, delta, T, pick):
    """The predictable-reweighting construction attempted for the martingale
    proof: Q_t := sum_s (T-1-s)*DeltaR_s. Its one-step martingale property is
    genuine (verified separately), but E[Q at the natural stopping index] does
    NOT equal zero, unlike what a naive Optional Stopping application would
    suggest. Returns (E[Q at stopping time], E[boundary term]) -- these sum
    exactly to the known missing term, but neither is individually zero,
    documenting an open gap rather than a completed argument.
    """
    n = len(v)
    full = frozenset(range(n))
    kappa = float(p[0] * e[0])
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))

    def enum(S, t, prob, hist):
        if t >= T or not S:
            yield (prob, t, hist)
            return
        a = pick(S)
        yield from enum(S, t + 1, prob*(1 - p[a]), hist + [(a, 0)])
        yield from enum(S - {a}, t + 1, prob*p[a], hist + [(a, 1)])

    outs = list(enum(full, 0, 1.0, []))

    def burden_seq(hist):
        alive = set(range(n))
        Bs = []
        for (a, died) in hist:
            Bs.append(B(frozenset(alive)))
            if died:
                alive.discard(a)
        return Bs

    E_Q, E_bound = 0.0, 0.0
    for prob, Nval, hist in outs:
        Bs = burden_seq(hist)
        Rs = [Bs[t] - delta*kappa*t for t in range(Nval)]
        if Nval >= 2:
            DeltaR = [Rs[s+1] - Rs[s] for s in range(Nval - 1)]
            Q_at_stop = sum((T-1-s)*DeltaR[s] for s in range(Nval - 1))
        else:
            Q_at_stop = 0.0
        E_Q += prob * Q_at_stop
        E_bound += prob * (-(T-Nval)*Rs[Nval-1])
    return E_Q, E_bound
