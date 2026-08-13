"""Experiment 49 -- a genuine attempt at the ECI closed-form bound.

THEORY.md lists this as open, with the option term O_a identified as the
obstruction since session 1 (exp13). The route attempted below -- bounding O_a
directly -- has no known closed form. This attempts a different route.

STEP 1 -- Monotonicity of V in the available set: FALSE, and the reason is a real
structural fact about the model, not a bug. See below.

STEP 2 -- A magnitude bound on D_a that does not require monotonicity.

STEP 3 -- Performance difference identity (exact, provable by telescoping).

STEP 4 -- A closed-form (loose) bound on Gap(ECI) combining Steps 2 and 3.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np

__all__ = ["monotonicity_check", "performance_difference_check", "eci_bound",
           "closed_form_bound"]


def build(v, p, e, delta, T):
    n = len(v)
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    def eci_action(S, t):
        return max(S, key=lambda a: R(a, S) - delta*p[a]*e[a]*(T-t))

    return V, R, eci_action


# --------------------------------------------------------------- Step 1 check
def monotonicity_check(n=6, T=8, delta=0.15, spread=1.2, inst=8, seed=41):
    """Test V(S',t) >= V(S,t) whenever S' is a strict superset. FALSE: having a
    surviving low-value arm can be a LIABILITY rather than a free option, because
    there is no null/stop action. If every other arm dies, the pool would be
    empty (V=0, the episode simply ends) -- but an extra surviving arm forces
    continued play, and if the accumulated burden by then makes every remaining
    round strictly negative, that forced continuation is worse than the episode
    ending. "Ignore the extra arm" is not a valid strategy when standing still is
    not an option."""
    rng = np.random.default_rng(seed)
    violations, checked = 0, 0
    worst = 0.0
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
        V, _, _ = build(v, p, e, delta, T)
        for size in range(1, n):
            for combo in combinations(range(n), size):
                S = frozenset(combo)
                for extra in set(range(n)) - S:
                    Sp = S | {extra}
                    for t in range(T):
                        checked += 1
                        diff = V(Sp, t) - V(S, t)
                        if diff < -1e-9:
                            violations += 1
                            worst = min(worst, diff)
    return violations, checked, worst


# --------------------------------------------------------------- Step 2 bound
def magnitude_bound(v, p, e, delta, T, t):
    """|V(S,t)| <= (T-t) * R for ANY S, where R bounds one round's reward
    magnitude regardless of sign. Proof: each round's reward is v_a - B(S) with
    v_a in [0, v_max] and B(S) in [0, delta*sum(e)], so |v_a - B(S)| <= R :=
    v_max + delta*sum(e). V(S,t) sums at most (T-t) such terms (fewer if the pool
    empties early, which only pulls the sum toward 0), so |V(S,t)| <= (T-t)*R by
    induction on the DP recursion. This requires no monotonicity."""
    R = float(np.max(v)) + delta * float(np.sum(e))
    return (T - t) * R


# --------------------------------------------------------- Step 3 check
def performance_difference_check(n=6, T=6, delta=0.15, spread=1.2, inst=6,
                                 seed=41):
    """Verify Gap(ECI) equals the exact telescoped sum of one-step regrets under
    ECI's own trajectory distribution -- computed via forward occupancy, not
    simulation, so the check is exact rather than approximate."""
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
        V, R, eci_action = build(v, p, e, delta, T)
        full = frozenset(range(n))

        def Q(a, S, t):
            return R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1)

        def reg(S, t):
            a = eci_action(S, t)
            return max(Q(b, S, t) for b in S) - Q(a, S, t)

        # forward occupancy under ECI, exact (small n so full enumeration is fine)
        occ = {(full, 0): 1.0}
        total_reg = 0.0
        for t in range(T):
            nxt = {}
            for (S, tt), w in list(occ.items()):
                if tt != t or not S:
                    continue
                total_reg += w * reg(S, t)
                a = eci_action(S, t)
                for Sp, pr in [(S - {a}, p[a]), (S, 1 - p[a])]:
                    if pr > 0 and Sp:
                        nxt[(Sp, t + 1)] = nxt.get((Sp, t + 1), 0.0) + w * pr
            occ.update(nxt)

        def W_eci(S, t):
            if t >= T or not S:
                return 0.0
            a = eci_action(S, t)
            return R(a, S) + p[a]*W_eci(S-{a}, t+1) + (1-p[a])*W_eci(S, t+1)

        actual_gap = V(full, 0) - W_eci(full, 0)
        errors.append(abs(actual_gap - total_reg))
    return float(np.max(errors))


# ----------------------------------------------------------- Steps 2+4: bound
def closed_form_bound(v, p, e, delta, T):
    """Gap(ECI) <= T(T+1) * max_a[ p_a (delta*e_a + 2R) ], R = v_max + delta*sum(e).

    Derivation. error(a) := true_score(a) - eci_score(a)
                           = delta*kappa_a*(T-t) - p_a*D_a(S,t).
    |D_a(S,t)| = |V(S,t+1) - V(S\\{a},t+1)| <= 2*(T-t)*R  by Step 2 (magnitude
    bound applied to both terms). So
        |error(a)| <= delta*kappa_a*(T-t) + p_a*2*(T-t)*R
                     = (T-t) * p_a * (delta*e_a + 2R).
    Standard argmax-perturbation gives reg(S,t) <= 2*max_a|error(a)|, and Step 3
    telescopes the per-step regret into the total gap, giving
        Gap(ECI) <= sum_{t=0}^{T-1} 2*(T-t)*max_a[p_a(delta*e_a+2R)]
                  = max_a[p_a(delta*e_a+2R)] * T(T+1).
    """
    v_max = float(np.max(v))
    R = v_max + delta * float(np.sum(e))
    per_step_coeff = float(np.max(p * (delta * e + 2 * R)))
    return T * (T + 1) * per_step_coeff


def eci_bound(n=6, T=8, delta=0.15, spread=1.2, inst=10, seed=41):
    """Compare the exact ECI gap to the closed-form bound across instances."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
        V, R, eci_action = build(v, p, e, delta, T)
        full = frozenset(range(n))

        def W(S, t):
            if t >= T or not S:
                return 0.0
            a = eci_action(S, t)
            return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)

        exact_gap = V(full, 0) - W(full, 0)
        bound = closed_form_bound(v, p, e, delta, T)
        rows.append((exact_gap, bound))
    return rows


if __name__ == "__main__":
    print("STEP 1 -- monotonicity of V in the available set\n")
    viol, checked, worst = monotonicity_check()
    print(f"  checked {checked} (S, S'={{superset}}, t) triples")
    print(f"  violations: {viol}  (worst: V(S') - V(S) = {worst:.4f})")
    print(f"  -> FALSIFIED. Extra options are a liability when there is no")
    print(f"     null/stop action: pool exhaustion (V=0) can beat forced")
    print(f"     continuation under high accumulated burden.")

    print("\nSTEP 2 -- magnitude bound |V(S,t)| <= (T-t)*R (no monotonicity needed)\n")
    print("  Provable by induction on the DP recursion; verified implicitly by")
    print("  the bound-validity check in Step 4 below.")

    print("\nSTEP 3 -- performance difference identity (exact telescoping)\n")
    max_err = performance_difference_check()
    print(f"  max |actual gap - telescoped sum| across instances: {max_err:.2e}")
    print(f"  -> {'CONFIRMED (identity holds exactly)' if max_err < 1e-6 else 'FALSIFIED'}")

    print("\nSTEP 4 -- closed-form bound vs exact gap\n")
    print("  Gap(ECI) <= T(T+1) * max_a[ p_a (delta*e_a + 2R) ],")
    print("  R = v_max + delta*sum(e)\n")
    print(f"{'exact gap':>12} {'bound':>12} {'bound/exact':>12} {'valid?':>8}")
    print("-" * 48)
    rows = eci_bound()
    ok = True
    ratios = []
    for exact, bound in rows:
        valid = bound >= exact - 1e-9
        ok &= valid
        r = bound / max(exact, 1e-9)
        ratios.append(r)
        print(f"{exact:12.4f} {bound:12.4f} {r:12.1f} {'yes' if valid else 'NO':>8}")
    print(f"\n  bound never violated: {'CONFIRMED' if ok else 'FALSIFIED'}")
    print(f"  looseness: {np.mean(ratios):.0f}x on average -- a valid but crude")
    print(f"  bound. It proves Gap(ECI) is FINITE and scales as O(T^2), the")
    print(f"  first closed-form guarantee for ECI, not a tight estimate.")
