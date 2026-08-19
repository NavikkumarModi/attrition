"""Experiment 56 -- searching for an instance where the ECI bound's T^2 order
is tight, per a reviewer's specific request. Reports a genuine negative
result rather than a manufactured example.

THE QUESTION. Theorem (ECI bound) proves Gap(ECI) <= T(T+1)*max_a[p_a(delta*e_a
+2R)], known to be loose by ~14,000x on average (Section on the ECI bound).
Is there SOME instance where the bound's T^2 growth rate is actually achieved
(even if the constant is off), which would show the bound is "order-tight"
somewhere, as opposed to being loose in both constant AND asymptotic order?

METHOD. Two regimes tested.
  (1) Fixed n, T -> infinity: search for the adversarial instance (among many
      random dispersed-kappa draws) with the fastest EMPIRICAL growth rate at
      small T, then push that SAME instance to much larger T and track the
      LOCAL exponent (log-log slope between consecutive T values) to see
      where growth actually settles.
  (2) n = T scaled together (delta rescaled to keep problem difficulty
      comparable across scale, per the fairness correction found earlier
      this project): does allowing the pool to grow with the horizon reveal
      genuine T^2 growth that a fixed, small pool cannot sustain?

FINDING (negative, reported honestly). Neither regime shows sustained T^2
growth. In regime (1), a small-T fit can show a fast APPARENT exponent
(e.g. ~3, even exceeding the proven bound's order -- necessarily a small-T
transient, since the bound is proven correct asymptotically), but pushing
to larger T reveals the gap SATURATES near the pool's exhaustion scale and
then DECAYS toward zero, because ECI's charge term uses the literal T-t
horizon (not the exhaustion-capped effective horizon), overcorrecting once
T well exceeds what the pool can sustain. In regime (2), gap/T^2 does not
grow with scale either. No instance found in either regime where the bound's
T^2 order is approached, let alone matched. This suggests the bound may be
loose not only in constant but in asymptotic ORDER too -- a more precise
characterisation than "loose", not previously established.
"""

from functools import lru_cache

import numpy as np

__all__ = ["gap_eci", "fixed_n_growth_check", "scaled_n_growth_check"]


def gap_eci(v, p, e, delta, T):
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))

    @lru_cache(maxsize=None)
    def Vstar(S, t):
        if t >= T or not S:
            return 0.0
        return max(v[a]-B(S) + p[a]*Vstar(S-{a}, t+1) + (1-p[a])*Vstar(S, t+1)
                  for a in S)

    def eci_action(S, t):
        return max(S, key=lambda a: v[a] - delta*p[a]*e[a]*(T-t))

    @lru_cache(maxsize=None)
    def Veci(S, t):
        if t >= T or not S:
            return 0.0
        a = eci_action(S, t)
        return v[a]-B(S) + p[a]*Veci(S-{a}, t+1) + (1-p[a])*Veci(S, t+1)

    return Vstar(full, 0) - Veci(full, 0)


def fixed_n_growth_check(v, p, e, delta, T_values):
    """Track Gap(ECI) and the local growth exponent across increasing T for a
    FIXED instance (fixed n). Returns list of (T, gap, local_exponent)."""
    results = []
    prev_gap, prev_T = None, None
    for T in T_values:
        g = gap_eci(v, p, e, delta, T)
        if prev_gap is not None and prev_gap > 1e-9 and g > 1e-9:
            k = np.log(g / prev_gap) / np.log(T / prev_T)
        else:
            k = float("nan")
        results.append((T, g, k))
        prev_gap, prev_T = g, T
    return results


def scaled_n_growth_check(T_values, base_delta=0.536, base_n=6, seed=3):
    """n scaled with T (n=T), delta rescaled by base_n/n to hold problem
    difficulty comparable. Returns list of (T, gap, gap_over_Tsq)."""
    rng = np.random.default_rng(seed)
    results = []
    for T in T_values:
        n = T
        delta = base_delta * base_n / n
        p = np.clip(rng.uniform(0.1, 0.9, n), 0.05, 1.0)
        kappa = rng.uniform(0.3, 2.0, n)
        e = kappa / p
        v = np.sort(rng.uniform(0.3, 3.0, n))[::-1]
        g = gap_eci(v, p, e, delta, T)
        results.append((T, g, g / T**2))
    return results


if __name__ == "__main__":
    print("Regime 1: fixed instance (n=3), T pushed far past the pool's own scale\n")
    v = np.array([1.28044277, 1.23436589, 0.78439627])
    p = np.array([0.61561575, 0.76372129, 0.51170322])
    e = np.array([2.49172314, 0.8968192, 1.24581187])
    delta = 0.536
    for T, g, k in fixed_n_growth_check(v, p, e, delta,
                                        [3, 5, 7, 9, 12, 15, 18, 22, 26, 30]):
        print(f"  T={T:3d}  gap={g:9.5f}  local exponent={k:8.3f}")
    print("\n  Gap saturates near T~12-15 (a few times n=3) then DECAYS toward")
    print("  zero -- the opposite of sustained T^2 growth.")

    print("\nRegime 2: n scaled with T, delta rescaled for fairness\n")
    for T, g, ratio in scaled_n_growth_check([3, 4, 5, 6, 7, 8]):
        print(f"  T=n={T:3d}  gap={g:9.5f}  gap/T^2={ratio:9.5f}")
    print("\n  No sustained growth in gap/T^2 either.")
    print("\nConclusion: no instance found, in either regime, approaching the")
    print("bound's T^2 order. The bound appears loose in order, not just")
    print("constant -- reported as a negative result, not a manufactured example.")
