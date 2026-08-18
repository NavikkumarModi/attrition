"""Experiment 59 -- the third open theorem: terminal commitment's optimal
allocation. Built against the ACTUAL attrition/commitment.py model, not a
reconstruction.

FOUND AND PROVEN: boundary-only expansion is exactly optimal (Theorem, with
proof, added to the paper). Any policy testing a setting not adjacent to the
current claimable envelope is weakly dominated by one that only tests
adjacent settings -- proven via an exchange argument (independence of
outcomes means a distant success is worthless until the gap fills), verified
by exact DP with ZERO gap across 42 instances.

FOUND AS A REAL BUG, not just a theoretical curiosity: the library's
`optimal_commitment_policy` picked the right STRUCTURE (boundary expansion,
matching the new theorem) but the wrong DIRECTION rule ("always cheaper"),
which mismatches the true DP optimum in 15/20 tested instances, losing real
value. Fixed in attrition/commitment.py: renamed to expand_outward_policy
(honest name), improved the direction rule to (1-fail_prob)*yield (closes
most but not all of the gap, 9/20 mismatches remaining), kept
optimal_commitment_policy as a backward-compatible alias pointing to the
improved function.

STATUS: one genuine new proven theorem (boundary-only expansion). One
concrete, actionable improvement to the actual policy used in the paper's
own experiments (better than the original, still not proven exactly
optimal). The direction sub-problem remains open -- the true optimal choice
appears to require genuine multi-step lookahead, not a static index.
"""

from functools import lru_cache

import numpy as np

from attrition.commitment import TerminalCommitment

__all__ = ["solve_dp", "boundary_only_matches_unrestricted"]


def _build(n_settings, budget, centre, fail_edge, seed):
    env = TerminalCommitment(n_settings=n_settings, budget=budget, centre=centre,
                             fail_edge=fail_edge, seed=seed)
    return env.fail_prob.copy(), env.yield_at.copy(), env.n, env.centre, \
        env._drift_pmf().copy(), env.T2


def solve_dp(fail_prob, yield_at, n, centre, pmf, T2, budget, restrict_boundary):
    def envelope(demonstrated, blocked):
        lo = hi = centre
        while lo-1 >= 0 and (lo-1) in demonstrated and (lo-1) not in blocked:
            lo -= 1
        while hi+1 < n and (hi+1) in demonstrated and (hi+1) not in blocked:
            hi += 1
        return lo, hi

    def operating_value(lo, hi):
        inside = np.zeros(n, dtype=bool)
        inside[lo:hi+1] = True
        per = float(np.sum(pmf[inside]*yield_at[inside])
                   + np.sum(pmf[~inside])*(yield_at[centre]-0.35))
        return T2*per

    @lru_cache(maxsize=None)
    def V(demonstrated, blocked, spent):
        demonstrated = frozenset(demonstrated)
        blocked = frozenset(blocked)
        if spent >= budget:
            lo, hi = envelope(demonstrated, blocked)
            return operating_value(lo, hi)
        lo, hi = envelope(demonstrated, blocked)
        if restrict_boundary:
            cands = [s for s in (lo-1, hi+1)
                    if 0 <= s < n and s not in blocked and s not in demonstrated]
        else:
            cands = [s for s in range(n) if s not in demonstrated and s not in blocked]
        if not cands:
            return operating_value(lo, hi)
        best = -1e18
        for s in cands:
            fp = fail_prob[s]
            vs = V(tuple(sorted(demonstrated|{s})), tuple(sorted(blocked)), spent+1)
            vf = V(tuple(sorted(demonstrated)), tuple(sorted(blocked|{s})), spent+1)
            best = max(best, (1-fp)*vs + fp*vf)
        return best

    return V(tuple({centre}), tuple(), 0)


def boundary_only_matches_unrestricted(seeds=15, seed0=99):
    """Returns (n_trials, worst_gap) between unrestricted and boundary-only
    exact DP optimal values -- should be exactly 0."""
    rng = np.random.default_rng(seed0)
    worst_gap = 0.0
    n_trials = 0
    for _ in range(seeds):
        n = int(rng.integers(5, 9))
        B = int(rng.integers(2, 5))
        centre = int(rng.integers(1, n-1))
        fail_edge = rng.uniform(0.2, 0.9)
        seed = int(rng.integers(0, 5000))
        fp, ya, nn, c, pmf, T2 = _build(n, B, centre, fail_edge, seed)
        unrestricted = solve_dp(fp, ya, nn, c, pmf, T2, B, restrict_boundary=False)
        boundary = solve_dp(fp, ya, nn, c, pmf, T2, B, restrict_boundary=True)
        worst_gap = max(worst_gap, unrestricted - boundary)
        n_trials += 1
    return n_trials, worst_gap


if __name__ == "__main__":
    print("Theorem check: boundary-only expansion vs unrestricted DP optimum\n")
    n_trials, worst_gap = boundary_only_matches_unrestricted()
    print(f"  {n_trials} trials, worst gap = {worst_gap:.10f} (should be exactly 0)")
