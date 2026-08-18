"""Terminal commitment: irreversible experimentation followed by a one-shot
declaration that fixes the feasible set forever.

This is a different object from the rest of the library. Everywhere else the action
set shrinks as a side effect of pulling arms. Here there is an additional, explicit
terminal action: after a finite budget of irreversible experiments the learner must
*declare* an operating envelope, and that declaration bounds every subsequent
action for the remaining horizon.

The motivating case is CMC process development under ICH Q8. A finite quantity of
API supports a finite number of process runs. Each run either demonstrates that a
parameter setting is viable or produces an out-of-spec batch. At filing, a design
space is declared: operating inside it needs no regulatory approval, operating
outside it triggers a variation procedure. The declared envelope is therefore a
permanent commitment made on incomplete evidence.

THE TENSION, which is what makes this a real problem rather than a wide-envelope
grab: a wider declared envelope is worth more operationally, because process drift
and raw-material variation can be absorbed without a filing. But an envelope can
only be claimed where viability was demonstrated, and demonstrating the edges means
running experiments at the edges -- precisely where failures occur, and where a
failure truncates the claimable region.

Ambition and damage travel together again, but now the damage lands on a terminal
decision rather than on the next reward.
"""

import numpy as np

__all__ = ["TerminalCommitment", "optimal_commitment_policy",
           "expand_outward_policy", "greedy_commitment_policy",
           "edge_first_policy", "evaluate_commitment_policy"]


class TerminalCommitment:
    """Finite experiment budget, then a one-shot envelope declaration.

    Parameters
    ----------
    n_settings : int          discretised process settings, indexed 0..n-1
    budget : int              number of irreversible experiments available
    operate_periods : int     length of the post-declaration operating phase
    centre : int              index of the nominal operating point
    fail_edge : float         out-of-spec probability at the outermost setting
    drift_sd : float          process drift, in setting-index units per period
    seed : int
    """

    def __init__(self, n_settings=9, budget=4, operate_periods=40, centre=4,
                 fail_edge=0.55, drift_sd=1.1, seed=0):
        self.n = n_settings
        self.budget = budget
        self.T2 = operate_periods
        self.centre = centre
        self.drift_sd = drift_sd
        rng = np.random.default_rng(seed)
        idx = np.arange(n_settings)
        # Yield rises monotonically with setting intensity -- hotter, faster,
        # more concentrated processes give more product. Failure probability
        # rises with it, since the same intensity pushes toward the spec limit.
        # This is the corr(v, kappa) > 0 structure: ambition and damage together.
        self.yield_at = 1.0 + 0.30 * (idx / (n_settings - 1))
        dist_up = np.clip(idx - centre, 0, None) / max(n_settings - 1 - centre, 1)
        dist_dn = np.clip(centre - idx, 0, None) / max(centre, 1)
        self.fail_prob = np.clip(0.02 + fail_edge * dist_up ** 1.7
                                 + 0.25 * fail_edge * dist_dn ** 2, 0.0, 0.95)
        self.rng = rng
        self._pmf = None
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.spent = 0
        self.demonstrated = {self.centre}     # nominal is known viable
        self.blocked = set()                  # settings shown out-of-spec
        return self.state()

    def state(self):
        return {"spent": self.spent,
                "demonstrated": sorted(self.demonstrated),
                "blocked": sorted(self.blocked)}

    # ---- experimentation phase --------------------------------------------
    def experiment(self, setting):
        """Run one irreversible experiment. Returns True if viability shown."""
        if self.spent >= self.budget:
            raise RuntimeError("experiment budget exhausted")
        self.spent += 1
        if self.rng.random() < self.fail_prob[setting]:
            self.blocked.add(setting)
            return False
        self.demonstrated.add(setting)
        return True

    # ---- terminal commitment ----------------------------------------------
    def claimable_envelope(self):
        """Widest contiguous interval of demonstrated settings containing centre,
        truncated by any blocked setting."""
        lo = hi = self.centre
        while lo - 1 >= 0 and (lo - 1) in self.demonstrated \
                and (lo - 1) not in self.blocked:
            lo -= 1
        while hi + 1 < self.n and (hi + 1) in self.demonstrated \
                and (hi + 1) not in self.blocked:
            hi += 1
        return lo, hi

    def _drift_pmf(self):
        """Probability the drifted process lands on each setting index.

        Discretised Gaussian drift about the nominal point, clipped to the grid.
        Computed once analytically rather than sampled, so evaluation is exact.
        """
        if getattr(self, "_pmf", None) is not None:
            return self._pmf
        from math import erf, sqrt
        idx = np.arange(self.n)
        cdf = lambda x: 0.5 * (1 + erf((x - self.centre) /
                                       (self.drift_sd * sqrt(2))))
        pmf = np.array([cdf(i + 0.5) - cdf(i - 0.5) for i in idx])
        pmf[0] += cdf(-0.5)                 # mass clipped to the low end
        pmf[-1] += 1.0 - cdf(self.n - 0.5)  # mass clipped to the high end
        self._pmf = pmf / pmf.sum()
        return self._pmf

    def operating_value(self, envelope=None):
        """Exact expected value of the operating phase inside the declared envelope.

        Each period the process drifts. If it lands inside the envelope the yield at
        the realised setting is collected; if outside, operation reverts to nominal
        and a variation cost is incurred.
        """
        lo, hi = self.claimable_envelope() if envelope is None else envelope
        pmf = self._drift_pmf()
        inside = np.zeros(self.n, dtype=bool)
        inside[lo:hi + 1] = True
        per_period = float(
            np.sum(pmf[inside] * self.yield_at[inside])
            + np.sum(pmf[~inside]) * (self.yield_at[self.centre] - 0.35))
        return self.T2 * per_period

    def width(self):
        lo, hi = self.claimable_envelope()
        return hi - lo + 1


# ------------------------------------------------------------------ policies
def greedy_commitment_policy(env):
    """Spend the budget on the settings with the highest immediate yield.

    This is the analogue of greedy: it optimises the quantity visible during the
    experimental phase and ignores what the declaration will need.
    """
    order = np.argsort(env.yield_at)[::-1]
    for s in order:
        if env.spent >= env.budget:
            break
        if int(s) in env.demonstrated or int(s) in env.blocked:
            continue
        env.experiment(int(s))


def edge_first_policy(env):
    """Spend the budget on the outermost settings, to claim the widest envelope."""
    order = np.argsort(-np.abs(np.arange(env.n) - env.centre))
    for s in order:
        if env.spent >= env.budget:
            break
        if int(s) in env.demonstrated or int(s) in env.blocked:
            continue
        env.experiment(int(s))


def expand_outward_policy(env):
    """Expand outward from the centre, one step at a time.

    Contiguity is what makes this the right structure: an envelope is an interval
    containing the nominal point, so demonstrating a distant setting is worthless
    unless everything between is also demonstrated. Expanding outward is provably
    optimal in its STRUCTURE -- see the boundary-only-expansion theorem in the
    long-form appendix -- but the DIRECTION choice below (favouring the side with
    higher expected value of a successful test, rather than simply the cheaper
    side) is a well-tested heuristic, not a proven-optimal rule: exact-DP checks
    show even this improved rule can still miss the true optimum, since the
    correct choice can depend on remaining budget in a way no static index
    captures. This function is deliberately NOT named 'optimal' for that reason.
    """
    while env.spent < env.budget:
        lo, hi = env.claimable_envelope()
        cands = []
        if lo - 1 >= 0 and (lo - 1) not in env.blocked:
            cands.append(lo - 1)
        if hi + 1 < env.n and (hi + 1) not in env.blocked:
            cands.append(hi + 1)
        if not cands:
            break
        # expected value of a successful test at each candidate, rather than
        # simply the cheaper (lower failure-probability) direction -- verified
        # to close most, but not all, of the gap to the true DP optimum
        s = max(cands, key=lambda c: (1 - env.fail_prob[c]) * env.yield_at[c])
        env.experiment(int(s))


# Backward-compatible alias: the old name overclaimed optimality this
# function's direction rule does not have. Kept so existing callers and
# experiment scripts referencing the old name do not break.
optimal_commitment_policy = expand_outward_policy


def evaluate_commitment_policy(policy, seeds=300, **kwargs):
    """Run a policy across seeds; return mean operating value and envelope width."""
    vals, widths, fails = [], [], []
    for s in range(seeds):
        env = TerminalCommitment(seed=s, **kwargs)
        policy(env)
        vals.append(env.operating_value())
        widths.append(env.width())
        fails.append(len(env.blocked))
    return (float(np.mean(vals)), float(np.mean(widths)),
            float(np.mean(fails)))
