"""Policies for bandits with consumable action sets.

All policies share the interface ``select(state) -> arm_index``.

Available:
    Greedy            highest immediate reward; the standard no-regret baseline
    ECI               externality-corrected index (Theorem 2)
    Conservative      ECI with a uniform upper-bound externality; requires no
                      per-arm knowledge (recommended default, see Theorem 3)
    SortByE           exact optimum in the pure-sequencing regime (Theorem 2b)
    Rollout           one step of policy improvement over a base policy
    ThompsonSampling  standard TS baseline
    UCB               standard UCB baseline
"""

import numpy as np

__all__ = ["State", "Greedy", "ECI", "Conservative", "SortByE", "Rollout",
           "ThompsonSampling", "UCB"]


class State:
    """Observable state passed to a policy at each decision point.

    Attributes
    ----------
    available : ndarray of int
        Indices of arms currently available.
    t, horizon : int
        Current round and total horizon.
    v_hat : ndarray
        Current estimate (or known value) of each arm's mean reward.
    p, e, delta : ndarray, ndarray, float
        Death probabilities, externality coefficients, externality scale.
        Policies that do not use these ignore them.
    counts : ndarray of int
        Number of pulls of each arm so far.
    """

    def __init__(self, available, t, horizon, v_hat, p=None, e=None,
                 delta=0.0, counts=None, sigma=0.1, rng=None):
        self.available = np.asarray(available, dtype=int)
        self.t, self.horizon = int(t), int(horizon)
        self.v_hat = np.asarray(v_hat, dtype=float)
        n = len(self.v_hat)
        self.p = np.ones(n) if p is None else np.asarray(p, dtype=float)
        self.e = np.zeros(n) if e is None else np.asarray(e, dtype=float)
        self.delta = float(delta)
        self.counts = np.zeros(n) if counts is None else np.asarray(counts)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def remaining(self):
        return max(self.horizon - self.t, 1)


class Policy:
    name = "policy"

    def select(self, state):
        raise NotImplementedError

    def scores(self, state):
        raise NotImplementedError

    def _argmax(self, state, s):
        return int(state.available[int(np.argmax(s))])


class Greedy(Policy):
    """Pull the highest-value available arm.

    Achieves zero regret against the best-available-arm benchmark by
    construction -- and, by Theorem 4, unbounded value loss.
    """
    name = "greedy"

    def scores(self, state):
        return state.v_hat[state.available]

    def select(self, state):
        return self._argmax(state, self.scores(state))


class ECI(Policy):
    """Externality-corrected index (Theorem 2).

        I(a, t) = v_a - delta * p_a * e_a * (T - t)

    The charge is the expected permanent burden the pull creates, times the
    number of rounds it will be paid over. Requires per-arm ``p`` and ``e``.
    """
    name = "eci"

    def __init__(self, kappa_scale=1.0):
        self.kappa_scale = float(kappa_scale)

    def scores(self, state):
        a = state.available
        kappa = state.p[a] * state.e[a] * self.kappa_scale
        return state.v_hat[a] - state.delta * kappa * state.remaining

    def select(self, state):
        return self._argmax(state, self.scores(state))


class Conservative(Policy):
    """ECI with a uniform upper-bound externality.

    By Theorem 3 the per-arm externalities cannot be estimated from experience:
    the observation and the damage are the same event. This policy therefore
    does not try. It charges every arm a specified upper bound ``e_bound``,
    which in experiments outperforms every learned estimator tested.

    This is the recommended default when the true externalities are unknown.
    """
    name = "conservative"

    def __init__(self, e_bound):
        self.e_bound = float(e_bound)

    def scores(self, state):
        a = state.available
        return (state.v_hat[a]
                - state.delta * state.p[a] * self.e_bound * state.remaining)

    def select(self, state):
        return self._argmax(state, self.scores(state))


class SortByE(Policy):
    """Exact optimum in the pure-sequencing regime (Theorem 2b).

    When every pull consumes (p = 1) and the horizon equals the pool size, the
    sum of arm values is a constant, so only the burden schedule matters and the
    optimal order is ascending in ``e``. Values are irrelevant.
    """
    name = "sort-by-e"

    def scores(self, state):
        return -state.e[state.available]

    def select(self, state):
        return self._argmax(state, self.scores(state))


class ThompsonSampling(Policy):
    """Gaussian Thompson Sampling. Departure- and externality-blind."""
    name = "thompson"

    def scores(self, state):
        a = state.available
        sd = state.sigma / np.sqrt(np.maximum(state.counts[a], 1))
        return state.v_hat[a] + state.rng.normal(0, 1, len(a)) * sd

    def select(self, state):
        return self._argmax(state, self.scores(state))


class UCB(Policy):
    """UCB1. Departure- and externality-blind."""
    name = "ucb"

    def __init__(self, c=1.0):
        self.c = float(c)

    def scores(self, state):
        a = state.available
        t = max(state.t, 1)
        return state.v_hat[a] + self.c * np.sqrt(
            2 * np.log(t) / np.maximum(state.counts[a], 1))

    def select(self, state):
        return self._argmax(state, self.scores(state))


class Rollout(Policy):
    """One step of policy improvement over a base policy.

    For each of the top-``width`` candidates by base score, simulate ``depth``
    forward episodes under the base policy and pick the best average return.
    Reaches 99.5-99.9% of the exact optimum where exact DP is computable.

    Cost is ``width * depth * horizon`` base-policy steps per decision.
    """
    name = "rollout"

    def __init__(self, base=None, width=4, depth=12, simulator=None):
        self.base = base if base is not None else ECI()
        self.width = int(width)
        self.depth = int(depth)
        self.simulator = simulator

    def select(self, state):
        if self.simulator is None:
            return self.base.select(state)
        a = state.available
        base_scores = self.base.scores(state)
        order = np.argsort(base_scores)[::-1][:self.width]
        best, best_arm = -np.inf, int(a[order[0]])
        for j in order:
            arm = int(a[j])
            q = np.mean([self.simulator(state, arm, self.base)
                         for _ in range(self.depth)])
            if q > best:
                best, best_arm = q, arm
        return best_arm
