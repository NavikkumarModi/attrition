"""Agents for the evolving-action-set bandit.

Baselines
  RandomAgent        -- uniform over available arms
  LinUCBAgent        -- standard LinUCB, ignores departure dynamics
  GreedyOracle       -- knows theta*, myopically picks best available arm

Proposed
  IPICAgent          -- information-per-irreversible-cost (v0 heuristic)

NOTE on the oracle: GreedyOracle is a *myopic placeholder*, not the Whittle-index
benchmark specified in the research spec. It knows theta* but does not plan
around departures, so it is neither an upper nor a lower bound on the true
optimal policy. Regret measured against it is diagnostic only.
"""

import numpy as np


class BaseAgent:
    name = "base"

    def reset(self, d):
        self.d = d

    def select(self, arms, env=None):
        raise NotImplementedError

    def update(self, arm, reward):
        pass


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, arms, env=None):
        return arms[self.rng.integers(len(arms))]


class GreedyOracle(BaseAgent):
    """Knows theta*. Myopic: best available arm, no lookahead over departures."""
    name = "oracle-myopic"

    def __init__(self, theta):
        self.theta = theta

    def select(self, arms, env=None):
        return max(arms, key=lambda a: self.theta @ a.x)


class LinUCBAgent(BaseAgent):
    """Standard LinUCB. Departure-blind."""
    name = "linucb"

    def __init__(self, d, alpha=1.0, lam=1.0):
        self.d = d
        self.alpha = alpha
        self.A = lam * np.eye(d)
        self.b = np.zeros(d)
        self._Ainv = np.linalg.inv(self.A)

    @property
    def theta_hat(self):
        return self._Ainv @ self.b

    def _score(self, x):
        th = self.theta_hat
        bonus = self.alpha * np.sqrt(max(x @ self._Ainv @ x, 0.0))
        return float(th @ x + bonus)

    def select(self, arms, env=None):
        return max(arms, key=lambda a: self._score(a.x))

    def update(self, arm, reward):
        x = arm.x
        self.A += np.outer(x, x)
        self.b += reward * x
        self._Ainv = np.linalg.inv(self.A)


class IPICAgent(LinUCBAgent):
    """Information-per-irreversible-cost.

    Standard optimism, minus a penalty for destroying an arm that is both
    valuable and scarce. The penalty is the estimated value the arm would
    still have delivered, weighted by the endogenous departure probability
    and by how replaceable it is (gap to the next-best available arm).

    v0 heuristic. p is assumed KNOWN here -- the identification problem is
    deliberately not addressed in this version so that the cost of *not*
    knowing p can be measured against it later.
    """
    name = "ipic"

    def __init__(self, d, p, alpha=1.0, lam=1.0, kappa=1.0, horizon=2000):
        super().__init__(d, alpha=alpha, lam=lam)
        self.p = p
        self.kappa = kappa
        self.horizon = horizon
        self.t = 0

    def select(self, arms, env=None):
        if len(arms) == 1:
            return arms[0]
        th = self.theta_hat
        vals = np.array([th @ a.x for a in arms])
        order = np.argsort(vals)[::-1]
        best, second = vals[order[0]], vals[order[1]]

        remaining = max(self.horizon - self.t, 1) / self.horizon

        scores = []
        for i, a in enumerate(arms):
            base = self._score(a.x)
            # value forgone if this arm is destroyed: how much better it is
            # than the fallback, scaled by how much horizon is left to use it
            fallback = second if vals[i] >= best - 1e-12 else best
            forgone = max(vals[i] - fallback, 0.0)
            penalty = self.kappa * self.p * forgone * remaining
            scores.append(base - penalty)
        return arms[int(np.argmax(scores))]

    def update(self, arm, reward):
        super().update(arm, reward)
        self.t += 1
