"""Evolving-action-set bandit environment.

Arms carry features in R^d. Reward is linear in features. Arms depart by two
mechanisms whose *cause is not revealed to the learner*:

  endogenous  -- a pulled arm departs with probability p
  exogenous   -- any available arm departs with probability mu each round

New arms arrive at Poisson rate lam. The learner observes departures but not
which mechanism fired: this is the latent-cause structure the project studies.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Arm:
    idx: int
    x: np.ndarray
    born: int


@dataclass
class EvolvingBanditEnv:
    d: int = 5
    n_init: int = 50
    p: float = 0.0            # endogenous departure prob (irreversibility)
    mu: float = 0.0           # exogenous per-round departure prob (churn)
    lam: float = 0.0          # expected arrivals per round
    sigma: float = 0.1        # reward noise
    horizon: int = 2000
    seed: int = 0

    theta: np.ndarray = field(init=False)
    arms: dict = field(init=False)
    t: int = field(init=False, default=0)
    _next_idx: int = field(init=False, default=0)
    # ground-truth departure log: (t, arm_idx, cause) with cause in {"endo","exo"}
    departures: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.theta = self.rng.normal(size=self.d)
        self.theta /= np.linalg.norm(self.theta)
        self.arms = {}
        self._next_idx = 0
        self.departures = []
        for _ in range(self.n_init):
            self._spawn()

    # ---- internals -------------------------------------------------------

    def _spawn(self):
        x = self.rng.normal(size=self.d)
        x /= np.linalg.norm(x)
        a = Arm(idx=self._next_idx, x=x, born=self.t)
        self.arms[a.idx] = a
        self._next_idx += 1
        return a

    def _mean(self, arm):
        return float(self.theta @ arm.x)

    # ---- public API ------------------------------------------------------

    @property
    def available(self):
        return list(self.arms.values())

    def best_available_mean(self):
        if not self.arms:
            return 0.0
        return max(self._mean(a) for a in self.arms.values())

    def step(self, arm_idx):
        """Pull an arm. Returns (reward, departed_set). Cause is NOT returned."""
        if arm_idx not in self.arms:
            raise KeyError(f"arm {arm_idx} not available")
        arm = self.arms[arm_idx]
        reward = self._mean(arm) + self.rng.normal(0, self.sigma)

        departed = set()

        # endogenous: only the pulled arm is at risk
        if self.rng.random() < self.p:
            departed.add(arm_idx)
            self.departures.append((self.t, arm_idx, "endo"))

        # exogenous: every available arm is at risk, independently
        if self.mu > 0:
            for i in list(self.arms.keys()):
                if i in departed:
                    continue
                if self.rng.random() < self.mu:
                    departed.add(i)
                    self.departures.append((self.t, i, "exo"))

        for i in departed:
            self.arms.pop(i, None)

        # arrivals
        if self.lam > 0:
            for _ in range(self.rng.poisson(self.lam)):
                self._spawn()

        self.t += 1
        return reward, departed

    def is_exhausted(self):
        return len(self.arms) == 0
