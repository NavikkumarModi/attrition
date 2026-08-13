"""Consumable-action-set bandit environment.

Pulling an arm may permanently destroy it, and each destroyed arm imposes a
permanent burden on every subsequent reward. This coupling -- consumption
creating a negative externality -- is what distinguishes the setting from
rotting, mortal, blocking, and knapsack bandits.
"""

import numpy as np

from .policies import State


class ConsumableBandit:
    """Bandit whose action set shrinks under the learner's own pulls.

    Parameters
    ----------
    v : ndarray            arm mean rewards
    p : ndarray            per-arm probability that a pull destroys the arm
    e : ndarray            per-arm externality coefficient
    delta : float          externality scale
    horizon : int          number of rounds
    sigma : float          reward noise standard deviation
    mu : float             exogenous per-round departure probability
    seed : int
    """

    def __init__(self, v, p, e, delta=0.05, horizon=100, sigma=0.1,
                 mu=0.0, seed=0):
        self.v = np.asarray(v, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.e = np.asarray(e, dtype=float)
        self.n = len(self.v)
        self.delta, self.horizon = float(delta), int(horizon)
        self.sigma, self.mu = float(sigma), float(mu)
        self.seed = seed
        self.reset()

    @classmethod
    def random(cls, n=20, k_spread=1.0, seed=0, **kw):
        r = np.random.default_rng(seed)
        v = np.sort(r.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(r.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + r.normal(0, k_spread, n), 0.0, None)
        return cls(v, p, e, seed=seed, **kw)

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self.alive = np.ones(self.n, dtype=bool)
        self.t = 0
        self.counts = np.zeros(self.n)
        self.v_hat = np.full(self.n, float(self.v.max()))   # optimistic init
        self._sums = np.zeros(self.n)
        return self.state()

    @property
    def burden(self):
        """Permanent penalty applied to every reward, from arms already dead."""
        return self.delta * float(self.e[~self.alive].sum())

    @property
    def kappa(self):
        """Expected marginal externality per pull. Governs optimal play (T1)."""
        return self.p * self.e

    def state(self, known=True):
        return State(available=np.flatnonzero(self.alive), t=self.t,
                     horizon=self.horizon,
                     v_hat=self.v if known else self.v_hat,
                     p=self.p, e=self.e, delta=self.delta,
                     counts=self.counts, sigma=self.sigma, rng=self.rng)

    def done(self):
        return self.t >= self.horizon or not self.alive.any()

    def step(self, arm):
        """Pull an arm. Returns (observed, realised_value, regret, destroyed)."""
        if not self.alive[arm]:
            raise ValueError(f"arm {arm} is not available")
        realised = float(self.v[arm] - self.burden)
        observed = realised + self.rng.normal(0, self.sigma)
        idx = np.flatnonzero(self.alive)
        best_available = float(np.max(self.v[idx] - self.burden))
        regret = best_available - realised

        destroyed = self.rng.random() < self.p[arm]
        if destroyed:
            self.alive[arm] = False
        if self.mu > 0:
            for i in np.flatnonzero(self.alive):
                if self.rng.random() < self.mu:
                    self.alive[i] = False

        self.counts[arm] += 1
        self._sums[arm] += observed
        self.v_hat[arm] = self._sums[arm] / self.counts[arm]
        self.t += 1
        return observed, realised, regret, destroyed


def run(env, policy, known=True, seed=None, log=False):
    """Run one episode. Returns a dict of value, regret, and survival stats.

    The two headline columns are ``value`` and ``regret``. Theorem 4 says they
    can move in opposite directions: a policy can hold regret at zero while
    driving value arbitrarily negative.
    """
    env.reset(seed)
    total_value, total_regret, rows = 0.0, 0.0, []
    while not env.done():
        st = env.state(known=known)
        arm = policy.select(st)
        obs, realised, regret, destroyed = env.step(arm)
        total_value += realised
        total_regret += regret
        if log:
            rows.append({"t": env.t - 1, "arm": arm, "value": realised,
                         "regret": regret, "destroyed": destroyed,
                         "alive": int(env.alive.sum())})
    return {"value": total_value, "regret": total_regret,
            "pulls": env.t, "arms_left": int(env.alive.sum()),
            "log": rows}


def compare(env_factory, policies, seeds=30, known=True):
    """Run several policies on matched environments. Returns {name: stats}."""
    out = {}
    for pol in policies:
        vals, regs, left = [], [], []
        for s in range(seeds):
            env = env_factory(s)
            r = run(env, pol, known=known, seed=s)
            vals.append(r["value"]); regs.append(r["regret"])
            left.append(r["arms_left"])
        out[pol.name] = {
            "value": float(np.mean(vals)),
            "value_se": float(np.std(vals) / np.sqrt(seeds)),
            "regret": float(np.mean(regs)),
            "arms_left": float(np.mean(left)),
        }
    return out
