"""Experiment 05 -- learning under asymmetric externality.

exp04 found greedy optimal in every scenario EXCEPT S5 (competitive release),
where consuming one class of arm imposes a burden on the whole system and the
other class does not. The distinguishing feature is *asymmetry*: in S4 the
degradation trajectory was independent of which arm was pulled, so greedy
survived; in S5 it was not.

This experiment asks the follow-on question: in the coupled regime, does a
preservation-aware learner beat a departure-blind one? If yes, the algorithm
design killed in session 2 is resurrected -- with a correct justification.

Coupled environment:
  arms 0..n-2  'sensitive'   -- consuming one adds `delta` burden to everything
  arm  n-1     'resistant'   -- consuming it adds no burden
  observed reward of arm a = v_a - delta * (# dead sensitive arms) + noise
"""

import numpy as np


class CoupledEnv:
    def __init__(self, n=16, d=4, p=0.6, delta=0.06, sigma=0.05,
                 horizon=200, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n, self.d, self.p, self.delta = n, d, p, delta
        self.sigma, self.horizon = sigma, horizon
        self.theta = self.rng.normal(size=d)
        self.theta /= np.linalg.norm(self.theta)
        X = self.rng.normal(size=(n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        self.X = X
        self.base = X @ self.theta + 1.0  # shift positive
        self.alive = np.ones(n, dtype=bool)
        self.sensitive = np.ones(n, dtype=bool)
        self.sensitive[-1] = False
        self.t = 0

    @property
    def burden(self):
        return self.delta * int(np.sum(~self.alive & self.sensitive))

    def value(self, a):
        return float(self.base[a] - self.burden)

    def available(self):
        return np.flatnonzero(self.alive)

    def best_value(self):
        idx = self.available()
        return max(self.value(a) for a in idx) if idx.size else 0.0

    def step(self, a):
        v = self.value(a)
        r = v + self.rng.normal(0, self.sigma)
        if self.rng.random() < self.p:
            self.alive[a] = False
        self.t += 1
        return r, v

    def done(self):
        return self.t >= self.horizon or not self.alive.any()


class Blind:
    """LinUCB on features. Departure- and burden-blind."""
    name = "linucb-blind"

    def __init__(self, env, alpha=0.5):
        self.d, self.alpha = env.d, alpha
        self.A = np.eye(env.d); self.b = np.zeros(env.d)
        self.Ai = np.eye(env.d)

    def select(self, env):
        idx = env.available()
        th = self.Ai @ self.b
        sc = [th @ env.X[a] + self.alpha * np.sqrt(env.X[a] @ self.Ai @ env.X[a])
              for a in idx]
        return int(idx[int(np.argmax(sc))])

    def update(self, env, a, r):
        x = env.X[a]
        self.A += np.outer(x, x); self.b += r * x
        self.Ai = np.linalg.inv(self.A)


class Preserving(Blind):
    """LinUCB with an externality charge on consuming sensitive arms.

    Killing a sensitive arm raises the burden by `delta` for EVERY remaining
    round, so the expected cost of pulling arm a is p * delta * (T - t), not a
    per-round constant. That horizon scaling is the whole point: early
    consumption is far more expensive than late consumption.

    Assumes knowledge of which arms are sensitive and of delta -- an
    oracle-informed upper bound on what preservation can buy.
    """
    name = "linucb-preserving"

    def __init__(self, env, alpha=0.5, kappa=1.0):
        super().__init__(env, alpha)
        self.kappa = kappa

    def select(self, env):
        idx = env.available()
        th = self.Ai @ self.b
        rem = max(env.horizon - env.t, 1)
        scores = []
        for a in idx:
            x = env.X[a]
            base = th @ x + self.alpha * np.sqrt(x @ self.Ai @ x)
            # expected externality: p * delta persisting for `rem` rounds,
            # expressed per-round so it is comparable to `base`
            charge = self.kappa * env.p * env.delta * rem if env.sensitive[a] else 0.0
            scores.append(base - charge / rem * rem / max(rem, 1) * rem)
            scores[-1] = base - self.kappa * env.p * env.delta * rem
        return int(idx[int(np.argmax(scores))])


def episode(env, agent):
    tot, opt = 0.0, 0.0
    while not env.done():
        a = agent.select(env)
        opt += env.best_value()
        r, v = env.step(a)
        agent.update(env, a, r)
        tot += v
    return tot, opt


def run(agent_cls, seeds=40, **kw):
    tots, opts = [], []
    for s in range(seeds):
        env = CoupledEnv(seed=s, **kw)
        ag = agent_cls(env)
        t, o = episode(env, ag)
        tots.append(t); opts.append(o)
    return np.mean(tots), np.std(tots)/np.sqrt(seeds), np.mean(opts)


if __name__ == "__main__":
    print(f"{'delta':>6} {'agent':>20} {'total value':>12} {'±se':>7} {'vs blind':>9}")
    print("-" * 60)
    for delta in [0.0, 0.02, 0.05, 0.10, 0.20]:
        base = None
        for cls in [Blind, Preserving]:
            m, se, o = run(cls, delta=delta)
            rel = "" if base is None else f"{100*(m-base)/abs(base):+8.1f}%"
            if base is None:
                base = m
            print(f"{delta:6.2f} {cls.name:>20} {m:12.3f} {se:7.3f} {rel:>9}")
