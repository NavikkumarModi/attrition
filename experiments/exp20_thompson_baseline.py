"""Experiment 20 -- the missing baseline: Thompson Sampling and UCB.

Every result so far compares against greedy or exact DP. A reviewer's first
question is whether standard bandit algorithms already handle this. They should
not -- TS and UCB are departure-blind, they model reward uncertainty, not the
consequence of consumption -- but the claim must be tested, not asserted.

Two settings:

  KNOWN    all algorithms told v, p, e. Isolates the POLICY question: does
           TS's exploration accidentally preserve arms? (TS with known means
           degenerates to greedy, so a randomised variant is used to test
           whether stochastic tie-breaking alone helps.)

  UNKNOWN  v estimated online. The realistic setting. TS and UCB run normally;
           the conservative default from exp19 is the proposed method.
"""

import numpy as np


class Env:
    def __init__(self, n=20, T=60, delta=0.03, sigma=0.1, spread=1.5, seed=0):
        r = np.random.default_rng(seed)
        self.n, self.T, self.delta, self.sigma = n, T, delta, sigma
        self.v = np.sort(r.uniform(0.4, 1.2, n))[::-1].copy()
        self.p = np.clip(r.uniform(0.3, 1.0, n), 0.05, 1.0)
        self.e = np.clip(1.0 + r.normal(0, spread, n), 0.0, None)

    def reset(self, seed):
        self.rng = np.random.default_rng(7000 + seed)
        self.alive = np.ones(self.n, dtype=bool)
        self.t = 0

    def burden(self):
        return self.delta * float(self.e[~self.alive].sum())

    def step(self, a):
        tv = self.v[a] - self.burden()
        obs = tv + self.rng.normal(0, self.sigma)
        if self.rng.random() < self.p[a]:
            self.alive[a] = False
        self.t += 1
        return obs, tv


def play(env, algo, seed, known=True):
    env.reset(seed)
    n = env.n
    mu = np.zeros(n); cnt = np.zeros(n)          # empirical mean / counts
    tot = 0.0
    while env.t < env.T and env.alive.any():
        idx = np.flatnonzero(env.alive)
        rem = env.T - env.t
        vhat = env.v if known else np.where(cnt > 0, mu, 1.5)   # optimistic init

        if algo == "greedy":
            s = vhat[idx]
        elif algo == "thompson":
            # posterior over mean reward; known-variance Gaussian
            post_sd = env.sigma / np.sqrt(np.maximum(cnt[idx], 1))
            s = vhat[idx] + (0.0 if known else 1.0) * env.rng.normal(
                0, 1, len(idx)) * post_sd
            if known:      # randomised tie-break variant
                s = vhat[idx] + env.rng.normal(0, env.sigma, len(idx))
        elif algo == "ucb":
            tt = max(env.t, 1)
            s = vhat[idx] + (0.0 if known else 1.0) * np.sqrt(
                2*np.log(tt)/np.maximum(cnt[idx], 1))
        elif algo == "index":
            s = vhat[idx] - env.delta*env.p[idx]*env.e[idx]*rem
        elif algo == "conservative":
            ebar = float(np.mean(env.e)) * 2.0
            s = vhat[idx] - env.delta*env.p[idx]*ebar*rem
        elif algo == "ts+index":
            post_sd = env.sigma / np.sqrt(np.maximum(cnt[idx], 1))
            draw = vhat[idx] + (0.0 if known else 1.0) * env.rng.normal(
                0, 1, len(idx)) * post_sd
            s = draw - env.delta*env.p[idx]*env.e[idx]*rem
        else:
            raise ValueError(algo)

        a = int(idx[int(np.argmax(s))])
        obs, tv = env.step(a)
        tot += tv
        cnt[a] += 1
        mu[a] += (obs - mu[a]) / cnt[a]
    return tot


def sweep(known, algos, spreads, seeds=40):
    print(f"\n{'KNOWN parameters' if known else 'UNKNOWN v (estimated online)'}"
          f"  -- total value, higher is better\n")
    hdr = f"{'spread':>7}" + "".join(f"{a:>13}" for a in algos)
    print(hdr); print("-" * len(hdr))
    for sp in spreads:
        row = f"{sp:7.2f}"
        for a in algos:
            vals = [play(Env(spread=sp, seed=s), a, s, known)
                    for s in range(seeds)]
            row += f"{np.mean(vals):13.3f}"
        print(row)


if __name__ == "__main__":
    sweep(True, ["greedy", "thompson", "ucb", "index"], [0.5, 1.5, 3.0])
    sweep(False, ["thompson", "ucb", "conservative", "ts+index"],
          [0.5, 1.5, 3.0])
