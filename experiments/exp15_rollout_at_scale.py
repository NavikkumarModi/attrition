"""Experiment 15 -- does rollout survive at scale?

exp14 showed rollout(index) reaching 99.5-99.9% of exact optimum, but exact
policy evaluation memoises over subsets and is exponential. For deployment the
base policy must be evaluated by simulation instead.

Monte Carlo rollout:
    for each candidate arm a (restricted to the top-m by index, for cost)
        simulate K episodes forward from the post-pull state under the base
        policy, average the returns
    pull the argmax

Cost per decision is m*K*T base-policy steps, which is polynomial -- so this
runs at n = 40 where exact DP cannot. No exact optimum is available at this
scale, so policies are compared against each other and against the index,
whose gap to optimum is known to be small from exp14.
"""

import numpy as np


class Problem:
    def __init__(self, n, T, delta, spread, seed):
        r = np.random.default_rng(seed)
        self.n, self.T, self.delta = n, T, delta
        self.v = np.sort(r.uniform(0.4, 1.2, n))[::-1].copy()
        self.p = np.clip(r.uniform(0.3, 1.0, n), 0.05, 1.0)
        self.e = np.clip(1.0 + r.normal(0, spread, n), 0.0, None)
        self.kappa = self.p * self.e

    def burden(self, alive):
        return self.delta * float(self.e[~alive].sum())


def base_index(prob, alive, t):
    """Deterministic base policy: index rule."""
    idx = np.flatnonzero(alive)
    sc = prob.v[idx] - prob.delta * prob.kappa[idx] * (prob.T - t)
    return int(idx[int(np.argmax(sc))])


def base_greedy(prob, alive, t):
    idx = np.flatnonzero(alive)
    return int(idx[int(np.argmax(prob.v[idx]))])


def simulate(prob, alive, t, pick, rng):
    """Roll the base policy forward to the horizon; return total true value."""
    alive = alive.copy()
    total = 0.0
    while t < prob.T and alive.any():
        a = pick(prob, alive, t)
        total += prob.v[a] - prob.burden(alive)
        if rng.random() < prob.p[a]:
            alive[a] = False
        t += 1
    return total


def run_policy(prob, kind, seed, m=4, K=12):
    rng = np.random.default_rng(5000 + seed)
    alive = np.ones(prob.n, dtype=bool)
    t, total = 0, 0.0
    while t < prob.T and alive.any():
        idx = np.flatnonzero(alive)
        if kind == "greedy":
            a = int(idx[int(np.argmax(prob.v[idx]))])
        elif kind == "index":
            a = base_index(prob, alive, t)
        elif kind == "rollout":
            sc = prob.v[idx] - prob.delta * prob.kappa[idx] * (prob.T - t)
            cand = idx[np.argsort(sc)[::-1][:m]]
            best, ba = -np.inf, int(cand[0])
            for a_c in cand:
                q = 0.0
                for _ in range(K):
                    nxt = alive.copy()
                    if rng.random() < prob.p[a_c]:
                        nxt[a_c] = False
                    q += simulate(prob, nxt, t + 1, base_index, rng)
                q = prob.v[a_c] - prob.burden(alive) + q / K
                if q > best:
                    best, ba = q, int(a_c)
            a = ba
        else:
            raise ValueError(kind)
        total += prob.v[a] - prob.burden(alive)
        if rng.random() < prob.p[a]:
            alive[a] = False
        t += 1
    return total


if __name__ == "__main__":
    SEEDS = 12
    print("total value (higher is better); n=40, exact DP infeasible\n")
    print(f"{'spread':>7} {'greedy':>9} {'index':>9} {'rollout':>9} "
          f"{'idx vs grd':>11} {'roll vs idx':>12}")
    print("-" * 62)
    for spread in [0.3, 0.8, 1.5]:
        res = {}
        for kind in ["greedy", "index", "rollout"]:
            vals = [run_policy(Problem(40, 40, 0.02, spread, s), kind, s)
                    for s in range(SEEDS)]
            res[kind] = float(np.mean(vals))
        g, i, r = res["greedy"], res["index"], res["rollout"]
        print(f"{spread:7.2f} {g:9.3f} {i:9.3f} {r:9.3f} "
              f"{100*(i-g)/abs(g):10.1f}% {100*(r-i)/abs(i):11.1f}%")
