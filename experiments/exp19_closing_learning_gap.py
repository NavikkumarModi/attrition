"""Experiment 19 -- closing the learning gap.

Theorem 3 says per-arm externalities e_i cannot be estimated: each arm supplies
exactly one before/after transition, so the sample budget is capped at ~n/p
regardless of horizon.

But T3 forbids estimating the FULL VECTOR e. It does not forbid acting well.
Three routes around it, in increasing order of assumption:

  R1  FEATURES. If e_a = <psi, z_a> for known features z_a in R^k with k << n,
      every death informs a shared k-vector rather than one coordinate. n deaths
      give n observations of a k-dimensional object. The T3 floor applies per
      coordinate of e, not per coordinate of psi.

  R2  CONSERVATIVE DEFAULT. Act on an upper bound e_hi instead of an estimate.
      Over-charging the externality costs some value but avoids ruin. Tests the
      practical claim "specify, don't learn".

  R3  ORACLE. Known e. Upper bound on what any method can reach.

Compared against greedy (no externality model at all) and per-arm OLS (exp10's
failed learner).
"""

import numpy as np


class World:
    """Externalities generated from features: e_a = <psi, z_a>."""

    def __init__(self, n=20, k=3, T=60, delta=0.03, sigma=0.05,
                 spread=1.0, seed=0):
        r = np.random.default_rng(seed)
        self.n, self.k, self.T = n, k, T
        self.delta, self.sigma = delta, sigma
        self.v = np.sort(r.uniform(0.4, 1.2, n))[::-1].copy()
        self.p = np.clip(r.uniform(0.3, 1.0, n), 0.05, 1.0)
        self.Z = np.abs(r.normal(size=(n, k)))          # features, known
        self.psi = np.abs(r.normal(size=k)) * spread    # unknown
        self.e = self.Z @ self.psi
        self.rng = r

    def reset(self, seed):
        self.rng = np.random.default_rng(9000 + seed)
        self.alive = np.ones(self.n, dtype=bool)
        self.t = 0

    def burden(self):
        return self.delta * float(self.e[~self.alive].sum())

    def step(self, a):
        true_v = self.v[a] - self.burden()
        obs = true_v + self.rng.normal(0, self.sigma)
        if self.rng.random() < self.p[a]:
            self.alive[a] = False
        self.t += 1
        return obs, true_v


def run(w, mode, seed, conservative_mult=2.0):
    w.reset(seed)
    n, k = w.n, w.k
    rows, ys = [], []
    psi_hat = np.zeros(k)
    e_arm_hat = np.zeros(n)
    p_hat = np.full(n, 0.5)
    pulls = np.zeros(n); deaths = np.zeros(n)
    total = 0.0

    while w.t < w.T and w.alive.any():
        idx = np.flatnonzero(w.alive)
        rem = w.T - w.t

        if mode == "greedy":
            e_use = np.zeros(n)
        elif mode == "oracle":
            e_use = w.e
        elif mode == "features":
            e_use = np.clip(w.Z @ psi_hat, 0, None)
        elif mode == "per-arm":
            e_use = e_arm_hat
        elif mode == "conservative":
            e_use = np.full(n, conservative_mult * float(np.mean(w.e)))
        else:
            raise ValueError(mode)

        p_use = w.p if mode in ("oracle", "conservative") else p_hat
        score = w.v[idx] - w.delta * p_use[idx] * e_use[idx] * rem
        if w.rng.random() < 0.08:
            a = int(w.rng.choice(idx))
        else:
            a = int(idx[int(np.argmax(score))])

        dead_before = ~w.alive.copy()
        obs, true_v = w.step(a)
        total += true_v
        pulls[a] += 1; deaths[a] += int(not w.alive[a] and dead_before[a] == False)

        # design row: v_a indicator, then either k feature cols or n arm cols
        if mode == "features":
            row = np.zeros(n + k)
            row[a] = 1.0
            row[n:] = -w.Z[dead_before].sum(axis=0)
            rows.append(row); ys.append(obs)
            if len(ys) >= 6 and len(ys) % 3 == 0:
                X = np.array(rows)
                beta = np.linalg.lstsq(X.T@X + 1e-2*np.eye(n+k),
                                       X.T@np.array(ys), rcond=None)[0]
                psi_hat = np.clip(beta[n:] / max(w.delta, 1e-9), 0, None)
        elif mode == "per-arm":
            row = np.zeros(2*n)
            row[a] = 1.0
            row[n:][dead_before] = -1.0
            rows.append(row); ys.append(obs)
            if len(ys) >= 6 and len(ys) % 3 == 0:
                X = np.array(rows)
                beta = np.linalg.lstsq(X.T@X + 1e-2*np.eye(2*n),
                                       X.T@np.array(ys), rcond=None)[0]
                e_arm_hat = np.clip(beta[n:] / max(w.delta, 1e-9), 0, None)

        p_hat = np.where(pulls > 0, (deaths + 1.0)/(pulls + 2.0), 0.5)

    return total


if __name__ == "__main__":
    SEEDS = 30
    print("Closing the learning gap: value achieved (higher better)\n")
    print(f"{'spread':>7} {'greedy':>8} {'per-arm':>8} {'conserv':>8} "
          f"{'features':>9} {'oracle':>8}   {'features captures':>18}")
    print("-" * 78)
    for spread in [0.5, 1.0, 2.0]:
        res = {}
        for mode in ["greedy", "per-arm", "conservative", "features", "oracle"]:
            vals = [run(World(spread=spread, seed=s), mode, s)
                    for s in range(SEEDS)]
            res[mode] = float(np.mean(vals))
        g, o = res["greedy"], res["oracle"]
        span = o - g
        cap = 100*(res["features"] - g)/span if abs(span) > 1e-9 else 0.0
        print(f"{spread:7.2f} {g:8.3f} {res['per-arm']:8.3f} "
              f"{res['conservative']:8.3f} {res['features']:9.3f} {o:8.3f}   "
              f"{cap:17.1f}%")
