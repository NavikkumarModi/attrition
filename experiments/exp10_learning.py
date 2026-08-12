"""Experiment 10 -- the learning problem.

Everything so far assumed v, p, e known. The actual bandit problem is to
estimate them online. The interesting structure:

  reward_t = v_{a_t} - delta * sum_i e_i * 1[arm i already dead at t] + noise

which is LINEAR in the unknowns (v_1..v_n, delta*e_1..delta*e_n). So least
squares recovers them jointly -- but only if the dead-set indicators carry
enough independent variation. When arms die together, their columns are
collinear and the individual e_i are not separately identified. That is the
credit-assignment version of the session-1 identification problem.

p_a is estimated from death frequency conditional on being pulled.

Policies compared:
  blind      -- greedy on estimated v, ignores externality entirely
  learned    -- index using estimated kappa = p_hat * e_hat
  oracle-idx -- index using TRUE kappa (upper bound on what learning can reach)
"""

import numpy as np


class Sim:
    def __init__(self, n=8, T=60, delta=0.12, sigma=0.05, spread=0.8, seed=0):
        r = np.random.default_rng(seed)
        self.n, self.T, self.delta, self.sigma = n, T, delta, sigma
        self.v = np.sort(r.uniform(0.4, 1.2, n))[::-1].copy()
        self.p = np.clip(r.uniform(0.5, 1.0, n), 0.05, 1.0)
        self.e = np.clip(1.0 + r.normal(0, spread, n), 0.0, None)
        self.rng = r

    def reset(self):
        self.alive = np.ones(self.n, dtype=bool)
        self.t = 0

    @property
    def burden(self):
        return self.delta * float(np.sum(self.e[~self.alive]))

    def step(self, a):
        true_val = self.v[a] - self.burden
        obs = true_val + self.rng.normal(0, self.sigma)
        died = self.rng.random() < self.p[a]
        if died:
            self.alive[a] = False
        self.t += 1
        return obs, true_val, died


def run(sim, policy, seed=0):
    sim.rng = np.random.default_rng(1000 + seed)
    sim.reset()
    n = sim.n
    # sufficient statistics
    X_rows, y = [], []
    pulls = np.zeros(n)
    deaths = np.zeros(n)
    total = 0.0

    v_hat = np.ones(n) * 0.8
    de_hat = np.ones(n) * sim.delta          # delta*e_i
    p_hat = np.ones(n) * 0.5

    while sim.t < sim.T and sim.alive.any():
        idx = np.flatnonzero(sim.alive)
        rem = sim.T - sim.t

        if policy == "blind":
            scores = v_hat[idx]
        elif policy == "learned":
            scores = v_hat[idx] - p_hat[idx] * de_hat[idx] * rem
        elif policy == "oracle-idx":
            scores = v_hat[idx] - sim.p[idx] * sim.delta * sim.e[idx] * rem
        else:
            raise ValueError(policy)

        # small forced exploration so estimates can move
        if sim.rng.random() < 0.10:
            a = int(sim.rng.choice(idx))
        else:
            a = int(idx[int(np.argmax(scores))])

        dead_before = ~sim.alive.copy()
        obs, true_val, died = sim.step(a)
        total += true_val

        row = np.zeros(2 * n)
        row[a] = 1.0                      # v_a
        row[n:][dead_before] = -1.0       # -sum delta*e_i over dead arms
        X_rows.append(row); y.append(obs)

        pulls[a] += 1; deaths[a] += int(died)

        # refit periodically
        if len(y) >= 8 and len(y) % 4 == 0:
            X = np.array(X_rows)
            beta = np.linalg.lstsq(X.T @ X + 1e-3 * np.eye(2 * n),
                                   X.T @ np.array(y), rcond=None)[0]
            v_hat = np.clip(beta[:n], 0.0, 2.0)
            de_hat = np.clip(beta[n:], 0.0, 1.0)
            p_hat = np.where(pulls > 0,
                             (deaths + 1.0) / (pulls + 2.0), 0.5)

    return total


if __name__ == "__main__":
    SEEDS = 40
    print(f"{'spread':>7} {'blind':>9} {'learned':>9} {'oracle-idx':>11} "
          f"{'learned vs blind':>17} {'gap to oracle':>14}")
    print("-" * 74)
    for spread in [0.0, 0.3, 0.6, 1.0]:
        res = {}
        for pol in ["blind", "learned", "oracle-idx"]:
            vals = []
            for s in range(SEEDS):
                sim = Sim(spread=spread, seed=s)
                vals.append(run(sim, pol, seed=s))
            res[pol] = float(np.mean(vals))
        b, l, o = res["blind"], res["learned"], res["oracle-idx"]
        lift = 100 * (l - b) / abs(b) if b != 0 else 0.0
        gap = 100 * (o - l) / abs(o) if o != 0 else 0.0
        print(f"{spread:7.2f} {b:9.3f} {l:9.3f} {o:11.3f} "
              f"{lift:16.1f}% {gap:13.1f}%")
