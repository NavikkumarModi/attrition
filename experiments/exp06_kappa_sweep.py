"""Experiment 06 -- can a penalty heuristic capture the DP's advantage?

exp04 showed the exact optimal policy beats greedy by 12-23% in the coupled
(competitive-release) regime. exp05 showed two penalty heuristics failing:
a constant charge was too weak, a horizon-scaled charge too strong.

Sweep the charge coefficient to test whether ANY member of this family works.
"""
import numpy as np
from experiments.exp05_coupled_learning import CoupledEnv, Blind, episode

class Charged(Blind):
    def __init__(self, env, alpha=0.5, kappa=0.0):
        super().__init__(env, alpha); self.kappa = kappa
    def select(self, env):
        idx = env.available(); th = self.Ai @ self.b
        best, bi = -np.inf, idx[0]
        for a in idx:
            x = env.X[a]
            s = th @ x + self.alpha*np.sqrt(x @ self.Ai @ x)
            if env.sensitive[a]:
                s -= self.kappa * env.p * env.delta
            if s > best: best, bi = s, int(a)
        return bi

def run(kappa, delta, seeds=40):
    tot = []
    for s in range(seeds):
        env = CoupledEnv(seed=s, delta=delta)
        t, _ = episode(env, Charged(env, kappa=kappa))
        tot.append(t)
    return np.mean(tot), np.std(tot)/np.sqrt(seeds)

def main():
    print(f"{'delta':>6} {'kappa':>7} {'value':>9} {'±se':>6} {'vs k=0':>8}")
    print("-"*42)
    for delta in [0.05, 0.10, 0.20]:
        base, _ = run(0.0, delta)
        for k in [0.0, 1.0, 5.0, 20.0, 50.0, 200.0]:
            m, se = run(k, delta)
            print(f"{delta:6.2f} {k:7.1f} {m:9.3f} {se:6.3f} "
                  f"{100*(m-base)/abs(base):+7.1f}%")
        print()


if __name__ == "__main__":
    main()
