"""Experiment 03 -- separate 'easier to learn' from 'better outcome'.

exp02 found per-pull regret FALLS as irreversibility rises (with arrivals).
Regret is measured against each instance's own optimum, so lower regret does
not imply better absolute performance. Report both.
"""
import numpy as np
from evolving_bandits.env import EvolvingBanditEnv
from evolving_bandits.agents import LinUCBAgent

SEEDS = 10

def run(p, lam, T=800, n_init=60, d=5, mu=0.0):
    reg, rew, orc = [], [], []
    for s in range(SEEDS):
        env = EvolvingBanditEnv(d=d, n_init=n_init, p=p, mu=mu, lam=lam,
                                sigma=0.1, horizon=T, seed=s)
        ag = LinUCBAgent(d=d, alpha=1.0)
        R, O, k = 0.0, 0.0, 0
        while env.t < env.horizon and not env.is_exhausted():
            arms = env.available
            best = env.best_available_mean()
            a = ag.select(arms)
            m = float(env.theta @ a.x)
            r, _ = env.step(a.idx); ag.update(a, r)
            R += m; O += best; k += 1
        reg.append((O - R) / k); rew.append(R / k); orc.append(O / k)
    return np.mean(reg), np.mean(rew), np.mean(orc)

print(f"{'lam':>5} {'p':>5} {'regret/pull':>12} {'reward/pull':>12} {'oracle/pull':>12}")
print("-" * 50)
for lam in [0.0, 1.0]:
    for p in [0.0, 0.1, 0.5, 1.0]:
        g, w, o = run(p, lam)
        print(f"{lam:5.1f} {p:5.2f} {g:12.4f} {w:12.4f} {o:12.4f}")
