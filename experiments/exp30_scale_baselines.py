"""Experiment 30 -- scale to n=50-100 with stronger baselines.

The review notes that comparisons against greedy/UCB/TS/conservative are too weak,
and that natural alternatives already incorporating resource consumption should be
included. Three are added:

  LAGRANGIAN   BwK-style. Treat accumulated burden as a budget consumed at rate
               p_a e_a and maintain a dual price lambda, updated by mirror descent
               on the budget violation. Selects argmax v_a - lambda * p_a e_a.
               This is the standard resource-aware bandit response, and is the
               closest existing method to ours.

  RATIO        Resource-aware heuristic: maximise value per unit of expected
               consumption, argmax v_a / (1 + p_a e_a). Common in knapsack-style
               practice.

  CONSERVATIVE-BOUND  Safe-bandit style: refuse any arm whose expected burden
               exceeds a threshold, then act greedily among the rest.

The point of interest is whether the Lagrangian method -- which does price
consumption -- recovers the gap. It should not, because a single global dual price
cannot express the horizon-dependence delta*kappa_a*(T-t): the correct charge
shrinks as the episode progresses, while lambda is a scalar.
"""

import numpy as np

from evolving_bandits import ConsumableBandit, run
from evolving_bandits.policies import Policy, Greedy, ECI, Conservative, \
    ThompsonSampling, UCB


class Lagrangian(Policy):
    """BwK-style dual price on expected consumption, updated online."""
    name = "lagrangian"

    def __init__(self, eta=0.05, target=None):
        self.eta = eta
        self.lam = 1.0
        self.target = target
        self._spent = 0.0

    def scores(self, state):
        a = state.available
        return state.v_hat[a] - self.lam * state.p[a] * state.e[a]

    def select(self, state):
        a = state.available
        s = self.scores(state)
        pick = int(a[int(np.argmax(s))])
        # dual update: if consumption is running ahead of a uniform schedule,
        # raise the price; else lower it
        budget_rate = (self.target if self.target is not None
                       else float(np.mean(state.p * state.e)))
        self._spent += state.p[pick] * state.e[pick]
        expected = budget_rate * (state.t + 1)
        self.lam = float(np.clip(
            self.lam + self.eta * (self._spent - expected), 0.0, 50.0))
        return pick


class Ratio(Policy):
    """Value per unit expected consumption."""
    name = "ratio"

    def scores(self, state):
        a = state.available
        return state.v_hat[a] / (1.0 + state.p[a] * state.e[a])

    def select(self, state):
        a = state.available
        return int(a[int(np.argmax(self.scores(state)))])


class ConservativeBound(Policy):
    """Safe-bandit style: exclude high-burden arms, then act greedily."""
    name = "safe-filter"

    def __init__(self, quantile=0.6):
        self.q = quantile

    def select(self, state):
        a = state.available
        kap = state.p[a] * state.e[a]
        thr = float(np.quantile(kap, self.q)) if len(a) > 2 else float(kap.max())
        allowed = a[kap <= thr]
        if len(allowed) == 0:
            allowed = a
        return int(allowed[int(np.argmax(state.v_hat[allowed]))])


def sweep(n, T, spread, delta, seeds=20, known=True):
    pols = [Greedy(), ThompsonSampling(), UCB(), Ratio(),
            ConservativeBound(), Lagrangian(), Conservative(e_bound=2.0), ECI()]
    out = {}
    for pol in pols:
        vals, regs = [], []
        for s in range(seeds):
            env = ConsumableBandit.random(n=n, k_spread=spread, delta=delta,
                                          horizon=T, seed=s)
            fresh = type(pol)(**({} if pol.name not in ("conservative",)
                                 else {"e_bound": 2.0}))
            r = run(env, fresh, known=known, seed=s)
            vals.append(r["value"]); regs.append(r["regret"])
        out[pol.name] = (float(np.mean(vals)),
                         float(np.std(vals)/np.sqrt(seeds)),
                         float(np.mean(regs)))
    return out


if __name__ == "__main__":
    for n, T, spread, delta in [(50, 100, 1.5, 0.010),
                                (100, 200, 1.5, 0.005),
                                (100, 200, 2.5, 0.005)]:
        print(f"\n=== n={n}, T={T}, std(e)={spread}, delta={delta} ===")
        res = sweep(n, T, spread, delta)
        base = res["greedy"][0]
        print(f"{'policy':>16} {'value':>10} {'±se':>7} {'regret':>9} "
              f"{'vs greedy':>10}")
        print("-" * 56)
        for k in sorted(res, key=lambda k: -res[k][0]):
            v, se, rg = res[k]
            rel = f"{100*(v-base)/abs(base):+9.1f}%" if k != "greedy" else ""
            print(f"{k:>16} {v:10.3f} {se:7.3f} {rg:9.3f} {rel:>10}")
