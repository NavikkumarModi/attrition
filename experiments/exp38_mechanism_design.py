"""Experiment 38 -- T-F: mechanism design when the externality cannot be measured.

Classical Pigouvian design charges each agent the marginal damage it causes. That
presumes the damage is measurable. Theorem 3 says here it is not: the information
about `e` is bounded by the number of transitions, which is the pool size, and no
amount of horizon, agents, or communication changes it.

So the question is whether a mechanism can work WITHOUT measurement.

The candidate: a UNIFORM tax. Charge every arm the same externality bound rather
than its true kappa_a. This requires no per-arm knowledge -- only a single number
that an operator can set from domain judgement. exp19 already found the uniform
conservative policy beat every learned estimator in the single-agent case. The
question here is whether it also repairs the price of anarchy.

If it does, the conclusion is unusually clean: the mechanism that cannot be
implemented (per-arm Pigouvian pricing) is not needed, because the one that can be
(a flat charge) recovers most of the benefit.
"""
import numpy as np
from experiments.exp32_multiagent_poa import (planner_value, decentralised_value,
                                              rule_greedy, rule_eci)

def make_uniform_tax(tax):
    """Charge every arm the same externality bound: no per-arm knowledge used."""
    def rule(idx, v, p, e, delta, T, t, b):
        return int(idx[np.argmax(v[idx] - delta * p[idx] * tax * (T - t))])
    return rule

def make_rank_based(scale):
    """Prices by the ORDERING of externalities rather than their values.

    IMPORTANT: this rule reads the ordering off the true `e`, so it is an ORACLE
    ordinal mechanism. It establishes what information SUFFICES -- ordinal is
    enough, cardinal is not required -- and says nothing about whether the ordering
    can be learned. exp39 shows it largely cannot: rank recovery saturates around
    tau = 0.68 even as noise goes to zero, because each arm still supplies one
    transition and arms dying early or late have almost no data on one side.

    The practical reading is therefore that the ordering must be SUPPLIED from
    domain knowledge. That is a weaker elicitation requirement than magnitudes, but
    it is still elicitation, not estimation.
    """
    def rule(idx, v, p, e, delta, T, t, b):
        rank = np.argsort(np.argsort(e)) / max(len(e) - 1, 1)
        return int(idx[np.argmax(v[idx]
                                 - delta * p[idx] * scale * rank[idx] * (T - t))])
    return rule


def make_ban_worst(frac):
    """Knows only WHICH arms are the worst offenders; bans them, then acts greedily."""
    def rule(idx, v, p, e, delta, T, t, b):
        thr = np.quantile(e, 1 - frac)
        ok = idx[e[idx] <= thr]
        if len(ok) == 0:
            ok = idx
        return int(ok[np.argmax(v[ok])])
    return rule

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    print("T-F: CAN A MECHANISM WORK WITHOUT MEASURING THE EXTERNALITY?\n")
    print("PoA = planner / decentralised.  Lower is better; 1.000 is optimal.\n")
    print(f"{'agents':>7} {'greedy':>9} {'uniform tax':>13} {'rank only':>11} "
          f"{'ban worst 1/3':>15} {'true kappa':>12}")
    print("-" * 72)
    for m in [2, 3]:
        acc = {k: [] for k in ["g","t","u","u_lo","u_hi"]}
        for _ in range(6):
            n, T, delta = 6, 5, 0.12
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, 1.2, n), 0.0, None)
            pl = planner_value(v, p, e, delta, T, m)
            ebar = float(np.mean(e))
            for key, rule in [("g", rule_greedy), ("t", rule_eci),
                              ("u", make_uniform_tax(ebar)),
                              ("u_lo", make_rank_based(2.0 * ebar)),
                              ("u_hi", make_ban_worst(1 / 3))]:
                d = decentralised_value(v, p, e, delta, T, m, rule)
                acc[key].append(pl / max(d, 1e-9))
        print(f"{m:7d} {np.mean(acc['g']):9.3f} {np.mean(acc['u']):13.3f} "
              f"{np.mean(acc['u_lo']):11.3f} {np.mean(acc['u_hi']):15.3f} "
              f"{np.mean(acc['t']):12.3f}")

    print("\n  A flat charge FAILS: it does not distinguish arms, so it shifts every")
    print("  preference equally and leaves the ordering that caused the problem")
    print("  intact. At m=2 it is worse than no charge at all.")
    print()
    print("  ORDINAL knowledge SUFFICES. Charging by rank -- or simply banning the")
    print("  worst third -- recovers nearly all of what true kappa pricing achieves.")
    print()
    print("  Caveat, established in exp39: these rules read the ordering off the")
    print("  true e, so they are ORACLE ordinal mechanisms. They show what")
    print("  information is enough, not that it can be learned -- rank recovery")
    print("  from data saturates near tau = 0.68 even at vanishing noise. The")
    print("  ordering must be supplied from domain knowledge. Weaker elicitation")
    print("  than magnitudes, but elicitation nonetheless.")
