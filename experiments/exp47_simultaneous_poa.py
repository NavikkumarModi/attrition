"""Experiment 47 -- a genuine price of anarchy, under simultaneous action.

exp46 showed the sequential model is degenerate. Under simultaneous action agents
cannot condition on each other's current choices, so two effects appear that
sequential play cannot express:

  COLLISION  several agents converge on the same attractive arm and destroy it
             faster than any intended; a coordinator would spread them out
  BLINDNESS  no agent sees the damage its peers do this round

The critical control: at constant kappa the single-agent Theorem 1 no longer forces
PoA = 1, because coordination is a separate problem from pricing. So this model
should show PoA > 1 even at zero dispersion -- which distinguishes a genuine
coordination cost from the pricing cost the earlier model conflated it with.
"""
import numpy as np
from attrition.simultaneous import (planner_value_simultaneous,
                                           decentralised_value_simultaneous,
                                           price_of_anarchy)

def cell(m, spread, delta=0.12, T=4, n=5, inst=6, seed=5, rule="greedy",
         tie_break="same"):
    rng = np.random.default_rng(seed)
    R, SD = [], []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
        e = (0.4 / p) if spread == 0.0 else \
            np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
        r, _, _ = price_of_anarchy(v, p, e, delta, T, m, rule=rule,
                                   tie_break=tie_break)
        R.append(r); SD.append(float(np.std(p * e)))
    return float(np.mean(R)), float(np.mean(SD))

if __name__ == "__main__":
    print("PRICE OF ANARCHY UNDER SIMULTANEOUS ACTION\n")
    print("Control: at zero kappa dispersion, pricing is irrelevant, so any PoA")
    print("above 1 is pure COORDINATION cost.\n")
    print(f"{'m':>3} {'spread':>8} {'std(kappa)':>11} {'PoA greedy':>12} "
          f"{'PoA ECI':>10} {'PoA spread-out':>16}")
    print("-" * 66)
    for m in [2, 3]:
        for spread in [0.0, 0.6, 1.4]:
            g, sd = cell(m, spread)
            ec, _ = cell(m, spread, rule="eci")
            sp, _ = cell(m, spread, tie_break="random")
            print(f"{m:3d} {spread:8.2f} {sd:11.3f} {g:12.4f} {ec:10.4f} "
                  f"{sp:16.4f}")

    print("\n  Two distinct costs are now separable:")
    print("    coordination -- visible at zero dispersion, fixed by spreading out")
    print("    pricing      -- visible only with dispersion, fixed by ECI")
    print("  The sequential model could express neither, because it let agents")
    print("  adapt within the round and so implicitly coordinated them.")
