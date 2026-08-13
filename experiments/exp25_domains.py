"""Experiment 25 -- the four domain instantiations, run end to end.

Each domain supplies its own (v, p, e, delta). No domain is illustrated; every
one is simulated, and every number below comes from a run.
"""
import numpy as np
from attrition import (agent_tools, adaptive_therapy, platform_trial,
                              design_space, Greedy, ECI, Conservative,
                              ThompsonSampling, run, DOMAIN_NOTES)

DOMAINS = [("agent_tools", agent_tools), ("adaptive_therapy", adaptive_therapy),
           ("platform_trial", platform_trial), ("design_space", design_space)]

def evaluate(factory, seeds=40):
    out = {}
    for pol_name, make in [("greedy", Greedy), ("thompson", ThompsonSampling),
                           ("conservative", None), ("eci", ECI)]:
        vals, regs, left = [], [], []
        for s in range(seeds):
            env = factory(seed=s)
            pol = (Conservative(e_bound=float(np.mean(env.e)) * 2.0)
                   if pol_name == "conservative" else make())
            r = run(env, pol, seed=s)
            vals.append(r["value"]); regs.append(r["regret"])
            left.append(r["arms_left"])
        out[pol_name] = (float(np.mean(vals)), float(np.mean(regs)),
                         float(np.mean(left)))
    return out

if __name__ == "__main__":
    for name, fac in DOMAINS:
        env = fac(seed=0)
        kap = env.kappa
        print(f"\n=== {name} ===")
        print(f"  arms {env.n}, std(kappa) = {np.std(kap):.3f}  "
              f"(T1: greedy optimal only if this is 0)")
        res = evaluate(fac)
        print(f"  {'policy':>13} {'value':>9} {'regret':>9} {'arms left':>10} "
              f"{'vs greedy':>10}")
        print("  " + "-"*54)
        base = res["greedy"][0]
        for pol, (v, r, l) in res.items():
            rel = "" if pol == "greedy" else f"{100*(v-base)/abs(base):+9.1f}%"
            print(f"  {pol:>13} {v:9.3f} {r:9.3f} {l:10.1f} {rel:>10}")
    print("\n\nFit notes:")
    for k, note in DOMAIN_NOTES.items():
        print(f"\n{k}:\n  {note}")
