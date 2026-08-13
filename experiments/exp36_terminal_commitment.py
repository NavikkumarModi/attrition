"""Experiment 36 -- terminal commitment: does the phenomenon appear here too?

A finite budget of irreversible experiments, then a one-shot declaration that fixes
the feasible set for the whole operating phase. The question is whether a policy
that optimises the *visible* experimental-phase quantity destroys terminal value,
as it does in the sequential setting.
"""
import numpy as np
from attrition.commitment import (TerminalCommitment,
                                         greedy_commitment_policy,
                                         edge_first_policy,
                                         optimal_commitment_policy,
                                         evaluate_commitment_policy)

POLICIES = [("greedy (best yield first)", greedy_commitment_policy),
            ("edge-first (max width)", edge_first_policy),
            ("expand-outward", optimal_commitment_policy)]

if __name__ == "__main__":
    print("TERMINAL COMMITMENT\n")
    print("Finite irreversible experiment budget, then a one-shot envelope")
    print("declaration that bounds all subsequent operation.\n")

    for budget in [2, 4, 6]:
        print(f"--- budget = {budget} experiments ---")
        print(f"{'policy':>28} {'operating value':>16} {'envelope width':>15} "
              f"{'failed runs':>12}")
        print("-" * 74)
        base = None
        for name, pol in POLICIES:
            v, w, f = evaluate_commitment_policy(pol, budget=budget)
            if base is None:
                base = v
            rel = "" if name.startswith("greedy") else \
                f"  ({100*(v-base)/abs(base):+.1f}%)"
            print(f"{name:>28} {v:16.3f}{rel:<10} {w:15.2f} {f:12.2f}")
        print()

    print("\nWHY GREEDY FAILS HERE\n")
    env = TerminalCommitment(seed=0)
    print(f"  yields      = {np.round(env.yield_at, 3)}")
    print(f"  fail probs  = {np.round(env.fail_prob, 3)}")
    print(f"  nominal     = index {env.centre}")
    print()
    print("  Greedy spends the budget on the highest-yield settings wherever they")
    print("  sit. But an envelope is an INTERVAL containing the nominal point, so")
    print("  a demonstrated setting far from centre is worthless unless everything")
    print("  between it and centre is also demonstrated. Greedy buys yield it can")
    print("  never claim.")
    print()
    print("  Edge-first buys width, but pays the highest failure probability per")
    print("  experiment, and a failure BLOCKS that setting permanently -- so an")
    print("  aggressive attempt to widen the envelope can narrow it instead.")
