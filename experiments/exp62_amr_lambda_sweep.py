"""Experiment 62 -- does a lambda-weighted community reward close the
regret-harm divergence once the antibiotic prescriber is decentralised?

Context (see paper/ discussion and the domain-extension proposal): an
independently developed simulator, abx_amr_simulator (Lee & Blumberg, 2026,
arXiv:2603.11369), fixes the antimicrobial-resistance externality with a
single dial on a single centralised policy:

    R_overall = (1 - lambda) * individual_reward + lambda * community_reward

Turning lambda up is supposed to make the policy internalise the harm it
does. But their model has exactly one decision-maker -- there is nothing to
coordinate. This repo's Theorem 6 (see THEORY.md) says that the moment you
decentralise into m simultaneously-acting agents, a genuine price of anarchy
appears from collision and blindness (see simultaneous.py's module
docstring) that reward-shaping on each agent's own objective cannot remove,
because the failure is coordination, not misaligned incentive.

This experiment transplants their lambda dial into attrition.simultaneous's
genuinely multi-agent setting (`rule="lambda_weighted"`) and asks the sharp
question directly: does dialing lambda toward the community term ever close
the harm gap once the prescriber is decentralised -- or does it shrink and
then hit a floor set by coordination failure alone?

Two numbers are tracked per cell, not one:
  reported value  -- what each agent's own dashboard would show (the score
                     it computed for the arm it chose; blind to the burden
                     term, see BLINDNESS in simultaneous.py)
  true system value -- what actually happened to the shared resistance pool
                     (realised reward, including the burden every agent's
                     score ignores)
The gap between them is the regret-harm divergence. Its floor as lambda -> 1
is the number that matters, not "ECI beats greedy" -- see the domain-
extension proposal's warning against making the wrong number the headline.

This is Layer 2 (ATTRITION translation) + Layer 3 (the multi-policy
comparison) of the planned AMR benchmark. Layer 1 (an independent
reproduction of abx_amr_simulator's own environment and a trained RL
baseline from it) is explicitly out of scope for this pass -- it requires
cloning and training their Stable-Baselines3 pipeline, which is real
additional cost, not free, and is left as follow-up work rather than faked.
"""
import numpy as np

from attrition.engines import derive_antibiotic_parameters
from attrition.simultaneous import (planner_value_simultaneous,
                                    decentralised_value_simultaneous)

# Same antibiotic-choice arms, budget, and prescriber count as
# examples/04_pharma_population.py, so this result is directly comparable to
# the existing pharma population demo rather than a fresh, incomparable setup.
DELTA, HORIZON, M = 0.35, 9, 3
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0]   # multiples of std(kappa)


def sweep():
    v, p, e, spectrum = derive_antibiotic_parameters()
    planner = planner_value_simultaneous(v, p, e, DELTA, HORIZON, M)

    # sanity check: lambda_weighted(lam=1, noise=0) must equal the existing
    # "eci" rule exactly -- both score v - delta*kappa*(T-t).
    eci = decentralised_value_simultaneous(v, p, e, DELTA, HORIZON, M,
                                           rule="eci")
    lam1 = decentralised_value_simultaneous(v, p, e, DELTA, HORIZON, M,
                                            rule="lambda_weighted", lam=1.0,
                                            obs_noise=0.0)
    assert abs(eci - lam1) < 1e-9, (
        f"lambda_weighted(lam=1) should equal eci exactly: {lam1} != {eci}")

    # collision-only floor: fully informed, lambda=0 (pure greedy) --
    # isolates coordination failure from any pricing effect at all.
    floor = decentralised_value_simultaneous(v, p, e, DELTA, HORIZON, M,
                                             rule="lambda_weighted", lam=0.0,
                                             obs_noise=0.0,
                                             return_details=True)

    results = {}
    for noise in NOISE_LEVELS:
        for lam in LAMBDAS:
            d = decentralised_value_simultaneous(
                v, p, e, DELTA, HORIZON, M, rule="lambda_weighted",
                lam=lam, obs_noise=noise, return_details=True, seeds=800)
            results[(noise, lam)] = d
    return planner, eci, floor, results, spectrum


def main():
    print(__doc__)
    planner, eci, floor, results, spectrum = sweep()
    print(f"antibiotic spectrum (narrow -> broad): "
          f"{[f'{s:.2f}' for s in spectrum]}")
    print(f"m={M} prescribers, horizon={HORIZON}, delta={DELTA}\n")
    print(f"planner (system optimum):            {planner:8.3f}")
    print(f"eci rule (existing baseline):         {eci:8.3f}")
    print(f"collision-only floor (lam=0, noise=0): "
          f"system={floor['system_value']:8.3f}  "
          f"reported={floor['reported_value']:8.3f}  "
          f"gap={floor['gap']:8.3f}\n")

    header = (f"{'noise':>6} {'lambda':>7} {'system_value':>13} "
              f"{'reported':>10} {'gap':>8} {'PoA (planner/system)':>22}")
    print(header)
    print("-" * len(header))
    for noise in NOISE_LEVELS:
        for lam in LAMBDAS:
            d = results[(noise, lam)]
            poa = planner / max(d["system_value"], 1e-9)
            print(f"{noise:6.1f} {lam:7.2f} {d['system_value']:13.3f} "
                  f"{d['reported_value']:10.3f} {d['gap']:8.3f} {poa:22.3f}")
        print()

    print("Reading the table:")
    print("  - within each noise block, system_value should rise toward the")
    print("    planner ceiling as lambda -> 1: reward-shaping genuinely helps.")
    print("  - the gap (reported - system) should shrink as lambda -> 1, but")
    print("    the question is whether it reaches ~0 or flattens above it.")
    print("  - if it flattens, that residual is the coordination cost this")
    print("    single-policy dial structurally cannot remove, because their")
    print("    model has no second agent to collide with.")
    print("  - rising noise should raise the floor further: partial")
    print("    observability of the community term makes even a")
    print("    fully-community-weighted agent misprice its own action.")


if __name__ == "__main__":
    main()
