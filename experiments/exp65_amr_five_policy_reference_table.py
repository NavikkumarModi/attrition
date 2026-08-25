"""Experiment 65 -- the "unified five-policy table" FINDINGS.md#26 recorded
as still open: "Private/greedy, ECI, lambda-weighted, and the planner
appear together in sections 24.1/25, in this repo's engine. The trained RL
baseline appears only in sections 24.2/24.4, in their environment. No
single run has all five side by side on common ground."

This closes that item the only honest way available: it puts all five
side by side, but does NOT pretend they share an axis. They don't, and
forcing them onto one does not create common ground -- it hides the fact
that there isn't any. Read down each half separately; read across only for
the qualitative pattern each half independently supports (reward-shaping
helps a single decision-maker; decentralising erodes what it bought),
never for a numeric ratio between the halves.

WHY THE TWO HALVES ARE NOT COMPARABLE, stated once, prominently, rather
than left for a footnote:

  - Different environments: this repo's `derive_antibiotic_parameters()`
    (m=3 prescribers, horizon=9 pulls, one antibiotic choice per pull) vs.
    `abx_amr_simulator`'s ABXAMREnv (5 patients/step, 500 steps/episode,
    continuous AMR_LeakyBalloon dynamics).
  - Different value semantics: this repo's `system_value`/`decentralised_
    value_simultaneous` is a sum of per-pull realised rewards net of a
    shared destruction burden; their "total reward"/"mean training reward"
    is their own RL reward function's scale, accumulated over a materially
    longer, denser episode.
  - Different agent counts and coordination models: this repo's
    `SimultaneousPool` blindness (burden visible only from the *next*
    round) vs. their sequential `env0.step(); env1.step(); env2.step()`
    stepping (FINDINGS.md#24.4's stated caveat) -- both decentralised and
    non-coordinating, but not bit-for-bit the same information structure.

None of that makes the two halves worthless side by side -- it means the
right comparison is qualitative (does reward-shaping help; does
decentralising erode it), not "is 12.09 bigger than -244.07."

In-repo numbers here are recomputed live from this repo's own engine
(`experiments/exp62_amr_lambda_sweep.py`'s machinery); external numbers are
reproduced from FINDINGS.md sections 24.2 and 24.4 verbatim, cited, not
recomputed (recomputing them requires cloning and training
`abx_amr_simulator` in a separate Python 3.11 environment -- see
`scripts/amr_decentralize_checkpoint.py`'s header for exact reproduction
commands).
"""
from attrition.engines import derive_antibiotic_parameters
from attrition.simultaneous import (planner_value_simultaneous,
                                    decentralised_value_simultaneous)

DELTA, HORIZON, M = 0.35, 9, 3

# FINDINGS.md#24.2, lambda=1.0 row (siloed / single centralised policy).
EXTERNAL_SILOED = {
    "mean_training_reward": -244.07,
    "final_amr_a": 0.7461,
    "final_amr_b": 0.2506,
    "patient_clinical_benefit": -1850.1,
}
# FINDINGS.md#24.4, "shared (real externality)" row -- the same checkpoint,
# decentralised into m=3 non-coordinating copies sharing one resistance pool.
EXTERNAL_SHARED = {
    "total_reward_3_agents": -1151.15,
    "final_amr_a": 0.9691,
    "final_amr_b": 0.1993,
}


def in_repo_half():
    v, p, e, spectrum = derive_antibiotic_parameters()
    planner = planner_value_simultaneous(v, p, e, DELTA, HORIZON, M)
    greedy = decentralised_value_simultaneous(v, p, e, DELTA, HORIZON, M,
                                              rule="greedy")
    eci = decentralised_value_simultaneous(v, p, e, DELTA, HORIZON, M,
                                           rule="eci")
    lam1 = decentralised_value_simultaneous(v, p, e, DELTA, HORIZON, M,
                                            rule="lambda_weighted", lam=1.0,
                                            obs_noise=0.0)
    assert abs(lam1 - eci) < 1e-9  # exp62's own sanity check, restated here
    return {"planner": planner, "greedy": greedy, "eci": eci,
           "lambda_weighted(lam=1)": lam1}


def main():
    print(__doc__)
    repo = in_repo_half()

    print("HALF 1 -- this repo's engine (system value, decentralised, "
          f"m={M}, horizon={HORIZON}, delta={DELTA}):\n")
    for name, val in repo.items():
        print(f"  {name:>26}  {val:8.3f}")
    print(f"\n  price of anarchy (planner/greedy): "
          f"{repo['planner']/repo['greedy']:.3f}")
    print(f"  price of anarchy (planner/eci):    "
          f"{repo['planner']/repo['eci']:.3f}\n")

    print("HALF 2 -- abx_amr_simulator's own environment, their trained "
          "PPO checkpoint (lambda=1.0), cited from FINDINGS.md#24.2/24.4, "
          "not recomputed here:\n")
    print("  siloed (single centralised policy, FINDINGS.md#24.2):")
    for name, val in EXTERNAL_SILOED.items():
        print(f"    {name:>26}  {val:8.3f}")
    print("  shared (m=3 decentralised copies, FINDINGS.md#24.4):")
    for name, val in EXTERNAL_SHARED.items():
        print(f"    {name:>26}  {val:8.3f}")

    print("\nQUALITATIVE reading across both halves (not a numeric "
          "comparison -- see the module docstring for why one is not "
          "possible):")
    print(f"  - HALF 1: eci/lambda_weighted(lam=1) recovers "
          f"{repo['eci']/repo['planner']*100:.1f}% of the planner ceiling; "
          f"greedy recovers only {repo['greedy']/repo['planner']*100:.1f}%.")
    print(f"  - HALF 2: decentralising the trained checkpoint moves "
          f"antibiotic A's resistance from "
          f"{EXTERNAL_SILOED['final_amr_a']:.1%} (siloed) to "
          f"{EXTERNAL_SHARED['final_amr_a']:.1%} (shared) -- most of the "
          f"way back to the no-stewardship level (99.97%, FINDINGS.md#24.2, "
          f"lambda=0).")
    print("  - Both halves show the same qualitative shape: a "
          "single-decision-maker fix (reward-shaping / lambda dial) "
          "recovers real value alone, and decentralising into "
          "non-coordinating copies erodes a large fraction of that "
          "recovery. Neither halves' magnitude transfers to the other; "
          "the shape does.")


if __name__ == "__main__":
    main()
