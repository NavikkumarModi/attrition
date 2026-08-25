"""Not runnable inside attrition's own environment -- this is a reproduction
script for FINDINGS.md#24.4, kept here for the record and for anyone who
wants to re-run it, not part of the attrition package or its dependency
tree. It needs, in a separate Python 3.11 environment (their pinned
torch==2.2.2 has no wheel for 3.13):

    git clone --depth 1 https://github.com/jl56923/abx_amr_simulator.git
    pip install -e abx_amr_simulator

...a `my_first_project/` scaffold as built in FINDINGS.md#24.2's
reproduction commands, and a trained `lambda1_*/checkpoints/best_model.zip`
in it (same commands). Then, from the directory containing both
`abx_amr_simulator/` and `my_first_project/`:

    python decentralize_checkpoint.py

Embeds the actual trained abx_amr_simulator checkpoint (lambda=1.0, the
stewardship-good policy from FINDINGS.md#24.2) as m=3 decentralised
prescribers sharing ONE resistance pool, using their own factory functions
and their own AMR_LeakyBalloon dynamics object -- not a re-implementation.

Two conditions, same trained policy, same obs shape, only one thing differs:

  SILOED    3 independent episodes, each prescriber has its OWN private
            pair of AMR balloons -- no externality between them.
  SHARED    3 prescribers, same trained policy, but their AMR balloons are
            the SAME object (injected via ABXAMREnv's own
            amr_dynamics_instances hook -- exactly the seam its docstring
            flags for "future multi-agent refactor").

The policy was trained single-agent, so it has no notion that other
prescribers are drawing on the same pool -- this is the direct empirical
answer to "what happens when a policy tuned to be good in isolation gets
decentralised," using their real checkpoint, not a proxy for it.

Caveat, stated up front: within a round the m=3 envs are stepped
sequentially (env0.step(), then env1.step(), then env2.step()), so a small
amount of within-round information leaks between them (env1's dose this
round is already reflected in the balloon env2 reads a moment later) --
this is not bit-for-bit the same BLINDNESS assumption
attrition.simultaneous.SimultaneousPool uses (burden visible only from the
next round). Decentralised and non-coordinating either way, but the
distinction is real and worth stating rather than glossing over.
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO = Path(__file__).parent / "abx_amr_simulator"
sys.path.insert(0, str(REPO / "src"))

from abx_amr_simulator.utils import load_config                # noqa: E402
from abx_amr_simulator.utils.factories import (                # noqa: E402
    create_reward_calculator, create_patient_generator,
    create_amr_dynamics)
from abx_amr_simulator.core.abx_amr_env import ABXAMREnv        # noqa: E402

PROJECT = Path(__file__).parent / "my_first_project"
CONFIG_PATH = PROJECT / "experiments/configs/umbrella_configs/base_experiment.yaml"
CHECKPOINT = sorted((PROJECT / "results").glob("lambda1_*/checkpoints/best_model.zip"))[-1]
N_AGENTS = 3
EPISODES_PER_CONDITION = 12


def _build_env(config, shared_amr_dynamics=None):
    """One ABXAMREnv, with its own reward calculator + patient generator, and
    either its own private AMR dynamics or a shared, injected dict."""
    import copy
    rc = create_reward_calculator(config)
    pg = create_patient_generator(config)
    env_kwargs = copy.deepcopy(config["environment"])
    env_kwargs.pop("patient_generator", None)
    env_kwargs.pop("visible_patient_attributes", None)
    amr_dynamics = shared_amr_dynamics if shared_amr_dynamics is not None \
        else create_amr_dynamics(config)
    if hasattr(pg, "bind_antibiotic_order"):
        pg.bind_antibiotic_order(antibiotic_names=list(rc.antibiotic_names))
    env_kwargs["reward_calculator"] = rc
    env_kwargs["patient_generator"] = pg
    env_kwargs["amr_dynamics_instances"] = amr_dynamics
    return ABXAMREnv(**env_kwargs), amr_dynamics


def run_condition(config, model, shared, seed0):
    """Run N_AGENTS decentralised copies of `model` for one full episode.
    `shared=True` -> one balloon dict injected into all N_AGENTS envs.
    `shared=False` -> each env gets its own private balloon dict.
    Returns (total_reward_across_agents, final_amr_levels_per_antibiotic)."""
    shared_dyn = create_amr_dynamics(config) if shared else None
    envs, dyns = [], []
    for i in range(N_AGENTS):
        env, dyn = _build_env(config, shared_amr_dynamics=shared_dyn)
        envs.append(env)
        dyns.append(dyn)

    obs = [env.reset(seed=seed0 + i)[0] for i, env in enumerate(envs)]
    total_reward = 0.0
    T = config["environment"]["max_time_steps"]
    for _t in range(T):
        for i, env in enumerate(envs):
            action, _ = model.predict(obs[i], deterministic=True)
            obs[i], reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)

    # dyns[i] is the same object across i when shared=True; dedupe.
    seen = {}
    for dyn in dyns:
        for abx, balloon in dyn.items():
            seen[abx] = balloon.get_volume()
    return total_reward, seen


def main():
    config = load_config(str(CONFIG_PATH))
    model = PPO.load(str(CHECKPOINT))
    print(f"loaded checkpoint: {CHECKPOINT}")
    print(f"m={N_AGENTS} decentralised prescribers, "
          f"{EPISODES_PER_CONDITION} episodes per condition\n")

    for label, shared in [("SILOED (private pools)", False),
                          ("SHARED (one pool, real externality)", True)]:
        rewards, finals_A, finals_B = [], [], []
        for ep in range(EPISODES_PER_CONDITION):
            tot, finals = run_condition(config, model, shared, seed0=1000 + ep * 10)
            rewards.append(tot)
            finals_A.append(finals["A"])
            finals_B.append(finals["B"])
        print(f"{label}")
        print(f"  total reward across {N_AGENTS} agents: "
              f"mean={np.mean(rewards):.2f} sd={np.std(rewards):.2f}")
        print(f"  final AMR level, antibiotic A: mean={np.mean(finals_A):.4f}")
        print(f"  final AMR level, antibiotic B: mean={np.mean(finals_B):.4f}")
        print()


if __name__ == "__main__":
    main()
