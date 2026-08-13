"""Run every named scenario and see which ones the phenomenon appears in --
and which it doesn't. `platform-trial` and `aligned-control` are negative
controls: dispersed but the phenomenon doesn't occur there, and the theory says
exactly why (see attrition.scenarios.SCENARIOS for the documented mechanism).

Scenarios use the Gym-style API (attrition.load), distinct from the
ConsumableBandit + run() pair used in 01_quickstart.py -- both are part of the
package; this example is also a working demonstration of the Gym interface.

Run:  python examples/02_scenarios.py
"""
import numpy as np

from attrition import SCENARIOS, load, describe


def rollout(env, policy_fn, seed):
    obs, info = env.reset(seed)
    total_regret = 0.0
    while True:
        idx = env.valid_actions()
        if len(idx) == 0:
            break
        a = policy_fn(env, obs, idx)
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    return info["total_value"], info["total_regret"]


def greedy_policy(env, obs, idx):
    # Uses the TRUE v, matching the paper's assumption of known values -- this
    # is what gives greedy its exactly-zero-regret property. Using the running
    # estimate in column 1 of `obs` instead gives a more realistic, noisier
    # "estimated greedy" whose regret is not exactly zero; try it by swapping
    # env.v for obs[:, 1] to see the difference.
    return int(idx[np.argmax(env.v[idx])])


def eci_policy(env, obs, idx):
    rem = max(env.horizon - env.t, 1)
    score = env.v[idx] - env.delta * env.p[idx] * env.e[idx] * rem
    return int(idx[np.argmax(score)])


def main():
    describe()
    print("Running each single-agent scenario, 20 seeds:\n")
    print(f"{'scenario':>20} {'greedy value':>13} {'greedy regret':>14} "
          f"{'ECI value':>10}")
    print("-" * 62)
    for name, spec in sorted(SCENARIOS.items()):
        if spec["agents"] > 1:
            continue
        gv, gr, ev = [], [], []
        for s in range(20):
            v, r = rollout(load(name, seed=s, reveal_externality=True),
                           greedy_policy, s)
            gv.append(v); gr.append(r)
            v2, _ = rollout(load(name, seed=s, reveal_externality=True),
                            eci_policy, s)
            ev.append(v2)
        print(f"{name:>20} {np.mean(gv):13.3f} {np.mean(gr):14.8f} "
              f"{np.mean(ev):10.3f}")


if __name__ == "__main__":
    main()
