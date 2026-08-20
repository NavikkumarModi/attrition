"""Nothing in this repo has ever run the multi-agent machinery past m=6
agents (checked: grepped every experiment, test, and example for n_agents=/
m= and the largest is 6, for the exact-planner PoA baseline in
simultaneous.py, which is inherently exponential in m and can't go further).
The simulation code itself has no such limit -- SimultaneousPool.step() and
simulate_population_simultaneous's per-round loop are both O(m). This asks,
for real, whether the two central multi-agent claims still hold at agent
counts an order of magnitude past anything previously tested:

  1. Theorem 6 (sequential equivalence): turn-taking m agents over a shared
     T-pull budget should behave EXACTLY like one learner over the same T
     pulls, for any m. Checked here up to m=200.

  2. Simultaneous action's genuine price of anarchy: does per-agent value
     degrade, stay flat, or collapse as agent density (m relative to pool
     size n) grows? This has only ever been checked at m<=6 against an
     exact planner baseline; that baseline is exponential and can't scale,
     so past m=6 this reports the empirical decentralised value and
     collision rate directly, honestly, without a baseline to compare
     against, rather than silently dropping the check.

Real data (antibiotic-stewardship-real, n=200 arms), not synthetic.

Run:  python experiments/exp61_population_scale.py
"""

import time

import numpy as np

from attrition import ConsumableBandit, Greedy, Population, SimultaneousPool, get_arrays
from attrition.population import simulate_population, simulate_population_simultaneous


def make_population(m):
    return Population({f"agent-{i}": Greedy() for i in range(m)})


def theorem6_check(v, p, e, delta, t_per_agent, m_values, seed=0):
    print("=" * 70)
    print("Theorem 6 at scale: m agents sharing a T-pull turn-taking budget")
    print("vs. one learner over the same T pulls (same seed, same policy).")
    print(f"{'m':>5}  {'horizon':>8}  {'pop value':>10}  {'solo value':>10}  "
          f"{'match':>7}  {'time (s)':>9}")
    for m in m_values:
        horizon = m * t_per_agent
        t0 = time.time()
        population = make_population(m)
        env = ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=seed)
        pop_result = simulate_population(env, population, log=False)
        elapsed = time.time() - t0

        solo_env = ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=seed)
        solo_result_value, solo_result_regret = 0.0, 0.0
        solo = Greedy()
        while not solo_env.done():
            arm = solo.select(solo_env.state())
            _, val, reg, _ = solo_env.step(arm)
            solo_result_value += val
            solo_result_regret += reg

        match = np.isclose(pop_result["system_value"], solo_result_value, atol=1e-9)
        print(f"{m:>5}  {horizon:>8}  {pop_result['system_value']:>10.4f}  "
              f"{solo_result_value:>10.4f}  {str(match):>7}  {elapsed:>9.3f}")
    print()


def simultaneous_scale_check(v, p, e, delta, rounds, m_values, seed=0):
    print("=" * 70)
    print(f"Simultaneous action at scale, {rounds} rounds, {len(v)} real arms.")
    print("No exact planner past m=6 (exponential) -- empirical only, honestly.")
    print(f"{'m':>5}  {'system value':>12}  {'value/agent':>12}  "
          f"{'collisions':>10}  {'arms left':>9}  {'time (s)':>9}")
    for m in m_values:
        t0 = time.time()
        population = make_population(m)
        pool = SimultaneousPool(v, p, e, delta=delta, horizon=rounds,
                                n_agents=m, seed=seed)
        result = simulate_population_simultaneous(pool, population, rounds=rounds,
                                                   log=True)
        elapsed = time.time() - t0

        # collisions, from the real trace this run actually produced -- not
        # a separately reimplemented approximation that could silently
        # diverge from what the simulation itself did.
        by_round = {}
        for row in result["trace"]:
            by_round.setdefault(row["t"], []).append(row["arm"])
        collisions = sum(len(arms) - len(set(arms)) for arms in by_round.values())

        arms_left = int(pool.alive.sum())
        print(f"{m:>5}  {result['system_value']:>12.3f}  "
              f"{result['system_value'] / m:>12.4f}  {collisions:>10}  "
              f"{arms_left:>9}  {elapsed:>9.3f}")
    print()


def main():
    print(__doc__)
    v, p, e, kw = get_arrays("antibiotic-stewardship-real")

    theorem6_check(v, p, e, delta=0.05, t_per_agent=3,
                  m_values=[2, 5, 10, 25, 50, 100, 200])

    simultaneous_scale_check(v, p, e, delta=0.05, rounds=5,
                             m_values=[2, 5, 10, 25, 50, 100, 190])


if __name__ == "__main__":
    main()
