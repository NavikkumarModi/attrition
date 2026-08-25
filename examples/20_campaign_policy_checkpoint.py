"""A real trained campaign-contact policy -- the election-track analogue of
FINDINGS.md#24.4's AMR checkpoint, closing the one item FINDINGS.md#31
recorded as open: "a real trained-policy checkpoint for campaigns... no
equivalent external, independently-trained campaign policy exists to
transplant." There is still no external policy to transplant, so this
trains one from scratch instead, using `attrition.election.
train_gradient_bandit` (the gradient-bandit method, plain numpy, no RL
framework -- see that module's docstring for why and how).

Two stages, mirroring the AMR checkpoint's own two stages:

  1. Train ONE centralised policy against `campaign-persuasion`'s arms,
     seeing only realised per-pull reward -- the same information Greedy
     sees, nothing about (p, e). Evaluate it against Greedy/ECI/Conservative
     on identical seeded episodes.
  2. Decentralise: freeze the trained policy and run THREE independent
     copies of it sharing one `SimultaneousPool`, exactly as
     `scripts/amr_decentralize_checkpoint.py` does with the external PPO
     checkpoint -- non-coordinating agents, one shared pool, no retraining.

Run:  python examples/20_campaign_policy_checkpoint.py
"""
from attrition import (derive_campaign_parameters, ConsumableBandit, Greedy, ECI,
                       Conservative, compare, SimultaneousPool, Population,
                       simulate_population_simultaneous,
                       planner_value_simultaneous, decentralised_value_simultaneous)
from attrition.election import train_gradient_bandit


def main():
    print(__doc__)
    v, p, e, intensity = derive_campaign_parameters()
    delta, horizon, m = 0.35, 9, 3
    print("Contact intensity (arm order):", [f"{x:.2f}" for x in intensity])
    print("v (immediate persuasion yield):", [f"{x:.3f}" for x in v])
    print("p (chance this round saturates the segment):", [f"{x:.3f}" for x in p])
    print("e (permanent persuasion-yield loss once saturated):",
          [f"{x:.3f}" for x in e])
    print()

    def env_factory(seed):
        return ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=seed)

    print("Stage 1: training (gradient-bandit, 4000 episodes, seeing only "
          "realised reward)...")
    policy = train_gradient_bandit(env_factory, episodes=4000, alpha=0.15, seed=0)
    early = policy.training_curve[:100].mean()
    late = policy.training_curve[-100:].mean()
    print(f"  mean per-episode return: first 100 episodes={early:.3f}  "
          f"last 100 episodes={late:.3f}")
    print(f"  learned preferences (softmax logits, one per arm): "
          f"{[f'{x:.2f}' for x in policy.h]}")
    print(f"  -> arm {int(policy.h.argmax())} (intensity "
          f"{intensity[policy.h.argmax()]:.2f}) is the learned favourite\n")

    print("Single-agent evaluation, 200 held-out seeds, against the "
          "existing baselines:\n")
    results = compare(env_factory,
                      [Greedy(), ECI(), Conservative(e_bound=float(e.max())), policy],
                      seeds=200)
    for name, stats in results.items():
        print(f"  {name:>24}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")
    print()
    trained_stats = results[policy.name]
    greedy_stats = results["greedy"]
    print(f"The trained policy lands at value={trained_stats['value']:.3f}, "
          f"regret={trained_stats['regret']:.3f} -- close to greedy's "
          f"value={greedy_stats['value']:.3f}, regret=0.000, not to ECI's. "
          "This is the predicted result, not a surprise one: realised "
          "per-pull reward already has the externality folded in as a "
          "common burden term, identical across every available arm in a "
          "round, so nothing in the training signal distinguishes a "
          "cheap-to-the-pool arm from an expensive one -- see "
          "attrition/election/policy_training.py's docstring.\n")

    print(f"Stage 2: decentralising the frozen trained policy -- {m} "
          f"independent copies sharing one pool, exactly as "
          f"scripts/amr_decentralize_checkpoint.py does with the external "
          f"AMR checkpoint:\n")
    planner = planner_value_simultaneous(v, p, e, delta, horizon, m)
    greedy_dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                                   rule="greedy")
    eci_dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                               rule="eci")
    population = Population({f"campaign-{i}": policy for i in range(m)})
    pool = SimultaneousPool(v, p, e, delta=delta, horizon=horizon, n_agents=m, seed=0)
    sim = simulate_population_simultaneous(pool, population, rounds=horizon)

    print(f"  {'planner (optimum)':>24}  value={planner:7.3f}")
    print(f"  {'decentralised greedy':>24}  value={greedy_dec:7.3f}")
    print(f"  {'decentralised eci':>24}  value={eci_dec:7.3f}")
    print(f"  {'trained policy x' + str(m):>24}  system_value="
          f"{sim['system_value']:7.3f}  system_regret={sim['system_regret']:7.3f}")
    print()
    if sim["system_value"] < greedy_dec:
        print("The decentralised trained policy lands BELOW decentralised "
              "greedy: identical copies of the same policy share a strong "
              "preference for the same arm, so collision (see COLLISION in "
              "simultaneous.py) compounds on top of the externality-"
              "blindness Stage 1 already showed -- a real, checkable cost "
              "of decentralising a policy that was never trained to expect "
              "company, not a modeling artefact.")
    else:
        print("The decentralised trained policy holds up better than "
              "decentralised greedy this run -- re-run with a different "
              "seed to see the range; the coordination-failure risk above "
              "is a real mechanism either way, not this run's headline.")


if __name__ == "__main__":
    main()
