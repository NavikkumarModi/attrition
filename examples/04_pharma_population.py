"""A population of LLM-driven prescriber agents sharing one population's
antimicrobial susceptibility -- see the "antibiotic-stewardship" scenario in
attrition.scenarios.SCENARIOS.

No API key required: this uses MockLLMClient, a deterministic offline stand-in,
by default. To use a real model, wrap your provider's SDK in a function and
pass `CallableLLMClient(that_function)` to `Population.from_personas` instead
-- see attrition/llm.py's docstring for the exact shape.

Run:  python examples/04_pharma_population.py
"""
from attrition import (derive_antibiotic_parameters, PHARMA_PERSONAS, Population,
                       MockLLMClient, ConsumableBandit, compare_population_to_baselines,
                       simulate_population_simultaneous, SimultaneousPool,
                       planner_value_simultaneous, decentralised_value_simultaneous,
                       Greedy, ECI)


def main():
    v, p, e, spectrum = derive_antibiotic_parameters()
    print(__doc__)
    print("Antibiotic choices (spectrum breadth):", [f"{s:.2f}" for s in spectrum])
    print("Derived v (immediate clearance):       ", [f"{x:.3f}" for x in v])
    print("Derived p (chance of selecting resistance):",
          [f"{x:.3f}" for x in p])
    print("Derived e (permanent loss once resistant):",
          [f"{x:.3f}" for x in e])
    print()

    prescribers = [PHARMA_PERSONAS["dr-conservative"],
                   PHARMA_PERSONAS["dr-balanced"],
                   PHARMA_PERSONAS["dr-aggressive"]]
    population = Population.from_personas(prescribers, client=MockLLMClient(seed=0))
    delta, horizon, m = 0.35, 9, len(prescribers)

    print(f"Turn-taking ({m} prescribers sharing one course budget of "
          f"{horizon} pulls -- Theorem 6 says this has no genuine price of "
          f"anarchy, so it's a fair budget-matched comparison to a single "
          f"policy):\n")
    result = compare_population_to_baselines(
        lambda s: ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=s),
        population, [Greedy(), ECI()], seeds=25)
    for name, stats in result.items():
        print(f"  {name:>12}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")
    print()

    print(f"Simultaneous action ({m} prescribers committing at once each of "
          f"{horizon} rounds -- the setting where a genuine price of "
          f"anarchy exists):\n")
    pool = SimultaneousPool(v, p, e, delta=delta, horizon=horizon, n_agents=m, seed=0)
    sim = simulate_population_simultaneous(pool, population, rounds=horizon)
    planner = planner_value_simultaneous(v, p, e, delta, horizon, m)
    greedy_dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                                   rule="greedy")
    eci_dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                               rule="eci")
    print(f"  {'llm population':>21}  system_value={sim['system_value']:7.3f}  "
          f"system_regret={sim['system_regret']:7.3f}")
    print(f"  {'planner (optimum)':>21}  value={planner:7.3f}")
    print(f"  {'decentralised greedy':>21}  value={greedy_dec:7.3f}")
    print(f"  {'decentralised eci':>21}  value={eci_dec:7.3f}")
    print()
    pop_value = sim["system_value"]
    if pop_value < greedy_dec:
        note = ("below even decentralised greedy: the personas' choices "
                "collided on the same arm often enough this run that "
                "coordination failure (COLLISION, see simultaneous.py), not "
                "just under-priced externality, is doing real damage")
    elif pop_value < eci_dec:
        note = ("between decentralised greedy and decentralised eci, as "
                "expected: persona risk_tolerance is buying back some -- "
                "but not all -- of what independent myopic prescribing loses")
    else:
        note = ("at or above decentralised eci: this population's "
                "risk-averse personas out-coordinated even the "
                "externality-corrected independent baseline")
    print(f"This run landed {note}.")
    print("Re-run with a different persona mix or MockLLMClient(seed=...) to "
          "see the range this can cover -- that range is the point: no-"
          "regret-is-not-no-harm is now emergent from persona-conditioned "
          "decisions, not read off a closed-form policy.")


if __name__ == "__main__":
    main()
