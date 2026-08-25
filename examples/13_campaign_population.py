"""A population of LLM-driven campaign-strategist agents sharing one
constituency's persuadable-voter segment -- the "campaign-persuasion"
scenario (attrition.scenarios.SCENARIOS), using the real margin-anchored
parameters from examples/12_real_persuasion.py rather than the engine's
synthetic default, so this is Phase 1+2 of the election track combined:
real data (attrition.election), the persuasion-commons theorem
(derive_campaign_parameters), and now genuine learning agents (Population)
choosing under persona-conditioned dispositions, not fixed Greedy/ECI
policies -- exactly mirroring how examples/04_pharma_population.py closes
the same loop for AMR.

No API key required: this uses MockLLMClient, a deterministic offline
stand-in, by default. To use a real model, wrap your provider's SDK in a
function and pass `CallableLLMClient(that_function)` to
`Population.from_personas` instead -- see attrition/llm.py's docstring for
the exact shape.

Run:  python examples/13_campaign_population.py [constituency name]
"""
import sys

from attrition import (CAMPAIGN_PERSONAS, Population, MockLLMClient,
                       ConsumableBandit, compare_population_to_baselines,
                       simulate_population_simultaneous, SimultaneousPool,
                       planner_value_simultaneous, decentralised_value_simultaneous,
                       Greedy, ECI)
from attrition.election import (derive_persuasion_parameters_from_constituency,
                                real_margin_pct, build_constituency)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Ahmedabad East"
    constituency = build_constituency(name)
    v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(name)
    margin = real_margin_pct(constituency)
    print(__doc__)
    print(f"Constituency: {constituency.name} ({constituency.state}), "
          f"real margin={margin:.2f}pp -> S0={s0:.3f}")
    print("Contact-intensity choices:                ",
          [f"{x:.2f}" for x in intensity])
    print("Derived v (immediate persuasion yield):    ",
          [f"{x:.3f}" for x in v])
    print("Derived p (chance of saturating a segment): ",
          [f"{x:.3f}" for x in p])
    print("Derived e (permanent yield lost once saturated):",
          [f"{x:.3f}" for x in e])
    print()

    campaigns = [CAMPAIGN_PERSONAS["campaign-disciplined"],
                CAMPAIGN_PERSONAS["campaign-balanced"],
                CAMPAIGN_PERSONAS["campaign-maximalist"]]
    population = Population.from_personas(campaigns, client=MockLLMClient(seed=0))
    delta, horizon, m = 0.35, 9, len(campaigns)

    print(f"Turn-taking ({m} campaigns sharing one contact budget of "
          f"{horizon} pulls -- Theorem 6 says this has no genuine price of "
          f"anarchy, so it's a fair budget-matched comparison to a single "
          f"policy):\n")
    result = compare_population_to_baselines(
        lambda s: ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=s),
        population, [Greedy(), ECI()], seeds=25)
    for name_, stats in result.items():
        print(f"  {name_:>12}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")
    print()

    print(f"Simultaneous action ({m} campaigns committing at once each of "
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
                "collided on the same intensity level often enough this "
                "run that coordination failure (COLLISION, see "
                "simultaneous.py), not just under-priced externality, is "
                "doing real damage")
    elif pop_value < eci_dec:
        note = ("between decentralised greedy and decentralised eci, as "
                "expected: persona risk_tolerance is buying back some -- "
                "but not all -- of what independent myopic campaigning loses")
    else:
        note = ("at or above decentralised eci: this population's "
                "discipline-weighted personas out-coordinated even the "
                "externality-corrected independent baseline")
    print(f"This run landed {note}.")
    print("Re-run with a different constituency name, persona mix, or "
          "MockLLMClient(seed=...) to see the range this can cover -- the "
          "same isomorphism as examples/04, now over a real constituency's "
          "real-margin-anchored persuadable pool instead of an antibiotic "
          "pool.")


if __name__ == "__main__":
    main()
