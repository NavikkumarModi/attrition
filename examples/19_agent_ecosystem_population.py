"""A population of LLM-driven tool-routing agents sharing one API quota
pool -- the `agent-ecosystem` world (`shared-quota`/`shared-quota-competing`
in `attrition.scenarios.SCENARIOS`), built the same way
`04_pharma_population.py` builds the AMR world: real personas,
`Population.from_personas`, `simulate_population_simultaneous`. Same engine,
same theorems, a different domain's labels on the same three arrays.

No API key required: this uses MockLLMClient, a deterministic offline stand-in,
by default. To use a real model, wrap your provider's SDK in a function and
pass `CallableLLMClient(that_function)` to `Population.from_personas` instead
-- see attrition/llm.py's docstring for the exact shape.

Run:  python examples/19_agent_ecosystem_population.py
"""
from attrition import (agent_tools, AGENT_PERSONAS, Population, MockLLMClient,
                       ConsumableBandit, compare_population_to_baselines,
                       simulate_population_simultaneous, SimultaneousPool,
                       planner_value_simultaneous, decentralised_value_simultaneous,
                       Greedy, ECI)


def main():
    env = agent_tools()
    v, p, e = env.v, env.p, env.e
    print(__doc__)
    print("Tools (roster order):                  ", env.labels)
    print("Derived v (task success rate):         ", [f"{x:.3f}" for x in v])
    print("Derived p (chance of tripping a limit/quota/key revocation):",
          [f"{x:.3f}" for x in p])
    print("Derived e (shared upstream infrastructure burned if it trips):",
          [f"{x:.3f}" for x in e])
    print()

    agents = [AGENT_PERSONAS["agent-conservative"],
              AGENT_PERSONAS["agent-balanced"],
              AGENT_PERSONAS["agent-aggressive"]]
    population = Population.from_personas(agents, client=MockLLMClient(seed=0))
    delta, horizon, m = 0.06, 40, len(agents)

    print(f"Turn-taking ({m} agents sharing one session's call budget of "
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

    print(f"Simultaneous action ({m} agents committing at once each of "
          f"{horizon} rounds -- e.g. concurrent sub-agents/sessions -- the "
          f"setting where a genuine price of anarchy exists):\n")
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
                "collided on the same tool often enough this run that "
                "coordination failure (COLLISION, see simultaneous.py), not "
                "just under-priced externality, is doing real damage")
    elif pop_value < eci_dec:
        note = ("between decentralised greedy and decentralised eci, as "
                "expected: persona risk_tolerance is buying back some -- "
                "but not all -- of what independent myopic tool routing loses")
    else:
        note = ("at or above decentralised eci: this population's "
                "risk-averse personas out-coordinated even the "
                "externality-corrected independent baseline")
    print(f"This run landed {note}.")
    print("Re-run with a different persona mix or MockLLMClient(seed=...) to "
          "see the range this can cover -- the same emergent no-regret-is-"
          "not-no-harm result already shown for AMR (examples/04) and "
          "elections (examples/13), now for the agent-ecosystem world.")


if __name__ == "__main__":
    main()
