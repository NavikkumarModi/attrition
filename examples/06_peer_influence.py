"""Same antibiotic-stewardship population as examples/04, but comparing two
observation regimes: prescribers who see nothing about each other (as in
examples/04) versus prescribers who can see what their peers chose last
round. The bandit itself -- value, destruction, externality -- is completely
unchanged between the two runs; only what each agent's prompt contains
differs.

MockLLMClient's peer-conformity term (`conformity=`) is a deliberate,
documented modeling choice for this offline stand-in -- a simple herding
rule that lets the peer-visibility plumbing be exercised end to end. It is
not a claim about how a real language model would respond to seeing its
peers' choices; that would need a real backend (see attrition/llm.py).

Run:  python examples/06_peer_influence.py
"""
from attrition import (AgentGraph, ConsumableBandit, MockLLMClient,
                       PHARMA_PERSONAS, Persona, Population, SimultaneousPool,
                       render_dashboard, simulate_population_simultaneous,
                       derive_antibiotic_parameters)


def main():
    print(__doc__)
    v, p, e, spectrum = derive_antibiotic_parameters()
    delta, horizon = 0.35, 9

    prescribers = [PHARMA_PERSONAS["dr-conservative"],
                   PHARMA_PERSONAS["dr-balanced"],
                   PHARMA_PERSONAS["dr-aggressive"],
                   PHARMA_PERSONAS["pharmacist-formulary"]]
    m = len(prescribers)
    agent_ids = [p_.name for p_ in prescribers]

    print(f"No peer visibility ({m} prescribers, no information about each "
          f"other -- same as examples/04):\n")
    pool_isolated = SimultaneousPool(v, p, e, delta=delta, horizon=horizon,
                                     n_agents=m, seed=0)
    population_isolated = Population.from_personas(prescribers,
                                                    client=MockLLMClient(seed=0))
    isolated = simulate_population_simultaneous(pool_isolated, population_isolated,
                                                rounds=horizon)
    print(f"  system_value={isolated['system_value']:.3f}  "
          f"system_regret={isolated['system_regret']:.3f}")
    print()

    print(f"With peer visibility (same {m} prescribers, full visibility "
          f"graph, MockLLMClient conformity=0.3):\n")
    graph = AgentGraph.complete(agent_ids)
    pool_networked = SimultaneousPool(v, p, e, delta=delta, horizon=horizon,
                                      n_agents=m, seed=0)
    population_networked = Population.from_personas(
        prescribers, client=MockLLMClient(seed=0, conformity=0.3))
    networked = simulate_population_simultaneous(pool_networked, population_networked,
                                                 rounds=horizon, graph=graph,
                                                 trace_store=None)
    print(f"  system_value={networked['system_value']:.3f}  "
          f"system_regret={networked['system_regret']:.3f}")
    print()

    delta_value = networked["system_value"] - isolated["system_value"]
    direction = "lower" if delta_value < 0 else "higher"
    print(f"Peer visibility produced {direction} system value "
          f"({delta_value:+.3f}) on this run. Herding onto the majority "
          f"choice can go either way here: it can accelerate convergence "
          f"onto the broad-spectrum arm (worse -- faster resistance), or it "
          f"can amplify whichever choice the risk-averse personas already "
          f"favoured (better). Re-run with different personas, conformity, "
          f"or graph shape (AgentGraph.ring/.star/.random) to see the range.")

    dash_path = render_dashboard(networked["trace"],
                                 path="/tmp/attrition_peer_influence.html",
                                 title="Peer-influence run", graph=graph)
    print(f"\nDashboard for the networked run written to {dash_path} "
          f"-- open it in a browser to see the network panel alongside the "
          f"value/burden curves.")


if __name__ == "__main__":
    main()
