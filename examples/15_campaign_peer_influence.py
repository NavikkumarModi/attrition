"""Campaign-to-campaign peer visibility, over a real, margin-anchored
constituency -- the same comparison examples/06_peer_influence.py runs for
prescribers, now for campaigns.

Scope, stated precisely: this is a *campaign* social network (do
campaigns see and herd toward each other's contact-intensity choices?),
using `attrition.network.AgentGraph`, already proven elsewhere in this
repo. It is NOT a *voter* social network (information/misinformation
propagating voter-to-voter through `attrition.election`'s synthetic
population) -- that is a materially different, larger, still-open piece of
the plan (see FINDINGS.md#31), not attempted here. Conflating the two would
overclaim what this script shows.

Run:  python examples/15_campaign_peer_influence.py [constituency name]
"""
import sys

from attrition import (AgentGraph, MockLLMClient, CAMPAIGN_PERSONAS,
                       Population, SimultaneousPool,
                       simulate_population_simultaneous)
from attrition.election import derive_persuasion_parameters_from_constituency


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Ahmedabad East"
    print(__doc__)
    v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(name)
    delta, horizon = 0.35, 9

    campaigns = [CAMPAIGN_PERSONAS["campaign-disciplined"],
                CAMPAIGN_PERSONAS["campaign-balanced"],
                CAMPAIGN_PERSONAS["campaign-maximalist"]]
    m = len(campaigns)
    agent_ids = [c.name for c in campaigns]

    print(f"Constituency: {name}, S0={s0:.3f} (real margin-anchored)\n")

    print(f"No peer visibility ({m} campaigns, no information about each "
          f"other -- same as examples/13):\n")
    pool_isolated = SimultaneousPool(v, p, e, delta=delta, horizon=horizon,
                                     n_agents=m, seed=0)
    population_isolated = Population.from_personas(campaigns,
                                                    client=MockLLMClient(seed=0))
    isolated = simulate_population_simultaneous(pool_isolated, population_isolated,
                                                rounds=horizon)
    print(f"  system_value={isolated['system_value']:.3f}  "
          f"system_regret={isolated['system_regret']:.3f}")
    print()

    print(f"With peer visibility (same {m} campaigns, full visibility "
          f"graph, MockLLMClient conformity=0.3):\n")
    graph = AgentGraph.complete(agent_ids)
    pool_networked = SimultaneousPool(v, p, e, delta=delta, horizon=horizon,
                                      n_agents=m, seed=0)
    population_networked = Population.from_personas(
        campaigns, client=MockLLMClient(seed=0, conformity=0.3))
    networked = simulate_population_simultaneous(pool_networked, population_networked,
                                                 rounds=horizon, graph=graph)
    print(f"  system_value={networked['system_value']:.3f}  "
          f"system_regret={networked['system_regret']:.3f}")
    print()

    delta_value = networked["system_value"] - isolated["system_value"]
    direction = "lower" if delta_value < 0 else "higher"
    print(f"Campaign-to-campaign peer visibility produced {direction} "
          f"system value ({delta_value:+.3f}) on this run. Herding onto "
          f"whichever intensity a rival campaign is seen using can go "
          f"either way: it can accelerate convergence onto the heaviest-"
          f"contact arm (worse -- faster saturation, echoing what real "
          f"campaigns describe as escalating 'contact wars'), or it can "
          f"amplify whichever choice the more disciplined personas already "
          f"favoured (better). Re-run with a different constituency name, "
          f"conformity, or graph shape (AgentGraph.ring/.star/.random) to "
          f"see the range -- and see the module docstring for what this "
          f"does NOT show (voter-to-voter propagation).")


if __name__ == "__main__":
    main()
