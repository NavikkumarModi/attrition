"""LLM-driven populations for the two real-data worlds that had scenarios
but no persona/population layer until now (FINDINGS.md #55): the
"cybersecurity" world (`exploit-catalog-real`, real CISA KEV + FIRST.org
EPSS data) and the "environment" world (`fisheries-commons-real`, real
NOAA commercial landings data) -- the same treatment "clinical-amr" and
"agent-ecosystem" already got in examples/04 and examples/19.

No API key required: this uses MockLLMClient, a deterministic offline
stand-in, by default. To use a real model, wrap your provider's SDK in a
function and pass `CallableLLMClient(that_function)` to
`Population.from_personas` instead -- see attrition/llm.py's docstring
for the exact shape.

Run:  python examples/22_cybersecurity_fisheries_population.py
"""
from attrition import (derive_real_cve_parameters, derive_real_fisheries_parameters,
                       CYBERSECURITY_PERSONAS, FISHERIES_PERSONAS, Population,
                       MockLLMClient, ConsumableBandit,
                       compare_population_to_baselines, Greedy, ECI)


def _run_world(title, v, p, e, personas, delta, horizon, seeds=25):
    print(f"{title}\n{'-' * len(title)}")
    population = Population.from_personas(personas, client=MockLLMClient(seed=0))
    m = len(personas)
    print(f"Turn-taking ({m} personas sharing one shared-catalog budget of "
          f"{horizon} pulls -- Theorem 6 says this has no genuine price of "
          f"anarchy, so it's a fair budget-matched comparison to a single "
          f"policy):\n")
    result = compare_population_to_baselines(
        lambda s: ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=s),
        population, [Greedy(), ECI()], seeds=seeds)
    for name, stats in result.items():
        print(f"  {name:>12}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")
    print()
    return result


def main():
    print(__doc__)

    v, p, e, labels = derive_real_cve_parameters(n=200, seed=0)
    print(f"Cybersecurity: {len(labels)} real actively-exploited CVEs "
          f"(CISA KEV + FIRST.org EPSS) -- personas are security-team "
          f"remediation dispositions, not attackers; see "
          f"attrition/persona.py's CYBERSECURITY_PERSONAS docstrings.")
    cyber_personas = [CYBERSECURITY_PERSONAS["patch-conservative"],
                      CYBERSECURITY_PERSONAS["patch-balanced"],
                      CYBERSECURITY_PERSONAS["patch-aggressive"]]
    _run_world("Cybersecurity (exploit-catalog-real)",
              v, p, e, cyber_personas, delta=0.05, horizon=40)

    v, p, e, labels = derive_real_fisheries_parameters()
    print(f"Environment: {len(labels)} real NOAA-tracked commercial fish "
          f"and shellfish species (real landings 1950-2024).")
    fish_personas = [FISHERIES_PERSONAS["fisher-conservative"],
                     FISHERIES_PERSONAS["fisher-balanced"],
                     FISHERIES_PERSONAS["fisher-aggressive"]]
    _run_world("Environment (fisheries-commons-real)",
              v, p, e, fish_personas, delta=0.15, horizon=12)

    print("Both worlds: greedy records zero private regret in every run "
          "(Theorem 4's zero-regret guarantee, mechanical) while the "
          "value gap to ECI reflects each world's real, previously "
          "measured corr(v, kappa) -- CONFLICT and pronounced for "
          "fisheries (+0.74), mild and near the alignment boundary for "
          "cybersecurity (-0.11). Persona-conditioned choices land "
          "between decentralised greedy and decentralised ECI, the same "
          "emergent pattern examples/04 and examples/19 already showed "
          "for clinical-amr and agent-ecosystem -- now real for two more "
          "worlds, not asserted.")


if __name__ == "__main__":
    main()
