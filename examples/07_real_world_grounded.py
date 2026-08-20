"""Three scenarios grounded in real, public, currently-maintained datasets --
not this repo's own simulations. See attrition.real_data.SOURCES for exact
citations and, importantly, which quantities are actually measured versus a
documented modeling proxy (only one of v/p/e is real in each case; the other
two are stated as proxies, not silently presented as equally real).

    antibiotic-stewardship-real   WHO GLASS-fed resistance surveillance,
                                  ~1,005 real country/year rows
    design-space-real             openFDA drug-recall data, 9 real failure
                                  categories across 15,556 recall records
    fisheries-commons-real        NOAA FOSS commercial landings, 16 real
                                  species across 1,143 species-year rows
    exploit-catalog-real          CISA KEV catalog joined with FIRST.org
                                  EPSS scores, 1,671 real actively-exploited
                                  CVEs
    agent_tools / shared-quota    real 2025-2026 agentic-AI survey data
                                  (Postman) and production rate-limit
                                  telemetry (Datadog) cited as supporting
                                  evidence for the mechanism -- NOT used to
                                  change any number, since both measure a
                                  different quantity than this roster's p
                                  (see domains.DOMAIN_NOTES["agent_tools"])

Run:  python examples/07_real_world_grounded.py
"""
import numpy as np

from attrition import (ConsumableBandit, Greedy, ECI, DOMAIN_NOTES,
                       PHARMA_PERSONAS, Population, MockLLMClient,
                       SimultaneousPool, SOURCES, compare,
                       derive_real_amr_parameters, derive_real_cmc_parameters,
                       derive_real_fisheries_parameters, derive_real_cve_parameters,
                       describe, simulate_population_simultaneous)


def _describe_gap(v, kappa, result):
    corr = float(np.corrcoef(v, kappa)[0, 1])
    gv, ev = result["greedy"]["value"], result["eci"]["value"]
    gap = (gv - ev) / max(abs(ev), 1e-9)
    if corr < -0.3:
        note = (f"corr(v, kappa) = {corr:+.3f}: the ALIGNED regime "
                f"(see platform-trial/aligned-control) -- greedy and ECI "
                f"should be close, and this run's value gap is "
                f"{gap:+.1%}.")
    elif corr > 0.3:
        note = (f"corr(v, kappa) = {corr:+.3f}: the CONFLICT regime -- "
                f"greedy should lose system value despite zero regret, "
                f"and this run's value gap is {gap:+.1%}.")
    else:
        note = (f"corr(v, kappa) = {corr:+.3f}: no strong alignment either "
                f"way; this run's value gap is {gap:+.1%}.")
    return note


def main():
    print(__doc__)

    print("=" * 70)
    print("Real WHO resistance rows (a few, for a sanity check):")
    v, p, e, labels = derive_real_amr_parameters(indicator="both", n=5, seed=0)
    for label, pi in zip(labels, p):
        print(f"  {label:>12}  measured resistance = {pi:.1%}")
    print(f"\n{SOURCES['who_amr']['fit']}\n")

    print("=" * 70)
    describe("antibiotic-stewardship-real")
    v_amr, p_amr, e_amr, _ = derive_real_amr_parameters(indicator="both", n=200, seed=0)
    env_amr = lambda seed: ConsumableBandit(v_amr, p_amr, e_amr, delta=0.05,
                                            horizon=40, seed=seed)
    result = compare(env_amr, [Greedy(), ECI()], seeds=15)
    print("n=200 real (country, year) arms, Greedy vs ECI, 15 seeds:")
    for name, stats in result.items():
        print(f"  {name:>8}  value={stats['value']:7.3f}  regret={stats['regret']:7.3f}")
    print(f"  {_describe_gap(v_amr, p_amr * e_amr, result)}")

    print("\nA small LLM population (5 agents) on a smaller real slice (n=50), "
          "simultaneous action:")
    v50, p50, e50, _ = derive_real_amr_parameters(indicator="both", n=50, seed=1)
    prescribers = [PHARMA_PERSONAS["dr-conservative"], PHARMA_PERSONAS["dr-balanced"],
                   PHARMA_PERSONAS["dr-aggressive"], PHARMA_PERSONAS["pharmacist-formulary"],
                   PHARMA_PERSONAS["patient-risk-averse"]]
    population = Population.from_personas(prescribers, client=MockLLMClient(seed=0))
    pool = SimultaneousPool(v50, p50, e50, delta=0.05, horizon=12,
                            n_agents=len(prescribers), seed=0)
    pop_result = simulate_population_simultaneous(pool, population, rounds=12)
    print(f"  system_value={pop_result['system_value']:.3f}  "
          f"system_regret={pop_result['system_regret']:.3f}")

    print("\n" + "=" * 70)
    describe("design-space-real")
    v_cmc, p_cmc, e_cmc, labels_cmc = derive_real_cmc_parameters()
    print("Real FDA failure categories:")
    for label, pi, ei in zip(labels_cmc, p_cmc, e_cmc):
        print(f"  {label:>12}  p(Class I)={pi:.3f}  severity_e={ei:.3f}  "
              f"kappa={pi*ei:.3f}")
    env_cmc = lambda seed: ConsumableBandit(
        v_cmc, p_cmc, e_cmc, delta=0.6, horizon=8, seed=seed)
    result_cmc = compare(env_cmc, [Greedy(), ECI()], seeds=200)
    print("Greedy vs ECI, 200 seeds:")
    for name, stats in result_cmc.items():
        print(f"  {name:>8}  value={stats['value']:7.3f} (se {stats['value_se']:.3f})  "
              f"regret={stats['regret']:7.3f}")
    print(f"  {_describe_gap(v_cmc, p_cmc * e_cmc, result_cmc)}")
    print("  (200 seeds, not 25, because the real gap here is small enough "
          "that a low seed count is dominated by Monte-Carlo noise -- see "
          "this scenario's expected_behaviour via describe() above.)")
    print(f"\n{SOURCES['fda_cmc']['fit']}\n")

    print("=" * 70)
    describe("fisheries-commons-real")
    v_fish, p_fish, e_fish, labels_fish = derive_real_fisheries_parameters()
    print("Real NOAA commercial landings, 16 species:")
    for label, pi, ei in zip(labels_fish, p_fish, e_fish):
        print(f"  {label:>20}  decline_freq={pi:.2f}  value_scale_e={ei:.2f}  "
              f"kappa={pi*ei:.3f}")
    env_fish = lambda seed: ConsumableBandit(
        v_fish, p_fish, e_fish, delta=0.15, horizon=12, seed=seed)
    result_fish = compare(env_fish, [Greedy(), ECI()], seeds=200)
    print("Greedy vs ECI, 200 seeds:")
    for name, stats in result_fish.items():
        print(f"  {name:>8}  value={stats['value']:7.3f} (se {stats['value_se']:.3f})  "
              f"regret={stats['regret']:7.3f}")
    print(f"  {_describe_gap(v_fish, p_fish * e_fish, result_fish)}")
    print(f"\n{SOURCES['noaa_fisheries']['fit']}\n")

    print("=" * 70)
    describe("exploit-catalog-real")
    v_cve, p_cve, e_cve, labels_cve = derive_real_cve_parameters(n=200, seed=0)
    print("A few real actively-exploited CVEs (n=200 subsample), for a sanity check:")
    for label, pi in list(zip(labels_cve, p_cve))[:5]:
        print(f"  {label:>16}  EPSS (30-day exploit probability) = {pi:.1%}")
    env_cve = lambda seed: ConsumableBandit(
        v_cve, p_cve, e_cve, delta=0.05, horizon=40, seed=seed)
    result_cve = compare(env_cve, [Greedy(), ECI()], seeds=200)
    print("Greedy vs ECI, 200 seeds:")
    for name, stats in result_cve.items():
        print(f"  {name:>8}  value={stats['value']:7.3f} (se {stats['value_se']:.3f})  "
              f"regret={stats['regret']:7.3f}")
    print(f"  {_describe_gap(v_cve, p_cve * e_cve, result_cve)}")
    print(f"\n{SOURCES['cisa_kev_epss']['fit']}\n")

    print("=" * 70)
    print("agent_tools / shared-quota: no numbers changed here -- see the "
          "citation and why it wasn't used to recalibrate anything:\n")
    print(DOMAIN_NOTES["agent_tools"])


if __name__ == "__main__":
    main()
