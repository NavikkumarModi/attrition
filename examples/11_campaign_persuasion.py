"""Election track, Phase 1 (abstract): does "no-regret is not no-harm"
transfer to campaigns competing over one shared, finite persuadable-voter
segment -- with no real constituency data attached yet (see
attrition.election for that separate, real-data population layer, and
FINDINGS.md's election-track sections for why the two aren't connected in
this pass).

`derive_campaign_parameters()` (attrition.engines) reuses the same
LotkaVolterraTumour two-compartment competitive-release engine already
used for oncology dose selection and antibiotic-stewardship, relabelled:
a persuadable/undecided voter segment plays the drug-sensitive
compartment's role, a saturated/unreachable segment plays the resistant
compartment's role. The registered scenario is "campaign-persuasion" --
see attrition.scenarios for its full description.

This script demonstrates the same result three ways, reusing existing,
already-proven machinery end to end -- no new theorem-proving code was
needed to get a persuasion-commons result, only a new (v, p, e) derivation
and a scenario registration:

  1. Turn-taking: greedy vs ECI, one campaign at a time.
  2. Multi-agent, turn-taking: Theorem 6 says m campaigns sharing one
     segment, acting in sequence, behave exactly like one campaign over
     m times the contact budget -- no genuine price of anarchy here.
  3. Multi-agent, simultaneous: several campaigns committing contact
     intensity at once, where a genuine price of anarchy exists (collision,
     blindness -- see attrition.simultaneous's module docstring), plus the
     lambda_weighted comparison built for the AMR extension, which now
     applies to this domain for free because it was built against the
     generic simultaneous-action machinery, not anything AMR-specific.

Run:  python examples/11_campaign_persuasion.py
"""
from attrition import (ConsumableBandit, Greedy, ECI, compare,
                       derive_campaign_parameters,
                       SimultaneousPool, planner_value_simultaneous,
                       decentralised_value_simultaneous)
from attrition.scenarios import get_arrays, describe


def main():
    describe("campaign-persuasion")

    v, p, e, kw = get_arrays("campaign-persuasion")
    delta, horizon = kw["delta"], kw["horizon"]

    print("1. Turn-taking, one campaign (Theorem 1/4: zero-regret ruin)\n")
    result = compare(lambda s: ConsumableBandit(v, p, e, delta=delta,
                                                horizon=horizon, seed=s),
                     [Greedy(), ECI()], seeds=400)
    for name, stats in result.items():
        print(f"  {name:>8}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")
    gv, ev = result["greedy"]["value"], result["eci"]["value"]
    print(f"\n  greedy is zero-regret by construction and still clears "
          f"{100*(ev-gv)/ev:.0f}% less value than ECI -- the campaign\n"
          f"  that never looks at the externality it's creating pays for "
          f"it in real persuasion yield, not in its own scorecard.\n")

    m = 3
    print(f"2. Turn-taking, {m} campaigns sharing one segment "
          f"(Theorem 6: no genuine price of anarchy)\n")
    print("   (see examples/04_pharma_population.py for the worked "
          "multi-agent turn-taking case -- the mechanism is identical, "
          "only the labels differ)\n")

    print(f"3. Simultaneous, {m} campaigns committing at once "
          f"(genuine price of anarchy)\n")
    planner = planner_value_simultaneous(v, p, e, delta, horizon, m)
    print(f"  {'planner (optimum)':>20}  value={planner:7.3f}")
    for rule in ("greedy", "eci"):
        dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                               rule=rule)
        print(f"  {'decentralised ' + rule:>20}  value={dec:7.3f}")
    for lam in (0.25, 0.5, 0.75, 1.0):
        d = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                             rule="lambda_weighted", lam=lam,
                                             return_details=True)
        print(f"  {'lambda=' + str(lam):>20}  system={d['system_value']:7.3f}"
              f"  reported={d['reported_value']:7.3f}  gap={d['gap']:7.3f}")
    print("\n  Same shape as the AMR lambda-sweep (experiments/"
          "exp62_amr_lambda_sweep.py): reward-shaping toward the community "
          "term narrows\n  the gap between what a campaign's own scorecard "
          "reports and what actually happens to the\n  segment's "
          "persuadability, but -- check the numbers above -- does not "
          "close it, for the same\n  coordination-cost reason (Theorem 6's "
          "collision/blindness) established for antibiotics.\n\n"
          "  Note: these numbers are numerically identical to exp62's "
          "antibiotic ones. That's not a\n  coincidence to read meaning "
          "into -- derive_campaign_parameters() reuses the same default\n"
          "  engine_kwargs unchanged, so the underlying dynamics ARE the "
          "same simulation, only\n  relabelled. Pass different intensity/"
          "engine_kwargs to see the numbers actually move.")


if __name__ == "__main__":
    main()
