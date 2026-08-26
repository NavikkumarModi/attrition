"""Experiment 66 -- the cross-world benchmark named early in this
session's punch list and never attempted until now: "does the same
policy transfer across domains?"

Two separate, both real, questions.

(a) A unified table. Every world's own Greedy-vs-ECI value gap has been
reported somewhere in this repo before -- scattered across FINDINGS.md
entries and individual example scripts, never assembled side by side in
one run. `unified_value_gap_table` builds that table directly from
`attrition.scenarios.SCENARIOS`, one real-data-preferring scenario per
registered world (excluding "synthetic", which is stress cases and
negative controls, not a domain).

(b) Transfer, not just comparison -- the harder, more interesting
question. `Conservative` (Theorem 3's recommended default) works by
charging every arm a single upper-bound externality `e_bound`,
deliberately not trying to learn per-arm externalities (which Theorem 3
shows cannot be estimated to arbitrary precision from experience). The
practical recipe used everywhere else in this repo (`election_api.py`,
`app.py`) is `e_bound = e.max()` -- calibrated FROM THAT WORLD'S OWN
arms. That is still domain-specific knowledge, even though it is not
per-arm knowledge. Transferring that raw number verbatim across domains
whose v/e live on completely different scales (a CVE catalog's v is
~0.3-1.5; a fish-species catalog's v is a different, unrelated unit)
would be a meaningless, unit-mismatched comparison, not a real test of
anything.

The genuinely domain-general version of "no domain-specific knowledge
needed" is checked instead: calibrate the RATIO `e_bound / v.max()` once,
on a single home world, then reuse that SAME ratio -- not the raw
e_bound number -- to derive every other world's e_bound from its own
v.max(), with no further look at that world's own e distribution.
`transfer_check` reports whether that one-world-calibrated, transferred
ratio actually holds up out of domain, against the honest baseline of
what a LOCALLY-calibrated Conservative (`e_bound = e.max()`, using that
world's own real e distribution) achieves in the same world. Reported as
measured -- including if transfer turns out NOT to hold up in some
world, which would be a real finding about the limits of "no
domain-specific knowledge needed", not a result to bury.
"""

import numpy as np

from attrition import ConsumableBandit, compare, Greedy, ECI, Conservative
from attrition.scenarios import get_arrays

# One single-agent-compatible, real-data-preferring scenario per
# registered world (excluding "synthetic": stress cases and negative
# controls, not a domain). Multi-agent-registered scenarios
# (campaign-persuasion, shared-quota) are re-run here through the plain
# single-decision-maker ConsumableBandit/compare() machinery -- a
# deliberate, stated reframing for this cross-world comparison, not a
# claim about population coordination, which is a different question
# examples/04, /13, /19, /22 already cover for these same domains.
DOMAIN_SCENARIOS = {
    "clinical-amr": "antibiotic-stewardship-real",
    "election": "campaign-persuasion",
    "agent-ecosystem": "shared-quota",
    "environment": "fisheries-commons-real",
    "cybersecurity": "exploit-catalog-real",
}

SEEDS = 200


def _env_factory(v, p, e, delta, horizon):
    return lambda seed: ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=seed)


def unified_value_gap_table():
    """(a) Greedy vs ECI, side by side, across every world -- assembled
    from SCENARIOS, not re-typed from prior reports."""
    rows = []
    for world, name in DOMAIN_SCENARIOS.items():
        v, p, e, kw = get_arrays(name)
        env_factory = _env_factory(v, p, e, kw["delta"], kw["horizon"])
        result = compare(env_factory, [Greedy(), ECI()], seeds=SEEDS)
        greedy_val, eci_val = result["greedy"]["value"], result["eci"]["value"]
        gap_pct = (100 * (eci_val - greedy_val) / abs(eci_val)
                  if eci_val else float("nan"))
        rows.append({"world": world, "scenario": name, "n_arms": len(v),
                    "greedy": greedy_val, "eci": eci_val, "gap_pct": gap_pct})
    return rows


def transfer_check(home_world="clinical-amr"):
    """(b) Calibrate `e_bound / v.max()` on `home_world` only; transfer
    that ratio (never the raw e_bound) to every other world; compare the
    transferred Conservative's value recovery against a locally-
    calibrated Conservative's, in every world including the home one
    (where transferred == local, by construction, a sanity check)."""
    home_name = DOMAIN_SCENARIOS[home_world]
    hv, hp, he, hkw = get_arrays(home_name)
    ratio = float(he.max() / hv.max())

    rows = []
    for world, name in DOMAIN_SCENARIOS.items():
        v, p, e, kw = get_arrays(name)
        env_factory = _env_factory(v, p, e, kw["delta"], kw["horizon"])

        local = Conservative(e_bound=float(e.max()))
        transferred = Conservative(e_bound=ratio * float(v.max()))
        transferred.name = "conservative-transferred"  # avoid colliding
                                                        # with `local` in
                                                        # compare()'s
                                                        # name-keyed dict

        result = compare(env_factory, [Greedy(), ECI(), local, transferred],
                         seeds=SEEDS)
        greedy_val = result["greedy"]["value"]
        eci_val = result["eci"]["value"]
        local_val = result["conservative"]["value"]
        transferred_val = result["conservative-transferred"]["value"]

        gap = eci_val - greedy_val
        local_recovery = ((local_val - greedy_val) / gap) if gap else float("nan")
        transferred_recovery = ((transferred_val - greedy_val) / gap) if gap else float("nan")

        rows.append({"world": world, "scenario": name, "is_home": world == home_world,
                    "greedy": greedy_val, "eci": eci_val,
                    "local": local_val, "transferred": transferred_val,
                    "local_recovery_pct": 100 * local_recovery,
                    "transferred_recovery_pct": 100 * transferred_recovery})
    return ratio, rows


def main():
    print(__doc__)

    print("(a) Unified cross-world value-gap table "
          f"(seeds={SEEDS}, Greedy vs ECI):\n")
    table = unified_value_gap_table()
    print(f"  {'world':16s} {'scenario':28s} {'arms':>5s} "
          f"{'greedy':>9s} {'eci':>9s} {'gap %':>7s}")
    for r in table:
        print(f"  {r['world']:16s} {r['scenario']:28s} {r['n_arms']:5d} "
              f"{r['greedy']:9.3f} {r['eci']:9.3f} {r['gap_pct']:6.1f}%")

    home = "clinical-amr"
    print(f"\n(b) Transfer check: 'e_bound / v.max()' ratio calibrated on "
          f"'{home}' only, reused unchanged for every other world's own "
          f"Conservative (seeds={SEEDS}):\n")
    ratio, rows = transfer_check(home_world=home)
    print(f"  transferred ratio (from {home}): {ratio:.4f}\n")
    print(f"  {'world':16s} {'local recovery':>15s} {'transferred recovery':>21s}")
    for r in rows:
        tag = "  <- home (ratio source)" if r["is_home"] else ""
        print(f"  {r['world']:16s} {r['local_recovery_pct']:14.1f}% "
              f"{r['transferred_recovery_pct']:20.1f}%{tag}")

    gaps = [abs(r["local_recovery_pct"] - r["transferred_recovery_pct"])
           for r in rows if not r["is_home"]]
    mean_gap = np.mean(gaps) if gaps else float("nan")
    print(f"\n  mean |local - transferred| recovery gap across non-home "
          f"worlds: {mean_gap:.1f} percentage points.")
    print("""
"Recovery" is the fraction of the Greedy-to-ECI value gap a Conservative
instance closes -- 100% would match the (per-arm-informed, in practice
unavailable) ECI oracle exactly; 0% would be no better than Greedy.
'local' recalibrates e_bound from each world's own real e distribution
(the recipe this repo uses everywhere else); 'transferred' reuses one
fixed ratio calibrated on a single home world and never looks at the
target world's own e values at all -- the actual test of whether
Theorem 3's 'no domain-specific knowledge needed' framing survives being
taken literally across genuinely different domains, not just within one.
This is a checked result, not an assumed one -- read the two recovery
columns above rather than this docstring's framing to see which way it
actually came out this run.
""")


if __name__ == "__main__":
    main()
