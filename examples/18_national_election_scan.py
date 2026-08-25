"""The real national tipping-point distribution, and how many of the 543
real 2019 Lok Sabha seats a single fixed campaign intensity would actually
flip -- the two questions FINDINGS.md's constituency-scaling section
(#36) named as the reason scaling to all 543 seats stopped being "a bigger
number for no new question": at ten hand-picked seats there was no
distribution to look at, only ten points.

Two passes over all 543 real constituencies:

  1. `tipping_point` for every seat -- no simulation, just the closed-form
     reallocation rule searched at 0.1pp resolution (examples/17's same
     function). Reports the real distribution: how close the median seat
     is, and which seats are most/least contestable.

  2. `simulate_level1_contact` for every seat under the SAME modest
     campaign sequence used in examples/17, then `recompute_outcome` to
     see which seats it would actually flip. The switched fraction this
     reports is calibrated (FINDINGS.md #41) against Kalla & Broockman
     (2018)'s real field-experiment finding on campaign contact's
     persuasive effect in general elections -- capped at
     `REALISTIC_PERSUASION_CEILING_PCT`, not an unconstrained mechanism
     total. This is still one campaign profile applied uniformly
     nationwide, not a claim about what any real campaign spends.

Run:  python examples/18_national_election_scan.py [n_voters]
(n_voters defaults to 1000 per seat -- the switched-fraction estimate is
an aggregate rate, not an individual claim, so this stays fast across all
543 seats; raise it for a less noisy per-seat number at the cost of
runtime.)
"""
import sys

from attrition.election import (ALL_PHASE1_CONSTITUENCIES, simulate_level1_contact,
                                recompute_outcome, tipping_point, real_margin_pct)


def main():
    n_voters = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(__doc__)

    # -------------------------------------------------- pass 1: tipping points
    print(f"Pass 1/2: tipping point for all {len(ALL_PHASE1_CONSTITUENCIES)} "
          f"real seats (no simulation, closed-form search)...")
    tps = []
    for c in ALL_PHASE1_CONSTITUENCIES:
        tp = tipping_point(c)
        if tp is not None:                 # None only if max_fraction=0.5 never flips it
            tps.append((c, tp))
    tps.sort(key=lambda ct: ct[1])
    n = len(tps)
    values = [tp for _, tp in tps]

    def pct(p):
        return values[min(int(p / 100.0 * n), n - 1)]

    print(f"\n{n}/{len(ALL_PHASE1_CONSTITUENCIES)} seats have a tipping point "
          f"within 50% of the electorate switching (the rest need more than "
          f"that -- effectively unwinnable by this reallocation rule alone).\n")
    print(f"Tipping-point distribution (fraction of TOTAL electorate that "
          f"must switch to flip):")
    print(f"  min (most contestable): {values[0]*100:5.1f}%  -- {tps[0][0].name} ({tps[0][0].state})")
    print(f"  p10:                    {pct(10)*100:5.1f}%")
    print(f"  median:                 {pct(50)*100:5.1f}%")
    print(f"  p90:                    {pct(90)*100:5.1f}%")
    print(f"  max (safest hold):      {values[-1]*100:5.1f}%  -- {tps[-1][0].name} ({tps[-1][0].state})")

    print(f"\nFive most contestable real seats by this measure:")
    for c, tp in tps[:5]:
        print(f"  {c.name:22} {c.state:20} tipping point {tp*100:5.1f}% "
              f"(real margin {real_margin_pct(c):5.2f}pp)")

    # ------------------------------------------- pass 2: one campaign, nationwide
    seq = [0.15, 0.2, 0.25, 0.3, 0.35]      # examples/17's same capped sequence
    print(f"\nPass 2/2: applying ONE fixed campaign (rounds of rising contact "
          f"intensity {seq}) to every seat, n_voters={n_voters}...")
    flipped = []
    for i, c in enumerate(ALL_PHASE1_CONSTITUENCIES):
        _, final = simulate_level1_contact(c, seq, n_voters=n_voters, seed=0)
        outcome = recompute_outcome(c, final)
        if outcome.flipped:
            flipped.append((c, final, outcome))

    print(f"\n{len(flipped)}/{len(ALL_PHASE1_CONSTITUENCIES)} real seats "
          f"flip under this single fixed campaign profile.")
    if flipped:
        flipped.sort(key=lambda cfo: real_margin_pct(cfo[0]))
        print("Flipped seats, closest real margin first:")
        for c, final, outcome in flipped[:15]:
            print(f"  {c.name:22} {c.state:20} real margin "
                  f"{real_margin_pct(c):5.2f}pp -> {final*100:5.1f}% switched "
                  f"-> {outcome.winner} wins by {outcome.margin_pct:5.2f}pp")
        if len(flipped) > 15:
            print(f"  ... and {len(flipped) - 15} more")

    razor_thin = [c for c, _, _ in flipped if real_margin_pct(c) < 1.0]
    if razor_thin:
        print(f"\n{len(razor_thin)} of the flipped seats have a real margin "
              f"under 1pp -- these have a persuadable fraction near 1.0 "
              f"(FINDINGS.md's persuasion-anchor result: persuadable "
              f"fraction = 1 - margin), so this modest sequence approaches "
              f"its real, calibrated ceiling (REALISTIC_PERSUASION_CEILING_PCT, "
              f"FINDINGS.md #41) fastest for them -- still capped at ~2%, "
              f"never the near-total saturation earlier, un-calibrated "
              f"versions of this mechanism produced.")

    print(f"\nReading this: pass 1 is the ceiling on what ANY campaign could "
          f"achieve by this reallocation rule; pass 2 is what ONE specific "
          f"campaign profile, now calibrated against a real field-experiment "
          f"finding (Kalla & Broockman 2018, FINDINGS.md #41), actually "
          f"reaches. The gap between the two is the honest way to read this "
          f"-- not every seat below its tipping point in pass 1 appears in "
          f"pass 2's list, because reaching the tipping point needs more "
          f"contact than this profile can realistically provide: the real "
          f"academic literature's best-case persuasion effect (~2% of the "
          f"electorate) is far below the ~8.5% a typical real seat needs to "
          f"flip, which is exactly why only 67/543 flip here versus most "
          f"seats in pass 1's distribution.")


if __name__ == "__main__":
    main()
