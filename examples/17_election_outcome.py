"""The Election Layer, closing the loop: does a simulated campaign's
persuasion effort actually flip who wins the real seat?

Nothing built before this connects Level 1/2's switched-fraction numbers
(examples/16) to an actual recomputed result -- this is validation ladder
level 4 ("intervention validation": does changing campaign/information
conditions produce plausible directional effects?), the one level of the
plan's six-level ladder not yet exercised.

Run:  python examples/17_election_outcome.py [constituency name]
"""
import sys

from attrition.election import (simulate_level1_contact, recompute_outcome,
                                tipping_point, real_margin_pct, build_constituency)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Ahmedabad East"
    constituency = build_constituency(name)
    margin = real_margin_pct(constituency)
    real_winner = constituency.parties[0]
    print(__doc__)
    print(f"Constituency: {name} ({constituency.state})")
    print(f"Real 2019 result: {real_winner} won by {margin:.2f}pp\n")

    tp = tipping_point(name)
    print(f"Tipping point: a campaign would need to swing "
          f"{tp*100:.1f}% of the total electorate to the runner-up "
          f"to flip this seat (all else equal).\n")

    # A modest, rising-intensity campaign. The switched fraction Level 1
    # reports is now calibrated (FINDINGS.md #41) against Kalla &
    # Broockman (2018)'s real field-experiment finding that campaign
    # contact's average persuasive effect on vote choice in a general
    # election is statistically indistinguishable from zero -- the
    # raw pressure/resistance mechanism still decides who's relatively
    # easier to move (a close seat, a non-incumbent), but the reported
    # magnitude is capped at REALISTIC_PERSUASION_CEILING_PCT (2.0%),
    # the rounded upper edge of that real study's confidence interval,
    # not an unconstrained mechanism total.
    seq = [0.15, 0.2, 0.25, 0.3, 0.35]
    fractions, final = simulate_level1_contact(name, seq, n_voters=5000, seed=0)
    print(f"Level 1 simulated campaign (n=5000, {len(seq)} rounds of "
          f"rising contact intensity): {final*100:.1f}% of voters switched "
          f"to the challenger by the final round.\n")

    outcome = recompute_outcome(name, final)
    verdict = "FLIPS the real result" if outcome.flipped else \
        "does NOT flip the real result"
    print(f"Recomputed outcome: {outcome.winner} wins with "
          f"{outcome.winner_share_pct:.2f}% (margin {outcome.margin_pct:.2f}pp)")
    print(f"-> this simulated campaign {verdict}.\n")

    print(f"Reading this: {final*100:.1f}% switched vs. a "
          f"{tp*100:.1f}% tipping point -- {'above' if final > tp else 'below'} "
          f"threshold, consistent with the recomputed outcome above.\n"
          f"Re-run with a different constituency name to see how real "
          f"margin (examples/12) sets both the tipping point AND how fast "
          f"Level 1 approaches it -- the same real anchor drives both "
          f"halves of this comparison, not two independent numbers.")


if __name__ == "__main__":
    main()
