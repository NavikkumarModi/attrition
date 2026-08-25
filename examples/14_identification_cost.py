"""The identification-cost result promised early in this project's
planning, closed out here: how much a campaign gives up to even know
whether its contact is helping or hurting a persuadable segment.

Not a new theorem -- Theorem 3 (THEORY.md, "estimation impossibility under
consumption") already proves the externality vector `e` cannot be
estimated to arbitrary precision no matter the horizon, because all
information about `e_i` enters through the reward-level shift at the exact
moment arm `i` "dies", and the episode's information budget is capped by
`E[N] = sum 1/p_i`, not by campaign duration. This script evaluates that
already-proven, closed-form floor at the real, heterogeneous `p` already
derived for every Phase-1 constituency -- see
`attrition/election/identification.py`.

Run:  python examples/14_identification_cost.py
"""
from attrition.election import PHASE1_IDENTIFICATION_FLOORS, ALL_PHASE1_CONSTITUENCIES


def main():
    print(__doc__)
    print(f"{'constituency':22} {'S0':>6} {'identifiable':>13} "
          f"{'E[N]':>8} {'SE floor':>10}")
    print("-" * 63)
    for c in ALL_PHASE1_CONSTITUENCIES:
        r = PHASE1_IDENTIFICATION_FLOORS[c.name]
        en = f"{r['expected_N']:.1f}" if r["expected_N"] is not None else "n/a"
        floor = f"{r['se_floor']:.5f}" if r["se_floor"] is not None else "n/a"
        print(f"{c.name:22} {r['s0']:6.3f} "
              f"{r['n_identifiable']}/{r['n_intensity_levels']:>10} "
              f"{en:>8} {floor:>10}")

    print("""
Two findings, not one:

  1. Only 2 of 5 contact-intensity levels EVER generate a saturation event
     within the horizon, at every single real constituency. For the other
     3, e_a is not merely hard to estimate -- it is not identifiable at
     any horizon, because no identifying event ever occurs. A campaign
     running light-to-moderate contact forever still never learns what
     heavy contact would have cost it.

  2. Where identification IS possible, the floor is real, strictly
     positive, and cannot be improved by running longer: it ranges from
     ~0.007 (Thiruvananthapuram, Bangalore Rural, Kolkata Dakshin -- the
     closest, most contestable real seats) to ~0.033 (Mumbai North, the
     widest real landslide). More-contestable seats give more identifying
     events (larger E[N]), so the floor there is lower -- a real,
     interpretable pattern, not noise.

Practical reading: a campaign cannot distinguish "my contact caused this
segment's persuasion to change" from "this segment was moving that way
regardless" without a deliberate holdout -- contacting some comparable
segment less, or not at all, specifically to observe the counterfactual
drift rate. That is the same "identification requires deliberate
abstention" logic this project's very first planning session drew from
Tsiatis-style competing-risks theory, now instantiated with real numbers
in a domain it was never proven in.
""")


if __name__ == "__main__":
    main()
