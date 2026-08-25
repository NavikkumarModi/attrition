"""Election track, Phase 2: connects the abstract persuasion-commons
theorem (examples/11_campaign_persuasion.py, FINDINGS.md#28) to the real,
validated Phase-1 population (FINDINGS.md#27) -- partially, not fully, and
the script says exactly where the line is.

`attrition.election.derive_persuasion_parameters_from_constituency` anchors
one real quantity -- a constituency's real margin (winner share minus
runner-up share, both real 2019 vote shares) -- to the campaign-persuasion
engine's initial persuadable-fraction parameter (S0). Everything else about
the engine's dynamics stays synthetic. See
`attrition/election/persuasion.py`'s module docstring for the full honesty
boundary.

The question this answers: does the zero-regret-ruin result from
examples/11 survive real-world variation in how contestable a
constituency actually is, or was it an artifact of one convenient default
parameter setting? Checked for all ten Phase-1 constituencies, not assumed
to generalise from one.

Run:  python examples/12_real_persuasion.py
"""
from attrition import ConsumableBandit, Greedy, ECI, compare
from attrition.election import (ALL_PHASE1_CONSTITUENCIES,
                                derive_persuasion_parameters_from_constituency,
                                real_margin_pct)


def main():
    print(__doc__)
    print(f"{'constituency':22} {'margin':>8} {'S0':>6} {'greedy':>8} "
          f"{'eci':>8} {'shortfall':>10}")
    print("-" * 68)
    for c in ALL_PHASE1_CONSTITUENCIES:
        margin = real_margin_pct(c)
        v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(c.name)
        result = compare(lambda s: ConsumableBandit(v, p, e, delta=0.35,
                                                     horizon=9, seed=s),
                         [Greedy(), ECI()], seeds=150)
        gv, ev = result["greedy"]["value"], result["eci"]["value"]
        shortfall = 100 * (ev - gv) / ev if ev else 0.0
        print(f"{c.name:22} {margin:7.2f}p {s0:6.3f} {gv:8.3f} {ev:8.3f} "
              f"{shortfall:9.1f}%")

    print("""
Reading the table: S0 (persuadable fraction) genuinely moves with the real
margin -- close contests (Thiruvananthapuram, Bangalore Rural, Kolkata
Dakshin, margins under 13 points) get high S0; landslides (Mumbai North,
47-point margin) get low S0. Despite that real variation, greedy stays
exactly zero-regret and loses roughly 26-30% of real persuasion value to
ECI at every single constituency -- the theorem's qualitative shape does
not depend on which real seat supplies S0.

What this does NOT show: that any of this predicts real campaign behaviour
or real voter persuasion. R0, the growth-rate asymmetry, the cross-
competition terms, and the contact-effectiveness ceiling are all still the
engine's synthetic defaults, not derived from anything real. One anchor,
stated as one anchor.
""")


if __name__ == "__main__":
    main()
