"""Experiment 63 -- does the regret-harm divergence survive when consumption
is no longer permanent?

exp62 showed the divergence survives decentralisation under exact,
permanent consumption (`SimultaneousPool`). Real antimicrobial resistance
does not binary-flip an antibiotic from usable to unusable -- it
accumulates, decays, and only asymptotes toward permanent loss (see
FINDINGS.md#24 and the domain-extension proposal's caution against
overclaiming AMR as literal irreversible arm consumption). `DegradingPool`
(attrition.simultaneous) generalises the binary alive/dead state into a
continuous, partially-recoverable one:

    s_a(t+1) = rho * s_a(t) + (1 - rho) * e_a * hit_a(t)

with rho the fraction of accrued harm that survives into next round.

This experiment asks the question directly, and reports what actually
happens rather than assuming the answer looks like exp62's. It does not:
the sign of the reported-vs-true gap flips depending on which scoring rule
is used, because ECI's charge term (`delta * p_a * e_a * (T - t)`) was
built assuming the arm's full externality applies for the rest of the
horizon once hit -- an assumption partial recovery breaks in the *opposite*
direction from what a naive "the divergence should persist" prediction
would suggest.
"""
import numpy as np

from attrition.engines import derive_antibiotic_parameters
from attrition.simultaneous import decentralised_value_degrading

DELTA, HORIZON, M = 0.35, 9, 3   # same as exp62, for direct comparability
RHOS = [0.0, 0.25, 0.5, 0.75, 0.95, 0.99]


def sweep():
    v, p, e, _ = derive_antibiotic_parameters()
    rows = {}
    for lam in (0.0, 1.0):
        for rho in RHOS:
            rows[(lam, rho)] = decentralised_value_degrading(
                v, p, e, rho, DELTA, HORIZON, M, rule="lambda_weighted",
                lam=lam, seeds=800, return_details=True)
    return rows


def main():
    print(__doc__)
    rows = sweep()

    for lam, label in [(0.0, "lam=0.0 (greedy -- score blind to burden)"),
                       (1.0, "lam=1.0 (eci-style -- score prices the "
                             "arm's FULL externality, as if permanent)")]:
        print(f"\n{label}\n" + "-" * len(label))
        print(f"{'rho':>6} {'system_value':>13} {'reported':>10} {'gap':>8}")
        for rho in RHOS:
            d = rows[(lam, rho)]
            print(f"{rho:6.2f} {d['system_value']:13.3f} "
                  f"{d['reported_value']:10.3f} {d['gap']:8.3f}")

    print("""
Reading this honestly, not by analogy to exp62:

  lam=0.0 -- the gap stays positive throughout (the blind score always
             overstates true value) but SHRINKS toward 0 as rho -> 1.
             At this horizon (T=9), high rho means accrued harm has not
             had time to build toward its ceiling yet, so realised burden
             stays low and the blind score's error shrinks -- a
             short-horizon effect, not evidence recovery makes blindness
             safe in general. A longer horizon would need re-checking,
             not assumed to extrapolate the same way.

  lam=1.0 -- the gap is NEGATIVE at every rho tested: the eci-style score
             prices each arm as though its externality were permanent
             the instant it is hit, which overcharges relative to what
             partial recovery actually costs. The agent under-uses arms
             it should be using -- a real, quantifiable cost of applying
             a permanent-loss-calibrated policy in a domain where loss is
             not actually permanent.

The headline claim from exp62 -- reward-shaping narrows a positive gap to a
positive floor -- does NOT simply transfer here. Partial recovery does not
just shrink the divergence; it can flip which side of "true harm" a given
policy errs on, depending on what assumption about persistence that policy's
pricing was built on. That is the actual finding, and it is a sharper,
more useful one than "the phenomenon survives" would have been.
""")


if __name__ == "__main__":
    main()
