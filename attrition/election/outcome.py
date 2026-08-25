"""The Election Layer: turns a simulated persuasion effect into an actual
seat outcome. Validation ladder level 4 (intervention validation -- "does
changing campaign/information conditions produce plausible directional
effects?"), the one level of the plan's six-level ladder not yet touched:
levels 1-3 (demographic, turnout, constituency/vote-share reproduction) are
covered by `population.validate_population`, which checks the synthetic
population reproduces reality; this module asks what happens when a
campaign changes it.

Nothing built before this module computes a winner from a persuasion
effect -- `agents.simulate_level1_contact` and `simulate_level2_contact`
report a switched fraction or a set of persona decisions, not a
recomputed result. This is a real, load-bearing gap being closed, not a
convenience wrapper.

The reallocation rule, stated once, explicitly: a `switched_fraction` of
the total electorate is added to the challenger's real baseline share; the
other parties' real shares (including the real winner's) shrink
proportionally so the total still sums to 100. That is a modelling choice
-- it assumes switched voters are drawn proportionally from every other
party, not specifically from the incumbent -- not a measured fact, and is
stated as one rather than left implicit.
"""

from dataclasses import dataclass

from .population import build_constituency

__all__ = ["Outcome", "recompute_outcome", "tipping_point"]


@dataclass(frozen=True)
class Outcome:
    winner: str
    winner_share_pct: float
    runner_up: str
    runner_up_share_pct: float
    margin_pct: float
    flipped: bool          # True if winner differs from the real result


def recompute_outcome(constituency, switched_fraction, challenger_party=None):
    """Real baseline shares, shifted by `switched_fraction` (in [0, 1], a
    fraction of the total electorate) moving to the challenger candidate
    (defaults to the real runner-up). See module docstring for the
    reallocation rule.

    Accounted per REAL CANDIDATE (by position), not per party label --
    a real bug caught scaling to all 543 seats: several independents can
    each carry the generic "IND" label, and an earlier version summed
    their votes into one pseudo-candidate, which could exceed the actual
    winner's real share (found at Ladakh and Mizoram, both real BJP/MNF
    wins an aggregated recount misreported as "IND" winning). Real
    political parties field exactly one candidate per seat, so
    per-candidate accounting changes nothing for them; it only matters for
    generic multi-candidate labels, which is exactly where aggregation was
    wrong.
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    parties = constituency.parties
    vshares = constituency.vote_share_pct
    n = len(parties)

    if challenger_party is None:
        challenger_idx = 1                     # the real runner-up candidate
    else:
        matches = [i for i, p in enumerate(parties) if p == challenger_party]
        if not matches:
            raise ValueError(f"{challenger_party!r} is not a candidate in "
                             f"{constituency.name}")
        # if the label matches more than one real candidate (independents),
        # anchor on whichever of them polled highest -- stated as the
        # deliberate choice it is, not left ambiguous.
        challenger_idx = max(matches, key=lambda i: vshares[i])

    remaining_total = 100.0 - vshares[challenger_idx]
    # A switched_fraction of the WHOLE electorate cannot claim more than
    # the share every other candidate currently holds -- clamped here, not
    # left to silently zero every other candidate out and hand the
    # challenger a fabricated ~100% (a real degenerate case caught by
    # actually running this against a high Level-1 switched_fraction, not
    # assumed away).
    gain = min(100.0 * float(switched_fraction), remaining_total)
    scale = max(remaining_total - gain, 0.0) / remaining_total if remaining_total > 0 else 0.0

    new_shares = [s * scale for s in vshares]
    new_shares[challenger_idx] = vshares[challenger_idx] + gain
    total = sum(new_shares)
    new_shares = [100.0 * s / total for s in new_shares]

    ranked = sorted(range(n), key=lambda i: new_shares[i], reverse=True)
    win_i, run_i = ranked[0], ranked[1]
    winner, winner_share = parties[win_i], new_shares[win_i]
    runner_up, runner_up_share = parties[run_i], new_shares[run_i]
    return Outcome(winner=winner, winner_share_pct=winner_share,
                  runner_up=runner_up, runner_up_share_pct=runner_up_share,
                  margin_pct=winner_share - runner_up_share,
                  flipped=(win_i != 0))


def tipping_point(constituency, challenger_party=None, resolution=0.001,
                  max_fraction=0.5):
    """The smallest `switched_fraction` (a fraction of the total
    electorate) at which the real winner flips to `challenger_party`, by
    linear search at `resolution` steps up to `max_fraction`. Returns
    `None` if no flip occurs within `max_fraction` -- reported as "no
    tipping point found in range", not silently treated as infinite.
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    f = 0.0
    while f <= max_fraction:
        outcome = recompute_outcome(constituency, f, challenger_party)
        if outcome.flipped:
            return f
        f += resolution
    return None
