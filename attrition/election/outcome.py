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
    fraction of the total electorate) moving to `challenger_party`
    (defaults to the real runner-up). See module docstring for the
    reallocation rule.
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    parties = list(constituency.parties)
    shares = {p: 0.0 for p in parties}
    for p, s in zip(constituency.parties, constituency.vote_share_pct):
        shares[p] += s   # sum, not overwrite -- duplicate labels (IND) exist

    real_winner = max(shares, key=shares.get)
    if challenger_party is None:
        challenger_party = sorted(shares, key=shares.get, reverse=True)[1]

    remaining_total = 100.0 - shares[challenger_party]
    # A switched_fraction of the WHOLE electorate cannot claim more than
    # the share every other party currently holds -- clamped here, not
    # left to silently zero every other party out and hand the challenger
    # a fabricated ~100% (a real degenerate case caught by actually
    # running this against a high Level-1 switched_fraction, not assumed
    # away).
    gain = min(100.0 * float(switched_fraction), remaining_total)
    scale = max(remaining_total - gain, 0.0) / remaining_total if remaining_total > 0 else 0.0
    new_shares = {}
    for p, s in shares.items():
        new_shares[p] = s * scale if p != challenger_party else s + gain
    # renormalise for float drift
    total = sum(new_shares.values())
    new_shares = {p: 100.0 * s / total for p, s in new_shares.items()}

    ranked = sorted(new_shares.items(), key=lambda kv: kv[1], reverse=True)
    winner, winner_share = ranked[0]
    runner_up, runner_up_share = ranked[1]
    return Outcome(winner=winner, winner_share_pct=winner_share,
                  runner_up=runner_up, runner_up_share_pct=runner_up_share,
                  margin_pct=winner_share - runner_up_share,
                  flipped=(winner != real_winner))


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
