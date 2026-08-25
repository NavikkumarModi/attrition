"""Level 0 of the election population engine: millions-scalable, no LLM
calls, statistical voters sampled to match real aggregate distributions.

This is Phase 1 of the plan discussed alongside the AMR extension (see
FINDINGS.md's election-track sections and docs/amr-stewardship-extension.md
for the sibling AMR writeup this mirrors the discipline of): population
infrastructure first, campaign/persuasion agents and any theorem claims
later, once this layer is validated on its own.

Observed vs. synthetic latent -- enforced here, not just documented
--------------------------------------------------------------------
Every field a `SyntheticVoter` carries in this module is drawn from, or
constrained to reproduce, a real published aggregate (electors, turnout,
party vote share, reservation category) from `attrition.election.lokdhaba`.
Nothing here invents what an individual voter believes, prioritises, or is
persuadable by -- issue priorities, persuadability, and information
exposure are Level 1/2 concerns (behavioural and LLM cognitive agents,
unbuilt) and must not be added to this module without a defensible
methodological basis for doing so at the individual level, per the
population-engine plan's explicit warning against exactly that.

A `SyntheticVoter` here is therefore intentionally thin: which real
constituency they're in, whether they turn out, and which party they vote
for if they do -- each drawn to match the constituency's real aggregate
numbers, and nothing else.
"""

from dataclasses import dataclass, field

import numpy as np

from .lokdhaba import get_constituency, load_constituencies

__all__ = ["SyntheticVoter", "Constituency", "build_constituency",
          "synthetic_voter_population", "validate_population",
          "ALL_PHASE1_CONSTITUENCIES"]


@dataclass(frozen=True)
class Constituency:
    """The observed/aggregate-constrained facts about one real constituency
    -- everything here comes straight from `lokdhaba.get_constituency`,
    nothing is synthesised."""
    state: str
    name: str
    note: str                  # non-empty only for the original ten pilot seats
    year: int
    constituency_type: str     # GEN / SC / ST
    electors: int
    turnout_pct: float
    parties: tuple             # party codes, ordered by real vote share (desc)
    vote_share_pct: tuple      # matching real vote share, sums to <= 100


def _constituency_from_row(row):
    cands = row["candidates"]
    return Constituency(
        state=row["state"], name=row["constituency"], note=row["note"],
        year=row["year"], constituency_type=row["constituency_type"],
        electors=row["electors"], turnout_pct=row["turnout_pct"],
        parties=tuple(c["party"] for c in cands),
        vote_share_pct=tuple(c["vote_share_pct"] for c in cands),
    )


def build_constituency(name, state=None):
    """Load one real constituency by name, e.g. build_constituency('Ahmedabad East').
    Three real names are ambiguous across states at national scale -- pass
    `state=` to disambiguate; see `lokdhaba.get_constituency`."""
    return _constituency_from_row(get_constituency(name, state=state))


# Built directly from one already-loaded pass over the snapshot, not via
# 543 individual build_constituency/get_constituency calls -- at national
# scale that would mean 543 linear scans over the same 543-row list (and,
# before load_constituencies() was cached, 543 full file re-reads). Real
# cost once this stopped being ten rows, not a hypothetical one.
ALL_PHASE1_CONSTITUENCIES = [_constituency_from_row(row)
                             for row in load_constituencies()]


@dataclass(frozen=True)
class SyntheticVoter:
    """One synthetic voter. `turned_out` and `voted_for` are the only
    stochastic content -- both drawn to match the constituency's real
    aggregate, not invented per-individual."""
    constituency: str
    turned_out: bool
    voted_for: str | None      # None if turned_out is False


def synthetic_voter_population(constituency, n_voters, seed=0):
    """Sample `n_voters` synthetic voters for one real constituency.

    Turnout is Bernoulli(turnout_pct/100) per voter -- independent draws
    reproducing the real aggregate turnout rate, not a real individual-level
    turnout model (no such data exists in the source). Among those who turn
    out, the party voted for is drawn from a categorical distribution over
    `constituency.parties` weighted by `constituency.vote_share_pct` -- this
    is exact by construction (see `validate_population` for the check, not
    an assumption), which is the honest way to state what Level 0 actually
    is: a population that reproduces known aggregates, not one that predicts
    them.
    """
    rng = np.random.default_rng(seed)
    turnout_p = constituency.turnout_pct / 100.0
    shares = np.asarray(constituency.vote_share_pct, dtype=float)
    shares = shares / shares.sum()   # renormalise (candidate list excludes NOTA-adjacent rounding)

    turned_out = rng.random(n_voters) < turnout_p
    n_out = int(turned_out.sum())
    choices = rng.choice(len(constituency.parties), size=n_out, p=shares)

    voters = []
    it = iter(range(n_out))
    for i in range(n_voters):
        if turned_out[i]:
            party = constituency.parties[choices[next(it)]]
            voters.append(SyntheticVoter(constituency.name, True, party))
        else:
            voters.append(SyntheticVoter(constituency.name, False, None))
    return voters


def validate_population(constituency, n_voters=200_000, seed=0, tol_pct=0.5):
    """Validation ladder, level 1 (demographic/aggregate reproduction --
    see the population-engine plan's six-level ladder): does the synthetic
    population reproduce the real turnout and vote-share numbers it was
    built from? True by construction of `synthetic_voter_population`, not a
    discovery -- this function exists so that claim is checked with an
    explicit tolerance and a real sample, not merely asserted.

    Returns a dict with the real vs. sampled turnout percentage and, for
    each party, the real vs. sampled vote share among those who turned out;
    raises AssertionError if any deviates by more than `tol_pct` points.
    """
    voters = synthetic_voter_population(constituency, n_voters, seed=seed)
    sampled_turnout_pct = 100.0 * sum(v.turned_out for v in voters) / n_voters
    assert abs(sampled_turnout_pct - constituency.turnout_pct) < tol_pct, (
        f"{constituency.name}: sampled turnout {sampled_turnout_pct:.2f}% vs "
        f"real {constituency.turnout_pct:.2f}%, exceeds tol={tol_pct}")

    voted = [v.voted_for for v in voters if v.turned_out]
    n_voted = len(voted)
    shares_real = np.asarray(constituency.vote_share_pct, dtype=float)
    shares_real = shares_real / shares_real.sum() * 100.0

    # Several candidates can share a label (independents are all "IND"), so
    # both the real and sampled shares must be aggregated by unique party
    # label before comparing -- comparing against one duplicate row's share
    # alone is a real bug, not a tolerance issue (caught during validation).
    real_by_party = {}
    for party, pct in zip(constituency.parties, shares_real):
        real_by_party[party] = real_by_party.get(party, 0.0) + float(pct)

    result = {"turnout_pct": {"real": constituency.turnout_pct,
                              "sampled": sampled_turnout_pct},
             "vote_share_pct": {}}
    for party, real_pct in real_by_party.items():
        sampled_pct = 100.0 * voted.count(party) / n_voted if n_voted else 0.0
        result["vote_share_pct"][party] = {"real": real_pct,
                                           "sampled": sampled_pct}
        assert abs(sampled_pct - real_pct) < tol_pct, (
            f"{constituency.name}/{party}: sampled {sampled_pct:.2f}% vs "
            f"real {real_pct:.2f}%, exceeds tol={tol_pct}")
    return result
