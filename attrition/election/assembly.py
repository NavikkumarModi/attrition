"""Real state Assembly (Vidhan Sabha) election results, rolled up to the
543 real Lok Sabha constituencies -- a second real data source joined to
the first, not a new voter mechanism and not invented per-voter data.

Two real sources, joined through a real administrative fact:

    TCPD/LokDhaba Assembly Election results (`attrition/data/
    tcpd_lokdhaba_ae.json`, `scripts/fetch_assembly_data.py`) -- real,
    candidate-level, one row per constituency's winner, one real
    constituency per Assembly Constituency (AC).

    DataMeet's AC-to-PC crosswalk (`attrition/data/
    datameet_ac_pc_crosswalk.json`, `scripts/fetch_assembly_boundaries.py`)
    -- which real Parliamentary Constituency (PC) each real AC belongs to.
    This is an EXACT administrative fact, not an approximation: by India's
    delimitation design, every Assembly Constituency belongs to exactly
    one Parliamentary Constituency, never split across two -- unlike
    district-level data (Census, SHRUG), which does not align cleanly to
    PC boundaries and would need real areal-interpolation approximation.
    That is the whole reason this module exists instead of a district-
    level join.

Real, disclosed limitations, not smoothed over:

1. State assembly elections happen on each state's own five-year cycle,
   not one national date -- there is no single real "2019 assembly
   election." `fetch_assembly_data.py` picks, per state, the real
   election year closest to 2019 and records the gap
   (`ASSEMBLY_YEAR_BY_STATE`, `year_gap_from_2019` in the raw snapshot);
   several states line up exactly (gap 0), most don't. Any rollup number
   here is therefore "as of that state's nearest real assembly election,"
   not "as of 2019" -- stated on every value this module returns, not
   just here.
2. The AC-name join between the two real sources is not perfect: 3,813 of
   4,115 real assembly constituencies (92.7%) match directly after real,
   checked state-name aliases (`_normalise`, `Odisha`/`Orissa`,
   `Uttarakhand`/`Uttarkhand`, and Telangana's ACs being mislabelled
   `Andhra Pradesh` in DataMeet's own crosswalk -- a disclosed error in
   that source, not fixed by inventing correct labels, just aliased for
   the join). The unmatched 302 are a long tail of individual spelling
   variants this module does not chase to 100%, the same honest-partial-
   coverage discipline `geo.py`'s PC-level join documents for itself
   before its alias table closed the gap -- here the gap is left open and
   reported, not closed by guessing.
3. Of the 3,813 matched ACs, 493 of the 543 real PCs (90.8%) resolve to at
   least one -- the remaining 50 real PCs have no assembly rollup data at
   all, reported as `None`, never silently defaulted to zero.

Nothing here invents an individual voter's demographic, party preference,
or any other latent attribute -- every number is a real aggregate over
real assembly election results, the same observed/aggregate-constrained
discipline `population.py` enforces for the Lok Sabha side.
"""
import json
import os
from functools import lru_cache

from .geo import _lookup_key, _normalise, _PC_NO_OVERRIDES
from .population import ALL_PHASE1_CONSTITUENCIES

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_AE_PATH = os.path.join(_DATA_DIR, "tcpd_lokdhaba_ae.json")
_CROSSWALK_PATH = os.path.join(_DATA_DIR, "datameet_ac_pc_crosswalk.json")

__all__ = ["ASSEMBLY_YEAR_BY_STATE", "assembly_rollup_for_seat",
          "all_assembly_rollups", "assembly_rollup_coverage"]

# Real state-name variants between the AE archive's State_Name column and
# DataMeet's crosswalk ST_NAME field -- checked against both live files,
# not guessed. Telangana is a real, disclosed error in DataMeet's own
# source (assembly-constituencies/README.md point 4): its ACs are still
# labelled "Andhra Pradesh".
_STATE_ALIASES_AC = {
    _normalise("Odisha"): _normalise("Orissa"),
    _normalise("Uttarakhand"): _normalise("Uttarkhand"),
    _normalise("Telangana"): _normalise("Andhra Pradesh"),
}


@lru_cache(maxsize=1)
def _load_ae():
    with open(_AE_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_crosswalk():
    with open(_CROSSWALK_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def ASSEMBLY_YEAR_BY_STATE():
    """Real {state: (year_used, year_gap_from_2019)}, read from the
    fetched snapshot itself rather than duplicated as a hand-typed
    constant that could drift out of sync with it."""
    out = {}
    for row in _load_ae():
        out[row["state"]] = (row["year"], row["year_gap_from_2019"])
    return out


@lru_cache(maxsize=1)
def _crosswalk_index():
    """(normalised AC state, normalised AC name) -> crosswalk row."""
    index = {}
    for row in _load_crosswalk():
        key = (_normalise(row["state"]), _normalise(row["ac_name"]))
        index.setdefault(key, row)
    return index


@lru_cache(maxsize=1)
def _real_seat_by_datameet_key():
    """DataMeet's own normalised (state, pc_name) key -> the real (state,
    name) seat it corresponds to -- built by running the REAL seat through
    `geo.py`'s existing TCPD-to-DataMeet name resolution
    (`_lookup_key`), so this module carries no second, potentially-
    drifting copy of that alias table."""
    out = {}
    for c in ALL_PHASE1_CONSTITUENCIES:
        out[_lookup_key(c.state, c.name)] = (c.state, c.name)
    return out


@lru_cache(maxsize=1)
def _ac_rows_by_real_seat():
    """(real state, real name) -> list of matched real AE rows for that
    seat's constituent Assembly Constituencies. Computed once; every
    public function in this module reads from this cache."""
    ae = _load_ae()
    cw_index = _crosswalk_index()
    seat_index = _real_seat_by_datameet_key()

    out = {}
    for row in ae:
        state_norm = _normalise(row["state"])
        state_norm = _STATE_ALIASES_AC.get(state_norm, state_norm)
        cw_row = cw_index.get((state_norm, _normalise(row["constituency"])))
        if cw_row is None or not cw_row["pc_name"]:
            continue
        pc_norm = _normalise(cw_row["pc_name"])
        pc_norm = _PC_NO_OVERRIDES.get((state_norm, cw_row["pc_no"]), pc_norm)
        real_seat = seat_index.get((state_norm, pc_norm))
        if real_seat is None:
            continue
        out.setdefault(real_seat, []).append(row)
    return out


def assembly_rollup_for_seat(state, name):
    """Real assembly-election rollup for one real Lok Sabha seat, or
    `None` if no constituent Assembly Constituency matched (50 of 543
    real seats, see module docstring point 3)."""
    rows = _ac_rows_by_real_seat().get((state, name))
    if not rows:
        return None
    n = len(rows)
    parties = [r["winner_party"] for r in rows]
    turnouts = [r["turnout_pct"] for r in rows if r["turnout_pct"] is not None]
    margins = [r["margin_pct"] for r in rows if r["margin_pct"] is not None]
    # Real, disclosed inconsistency in the raw TCPD archive: winner_sex
    # uses "F"/"M" for some state/year combinations and "FEMALE"/"MALE"
    # for others (checked, not assumed) -- both forms accepted here.
    women = sum(1 for r in rows if r["winner_sex"] in ("F", "FEMALE"))
    incumbents = sum(1 for r in rows if r["winner_incumbent"])
    years = sorted({r["year"] for r in rows})
    return {
        "n_constituent_acs_matched": n,
        "party_diversity": len(set(parties)),
        "most_common_party": max(set(parties), key=parties.count),
        "most_common_party_fraction": parties.count(max(set(parties), key=parties.count)) / n,
        "mean_turnout_pct": sum(turnouts) / len(turnouts) if turnouts else None,
        "mean_margin_pct": sum(margins) / len(margins) if margins else None,
        "woman_winner_fraction": women / n,
        "incumbent_winner_fraction": incumbents / n,
        "assembly_years_used": years,
    }


def all_assembly_rollups():
    """{(state, name): rollup} for every real seat with at least one
    matched constituent AC -- the 493/543 seats `assembly_rollup_
    coverage` reports."""
    return {seat: assembly_rollup_for_seat(*seat)
           for seat in _ac_rows_by_real_seat()}


def assembly_rollup_coverage():
    """Real, checked coverage numbers for this join -- see module
    docstring points 2-3. Computed from the actual data every call
    (cheap: a few dict lookups), never hand-typed, so this can't drift
    out of sync with what the join actually produces."""
    ae = _load_ae()
    cw_index = _crosswalk_index()
    matched_acs = 0
    for row in ae:
        state_norm = _normalise(row["state"])
        state_norm = _STATE_ALIASES_AC.get(state_norm, state_norm)
        if (state_norm, _normalise(row["constituency"])) in cw_index:
            matched_acs += 1
    return {
        "n_ac_rows_total": len(ae),
        "n_ac_rows_matched_to_crosswalk": matched_acs,
        "n_real_seats_total": len(ALL_PHASE1_CONSTITUENCIES),
        "n_real_seats_with_rollup": len(_ac_rows_by_real_seat()),
    }
