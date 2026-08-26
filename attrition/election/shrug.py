"""Real SHRUG (Socioeconomic High-resolution Rural-Urban Geographic
platform for India) socioeconomic indicators, rolled up to the 543 real
Lok Sabha constituencies -- a fourth real data source (after Assembly
results, Census 2011 district demographics, and the Census-derived state
Assembly rollup), and the first genuinely ECONOMIC one: real 2012 SECC
consumption/poverty estimates and a 2021 satellite-derived wealth proxy,
neither of which anything in this repo had before. See
`scripts/process_shrug_data.py` for the source processing and real
disclosed limitations on what each field means.

Joined through the SAME real DataMeet AC-to-PC crosswalk `assembly.py`
and `census.py` already use, but with one real, checked difference from
both: SHRUG's data is keyed to the 2008 DELIMITATION's Assembly
Constituency boundaries specifically (`ac08_id`), and DataMeet's
crosswalk itself carries a real `status` field marking 524 of its 4,182
rows "Pre delimitation" -- older AC boundaries for states the 2002
Delimitation Act's redraw did not reach on the same timeline (J&K,
Assam, Jharkhand, and others). Joining SHRUG's post-2008-delimitation AC
names against a pre-delimitation row would match the wrong constituency
under a name that may not even mean the same boundary -- so this
module's own crosswalk index EXCLUDES "Pre delimitation" rows, a
narrower filter than `assembly.py`'s (which does not filter on this
field, a pre-existing, unrelated design choice of that module, not
touched here). This costs real coverage for the affected states,
reported honestly via `shrug_rollup_coverage()`, not hidden by a looser
join that would silently mismatch boundaries.

Real, disclosed indicator-level limitations, not smoothed over:

1. Rural and urban SECC consumption/poverty figures are kept SEPARATE
   per constituency, never combined into one number -- the same
   discipline `census.py` states for its own district rollups. A
   constituency's rural or urban SECC figure is `None` when that segment
   has no matched SHRUG row, never defaulted to zero or silently
   averaged with the other segment.
2. `pmgsy_pct_villages_connected` is a village-count coverage rate
   (population-fragment-weighted, see the processing script), not a
   claim about total road length or quality -- a village with one very
   short completed road counts identically to one with a much longer
   one.
3. `facebook_mean_rwi` is a relative wealth INDEX (zero-centred, no
   fixed unit), useful for comparing constituencies against each other
   or against the national distribution, not interpretable as an
   absolute rupee figure the way `secc_cons_pc_rural`/`_urban` are.

Nothing here invents an individual voter's demographic or economic
attribute -- every number is a real, published constituency-level
aggregate, the same observed/aggregate-constrained discipline every
other module in this package enforces.
"""
import json
import os
from functools import lru_cache

from .geo import _lookup_key, _normalise, _PC_NO_OVERRIDES
from .population import ALL_PHASE1_CONSTITUENCIES

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_SHRUG_PATH = os.path.join(_DATA_DIR, "shrug_con08.json")
_CROSSWALK_PATH = os.path.join(_DATA_DIR, "datameet_ac_pc_crosswalk.json")

__all__ = ["shrug_rollup_for_seat", "all_shrug_rollups", "shrug_rollup_coverage"]

# Real state-name variants between SHRUG's pc01_state_name field and
# DataMeet's ST_NAME field -- the same two real aliases `assembly.py`/
# `census.py` document for their own joins against this crosswalk (both
# draw from the same field).
_STATE_ALIASES_SHRUG = {
    _normalise("Odisha"): _normalise("Orissa"),
    _normalise("Uttarakhand"): _normalise("Uttarkhand"),
}


@lru_cache(maxsize=1)
def _load_shrug():
    with open(_SHRUG_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _shrug_index():
    """(normalised SHRUG state, normalised SHRUG ac_name) -> SHRUG row."""
    index = {}
    for row in _load_shrug():
        key = (_normalise(row["state"]), _normalise(row["ac_name"]))
        index.setdefault(key, row)
    return index


@lru_cache(maxsize=1)
def _post_delimitation_crosswalk():
    with open(_CROSSWALK_PATH) as f:
        rows = json.load(f)
    return [r for r in rows if r.get("status") != "Pre delimitation"]


@lru_cache(maxsize=1)
def _real_seat_by_datameet_key():
    out = {}
    for c in ALL_PHASE1_CONSTITUENCIES:
        out[_lookup_key(c.state, c.name)] = (c.state, c.name)
    return out


@lru_cache(maxsize=1)
def _shrug_rows_by_real_seat():
    """(real state, real name) -> list of matched real SHRUG AC rows for
    that seat's constituent 2008-delimitation Assembly Constituencies."""
    cw = _post_delimitation_crosswalk()
    seat_index = _real_seat_by_datameet_key()
    shrug_index = _shrug_index()

    out = {}
    for row in cw:
        if not row["pc_name"] or not row["ac_name"]:
            continue
        state_norm = _normalise(row["state"])
        state_norm_aliased = _STATE_ALIASES_SHRUG.get(state_norm, state_norm)
        pc_norm = _normalise(row["pc_name"])
        pc_norm = _PC_NO_OVERRIDES.get((state_norm, row["pc_no"]), pc_norm)
        real_seat = seat_index.get((state_norm, pc_norm))
        if real_seat is None:
            continue
        shrug_row = shrug_index.get((state_norm_aliased, _normalise(row["ac_name"])))
        if shrug_row is None:
            continue
        bucket = out.setdefault(real_seat, {})
        bucket[shrug_row["ac08_id"]] = shrug_row
    return {seat: list(acs.values()) for seat, acs in out.items()}


def _sum_field(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return sum(vals) if vals else None


def _population_weighted_mean(rows, field, weight_field="population_total"):
    pairs = [(r[field], r[weight_field]) for r in rows
            if r.get(field) is not None and r.get(weight_field)]
    if not pairs:
        return None
    total_weight = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_weight


def shrug_rollup_for_seat(state, name):
    """Real SHRUG indicators for one real Lok Sabha seat's constituent
    2008-delimitation Assembly Constituency/Constituencies, or `None` if
    none matched. Population totals/counts are SUMMED across constituent
    ACs (they are real, disjoint counts, unlike Census district data --
    see `census.py`); rate/index fields (poverty rate, consumption per
    capita, wealth index, PMGSY connectivity) are population-weighted
    means, not simple averages, so a constituency's few large ACs are
    not swamped by many small ones."""
    rows = _shrug_rows_by_real_seat().get((state, name))
    if not rows:
        return None
    return {
        "n_constituent_acs_matched": len(rows),
        "constituent_acs": rows,
        "population_total": _sum_field(rows, "population_total"),
        "n_households": _sum_field(rows, "n_households"),
        "population_sc": _sum_field(rows, "population_sc"),
        "population_st": _sum_field(rows, "population_st"),
        "population_literate": _sum_field(rows, "population_literate"),
        "secc_cons_pc_rural": _population_weighted_mean(rows, "secc_cons_pc_rural"),
        "secc_pov_rate_rural": _population_weighted_mean(rows, "secc_pov_rate_rural"),
        "secc_cons_pc_urban": _population_weighted_mean(rows, "secc_cons_pc_urban"),
        "secc_pov_rate_urban": _population_weighted_mean(rows, "secc_pov_rate_urban"),
        "facebook_mean_rwi": _population_weighted_mean(rows, "facebook_mean_rwi"),
        "pmgsy_pct_villages_connected": _population_weighted_mean(
            rows, "pmgsy_pct_villages_connected"),
    }


def all_shrug_rollups():
    return {seat: shrug_rollup_for_seat(*seat)
           for seat in _shrug_rows_by_real_seat()}


def shrug_rollup_coverage():
    """Real, checked coverage numbers for this join -- computed from the
    actual data every call, never hand-typed, so this can't drift out of
    sync with what the join actually produces."""
    return {
        "n_acs_total": len(_load_shrug()),
        "n_real_seats_total": len(ALL_PHASE1_CONSTITUENCIES),
        "n_real_seats_with_rollup": len(_shrug_rows_by_real_seat()),
    }
