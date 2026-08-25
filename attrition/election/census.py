"""Real Census 2011 district-level demographics (population, households,
area, density -- table A-1, see `scripts/process_census_data.py`), rolled
up to the 543 real Lok Sabha constituencies via the SAME real DataMeet
AC-to-PC crosswalk `assembly.py` already uses, this time reading its
`district` field instead of its `pc_name`-to-election-result path.

A real, deliberately NOT smoothed-over limitation: a Parliamentary
Constituency frequently spans PARTS of more than one district (median 2,
up to 8 in this real crosswalk), and a district frequently spans more
than one PC. So this module does NOT report a single "this seat's
population is N" number -- doing that by summing whole-district
populations across a multi-district seat would double-count every
person in a district shared with another PC, a fabricated precision this
project does not manufacture. Instead it reports the real, unmodified
per-district figures for every constituent district, plus a MEAN (not
sum) of density/rate-like fields across them -- a real, honest
descriptive statistic, not a population total.

Real, checked coverage: 486 of 543 real seats (89.5%) resolve to at least
one real constituent district via this crosswalk -- the same real
name-matching gap `assembly.py` documents for its own join (Odisha/
Orissa, Uttarakhand/Uttarkhand aliases; J&K's newer post-2011 districts
not present in DataMeet's pre-delimitation-era AC list for that state;
NCT of Delhi's districts absent from the crosswalk entirely). Reported
via `census_rollup_coverage()`, never silently defaulted to zero.

No per-voter demographic data is introduced -- every number is a real,
published district aggregate, the same observed/aggregate-constrained
discipline every other module in this package enforces.
"""
import json
import os
import re
import statistics
from functools import lru_cache

from .geo import _lookup_key, _normalise, _PC_NO_OVERRIDES
from .population import ALL_PHASE1_CONSTITUENCIES

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_DISTRICT_PATH = os.path.join(_DATA_DIR, "census_2011_district.json")
_CROSSWALK_PATH = os.path.join(_DATA_DIR, "datameet_ac_pc_crosswalk.json")

__all__ = ["census_rollup_for_seat", "all_census_rollups", "census_rollup_coverage"]

# Real state-name variant checked against both live files (same aliases
# `assembly.py` documents; a district-level join hits the same two real
# spelling issues in this crosswalk, not a coincidence -- both draw from
# the same `ST_NAME` field).
_STATE_ALIASES_DISTRICT = {
    _normalise("Odisha"): _normalise("Orissa"),
    _normalise("Uttarakhand"): _normalise("Uttarkhand"),
}


def _clean_state_name(name):
    return re.sub(r"[@&$]", "", name).strip()


@lru_cache(maxsize=1)
def _load_districts():
    with open(_DISTRICT_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _district_index():
    """(normalised state, normalised district) -> district record."""
    index = {}
    for row in _load_districts():
        key = (_normalise(row["state"]), _normalise(row["district"]))
        index[key] = row
    return index


@lru_cache(maxsize=1)
def _real_seat_by_datameet_key():
    out = {}
    for c in ALL_PHASE1_CONSTITUENCIES:
        out[_lookup_key(c.state, c.name)] = (c.state, c.name)
    return out


@lru_cache(maxsize=1)
def _districts_by_real_seat():
    """(real state, real name) -> list of real district records for that
    seat's constituent districts (deduplicated by (state, district))."""
    cw = json.load(open(_CROSSWALK_PATH))
    seat_index = _real_seat_by_datameet_key()
    d_index = _district_index()

    out = {}
    for row in cw:
        if not row["pc_name"] or not row["district"]:
            continue
        state_norm = _normalise(row["state"])
        state_norm_aliased = _STATE_ALIASES_DISTRICT.get(state_norm, state_norm)
        pc_norm = _normalise(row["pc_name"])
        pc_norm = _PC_NO_OVERRIDES.get((state_norm, row["pc_no"]), pc_norm)
        real_seat = seat_index.get((state_norm, pc_norm))
        if real_seat is None:
            continue
        district_row = d_index.get((state_norm_aliased, _normalise(row["district"])))
        if district_row is None:
            continue
        bucket = out.setdefault(real_seat, {})
        bucket[(district_row["state"], district_row["district"])] = district_row
    return {seat: list(districts.values()) for seat, districts in out.items()}


def census_rollup_for_seat(state, name):
    """Real Census 2011 district data for one real seat's constituent
    district(s), or `None` if none matched. Returns the real per-district
    records unmodified (`constituent_districts`) plus honest MEAN (not
    summed -- see module docstring) descriptive statistics."""
    districts = _districts_by_real_seat().get((state, name))
    if not districts:
        return None
    densities = [d["population_density_per_sq_km"] for d in districts]
    return {
        "n_constituent_districts": len(districts),
        "constituent_districts": districts,
        "mean_population_density_per_sq_km": statistics.fmean(densities),
        "mean_households_per_district": statistics.fmean(
            d["n_households"] for d in districts),
    }


def all_census_rollups():
    return {seat: census_rollup_for_seat(*seat)
           for seat in _districts_by_real_seat()}


def census_rollup_coverage():
    return {
        "n_districts_total": len(_load_districts()),
        "n_real_seats_total": len(ALL_PHASE1_CONSTITUENCIES),
        "n_real_seats_with_rollup": len(_districts_by_real_seat()),
    }
