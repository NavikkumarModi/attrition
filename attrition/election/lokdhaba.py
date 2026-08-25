"""Real Lok Sabha constituency data: the observed/aggregate-constrained
layer of the election population engine (Phase 1 -- see
docs/two-extensions plan and FINDINGS.md's election-track sections).

Checked-in snapshot at attrition/data/tcpd_lokdhaba_ge.json, refreshed by
scripts/fetch_election_data.py -- same regenerate-from-live-source
discipline as attrition/real_data.py, same honesty requirement: this is
real, candidate-level 2019 Lok Sabha (General Election) data, all 543
constituencies nationally (scaled up from the original ten-seat pilot --
see FINDINGS.md's election-track sections for why that scaling needed a
real question attached before it was worth doing).

SOURCE -- read before treating this as more than it is:
"""

import json
import os
from functools import lru_cache

__all__ = ["SOURCE", "load_constituencies", "get_constituency"]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

SOURCE = {
    "publisher": "Trivedi Centre for Political Data, Ashoka University "
                "(LokDhaba)",
    "url": ("https://github.com/tcpd/tcpd-ld-data-archive "
           "(per-state gzipped CSV mirror, \"LD data as on June 2023\")"),
    "license": "freely downloadable for any purpose -- see "
              "https://lokdhaba.ashoka.edu.in",
    "fetched": "2026-08-25",
    "note": ("The live lokdhaba.ashoka.edu.in query API was returning 502 "
            "Bad Gateway when this was fetched, independent of query size "
            "-- this GitHub-hosted archive was what was actually usable. "
            "All fields here (electors, turnout, vote share, party, "
            "reservation category) are REAL, published results, not "
            "proxies or synthetic values -- this is the observed/aggregate-"
            "constrained layer only. Issue priorities, persuadability, and "
            "any other individual-voter latent attribute are NOT in this "
            "file and must not be invented and attributed to it; see "
            "attrition/election/population.py's module docstring for "
            "where that distinction is enforced."),
}


@lru_cache(maxsize=1)
def _load_constituencies_cached():
    path = os.path.join(_DATA_DIR, "tcpd_lokdhaba_ge.json")
    with open(path) as f:
        return tuple(json.load(f))


def load_constituencies():
    """All 543 real Lok Sabha constituencies (2019 General Election), as a
    list of dicts straight from the checked-in snapshot (state,
    constituency, note, year, constituency_type, electors, turnout_pct,
    n_candidates, enop, candidates). Cached after the first call -- at 543
    seats, re-reading and re-parsing a ~1MB JSON file on every call (the
    original ten-seat version's cost was negligible; this one measurably
    isn't) would make every `build_constituency` call pay for a full file
    load, including the 543 calls `ALL_PHASE1_CONSTITUENCIES` makes at
    import time.
    """
    return list(_load_constituencies_cached())


def get_constituency(name, state=None):
    """One constituency by name (case-insensitive), e.g.
    get_constituency('Ahmedabad East'). Three real names are ambiguous
    across states at national scale -- Aurangabad (Maharashtra and Bihar),
    Maharajganj (Uttar Pradesh and Bihar), Hamirpur (Uttar Pradesh and
    Himachal Pradesh), found by checking, not assumed absent. An ambiguous
    name without `state` raises rather than silently returning whichever
    row happens to come first.
    """
    rows = load_constituencies()
    key = name.strip().upper()
    matches = [row for row in rows if row["constituency"] == key]
    if state is not None:
        state_key = state.strip().upper()
        matches = [row for row in matches
                  if row["state"].strip().upper() == state_key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        states = [row["state"] for row in matches]
        raise KeyError(f"{name!r} is ambiguous across states {states} -- "
                       f"pass state= to disambiguate, "
                       f"e.g. get_constituency({name!r}, state={states[0]!r})")
    available = sorted(set(r["constituency"] for r in rows))
    raise KeyError(f"unknown constituency {name!r}; "
                   f"{len(available)} available, e.g. {available[:5]}")
