"""Real Lok Sabha constituency data: the observed/aggregate-constrained
layer of the election population engine (Phase 1 -- see
docs/two-extensions plan and FINDINGS.md's election-track sections).

Checked-in snapshot at attrition/data/tcpd_lokdhaba_ge.json, refreshed by
scripts/fetch_election_data.py -- same regenerate-from-live-source
discipline as attrition/real_data.py, same honesty requirement: this is
real, candidate-level 2019 Lok Sabha (General Election) data from ten
deliberately different constituencies, not a representative national
sample and not synthetic.

SOURCE -- read before treating this as more than it is:
"""

import json
import os

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


def load_constituencies():
    """All ten Phase-1 constituencies, as a list of dicts straight from the
    checked-in snapshot (state, constituency, note, year, constituency_type,
    electors, turnout_pct, n_candidates, enop, candidates)."""
    path = os.path.join(_DATA_DIR, "tcpd_lokdhaba_ge.json")
    with open(path) as f:
        return json.load(f)


def get_constituency(name):
    """One constituency by name (case-insensitive), e.g. get_constituency('Ahmedabad East')."""
    rows = load_constituencies()
    key = name.strip().upper()
    for row in rows:
        if row["constituency"] == key:
            return row
    available = [r["constituency"] for r in rows]
    raise KeyError(f"unknown constituency {name!r}; available: {available}")
