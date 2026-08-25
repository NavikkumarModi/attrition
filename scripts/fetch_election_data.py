"""Refetch the Lok Sabha constituency snapshot used by attrition/election/,
from the live public source, and overwrite attrition/data/tcpd_lokdhaba_ge.json.

Source: Trivedi Centre for Political Data, Ashoka University -- LokDhaba,
mirrored as gzipped CSVs at
    https://github.com/tcpd/tcpd-ld-data-archive
("LD data as on June 2023"). The live lokdhaba.ashoka.edu.in site has its
own query API (POST /api/data/api/v2.0/getDerivedData) but it was returning
502 Bad Gateway when this script was written (2026-08-25) regardless of
query size -- this archive is what was actually usable, and it is arguably
more reproducible anyway: static, versioned files instead of a live query
against infrastructure that may or may not be up when someone else runs
this.

Real, candidate-level election results -- constituency, electors, turnout,
vote share, party, margin, reservation category, not synthetic. See
`attrition/election/lokdhaba.py`'s module docstring for the observed-vs-
synthetic-latent distinction this feeds into.

Also captures real candidate-biography fields the archive already carries
but the original fetch discarded: sex, MyNeta-affidavit education tier,
TCPD-coded profession, incumbency, terms served, and turncoat/recontest
status. These are observed facts about the real candidates, not invented
individual-voter latents -- the same observed/aggregate-constrained
discipline `population.py` already enforces, just extended to who the
real candidates were, not just how many votes they got.

**All 543 real Lok Sabha constituencies**, 2019 General Election (the most
recent full general election in the archive) -- not the original ten
hand-picked "deliberately different environments" seats from this
project's Phase 1 pilot. `downloads/All_States/All_States_GE.csv.gz` is
the archive's own combined national file (one fetch instead of 36
per-state ones); filtering to `Year == 2019`, `Election_Type == "Lok Sabha
Election (GE)"`, `Poll_No == "0"` (the main poll, excluding scattered
later by-elections that share the GE election-type code) gives exactly
543 seats across all 36 states/UTs as they existed in 2019 -- checked, not
assumed; see `scripts/fetch_election_data.py`'s own run output.

The original ten seats keep their hand-written diversity note (urban/
rural/northeast, a curatorial label, not a data claim); every other seat's
note is left empty rather than guessed.

Only the 2019 cross-section is kept, not every historical election -- a
time series is a different, unattempted piece of the plan (temporal
validation), not this one.

stdlib only (urllib, gzip, csv, json) -- same policy as fetch_real_data.py.

Run:  python scripts/fetch_election_data.py
"""

import csv
import gzip
import io
import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "attrition", "data")

ARCHIVE_BASE = ("https://raw.githubusercontent.com/tcpd/tcpd-ld-data-archive"
                "/main/downloads")
NATIONAL_FILE = f"{ARCHIVE_BASE}/All_States/All_States_GE.csv.gz"

TARGET_YEAR = 2019

# The original Phase-1 pilot's ten seats, kept as hand-written diversity
# notes on top of the full 543 -- a curatorial label, not a data claim.
PILOT_NOTES = {
    "AHMEDABAD EAST": "urban",
    "MUMBAI NORTH": "urban",
    "LUCKNOW": "urban",
    "AZAMGARH": "rural",
    "KOLKATA DAKSHIN": "urban",
    "CHENNAI CENTRAL": "urban",
    "BANGALORE RURAL": "rural",
    "PATNA SAHIB": "urban",
    "THIRUVANANTHAPURAM": "urban",
    "GAUHATI": "northeast",
}


def _fetch_national_rows():
    with urllib.request.urlopen(NATIONAL_FILE, timeout=60) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main():
    rows = _fetch_national_rows()
    main_poll_2019 = [r for r in rows
                     if int(r["Year"]) == TARGET_YEAR
                     and r["Election_Type"] == "Lok Sabha Election (GE)"
                     and r["Poll_No"] == "0"]

    by_seat = {}
    for r in main_poll_2019:
        key = (r["State_Name"], r["Constituency_Name"])
        by_seat.setdefault(key, []).append(r)

    def _bool(v):
        return v == "TRUE" if v in ("TRUE", "FALSE") else None

    def _int(v):
        return int(v) if v else None

    snapshot = []
    for (state, name), candidates in sorted(by_seat.items()):
        candidates = sorted(candidates, key=lambda r: int(r["Position"]))
        top = candidates[0]
        snapshot.append({
            "state": state.replace("_", " "),
            "constituency": name,
            "note": PILOT_NOTES.get(name, ""),
            "year": TARGET_YEAR,
            "constituency_type": top["Constituency_Type"],  # GEN / SC / ST
            "sub_region": top["Sub_Region"] or None,
            "electors": int(top["Electors"]),
            "turnout_pct": float(top["Turnout_Percentage"]),
            "n_candidates": int(top["N_Cand"]),
            "enop": float(top["ENOP"]) if top["ENOP"] else None,
            "candidates": [
                {
                    "position": int(c["Position"]),
                    "party": c["Party"],
                    "votes": int(c["Votes"]),
                    "vote_share_pct": float(c["Vote_Share_Percentage"]),
                    "sex": c["Sex"] or None,
                    "education": c["MyNeta_education"] or None,
                    "profession": c["TCPD_Prof_Main"] or None,
                    "profession_desc": c["TCPD_Prof_Main_Desc"] or None,
                    "incumbent": _bool(c["Incumbent"]),
                    "no_terms": _int(c["No_Terms"]),
                    "turncoat": _bool(c["Turncoat"]),
                    "recontest": _bool(c["Recontest"]),
                }
                for c in candidates
            ],
        })

    out = os.path.join(DATA_DIR, "tcpd_lokdhaba_ge.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"wrote {out}")
    print(f"  {len(snapshot)} constituencies, {len(main_poll_2019)} candidate rows")
    print(f"  {len(set(s['state'] for s in snapshot))} states/UTs")
    n_reserved = sum(1 for s in snapshot if s["constituency_type"] != "GEN")
    print(f"  {n_reserved} reserved (SC/ST) seats")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
