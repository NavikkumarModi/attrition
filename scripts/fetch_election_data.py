"""Refetch the Lok Sabha constituency snapshot used by attrition/election/,
from the live public source, and overwrite attrition/data/tcpd_lokdhaba_ge.json.

Source: Trivedi Centre for Political Data, Ashoka University -- LokDhaba,
mirrored as per-state gzipped CSVs at
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

Ten constituencies, Phase 1 of the election-track population engine plan:
deliberately different environments, not a representative sample -- see
CONSTITUENCIES below and attrition/election/lokdhaba.py.

Only the most recent Lok Sabha (General Election, "GE") row per candidate in
each selected constituency is kept, not every historical election -- Phase 1
needs one real, current cross-section per constituency, not a time series
(that is Phase 5, temporal validation, and is not attempted here).

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

# (state, constituency name as it appears in the CSV, one-line note on why
# this seat, out of the state's full list, was picked for Phase 1 diversity)
CONSTITUENCIES = [
    ("Gujarat", "AHMEDABAD EAST", "urban"),
    ("Maharashtra", "MUMBAI NORTH", "urban"),
    ("Uttar_Pradesh", "LUCKNOW", "urban"),
    ("Uttar_Pradesh", "AZAMGARH", "rural"),
    ("West_Bengal", "KOLKATA DAKSHIN", "urban"),
    ("Tamil_Nadu", "CHENNAI CENTRAL", "urban"),
    ("Karnataka", "BANGALORE RURAL", "rural"),
    ("Bihar", "PATNA SAHIB", "urban"),
    ("Kerala", "THIRUVANANTHAPURAM", "urban"),
    ("Assam", "GAUHATI", "northeast"),
]


def _fetch_state_ge_rows(state):
    url = f"{ARCHIVE_BASE}/{state}/{state}_GE.csv.gz"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _latest_year_for(rows, constituency_name):
    years = {int(r["Year"]) for r in rows
             if r["Constituency_Name"] == constituency_name
             and r["Election_Type"] == "Lok Sabha Election (GE)"}
    if not years:
        raise ValueError(f"no GE rows found for {constituency_name!r}")
    return max(years)


def main():
    snapshot = []
    for state, name, note in CONSTITUENCIES:
        rows = _fetch_state_ge_rows(state)
        year = _latest_year_for(rows, name)
        candidates = [r for r in rows
                     if r["Constituency_Name"] == name and int(r["Year"]) == year
                     and r["Election_Type"] == "Lok Sabha Election (GE)"]
        candidates.sort(key=lambda r: int(r["Position"]))
        top = candidates[0]
        snapshot.append({
            "state": state.replace("_", " "),
            "constituency": name,
            "note": note,
            "year": year,
            "constituency_type": top["Constituency_Type"],  # GEN / SC / ST
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
                }
                for c in candidates
            ],
        })
        print(f"{state}/{name}: {year}, {top['Electors']} electors, "
              f"{top['Turnout_Percentage']}% turnout, "
              f"{len(candidates)} candidates")

    out = os.path.join(DATA_DIR, "tcpd_lokdhaba_ge.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"\nwrote {out} ({len(snapshot)} constituencies)")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
