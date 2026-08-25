"""Fetch real state Assembly (Vidhan Sabha) election results, from the same
source and archive already trusted for the Lok Sabha snapshot
(`attrition/election/lokdhaba.py`), and write
attrition/data/tcpd_lokdhaba_ae.json.

Source: Trivedi Centre for Political Data, Ashoka University -- LokDhaba,
same mirror as `fetch_election_data.py`:
    https://github.com/tcpd/tcpd-ld-data-archive
    downloads/All_States/All_States_AE.csv.gz ("AE" = Assembly Election)

This exists to build a REAL rollup of assembly-level results up to the 543
Lok Sabha constituencies (`attrition/election/assembly.py`), via DataMeet's
Assembly-Constituency-to-Parliamentary-Constituency crosswalk (`fetch_
assembly_boundaries.py`) -- not to invent per-voter demographic data
(there is none in either source; see `lokdhaba.py`'s module docstring).

Real, unavoidable methodological point, stated here rather than glossed
over: unlike the Lok Sabha election, which happens on one national date,
each state's Assembly election happens on its own five-year cycle. There
is no single "2019 state assembly election" for the whole country. This
script picks, PER STATE, the assembly election year closest to 2019 (ties
broken toward the earlier year) and records how far that year actually is
from 2019 -- from 0 years (several states held assembly polls in 2019
itself, some concurrently with the Lok Sabha election) up to several years
either side. That gap is written into the output and must be surfaced
wherever this data is used, not silently treated as "the same year."

stdlib only (urllib, gzip, csv, json) -- same policy as fetch_election_data.py.

Run:  python scripts/fetch_assembly_data.py
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
NATIONAL_FILE = f"{ARCHIVE_BASE}/All_States/All_States_AE.csv.gz"

TARGET_YEAR = 2019

# The archive's state names, exactly as they appear in the AE file's
# State_Name column (underscores for spaces) -- restricted to the real,
# current 36 states/UTs `tcpd_lokdhaba_ge.json` already carries, checked
# by intersecting the two files' state lists, not assumed. Excludes
# historical/superseded entries the raw archive still carries under old
# names (e.g. "Madras" pre-1969, "Mysore" pre-1973, "Goa_Daman_&_Diu"
# pre-1987 Goa statehood) -- real archive rows, but for a state identity
# that no longer exists and is already covered under its current name.
_CURRENT_STATES = {
    "Andhra_Pradesh", "Arunachal_Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal_Pradesh",
    "Jammu_&_Kashmir", "Jharkhand", "Karnataka", "Kerala", "Madhya_Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil_Nadu", "Telangana",
    "Tripura", "Uttar_Pradesh", "Uttarakhand", "West_Bengal",
    # UTs without their own legislative assembly (Andaman & Nicobar,
    # Chandigarh, Dadra & Nagar Haveli, Daman & Diu, Lakshadweep) have no
    # AE rows at all -- real, expected, not an omission here.
}


def _fetch_national_rows():
    with urllib.request.urlopen(NATIONAL_FILE, timeout=120) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _bool(v):
    return v == "TRUE" if v in ("TRUE", "FALSE") else None


def main():
    rows = _fetch_national_rows()
    main_poll = [r for r in rows
                if r["Poll_No"] == "0" and r["State_Name"] in _CURRENT_STATES]

    # Per real state, the real assembly-election year closest to 2019 --
    # computed from the data itself, not assumed.
    years_by_state = {}
    for r in main_poll:
        state = r["State_Name"]
        years_by_state.setdefault(state, set()).add(int(r["Year"]))
    target_year_by_state = {
        state: min(years, key=lambda y: (abs(y - TARGET_YEAR), y))
        for state, years in years_by_state.items()
    }

    selected = [r for r in main_poll
               if int(r["Year"]) == target_year_by_state.get(r["State_Name"])]

    by_seat = {}
    for r in selected:
        key = (r["State_Name"], r["Constituency_Name"])
        by_seat.setdefault(key, []).append(r)

    snapshot = []
    for (state, name), candidates in sorted(by_seat.items()):
        candidates = sorted(candidates, key=lambda r: int(r["Position"]))
        top = candidates[0]
        year = int(top["Year"])
        snapshot.append({
            "state": state.replace("_", " "),
            "constituency": name,
            "year": year,
            "year_gap_from_2019": year - TARGET_YEAR,
            "constituency_type": top["Constituency_Type"],
            "electors": int(top["Electors"]) if top["Electors"] else None,
            "turnout_pct": float(top["Turnout_Percentage"]) if top["Turnout_Percentage"] else None,
            "margin_pct": float(top["Margin_Percentage"]) if top["Margin_Percentage"] else None,
            "winner_party": top["Party"],
            "winner_sex": top["Sex"] or None,
            "winner_incumbent": _bool(top["Incumbent"]),
        })

    out = os.path.join(DATA_DIR, "tcpd_lokdhaba_ae.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"wrote {out}")
    print(f"  {len(snapshot)} assembly constituencies")
    print(f"  {len(set(s['state'] for s in snapshot))} states")
    print("  year used per state (gap from 2019):")
    for state in sorted(target_year_by_state):
        y = target_year_by_state[state]
        print(f"    {state:30s} {y} ({y - TARGET_YEAR:+d})")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
