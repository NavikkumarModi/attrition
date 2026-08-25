"""Refetch a PRIOR Lok Sabha election's constituency snapshot -- same
source, same schema, same fields as `fetch_election_data.py`'s 2019
snapshot, just a different `Year` filter -- and write it to
attrition/data/tcpd_lokdhaba_ge_<year>.json.

Why a second real snapshot, checked in rather than fetched live each
time: `attrition/election/forecast.py` needs a real prior election's
results as FEATURES to forecast the next one from, and the test suite's
"offline only" policy (see `tests/test_election.py`'s module docstring)
means that data has to be checked in, not live-fetched inside a test run
-- the same reason `tcpd_lokdhaba_ge.json` (2019) is checked in rather
than fetched on import.

2014 is the one prior real cross-section this project uses (the general
election immediately before 2019, at the same post-2008 delimitation --
see `scripts/calibrate_campaign_magnitudes.py`, which already fetches it
live for a one-off calibration check; this script checks the same real
data in permanently for `forecast.py` to depend on offline).

stdlib only, same policy as fetch_election_data.py.

Run:  python scripts/fetch_election_data_historical.py [year]
(defaults to 2014)
"""
import csv
import gzip
import io
import json
import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "attrition", "data")

ARCHIVE_BASE = ("https://raw.githubusercontent.com/tcpd/tcpd-ld-data-archive"
                "/main/downloads")
NATIONAL_FILE = f"{ARCHIVE_BASE}/All_States/All_States_GE.csv.gz"


def _fetch_national_rows():
    with urllib.request.urlopen(NATIONAL_FILE, timeout=60) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _bool(v):
    return v == "TRUE" if v in ("TRUE", "FALSE") else None


def _int(v):
    return int(v) if v else None


def build_snapshot(rows, year):
    """Same field set as fetch_election_data.py's 2019 snapshot, for one
    arbitrary real year -- shared schema so `forecast.py` and
    `attrition.election.population._constituency_from_row` can read
    either file identically."""
    main_poll = [r for r in rows
                if int(r["Year"]) == year
                and r["Election_Type"] == "Lok Sabha Election (GE)"
                and r["Poll_No"] == "0"]

    by_seat = {}
    for r in main_poll:
        key = (r["State_Name"], r["Constituency_Name"])
        by_seat.setdefault(key, []).append(r)

    snapshot = []
    for (state, name), candidates in sorted(by_seat.items()):
        candidates = sorted(candidates, key=lambda r: int(r["Position"]))
        top = candidates[0]
        snapshot.append({
            "state": state.replace("_", " "),
            "constituency": name,
            "note": "",
            "year": year,
            "constituency_type": top["Constituency_Type"],
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
    return snapshot, main_poll


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2014
    rows = _fetch_national_rows()
    snapshot, main_poll = build_snapshot(rows, year)

    out = os.path.join(DATA_DIR, f"tcpd_lokdhaba_ge_{year}.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"wrote {out}")
    print(f"  {len(snapshot)} constituencies, {len(main_poll)} candidate rows")
    print(f"  {len(set(s['state'] for s in snapshot))} states/UTs")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
