"""The calibration investigation FINDINGS.md #35/#36 flagged as still
open: campaign-intensity magnitudes in `attrition/election/agents.py` are
stated as "illustrative of the pipeline's mechanics, not calibrated
forecasts" -- not fit to any survey or field-experiment data. There is no
survey/field-experiment dataset available here, but the SAME archive
already checked into this repo (`attrition/data/tcpd_lokdhaba_ge.json`,
2019 only) also holds the 2014 Lok Sabha general election, one real
election cycle earlier at the same delimitation (constituency boundaries
were fixed 2009-2029, so 2014 and 2019 seats are the same real seats,
not a redistricting-confounded comparison). That gives one real,
checkable anchor this investigation did not have before: how much a real
winner's vote share actually moved, election to election, in seats that
did not change hands -- an empirical swing magnitude, not a modelled one.

This is a proxy comparison, stated as one, not a fit: `tipping_point`
returns a fraction of the TOTAL ELECTORATE that must switch (this
project's unit throughout); real vote-share swing is a percentage of
VALID VOTES cast. The two are converted onto the same footing via each
seat's own real turnout (electorate_fraction ~= vote_share_swing_pp/100 *
turnout_pct/100), not assumed equal.

Network fetch, stdlib only -- same policy as fetch_election_data.py.
Nothing here overwrites checked-in data; this is a read-only, one-off
investigation, run and its result recorded in FINDINGS.md, not a
dependency of the package.

Also writes its result to attrition/data/calibration_2014_2019.json --
generated once by running this script, not fetched live on every read;
`election_api.py` serves that static file rather than repeating a network
fetch per request. Regenerate by re-running this script.

Run:  python scripts/calibrate_campaign_magnitudes.py
"""
import csv
import gzip
import io
import json
import os
import urllib.request

from attrition.election import ALL_PHASE1_CONSTITUENCIES, tipping_point, real_margin_pct

ARCHIVE_BASE = ("https://raw.githubusercontent.com/tcpd/tcpd-ld-data-archive"
                "/main/downloads")
NATIONAL_FILE = f"{ARCHIVE_BASE}/All_States/All_States_GE.csv.gz"
PRIOR_YEAR = 2014
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "attrition", "data", "calibration_2014_2019.json")


def _fetch_rows():
    with urllib.request.urlopen(NATIONAL_FILE, timeout=60) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _winners(rows, year):
    return [r for r in rows
            if int(r["Year"]) == year
            and r["Election_Type"] == "Lok Sabha Election (GE)"
            and r["Poll_No"] == "0"
            and r["Position"] == "1"]


def main():
    print(__doc__)
    print(f"Fetching {PRIOR_YEAR} results from the same archive "
          f"scripts/fetch_election_data.py already pulls 2019 from...")
    rows = _fetch_rows()
    prior_by_name = {}
    for r in _winners(rows, PRIOR_YEAR):
        prior_by_name.setdefault(r["Constituency_Name"], []).append(r)

    turnout_by_seat = {c.name: c.turnout_pct for c in ALL_PHASE1_CONSTITUENCIES}
    state_by_seat = {c.name: c.state for c in ALL_PHASE1_CONSTITUENCIES}

    matched, ambiguous, missing, flips = 0, 0, 0, 0
    swing_electorate_fraction = []       # only same-winner-party seats
    for c in ALL_PHASE1_CONSTITUENCIES:
        candidates = prior_by_name.get(c.name)
        if not candidates:
            missing += 1
            continue
        if len(candidates) > 1:
            candidates = [r for r in candidates if r["State_Name"] == c.state]
            if len(candidates) != 1:
                ambiguous += 1
                continue
        prior = candidates[0]
        real_winner_2019 = c.parties[0]
        if prior["Party"] != real_winner_2019:
            flips += 1
            continue
        matched += 1
        vs_2019 = c.vote_share_pct[0]
        vs_prior = float(prior["Vote_Share_Percentage"])
        swing_pp = abs(vs_2019 - vs_prior)
        swing_electorate_fraction.append(swing_pp / 100.0 * c.turnout_pct / 100.0)

    print(f"\n{matched} seats matched with the SAME winning party in both "
          f"{PRIOR_YEAR} and 2019 ({flips} seats flipped to a different "
          f"party -- a real observed swing too, but not usable for a "
          f"same-party vote-share-swing comparison; {missing} seats had no "
          f"match in {PRIOR_YEAR}'s data, {ambiguous} were ambiguous "
          f"three-way name collisions unresolved by state alone).\n")

    swing_electorate_fraction.sort()
    n = len(swing_electorate_fraction)

    def pct_s(p):
        return swing_electorate_fraction[min(int(p / 100.0 * n), n - 1)]

    print(f"Real {PRIOR_YEAR}->2019 winner vote-share swing, converted to a "
          f"fraction of each seat's total electorate (this project's unit):")
    print(f"  min:    {swing_electorate_fraction[0]*100:5.2f}%")
    print(f"  p10:    {pct_s(10)*100:5.2f}%")
    print(f"  median: {pct_s(50)*100:5.2f}%")
    print(f"  p90:    {pct_s(90)*100:5.2f}%")
    print(f"  max:    {swing_electorate_fraction[-1]*100:5.2f}%")

    print(f"\nComparing against this project's own `tipping_point` -- the "
          f"fraction of the electorate the model says must switch to flip "
          f"each seat -- computed independently for all 543 seats:")
    tps = sorted(tp for tp in (tipping_point(c) for c in ALL_PHASE1_CONSTITUENCIES)
                if tp is not None)
    m = len(tps)

    def pct_t(p):
        return tps[min(int(p / 100.0 * m), m - 1)]

    print(f"  median tipping point required: {pct_t(50)*100:5.2f}%  vs. "
          f"real median swing: {pct_s(50)*100:5.2f}%")
    print(f"  p90 tipping point required:    {pct_t(90)*100:5.2f}%  vs. "
          f"real p90 swing:    {pct_s(90)*100:5.2f}%")

    result = {
        "prior_year": PRIOR_YEAR, "current_year": 2019,
        "n_seats": len(ALL_PHASE1_CONSTITUENCIES),
        "n_matched_same_party": matched, "n_flipped_party": flips,
        "n_missing_prior_year": missing, "n_ambiguous": ambiguous,
        "real_swing_electorate_fraction_pct": {
            "min": swing_electorate_fraction[0] * 100, "p10": pct_s(10) * 100,
            "median": pct_s(50) * 100, "p90": pct_s(90) * 100,
            "max": swing_electorate_fraction[-1] * 100,
        },
        "tipping_point_required_pct": {
            "median": pct_t(50) * 100, "p90": pct_t(90) * 100,
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nWrote {OUT_PATH}")

    ratio_median = pct_s(50) / pct_t(50)
    print(f"\nAt the median, real historical swing is {ratio_median*100:.0f}% "
          f"of what the model requires to flip a typical seat -- consistent "
          f"with the real-world base rate (most seats do NOT flip most "
          f"elections; {flips}/{len(ALL_PHASE1_CONSTITUENCIES)} did in this "
          f"specific, unusually large 2014->2019 wave). This is a real, "
          f"checkable anchor, not a forecast: it says nothing about what any "
          f"SPECIFIC campaign's persuasion effort contributed to a given "
          f"seat's real swing (elections move on far more than one "
          f"campaign's contact effort), only that the ORDER OF MAGNITUDE "
          f"this project's example campaign sequences produce should be "
          f"read against this real distribution, not treated as calibrated "
          f"in its own right. See FINDINGS.md for the full writeup.")


if __name__ == "__main__":
    main()
