"""Process a real, manually-downloaded Census 2011 district-level table
into attrition/data/census_2011_district.json.

Source: Registrar General & Census Commissioner, India (Ministry of Home
Affairs) -- Census 2011, table A-1 "Number of Villages, Towns,
Households, Population and Area", downloaded by hand from
censusindia.gov.in's Population Finder. Unlike every other real-data
fetch script in this repo (`fetch_election_data.py`,
`fetch_assembly_data.py`, `fetch_real_data.py`, ...), this one is NOT a
`urllib` fetch from a stable URL -- there isn't one. censusindia.gov.in
serves this table through a client-heavy portal with no scriptable bulk
endpoint, and data.gov.in's own catalog entry for it says plainly "the
data for this resource is hosted and maintained by the respective
Ministry/Department" and is not proxied through their API without a
registered key. So the raw file has to be downloaded by a person through
the site's own UI first (National Data Sharing and Accessibility Policy
license -- open, just not scriptable) and placed at
`data/A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx`
(repo-root `data/`, NOT `attrition/data/` -- that directory holds
already-processed, checked-in snapshots; this one holds the raw manual
download this script consumes). That distinction is deliberate, not a
typo.

Real, disclosed limitation on WHICH Census tables this covers: four
other real, official tables were downloaded alongside A-1 (A-3 village-
size distribution; B-1 worker classification; C-3 Appendix marital
status by religion; C-8 Appendix education level) but are NOT processed
by this script or joined anywhere yet -- checked, not assumed: B-1 and
C-8 in this specific download are TOWN-level cuts (urban areas only, no
rural rows), so summing them to a district total would silently
undercount every district's real population; C-3 in this specific
download is STATE-level only (District Code is '000' throughout), a
coarser real number than the seat-level rollup this module builds for
A-1. Using any of the three as-is would misrepresent their real
coverage. Only A-1 (and A-3, structurally identical to A-1) are genuinely
clean district-level tables with real district Total/Rural/Urban rows
for the whole country, which is why only A-1 is processed here.

District-to-seat join: see `attrition/election/census.py` -- reuses the
real `district` field DataMeet's AC-to-PC crosswalk already carries
(`datameet_ac_pc_crosswalk.json`, fetched for FINDINGS.md#49), not a new
geographic file.

Run:  pip install "attrition[census]"  (openpyxl)
      python scripts/process_census_data.py
"""
import json
import os
import re

import openpyxl

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_PATH = os.path.join(
    REPO_ROOT, "data",
    "A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx")
OUT_PATH = os.path.join(
    REPO_ROOT, "attrition", "data", "census_2011_district.json")


def _clean_state_name(name):
    """Real A-1 state names carry footnote markers ('@', '&', trailing
    '&') the crosswalk's state names don't -- stripped here, not
    silently left to break every district match for that state."""
    return re.sub(r"[@&$]", "", name).strip()


def main():
    wb = openpyxl.load_workbook(RAW_PATH, read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))

    state_names = {}
    for r in rows[4:]:
        if r[3] == "STATE" and r[5] == "Total":
            state_names[r[0]] = _clean_state_name(r[4])

    districts = []
    for r in rows[4:]:
        if r[3] != "DISTRICT" or r[5] != "Total":
            continue
        state_code = r[0]
        districts.append({
            "state": state_names.get(state_code, r[3]),
            "district": r[4].strip(),
            "n_villages_inhabited": r[6],
            "n_villages_uninhabited": r[7],
            "n_towns": r[8],
            "n_households": r[9],
            "population_total": r[10],
            "population_male": r[11],
            "population_female": r[12],
            "area_sq_km": r[13],
            "population_density_per_sq_km": r[14],
        })

    with open(OUT_PATH, "w") as f:
        json.dump(districts, f, indent=1)
    print(f"wrote {OUT_PATH}")
    print(f"  {len(districts)} real districts")
    print(f"  {len(set(d['state'] for d in districts))} states/UTs")
    total_pop = sum(d["population_total"] for d in districts)
    print(f"  total population across districts: {total_pop:,}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    main()
