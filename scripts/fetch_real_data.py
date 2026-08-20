"""Refetch the real-world snapshots used by attrition/real_data.py, from the
live public sources, and overwrite attrition/data/*.json.

Sources (see attrition/real_data.py's SOURCES dict for citations/caveats):
    WHO Global Health Observatory API  -- MRSA and E. coli resistance rates
    openFDA drug enforcement API       -- recall classification by failure
                                          category (server-side aggregation,
                                          not raw-row download)
    NOAA Fisheries FOSS landings API   -- real commercial landings (pounds,
                                          dollars) per species per year

stdlib `urllib.request` only -- no new dependency, here or anywhere else in
the package. Same regenerate-from-live-source philosophy as
docs/make_readme_images.py and paper/make_figures.py: the checked-in
snapshot is what the library actually reads, this script is how you refresh
it.

Run:  python scripts/fetch_real_data.py
"""

import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "attrition", "data")

GHO_INDICATORS = {
    "mrsa": "AMR_INFECT_MRSA",
    "ecoli": "AMR_INFECT_ECOLI",
}

FDA_CATEGORIES = ["CGMP", "sterility", "contamina*", "particulate", "potency",
                  "dissolution", "stability", "mislabel*", "microbial"]

# 16 major U.S. commercial species spanning multiple coasts and price points,
# each confirmed present in NOAA FOSS's LANDINGS table (ts_afs_name values).
FOSS_SPECIES = [
    "COD, ATLANTIC", "SALMON, CHINOOK", "MENHADEN, ATLANTIC", "HADDOCK",
    "LOBSTER, AMERICAN", "CRAB, BLUE", "FLOUNDER, SUMMER", "HERRING, ATLANTIC",
    "HALIBUT, PACIFIC", "SCALLOP, SEA", "TUNA, ALBACORE", "SWORDFISH",
    "BASS, STRIPED", "SNAPPER, RED", "MACKEREL, ATLANTIC", "CRAB, DUNGENESS",
]


def _get_json(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def fetch_who_amr(indicator_key, indicator_code):
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    data = _get_json(url)
    rows = []
    for r in data["value"]:
        if r.get("SpatialDimType") != "COUNTRY" or r.get("NumericValue") is None:
            continue
        rows.append({
            "country": r["SpatialDim"],
            "region": r.get("ParentLocation"),
            "year": r["TimeDim"],
            "resistance_pct": r["NumericValue"],
        })
    out = os.path.join(DATA_DIR, f"who_amr_{indicator_key}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}  ({len(rows)} rows)")
    return len(rows)


def fetch_fda_cmc_categories():
    categories = []
    for kw in FDA_CATEGORIES:
        search = urllib.parse.quote(f'reason_for_recall:{kw}')
        url = (f"https://api.fda.gov/drug/enforcement.json"
               f"?search={search}&count=classification.exact")
        data = _get_json(url)
        counts = {r["term"]: r["count"] for r in data.get("results", [])}
        total = sum(counts.values())
        categories.append({
            "category": kw.strip("*"),
            "class_i": counts.get("Class I", 0),
            "class_ii": counts.get("Class II", 0),
            "class_iii": counts.get("Class III", 0),
            "total": total,
        })
    out = os.path.join(DATA_DIR, "fda_cmc_categories.json")
    with open(out, "w") as f:
        json.dump(categories, f, indent=1)
    total_incidents = sum(c["total"] for c in categories)
    print(f"wrote {out}  ({len(categories)} categories, "
          f"{total_incidents} incidents)")
    return len(categories)


def _fetch_species_rows(name, limit=1000):
    rows, offset = [], 0
    while True:
        q = json.dumps({"ts_afs_name": name, "collection": "Commercial"})
        params = urllib.parse.urlencode({"q": q, "limit": limit, "offset": offset})
        url = f"https://apps-st.fisheries.noaa.gov/ods/foss/landings/?{params}"
        data = _get_json(url)
        rows.extend(data["items"])
        if not data.get("hasMore"):
            break
        offset += limit
    return rows


def fetch_fisheries_landings():
    """NOAA FOSS's LANDINGS table is row-level (per species/state/year), so
    this aggregates to national per-species-per-year totals before writing
    the snapshot -- attrition/real_data.py's derive function reads that
    aggregated shape, not raw rows.
    """
    species_out = []
    for name in FOSS_SPECIES:
        rows = _fetch_species_rows(name)
        by_year = defaultdict(lambda: [0.0, 0.0])
        for r in rows:
            if r["pounds"] is None or r["year"] is None:
                continue
            by_year[r["year"]][0] += r["pounds"]
            by_year[r["year"]][1] += (r["dollars"] or 0)
        years = sorted(by_year)
        species_out.append({
            "species": name,
            "scientific_name": next(
                (r["ts_scientific_name"] for r in rows if r["ts_scientific_name"]),
                None),
            "annual": [{"year": y, "pounds": by_year[y][0], "dollars": by_year[y][1]}
                      for y in years],
        })
        print(f"  {name}: {len(years)} years, {len(rows)} raw rows")
        time.sleep(0.3)   # a light courtesy delay, not a rate-limit workaround

    out = os.path.join(DATA_DIR, "noaa_fisheries_landings.json")
    with open(out, "w") as f:
        json.dump(species_out, f, indent=1)
    total_years = sum(len(s["annual"]) for s in species_out)
    print(f"wrote {out}  ({len(species_out)} species, {total_years} species-year rows)")
    return len(species_out)


def fetch_cisa_kev_epss():
    """CISA's KEV feed is one JSON GET, no auth. FIRST.org's EPSS API takes
    a comma-separated `cve=` batch, but batches above ~100 CVE IDs silently
    return far fewer matches than requested (a URL-length effect, found by
    testing -- not documented anywhere) -- keep batches at 50, confirmed to
    reliably return 100% matches.
    """
    with urllib.request.urlopen(
            "https://www.cisa.gov/sites/default/files/feeds/"
            "known_exploited_vulnerabilities.json", timeout=30) as resp:
        kev = json.load(resp)["vulnerabilities"]

    epss = {}
    batch_size = 50
    for i in range(0, len(kev), batch_size):
        batch = [v["cveID"] for v in kev[i:i + batch_size]]
        q = ",".join(batch)
        url = (f"https://api.first.org/data/v1/epss?cve={q}"
              f"&limit={batch_size + 10}")
        data = _get_json(url)
        for row in data["data"]:
            epss[row["cve"]] = float(row["epss"])
        time.sleep(0.1)

    rows = []
    for item in kev:
        cve = item["cveID"]
        if cve not in epss:
            continue
        rows.append({
            "cve": cve,
            "vendor_project": item["vendorProject"],
            "product": item["product"],
            "date_added": item["dateAdded"],
            "known_ransomware_use": item["knownRansomwareCampaignUse"] == "Known",
            "epss": epss[cve],
        })
    out = os.path.join(DATA_DIR, "cisa_kev_epss.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}  ({len(rows)} CVEs, {len(kev) - len(rows)} unmatched)")
    return len(rows)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, code in GHO_INDICATORS.items():
        fetch_who_amr(key, code)
    fetch_fda_cmc_categories()
    fetch_fisheries_landings()
    fetch_cisa_kev_epss()
