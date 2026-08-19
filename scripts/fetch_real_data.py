"""Refetch the real-world snapshots used by attrition/real_data.py, from the
live public sources, and overwrite attrition/data/*.json.

Sources (see attrition/real_data.py's SOURCES dict for citations/caveats):
    WHO Global Health Observatory API  -- MRSA and E. coli resistance rates
    openFDA drug enforcement API       -- recall classification by failure
                                          category (server-side aggregation,
                                          not raw-row download)

stdlib `urllib.request` only -- no new dependency, here or anywhere else in
the package. Same regenerate-from-live-source philosophy as
docs/make_readme_images.py and paper/make_figures.py: the checked-in
snapshot is what the library actually reads, this script is how you refresh
it.

Run:  python scripts/fetch_real_data.py
"""

import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "attrition", "data")

GHO_INDICATORS = {
    "mrsa": "AMR_INFECT_MRSA",
    "ecoli": "AMR_INFECT_ECOLI",
}

FDA_CATEGORIES = ["CGMP", "sterility", "contamina*", "particulate", "potency",
                  "dissolution", "stability", "mislabel*", "microbial"]


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


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, code in GHO_INDICATORS.items():
        fetch_who_amr(key, code)
    fetch_fda_cmc_categories()
