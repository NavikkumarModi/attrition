"""Refetch the Lok Sabha constituency boundary GeoJSON used by the
frontend's national map, from the live public source, and overwrite
attrition/data/india_pc_2019_boundaries.geojson.

Source: DataMeet (India's open-data/open-mapping community),
`datameet/maps` on GitHub -- the "simplified" (<2MB) parliamentary-
constituency file, built for exactly this kind of web use. MIT-licensed
repository; the dataset itself is CC BY 4.0 (attribution required, no
other restriction) -- see SOURCE below.

543 real constituency polygons, 2019 delimitation -- the same real seats
`attrition/election/lokdhaba.py` already carries results for. This file
adds GEOMETRY only; `attrition/election/geo.py` is what joins it to the
real election data by name, since the two sources spell some constituency
names differently (confirmed by attempting the join, not assumed) -- see
that module's docstring for the alias table this produced.

stdlib only, same policy as fetch_election_data.py.

Run:  python scripts/fetch_constituency_boundaries.py
"""
import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "attrition", "data")
URL = ("https://raw.githubusercontent.com/datameet/maps/master/"
      "parliamentary-constituencies/india_pc_2019_simplified.geojson")


def main():
    with urllib.request.urlopen(URL, timeout=60) as resp:
        raw = resp.read()
    geo = json.loads(raw)
    out = os.path.join(DATA_DIR, "india_pc_2019_boundaries.geojson")
    with open(out, "w") as f:
        json.dump(geo, f)
    print(f"wrote {out}")
    print(f"  {len(geo['features'])} constituency polygons")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
