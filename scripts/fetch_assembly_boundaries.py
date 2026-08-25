"""Fetch DataMeet's real Assembly-Constituency-to-Parliamentary-
Constituency crosswalk and write attrition/data/datameet_ac_pc_crosswalk.json.

Source: DataMeet (India open-data/open-mapping community), same publisher
and repository already used for the real Lok Sabha boundary polygons
(`attrition/election/geo.py`):
    https://github.com/datameet/maps/tree/master/assembly-constituencies
    license: CC BY 2.5 India -- attribution required

Only the ATTRIBUTE TABLE is fetched here (`India_AC.dbf`), not the
boundary geometry (`.shp`) -- this crosswalk only needs which real
Assembly Constituency belongs to which real Parliamentary Constituency,
a real, exact, many-ACs-to-one-PC administrative fact (by India's
delimitation design an Assembly Constituency is never split across two
Parliamentary Constituencies), not an approximation. No GIS/areal-
interpolation library needed, and none is added -- `.dbf` is a plain
binary record format, parsed here with `struct` (stdlib only, same policy
as every other fetch script in this repo).

DataMeet's own stated caveats (`assembly-constituencies/README.md`),
carried through here rather than silently dropped:
  - Jammu & Kashmir, Jharkhand, Assam, Manipur, Nagaland, and Arunachal
    Pradesh's Assembly Constituencies "appear to be pre-delimitation
    boundaries" (STATUS field, kept in the output).
  - Telangana's Assembly Constituencies are still attributed to Andhra
    Pradesh in this file (a real, disclosed error in the source, not
    fixed here -- see `attrition/election/assembly.py` for how the join
    handles it).
  - 34 of 4,182 records have a blank PC_NAME (checked, not assumed) --
    kept in the output as null, not guessed.

Run:  python scripts/fetch_assembly_boundaries.py
"""

import json
import os
import struct
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "attrition", "data")

URL = ("https://raw.githubusercontent.com/datameet/maps/master/"
      "assembly-constituencies/India_AC.dbf")

_FIELDS_KEPT = ("ST_NAME", "DIST_NAME", "AC_NO", "AC_NAME",
               "PC_NO", "PC_NAME", "STATUS")


def _parse_dbf(raw):
    """Minimal dBase III attribute-table reader -- header, field
    descriptors, then one fixed-width record per row. `raw[0] == b'*'`
    marks a soft-deleted record (skipped, per the format spec)."""
    _, _, _, _, n_records, header_len, record_len = struct.unpack(
        "<BBBBIHH", raw[:12])
    fields = []
    offset = 32
    while raw[offset:offset + 1] != b"\r":
        field_desc = raw[offset:offset + 32]
        name = field_desc[0:11].split(b"\x00")[0].decode("latin1")
        length = field_desc[16]
        fields.append((name, length))
        offset += 32

    records = []
    pos = header_len
    for _ in range(n_records):
        row_bytes = raw[pos:pos + record_len]
        pos += record_len
        if len(row_bytes) < record_len or row_bytes[0:1] == b"*":
            continue
        cursor = 1
        row = {}
        for name, length in fields:
            row[name] = row_bytes[cursor:cursor + length].decode("latin1").strip()
            cursor += length
        records.append(row)
    return records


def main():
    with urllib.request.urlopen(URL, timeout=60) as resp:
        raw = resp.read()
    records = _parse_dbf(raw)

    crosswalk = [
        {
            "state": r["ST_NAME"],
            "district": r["DIST_NAME"] or None,
            "ac_no": int(r["AC_NO"]) if r["AC_NO"] else None,
            "ac_name": r["AC_NAME"],
            "pc_no": int(r["PC_NO"]) if r["PC_NO"] else None,
            "pc_name": r["PC_NAME"] or None,
            "status": r["STATUS"] or None,
        }
        for r in records
    ]

    out = os.path.join(DATA_DIR, "datameet_ac_pc_crosswalk.json")
    with open(out, "w") as f:
        json.dump(crosswalk, f, indent=1)
    print(f"wrote {out}")
    print(f"  {len(crosswalk)} assembly constituencies")
    print(f"  {len(set(c['state'] for c in crosswalk))} states")
    n_blank_pc = sum(1 for c in crosswalk if not c["pc_name"])
    print(f"  {n_blank_pc} with no PC_NAME in the source (kept as null)")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
