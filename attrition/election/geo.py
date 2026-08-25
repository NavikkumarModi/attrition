"""Joins the real election data (`lokdhaba.py`, TCPD/LokDhaba) to the real
constituency boundary polygons (`attrition/data/india_pc_2019_boundaries.geojson`,
DataMeet -- see `scripts/fetch_constituency_boundaries.py`) for the
frontend's national map. Geometry only lives here; no election results are
computed or altered.

SOURCE (boundaries):
    publisher: DataMeet (India open-data/open-mapping community)
    url: https://github.com/datameet/maps/tree/master/parliamentary-constituencies
    license: CC BY 4.0 -- attribution required, reused here under that term
    note: "simplified" (<2MB) file, built by DataMeet for web use

The two real sources spell some real constituency names differently --
found by actually attempting the join (506/543 matched on a direct,
case-insensitive name+state match), not assumed to line up. The gaps
break into three real categories, not one:

1. Reservation-category suffixes TCPD carries and DataMeet doesn't
   (`"GAYA (SC)"` vs `"Gaya"`) -- stripped before comparing.
2. Genuine spelling variants between the two real sources for the same
   real seat (`"BELGAUM"` vs `"Belagavi"`, `"HARDWAR"` vs `"Haridwar"`) --
   an explicit, checked alias table below, not a fuzzy-match guess.
3. One confirmed data-quality bug in the boundary file itself: Maharashtra
   pc_no 31 is labelled `"Mumbai South"` -- identical to pc_no 30 (same
   name, same wikidata_qid, Q12446954) -- when it is really Mumbai South
   Central (a different real seat). Fixed by pc_no, not silently left
   ambiguous or dropped.

State names also differ for two real states/UTs: DataMeet's
`"Andaman & Nicobar"` (TCPD: `"Andaman & Nicobar Islands"`) and
`"Orissa"` (TCPD/current usage: `"Odisha"`) -- both real, both checked,
not typos in either source.

All names are compared after `_normalise` (uppercase, reservation suffix
stripped, `&` spelled out, then every non-alphanumeric character removed)
-- so hyphens, extra spaces, and "&" vs "and" never cause a false gap;
the alias tables below hold ONLY genuine spelling differences left over
after that.

Verified: all 543 real seats resolve to a real boundary polygon after
these fixes -- see `test_geo_join_matches_all_543_real_seats`.
"""
import json
import os
import re
from functools import lru_cache

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_BOUNDARIES_PATH = os.path.join(_DATA_DIR, "india_pc_2019_boundaries.geojson")

__all__ = ["load_boundaries", "geojson_feature_collection"]


def _normalise(name):
    name = name.strip().upper()
    name = re.sub(r"\s*\((?:SC|ST|GEN)\)\s*$", "", name)   # reservation suffix
    name = name.replace("&", " AND ")
    return re.sub(r"[^A-Z0-9]", "", name)                  # punctuation/whitespace-insensitive


# TCPD state name -> DataMeet state name, both already run through
# `_normalise` -- only where they genuinely differ (checked against the
# live DataMeet file, not guessed).
_STATE_ALIASES = {
    _normalise("Odisha"): _normalise("Orissa"),
    _normalise("Andaman & Nicobar Islands"): _normalise("Andaman & Nicobar"),
}

# (normalised TCPD state, normalised TCPD constituency name) -> normalised
# DataMeet constituency name -- real spelling variants between the two
# sources, found by diffing the unmatched 37 after the automatic
# punctuation-insensitive match above.
_NAME_ALIASES = {
    (_normalise("Andhra Pradesh"), _normalise("Anantapur")): _normalise("Anantapuramu"),
    (_normalise("Andhra Pradesh"), _normalise("Aruku")): _normalise("Araku"),
    (_normalise("Karnataka"), _normalise("Belgaum")): _normalise("Belagavi"),
    (_normalise("Karnataka"), _normalise("Chikkballapur")): _normalise("Chikballapur"),
    (_normalise("Karnataka"), _normalise("Chikkodi")): _normalise("Chikodi"),
    (_normalise("Karnataka"), _normalise("Davanagere")): _normalise("Davangere"),
    (_normalise("Karnataka"), _normalise("Hassan")): _normalise("Haasan"),
    (_normalise("Kerala"), _normalise("Mavelikkara")): _normalise("Mavelikara"),
    (_normalise("Madhya Pradesh"), _normalise("Mandsour")): _normalise("Mandsaur"),
    (_normalise("Maharashtra"), _normalise("Ahmadnagar")): _normalise("Ahmednagar"),
    (_normalise("Punjab"), _normalise("Firozpur")): _normalise("Firozepur"),
    (_normalise("Tamil Nadu"), _normalise("Kanniyakumari")): _normalise("Kanyakumari"),
    (_normalise("Tamil Nadu"), _normalise("Mayiladuthurai")): _normalise("Mayiladuturai"),
    (_normalise("Tamil Nadu"), _normalise("Thoothukkudi")): _normalise("Thoothukudi"),
    (_normalise("Telangana"), _normalise("Bhongir")): _normalise("Bhuvanagiri"),
    (_normalise("Telangana"), _normalise("Mahbubnagar")): _normalise("Mahabubnagar"),
    (_normalise("Telangana"), _normalise("Peddapalle")): _normalise("Peddapalli"),
    (_normalise("Telangana"), _normalise("Secundrabad")): _normalise("Secunderabad"),
    (_normalise("Uttarakhand"), _normalise("Hardwar")): _normalise("Haridwar"),
    (_normalise("West Bengal"), _normalise("Srerampur")): _normalise("Sreerampur"),
    (_normalise("Chhattisgarh"), _normalise("Janjgir-Champa")): _normalise("Janjgir"),
    (_normalise("Chhattisgarh"), _normalise("Sarguja")): _normalise("Surguja"),
    (_normalise("Jharkhand"), _normalise("Palamau")): _normalise("Palamu"),
}

# Confirmed data-quality bug in the boundary file (module docstring, point
# 3): Maharashtra pc_no 31 is mislabelled "Mumbai South" -- a duplicate of
# pc_no 30's name and wikidata_qid -- when it is really Mumbai South
# Central. Resolved by pc_no, checked, not a guess.
_PC_NO_OVERRIDES = {
    (_normalise("Maharashtra"), 31): _normalise("Mumbai South Central"),
}


# Genuinely inverted slivers measure around -5e-7 to -2.5e-6 (checked,
# Bharatpur's 30 real cases); some entirely degenerate zero-area rings
# (repeated/collinear points) land within +-1e-15 of zero from pure
# floating-point summation noise, not a real winding direction either
# way. The threshold sits well below the real cases and well above the
# noise floor so it never mistakes one for the other.
_INVERTED_AREA_EPS = -1e-10


def _ring_signed_area(ring):
    """Planar (lon/lat) shoelace signed area -- positive for
    counter-clockwise winding, negative for clockwise, per the GeoJSON
    (RFC 7946) convention exterior rings are supposed to follow."""
    total = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        total += (x2 - x1) * (y2 + y1)
    return total / 2.0


def _fix_polygon_winding(geometry):
    """Reverses any single-ring polygon part whose winding is inverted
    (negative signed area, i.e. clockwise) -- a real, checked bug in the
    boundary file: one constituency (Rajasthan/Bharatpur, 12 of its 602
    MultiPolygon parts) has clockwise-wound single-ring parts that a
    Mercator projection renders as "everything except this sliver"
    instead of the small real sliver itself -- a solid rectangle
    swallowing the whole map, found by rendering it, not by inspection.
    Scanned across all 543 features (`test_geo_fixes_bharatpur_inverted_
    winding_and_only_bharatpur`) -- Bharatpur is the only one affected, so
    this is a narrow, targeted fix, not a blanket reprocessing step.
    Multi-ring parts (real exterior+hole pairs elsewhere in the file) are
    left untouched -- reversing a ring's winding without knowing which
    ring is the hole would be a guess, not a fix.
    """
    if geometry["type"] == "Polygon":
        polys = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polys = geometry["coordinates"]
    else:
        return geometry

    changed = False
    fixed_polys = []
    for part in polys:
        if len(part) == 1 and _ring_signed_area(part[0]) < _INVERTED_AREA_EPS:
            fixed_polys.append([list(reversed(part[0]))])
            changed = True
        else:
            fixed_polys.append(part)
    if not changed:
        return geometry
    coordinates = fixed_polys[0] if geometry["type"] == "Polygon" else fixed_polys
    return {**geometry, "coordinates": coordinates}


@lru_cache(maxsize=1)
def load_boundaries():
    """The checked-in DataMeet snapshot, as a parsed GeoJSON dict --
    real geometry, not simplified beyond what DataMeet's own "simplified"
    file already is."""
    with open(_BOUNDARIES_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _geo_index():
    geo = load_boundaries()
    index = {}
    for feature in geo["features"]:
        p = feature["properties"]
        state_norm = _normalise(p["st_name"])
        pc_no = p.get("pc_no")
        override = _PC_NO_OVERRIDES.get((state_norm, pc_no))
        name_norm = override if override else _normalise(p["pc_name"])
        index.setdefault((state_norm, name_norm), []).append(feature)
    return index


def _lookup_key(state, name):
    state_norm = _normalise(state)
    state_norm = _STATE_ALIASES.get(state_norm, state_norm)
    name_norm = _normalise(name)
    name_norm = _NAME_ALIASES.get((state_norm, name_norm), name_norm)
    return state_norm, name_norm


def geojson_feature_collection(seats):
    """A GeoJSON `FeatureCollection` with each real constituency's boundary
    geometry, `properties` replaced by whatever's in `seats` (a dict keyed
    by (state, name) -> properties dict, e.g. `election_api.py`'s seat
    summaries) plus the matched boundary's own state/pc_name for
    traceability. Seats with no matching boundary are reported
    (`unmatched`), not silently dropped.
    """
    index = _geo_index()
    features = []
    unmatched = []
    for (state, name), props in seats.items():
        key = _lookup_key(state, name)
        matches = index.get(key)
        if not matches:
            unmatched.append((state, name))
            continue
        geometry = _fix_polygon_winding(matches[0]["geometry"])
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {**props, "_boundary_state": matches[0]["properties"]["st_name"],
                           "_boundary_pc_name": matches[0]["properties"]["pc_name"]},
        })
    return {"type": "FeatureCollection", "features": features}, unmatched
