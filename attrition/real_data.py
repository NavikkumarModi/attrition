"""Real-world data: a third tier alongside domains.py (hand-set) and
engines.py (mechanistic simulation). These (v, p, e) triples are derived from
actual measurements published by WHO, the FDA, and NOAA, not invented for
this repo -- checked-in snapshots under attrition/data/, refreshed by
scripts/fetch_real_data.py.

Read the fit-quality caveats in SOURCES before treating any of this as more
than it is: in every domain here, only ONE of (v, p, e) is a real measured
quantity (or, for fisheries, a ratio of two real raw counts -- the same
"computed from real, cited numbers" standard fda_cmc's p already uses, not a
weaker one introduced just for this domain). The other two are documented
proxies, stated as such, not silently presented as equally real -- the same
honesty pattern as domains.py's DOMAIN_NOTES.
"""

import json
import os

import numpy as np

__all__ = ["SOURCES", "derive_real_amr_parameters", "derive_real_cmc_parameters",
          "derive_real_fisheries_parameters"]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SOURCES = {
    "who_amr": {
        "url": "https://ghoapi.azureedge.net/api/ (indicators AMR_INFECT_MRSA, "
              "AMR_INFECT_ECOLI)",
        "publisher": "World Health Organization, Global Health Observatory",
        "license": "WHO open data (see https://www.who.int/about/policies/publishing)",
        "fetched": "2026-08-19",
        "fit": ("p_a is REAL: the actual reported proportion of bloodstream "
               "infections resistant to the relevant antibiotic, per "
               "country/year. v_a and e_a are documented proxies derived "
               "from p_a -- WHO surveillance reports resistance rates, not "
               "clinical treatment value or externality magnitude, so those "
               "two are NOT independently measured."),
    },
    "fda_cmc": {
        "url": "https://api.fda.gov/drug/enforcement.json "
              "(server-side count= aggregation by reason_for_recall keyword)",
        "publisher": "U.S. Food and Drug Administration, openFDA",
        "license": "U.S. government work, public domain "
                  "(see https://open.fda.gov/license/)",
        "fetched": "2026-08-19",
        "fit": ("p_a is REAL: the actual fraction of recalls in that failure "
               "category classified Class I (most severe) among 15,556 real "
               "recall records across 9 categories. e_a uses a stated "
               "severity weighting (Class I/II/III) that is a modeling "
               "choice, not an FDA-provided cost; v_a is a documented proxy "
               "from each category's real incident share, not a measured "
               "process yield."),
    },
    "noaa_fisheries": {
        "url": "https://apps-st.fisheries.noaa.gov/ods/foss/landings/ "
              "(Fisheries One Stop Shop, LANDINGS table, filtered to "
              "collection=Commercial per species)",
        "publisher": "NOAA Fisheries, Office of Science and Technology",
        "license": "U.S. government work, public domain",
        "fetched": "2026-08-20",
        "fit": ("p_a is REAL, in the same computed-ratio sense as fda_cmc's "
               "p_a: the actual fraction of year-over-year transitions in "
               "which a species' national commercial landings (pounds) fell "
               "versus the prior year, computed from 1,143 real "
               "(species, year) landings records across 16 species, most "
               "spanning 1950-2024. v_a (mean real price per landed pound, "
               "dollars/pounds -- both raw reported fields) and e_a "
               "(a species' total real landed value, normalized to the "
               "16-species mean, as a proxy for the economic scale lost if "
               "that stock collapsed) are documented proxies for what this "
               "bandit's value and externality mean for a real fishery, not "
               "NOAA-provided cost or severity figures."),
    },
}

_SEVERITY_WEIGHT = {"class_i": 3.0, "class_ii": 1.0, "class_iii": 0.3}


def _load(name):
    with open(os.path.join(_DATA_DIR, name)) as f:
        return json.load(f)


def derive_real_amr_parameters(indicator="both", n=None, seed=0):
    """Real (v, p, e) from WHO GLASS-fed antimicrobial-resistance surveillance.

    Each arm is one (country, year) row. `p_a` is the real measured
    resistance proportion for that row. `v_a = 1 - p_a` and
    `e_a = p_a` are documented proxies (see SOURCES["who_amr"]["fit"]):
    a resistant-heavy context is modeled as lower immediate clearance value
    and larger permanent loss if resistance is allowed to entrench further --
    a plausible reading of the real number, not itself a WHO measurement.

    Important consequence of that specific choice, checked rather than
    assumed: `kappa = p*e = p^2` is then a deterministic, monotone function
    of `v = 1-p`, so `corr(v, kappa)` comes out close to -1 (verified: -0.96
    at n=200, seed=0) regardless of which real rows feed it. That is exactly
    the ALIGNED regime this repo's own `platform-trial`/`aligned-control`
    scenarios document as the negative control where greedy is already
    near-optimal -- so greedy and ECI will typically coincide here, by
    construction of the proxy formula, not because of anything discovered
    about real-world antibiotic resistance. This is reported plainly rather
    than replaced with a formula tuned to manufacture divergence.

    `indicator`: "mrsa", "ecoli", or "both" (default, concatenates both).
    `n`: optional subsample size (seeded) for a tractable large-N demo;
    `None` uses every available row (~1,005 for "both").
    """
    if indicator == "both":
        rows = _load("who_amr_mrsa.json") + _load("who_amr_ecoli.json")
    elif indicator == "mrsa":
        rows = _load("who_amr_mrsa.json")
    elif indicator == "ecoli":
        rows = _load("who_amr_ecoli.json")
    else:
        raise ValueError(f"unknown indicator {indicator!r}; "
                         f"expected 'mrsa', 'ecoli', or 'both'")

    if n is not None and n < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in idx]

    p = np.clip(np.array([r["resistance_pct"] / 100.0 for r in rows]), 1e-6, 1.0)
    v = np.clip(1.0 - p, 0.05, None)
    e = p.copy()
    labels = [f"{r['country']}-{r['year']}" for r in rows]
    return v, p, e, labels


def derive_real_cmc_parameters():
    """Real (v, p, e) from openFDA drug-recall classification by failure
    category. Each arm is one failure category (CGMP, sterility,
    contamination, ...). `p_a` is the real fraction of that category's
    recalls classified Class I. `e_a`/`v_a` are documented proxies (see
    SOURCES["fda_cmc"]["fit"]).
    """
    cats = _load("fda_cmc_categories.json")
    p, e, v, labels = [], [], [], []
    total_all = sum(c["total"] for c in cats)
    for c in cats:
        total = max(c["total"], 1)
        p.append(c["class_i"] / total)
        severity = sum(_SEVERITY_WEIGHT[k] * c[k]
                       for k in ("class_i", "class_ii", "class_iii")) / total
        e.append(severity)
        v.append(0.4 + 0.6 * (c["total"] / total_all))
        labels.append(c["category"])
    return (np.array(v), np.clip(np.array(p), 1e-6, 1.0), np.array(e), labels)


def derive_real_fisheries_parameters(min_years=10):
    """Real (v, p, e) from NOAA FOSS commercial landings. Each arm is one
    species. `p_a` is the real year-over-year landings-decline frequency
    (see SOURCES["noaa_fisheries"]["fit"]); `v_a`/`e_a` are documented
    proxies from real price and total-value data.

    Unlike antibiotic-stewardship-real's proxy (which mechanically forces
    the ALIGNED regime -- see derive_real_amr_parameters's docstring), this
    one lands in the opposite, more consequential regime by nothing more
    than what the real data says: `corr(v, kappa)` comes out positive
    (checked, not assumed: +0.74 across all 16 species) -- the species with
    the highest real per-pound price and total value (lobster, sea scallop)
    are also the ones with the highest kappa. That is exactly the CONFLICT
    regime this repo's central claim is about: greedy naturally prefers
    high-value arms, and here the high-value arms are the ones a real
    fishery can least afford to over-exploit.

    `min_years`: drop any species with fewer than this many years of
    landings data (only affects Atlantic menhaden in the current snapshot,
    at 18 years -- every other species has 75).
    """
    species = _load("noaa_fisheries_landings.json")
    v, p, e_raw, labels = [], [], [], []
    for sp in species:
        annual = sp["annual"]
        if len(annual) < min_years:
            continue
        pounds = [a["pounds"] for a in annual]
        dollars = [a["dollars"] for a in annual]
        declines = sum(1 for i in range(1, len(pounds)) if pounds[i] < pounds[i - 1])
        p.append(declines / (len(pounds) - 1))
        prices = [d / lb for lb, d in zip(pounds, dollars) if lb > 0]
        v.append(sum(prices) / len(prices))
        e_raw.append(sum(dollars))
        labels.append(sp["species"])
    e_raw = np.array(e_raw)
    e = e_raw / e_raw.mean()
    return np.array(v), np.clip(np.array(p), 1e-6, 1.0), e, labels
