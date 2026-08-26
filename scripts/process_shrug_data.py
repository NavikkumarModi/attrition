"""Process real, manually-downloaded SHRUG (Socioeconomic High-resolution
Rural-Urban Geographic platform for India) tables into
attrition/data/shrug_con08.json -- one row per real 2008-delimitation
Assembly Constituency (AC), joining four indicator categories the
election track had none of before: demographic (real Census 2011,
finer-grained than the district-level table `process_census_data.py`
already processes), economic (real 2012 Socio-Economic Caste Census
consumption/poverty estimates -- genuinely new indicator TYPE, nothing
like it existed in this repo), a satellite-derived wealth proxy (2021
Facebook Relative Wealth Index), and infrastructure (PMGSY rural road
connectivity).

Source: Development Data Lab, https://devdatalab.org/shrug_download/,
SHRUG v2.2 ("pakora"). Licensed CC BY-NC-SA 4.0. Like Census
(`process_census_data.py`), this is NOT a stable scriptable URL -- SHRUG
serves signed, expiring S3 links generated fresh per page visit, not a
public API -- so the raw CSVs must be downloaded by hand from the page
above and placed at `data/shrug/` (repo-root `data/`, gitignored, same
"raw manual download" convention as Census; NOT `attrition/data/`, which
holds only this script's small, processed, checked-in output):

    data/shrug/ac08_name_key.csv               (Constituency keys module)
    data/shrug/shrid_frag_con08_key.csv         (Constituency keys module)
    data/shrug/pc11_pca_clean_con08.csv         (Population Census module)
    data/shrug/secc_cons_rural_con08.csv        (SECC consumption module)
    data/shrug/secc_cons_urban_con08.csv        (SECC consumption module)
    data/shrug/facebook_rwi_con08.csv           (Facebook wealth module)
    data/shrug/pmgsy_2015_shrid.csv             (PMGSY roads module)

Real design choice, checked before writing anything: SHRUG already
publishes most of these modules PRE-AGGREGATED to `ac08_id` (2008-
delimitation Assembly Constituency) -- no shrid-level rollup needed for
demographic, economic, or wealth-proxy data, so nothing here re-derives
what SHRUG itself already computed and QA'd. PMGSY is the one exception:
it ships only at shrid (village) level, so this script aggregates it
using SHRUG's OWN `shrid_frag_con08_key.csv` population-fragment weights
(`fragment_wt_con08_norm`) -- the same weighting scheme SHRUG's own
con08-level tables are built with, not an invented one. The metric
computed is `pmgsy_pct_villages_connected`: the population-weighted
share of a constituency's villages with at least one completed (new or
upgraded) PMGSY road as of the 2015 snapshot -- a real connectivity
coverage rate, not a summed road length (summing "earliest road per
village" lengths across villages would be a harder-to-interpret,
somewhat arbitrary number; a coverage rate is not).

Two rural/urban indicator pairs (SECC consumption, poverty rate) are
kept SEPARATE per constituency, never combined into one number -- the
same "don't fabricate a combined figure without a real population-
weighting justification" discipline `census.py`'s module docstring
states for district rollups. A constituency missing one segment (e.g. a
fully rural AC with no urban SECC row) simply has that field absent, not
defaulted to zero or averaged away.

Run:  pip install "attrition[analysis]"   (pandas)
      python scripts/process_shrug_data.py
"""
import json
import os

import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(REPO_ROOT, "data", "shrug")
OUT_PATH = os.path.join(REPO_ROOT, "attrition", "data", "shrug_con08.json")


def _pmgsy_connectivity_by_ac():
    """Population-fragment-weighted share of villages with a completed
    PMGSY road, per ac08_id -- the one indicator here that needs
    shrid-to-AC aggregation, done with SHRUG's own published weights."""
    pmgsy = pd.read_csv(os.path.join(RAW_DIR, "pmgsy_2015_shrid.csv"),
                        usecols=["shrid2", "road_comp_date_new", "road_comp_date_upg"])
    pmgsy["connected"] = (pmgsy["road_comp_date_new"].notna()
                          | pmgsy["road_comp_date_upg"].notna()).astype(float)

    weights = pd.read_csv(
        os.path.join(RAW_DIR, "shrid_frag_con08_key.csv"),
        usecols=["shrid2", "ac08_id", "fragment_wt_con08_norm"])

    merged = weights.merge(pmgsy[["shrid2", "connected"]], on="shrid2", how="inner")
    merged["weighted_connected"] = merged["connected"] * merged["fragment_wt_con08_norm"]

    grouped = merged.groupby("ac08_id").agg(
        pmgsy_pct_villages_connected=("weighted_connected", "sum"),
        pmgsy_weight_total=("fragment_wt_con08_norm", "sum"),
        pmgsy_n_villages_matched=("shrid2", "count"),
    )
    # normalise by the actual matched weight, not assumed-1.0 -- a shrid's
    # con08 fragment weights need not sum to exactly 1 across every AC it
    # touches once rows with no PMGSY match are dropped by the inner join.
    grouped["pmgsy_pct_villages_connected"] = (
        grouped["pmgsy_pct_villages_connected"] / grouped["pmgsy_weight_total"])
    return grouped.drop(columns="pmgsy_weight_total").reset_index()


def main():
    keys = pd.read_csv(os.path.join(RAW_DIR, "ac08_name_key.csv"),
                       usecols=["ac08_id", "ac08_name", "pc01_state_name"])

    pca = pd.read_csv(os.path.join(RAW_DIR, "pc11_pca_clean_con08.csv"), usecols=[
        "ac08_id", "pc11_pca_tot_p", "pc11_pca_tot_m", "pc11_pca_tot_f",
        "pc11_pca_no_hh", "pc11_pca_p_sc", "pc11_pca_p_st",
        "pc11_pca_p_lit", "pc11_pca_p_ill", "pc11_pca_tot_work_p"])
    pca = pca.rename(columns={
        "pc11_pca_tot_p": "population_total", "pc11_pca_tot_m": "population_male",
        "pc11_pca_tot_f": "population_female", "pc11_pca_no_hh": "n_households",
        "pc11_pca_p_sc": "population_sc", "pc11_pca_p_st": "population_st",
        "pc11_pca_p_lit": "population_literate", "pc11_pca_p_ill": "population_illiterate",
        "pc11_pca_tot_work_p": "population_working"})

    secc_rural = pd.read_csv(os.path.join(RAW_DIR, "secc_cons_rural_con08.csv"), usecols=[
        "ac08_id", "secc_cons_pc_rural", "secc_pov_rate_rural"])
    secc_urban = pd.read_csv(os.path.join(RAW_DIR, "secc_cons_urban_con08.csv"), usecols=[
        "ac08_id", "secc_cons_pc_urban", "secc_pov_rate_urban"])

    facebook = pd.read_csv(os.path.join(RAW_DIR, "facebook_rwi_con08.csv"), usecols=[
        "ac08_id", "facebook_mean_rwi"])

    pmgsy = _pmgsy_connectivity_by_ac()

    merged = keys
    for df in (pca, secc_rural, secc_urban, facebook, pmgsy):
        merged = merged.merge(df, on="ac08_id", how="left")

    merged = merged.rename(columns={"ac08_name": "ac_name", "pc01_state_name": "state"})
    records = json.loads(merged.to_json(orient="records"))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=1)

    print(f"wrote {OUT_PATH}")
    print(f"  {len(records)} real Assembly Constituencies (2008 delimitation)")
    n_secc_rural = merged["secc_cons_pc_rural"].notna().sum()
    n_secc_urban = merged["secc_cons_pc_urban"].notna().sum()
    n_facebook = merged["facebook_mean_rwi"].notna().sum()
    n_pmgsy = merged["pmgsy_pct_villages_connected"].notna().sum()
    print(f"  SECC rural consumption matched: {n_secc_rural}")
    print(f"  SECC urban consumption matched: {n_secc_urban}")
    print(f"  Facebook wealth index matched:  {n_facebook}")
    print(f"  PMGSY connectivity matched:     {n_pmgsy}")


if __name__ == "__main__":
    main()
