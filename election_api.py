"""A thin FastAPI wrapper around `attrition` -- originally scoped to just
the election track, now the backend for the unified frontend (`frontend/`)
covering every world (see `attrition.scenarios.worlds()`): election,
clinical-amr, agent-ecosystem, environment, cybersecurity, synthetic. No
new simulation logic lives here, only JSON plumbing over functions already
proven by `tests/`, `examples/`, and `experiments/`.

Run:  uvicorn election_api:app --reload --port 8000
(pip install fastapi "uvicorn[standard]" -- not in requirements.txt, since
the core package and its Streamlit app (`app.py`, still the simpler
single-file deployment option -- see README's "Interactive app" section)
don't need a web server to run the simulator itself; this is an
additional, optional surface.)
"""
import dataclasses
import json
import os
import statistics
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from attrition import (ConsumableBandit, Greedy, ECI, SECI, UCB, ThompsonSampling,
                       Conservative, compare, run, Population, simulate_population,
                       SimultaneousPool, simulate_population_simultaneous,
                       planner_value_simultaneous, decentralised_value_simultaneous,
                       derive_campaign_parameters, SCENARIOS, get_arrays, worlds,
                       WORLDS, PHARMA_PERSONAS, CAMPAIGN_PERSONAS, AGENT_PERSONAS,
                       CYBERSECURITY_PERSONAS, FISHERIES_PERSONAS,
                       CallableLLMClient, AgentGraph)
from attrition.llm_policy import _CHOICE
from attrition.election import (ALL_PHASE1_CONSTITUENCIES, build_constituency,
                                real_margin_pct, tipping_point, recompute_outcome,
                                simulate_level1_contact, simulate_level2_contact,
                                simulate_level1_contact_with_network,
                                get_constituency, geojson_feature_collection,
                                greedy_allocation, even_allocation,
                                leave_one_state_out_backtest, train_gradient_bandit,
                                assembly_rollup_for_seat, assembly_rollup_coverage,
                                ASSEMBLY_YEAR_BY_STATE,
                                census_rollup_for_seat, census_rollup_coverage,
                                simulate_level1_contact_by_district,
                                shrug_rollup_for_seat, shrug_rollup_coverage)
from experiments.exp65_amr_five_policy_reference_table import (
    in_repo_half, EXTERNAL_SILOED, EXTERNAL_SHARED)

app = FastAPI(title="ATTRITION API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "attrition", "data", "calibration_2014_2019.json")


def _seat_id(c):
    return f"{c.state}|{c.name}"


@lru_cache(maxsize=1)
def _tipping_points():
    """Computed once per process, not per request -- ~500 steps x 543
    seats of pure arithmetic (a few seconds), identical to
    examples/18_national_election_scan.py's pass 1."""
    return {_seat_id(c): tipping_point(c) for c in ALL_PHASE1_CONSTITUENCIES}


def _summary(c):
    tp = _tipping_points()[_seat_id(c)]
    return {
        "id": _seat_id(c), "state": c.state, "name": c.name, "note": c.note,
        "constituency_type": c.constituency_type, "sub_region": c.sub_region,
        "electors": c.electors, "turnout_pct": c.turnout_pct,
        "real_margin_pct": real_margin_pct(c),
        "tipping_point_pct": tp * 100 if tp is not None else None,
        "winner_party": c.parties[0], "winner_share_pct": c.vote_share_pct[0],
        "runner_up_party": c.parties[1], "runner_up_share_pct": c.vote_share_pct[1],
        "winner_sex": c.candidate_sex[0], "winner_education": c.candidate_education[0],
        "winner_profession": c.candidate_profession[0],
        "winner_incumbent": c.candidate_incumbent[0],
        "winner_no_terms": c.candidate_no_terms[0],
        "winner_turncoat": c.candidate_turncoat[0],
    }


@app.get("/api/constituencies")
def list_constituencies():
    """All 543 real seats, summarised -- enough for the national overview
    table/scatter without a per-seat round trip."""
    return [_summary(c) for c in ALL_PHASE1_CONSTITUENCIES]


@lru_cache(maxsize=1)
@app.get("/api/constituencies.geojson")
def constituencies_geojson():
    """All 543 real seats, joined to their real boundary polygons
    (`attrition.election.geo`, DataMeet -- CC BY 4.0, attribution required)
    for the frontend's national map. Computed once per process; the join
    itself is deterministic and doesn't depend on request state."""
    seats = {(c.state, c.name): _summary(c) for c in ALL_PHASE1_CONSTITUENCIES}
    fc, unmatched = geojson_feature_collection(seats)
    if unmatched:
        raise HTTPException(status_code=500,
                            detail=f"{len(unmatched)} seats failed to match a "
                                   f"boundary polygon: {unmatched[:5]}")
    return fc


def _resolve(state, name):
    try:
        return build_constituency(name, state=state)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/constituencies/{state}/{name}")
def constituency_detail(state: str, name: str):
    c = _resolve(state, name)
    detail = _summary(c)
    detail["parties"] = list(c.parties)
    detail["vote_share_pct"] = list(c.vote_share_pct)
    return detail


@app.get("/api/constituencies/{state}/{name}/assembly-rollup")
def constituency_assembly_rollup(state: str, name: str):
    """Real state Assembly election results, rolled up from this real
    seat's constituent Assembly Constituencies via DataMeet's real
    AC-to-PC crosswalk (`attrition.election.assembly`) -- not a new
    mechanism, a second real data source joined to the first. `null` for
    the 50/543 real seats with no matched constituent AC (see that
    module's docstring for the honest, disclosed coverage numbers)."""
    _resolve(state, name)  # 404s on an unknown seat before touching the rollup
    rollup = assembly_rollup_for_seat(state, name)
    return {"seat": {"state": state, "name": name}, "rollup": rollup}


@lru_cache(maxsize=1)
@app.get("/api/assembly-rollup/coverage")
def assembly_rollup_coverage_endpoint():
    """Real, checked coverage numbers for the AE-to-PC join, plus the real
    assembly-election year used per state and its gap from 2019 -- state
    assembly elections happen on each state's own cycle, not one national
    date, so this must be surfaced wherever rollup numbers are shown, not
    silently treated as "the same year" as the Lok Sabha data."""
    coverage = assembly_rollup_coverage()
    coverage["year_by_state"] = {
        state: {"year": year, "gap_from_2019": gap}
        for state, (year, gap) in ASSEMBLY_YEAR_BY_STATE().items()
    }
    return coverage


@app.get("/api/constituencies/{state}/{name}/census-rollup")
def constituency_census_rollup(state: str, name: str):
    """Real Census 2011 district-level demographics (population,
    households, area, density -- table A-1) for this real seat's
    constituent district(s), via the same real DataMeet AC-to-PC
    crosswalk `assembly-rollup` uses, this time joined on its `district`
    field (`attrition.election.census`). A seat spanning more than one
    district reports each district's real, UNMODIFIED figures plus a
    mean (never a sum -- see that module's docstring for why summing
    would double-count) -- `null` for the ~19% of real seats with no
    matched constituent district."""
    _resolve(state, name)
    rollup = census_rollup_for_seat(state, name)
    return {"seat": {"state": state, "name": name}, "rollup": rollup}


@lru_cache(maxsize=1)
@app.get("/api/census-rollup/coverage")
def census_rollup_coverage_endpoint():
    """Real, checked coverage numbers for the district-to-PC join."""
    return census_rollup_coverage()


@app.get("/api/constituencies/{state}/{name}/shrug-rollup")
def constituency_shrug_rollup(state: str, name: str):
    """Real SHRUG (Socioeconomic High-resolution Rural-Urban Geographic
    platform) indicators for this real seat's constituent 2008-
    delimitation Assembly Constituency/Constituencies -- demographic
    (Census 2011), economic (2012 SECC consumption/poverty, the first
    genuinely economic indicator in this repo), a 2021 satellite-derived
    wealth proxy, and PMGSY rural road connectivity
    (`attrition.election.shrug`). Population counts are real sums;
    rate/index fields are population-weighted means -- see that module's
    docstring. `null` for real seats with no matched post-2008-
    delimitation AC (a stricter, disclosed filter than assembly-rollup/
    census-rollup use, see FINDINGS.md #57)."""
    _resolve(state, name)
    rollup = shrug_rollup_for_seat(state, name)
    return {"seat": {"state": state, "name": name}, "rollup": rollup}


@lru_cache(maxsize=1)
@app.get("/api/shrug-rollup/coverage")
def shrug_rollup_coverage_endpoint():
    """Real, checked coverage numbers for the SHRUG AC-to-PC join."""
    return shrug_rollup_coverage()


class SimulateRequest(BaseModel):
    state: str
    name: str
    intensity_sequence: list[float] = [0.15, 0.2, 0.25, 0.3, 0.35]
    n_voters: int = 3000
    seed: int = 0
    include_level2: bool = True
    # -- granular controls added for FINDINGS.md#47; see that section and
    # this file's module docstring for what's real vs. what would require
    # inventing data the source doesn't have.
    n_seeds: int = 1                          # >1 -> flip-probability sweep
    use_network: bool = False                 # voter-to-voter peer propagation
    peer_conversion_prob: float = 0.05
    graph_density: float = 0.02                # edge probability -- exact
                                               # Erdos-Renyi, used as-is below
                                               # _LARGE_POPULATION_THRESHOLD
    mean_degree: Optional[float] = None       # expected neighbors/voter --
                                               # used instead of graph_density
                                               # at/above that threshold (or
                                               # if set explicitly at any
                                               # size); see FINDINGS.md#52
                                               # for why a fixed edge
                                               # probability stops making
                                               # sense at real scale
    incumbent_override: Optional[bool] = None  # counterfactual: real real seat's
                                               # actual incumbency status overridden
    margin_override_pct: Optional[float] = None  # counterfactual real margin
    use_district_granularity: bool = False    # split n_voters across real
                                               # constituent districts (FINDINGS.md#51)


def _with_overrides(c, incumbent_override, margin_override_pct):
    """Build a counterfactual `Constituency` (a real seat with one or two
    real, already-causally-wired fields overridden) via `dataclasses.
    replace` -- never inventing a field the real data doesn't have, only
    hypothesising a different value for one that's already real and already
    used by `mean_resistance_for`/`persuadable_fraction_from_margin`.
    `Constituency` is `@dataclass(frozen=True)`, so this never mutates the
    real, cached `ALL_PHASE1_CONSTITUENCIES` entries."""
    if incumbent_override is not None:
        ci = list(c.candidate_incumbent)
        ci[0] = bool(incumbent_override)
        c = dataclasses.replace(c, candidate_incumbent=tuple(ci))
    if margin_override_pct is not None:
        # Symmetric reallocation between the real winner and real runner-up
        # only -- every other real candidate's share is left untouched, and
        # winner+runner_up's combined share is preserved. Stated as the
        # deliberate reallocation rule it is, the same discipline
        # outcome.py's own reallocation rule documents for itself.
        shares = list(c.vote_share_pct)
        current_margin = shares[0] - shares[1]
        delta = float(margin_override_pct) - current_margin
        shares[0] = max(0.0, shares[0] + delta / 2.0)
        shares[1] = max(0.0, shares[1] - delta / 2.0)
        c = dataclasses.replace(c, vote_share_pct=tuple(shares))
    return c


def _seat_summary_live(c):
    """Like `_summary`, but recomputes `tipping_point` directly from `c`
    instead of the national `_tipping_points()` cache -- required whenever
    `c` might be a counterfactual copy (overridden margin/incumbency), since
    the cache is keyed by the real seat's identity and would silently
    return the REAL seat's tipping point for a hypothetical one otherwise.
    Cheap enough for a single seat (~10-20ms) to always do live here."""
    tp = tipping_point(c)
    return {
        "id": _seat_id(c), "state": c.state, "name": c.name, "note": c.note,
        "constituency_type": c.constituency_type, "sub_region": c.sub_region,
        "electors": c.electors, "turnout_pct": c.turnout_pct,
        "real_margin_pct": real_margin_pct(c),
        "tipping_point_pct": tp * 100 if tp is not None else None,
        "winner_party": c.parties[0], "winner_share_pct": c.vote_share_pct[0],
        "runner_up_party": c.parties[1], "runner_up_share_pct": c.vote_share_pct[1],
        "winner_incumbent": c.candidate_incumbent[0],
    }


# Real, benchmarked bounds (FINDINGS.md#52), not arbitrary round numbers:
# plain/district-granularity Level 1 (agents.py's per-voter Python loop)
# runs 1,000,000 voters x 5 rounds in ~0.5s. The network path is bounded
# lower because graph CONSTRUCTION, not the contact simulation itself,
# dominates its cost -- AgentGraph.random_by_degree builds a 500,000-node,
# mean-degree-10 graph in ~2s; the O(n^2) AgentGraph.random this path used
# to always use is not an option at these sizes at all (see network.py's
# docstring: n=1,000,000 at graph_density=0.02 implies ~10 billion edges,
# not merely slow to build).
_MAX_N_VOTERS = 1_000_000
_MAX_N_VOTERS_NETWORK = 500_000
_LARGE_POPULATION_THRESHOLD = 5_000  # AgentGraph.random's O(n^2) pair-test
                                     # is still fast below this; above it,
                                     # switch to the mean-degree generator
_MAX_MEAN_DEGREE = 100.0             # sanity cap: mean_degree, explicit or
                                     # derived, is clamped here
_DEFAULT_LARGE_POPULATION_MEAN_DEGREE = 10.0  # NOT derived from
    # graph_density * n_voters: that formula reproduces graph_density's
    # real meaning only near the small-n scale it was calibrated at (e.g.
    # 0.02 * 500 = 10, this same constant); above the threshold it would
    # already exceed _MAX_MEAN_DEGREE and get silently clamped for almost
    # any n_voters, making graph_density look like it still controls
    # density when it's actually just hitting the cap every time. A fixed
    # default is the honest choice once the parameterisation itself has
    # changed (see network.py's AgentGraph.random_by_degree docstring).


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Level 1 (always) + Level 2 (optional) campaign simulation over one
    real constituency, plus the recomputed outcome -- the same three
    functions `examples/17_election_outcome.py` chains by hand, now with
    granular controls: voter-to-voter network propagation
    (`attrition.election.simulate_level1_contact_with_network`), a
    flip-probability sweep across `n_seeds` seeds instead of one
    deterministic run, two real counterfactual overrides (incumbency,
    margin), and population granularity -- splitting `n_voters` across the
    seat's real constituent districts instead of one undifferentiated pool
    (`attrition.election.simulate_level1_contact_by_district`, FINDINGS.md
    #51) -- see `_with_overrides`'s and that function's docstrings for
    exactly what each control does and doesn't claim. `use_network` and
    `use_district_granularity` CAN now be combined (FINDINGS.md#53):
    propagation runs independently within each constituent district (no
    cross-district edges -- a real, stated scoping choice, see
    `simulate_level1_contact_by_district`'s docstring), not seat-wide."""
    c = _resolve(req.state, req.name)
    c = _with_overrides(c, req.incumbent_override, req.margin_override_pct)
    n_voters_cap = _MAX_N_VOTERS_NETWORK if req.use_network else _MAX_N_VOTERS
    n_voters = max(50, min(req.n_voters, n_voters_cap))
    n_seeds = max(1, min(req.n_seeds, 50))

    def run_once(seed):
        """Returns (fractions_by_round_or_None, final, extra) where
        `extra` is a small dict tagged by `kind` ("network"/"district"/
        None) -- a tagged structure instead of a bare positional value,
        since the two granular paths return shapes too different
        (a float vs. a list of per-district results) to safely share one
        untagged slot."""
        if req.use_district_granularity:
            mean_degree = min(req.mean_degree or _DEFAULT_LARGE_POPULATION_MEAN_DEGREE,
                              _MAX_MEAN_DEGREE)
            per_district, final = simulate_level1_contact_by_district(
                c, req.intensity_sequence, n_voters=n_voters, seed=seed,
                use_network=req.use_network,
                peer_conversion_prob=req.peer_conversion_prob,
                mean_degree=mean_degree)
            return None, final, {"kind": "district", "per_district": per_district,
                                 "network_within_district": req.use_network}
        if req.use_network:
            voter_ids = [f"v{i}" for i in range(n_voters)]
            if req.mean_degree is not None or n_voters >= _LARGE_POPULATION_THRESHOLD:
                mean_degree = (req.mean_degree if req.mean_degree is not None
                              else _DEFAULT_LARGE_POPULATION_MEAN_DEGREE)
                mean_degree = min(mean_degree, _MAX_MEAN_DEGREE)
                graph = AgentGraph.random_by_degree(voter_ids, mean_degree=mean_degree,
                                                    seed=seed)
                graph_method, graph_param = "random_by_degree", mean_degree
            else:
                graph = AgentGraph.random(voter_ids, p=req.graph_density, seed=seed)
                graph_method, graph_param = "random", req.graph_density
            fractions, final, direct_only = simulate_level1_contact_with_network(
                c, req.intensity_sequence, n_voters=n_voters, graph=graph,
                peer_conversion_prob=req.peer_conversion_prob, seed=seed)
            return fractions, final, {"kind": "network", "direct_only": direct_only,
                                      "graph_method": graph_method, "graph_param": graph_param}
        fractions, final = simulate_level1_contact(
            c, req.intensity_sequence, n_voters=n_voters, seed=seed)
        return fractions, final, None

    fractions, final, extra = run_once(req.seed)
    outcome = recompute_outcome(c, final)
    result = {
        "seat": _seat_summary_live(c),
        "level1_switched_fraction_by_round": fractions,
        "level1_final_switched_fraction": final,
        "outcome": {
            "winner": outcome.winner, "winner_share_pct": outcome.winner_share_pct,
            "runner_up": outcome.runner_up, "runner_up_share_pct": outcome.runner_up_share_pct,
            "margin_pct": outcome.margin_pct, "flipped": outcome.flipped,
        },
    }
    if extra and extra["kind"] == "network":
        result["network"] = {"final_switched_fraction": final,
                             "direct_only_final_switched_fraction": extra["direct_only"],
                             "peer_conversion_prob": req.peer_conversion_prob,
                             "graph_density": req.graph_density,
                             "graph_method": extra["graph_method"],
                             "graph_param": extra["graph_param"]}
    elif extra and extra["kind"] == "district":
        result["district_granularity"] = {"per_district": extra["per_district"],
                                          "combined_final_switched_fraction": final,
                                          "network_within_district": extra["network_within_district"]}
    if req.include_level2:
        result["level2_decisions"] = simulate_level2_contact(c, req.intensity_sequence)

    if n_seeds > 1:
        flips = 0
        finals = []
        for s in range(n_seeds):
            _, final_s, _ = run_once(s)
            outcome_s = recompute_outcome(c, final_s)
            flips += int(outcome_s.flipped)
            finals.append(final_s)
        result["flip_probability"] = {
            "n_seeds": n_seeds, "flip_rate": flips / n_seeds,
            "final_switched_fraction_mean": sum(finals) / n_seeds,
            "final_switched_fraction_min": min(finals),
            "final_switched_fraction_max": max(finals),
        }
    return result


@app.get("/api/national-scan")
@lru_cache(maxsize=1)
def national_scan():
    """The distribution `examples/18_national_election_scan.py`'s pass 1
    reports -- tipping points across all 543 real seats -- as JSON."""
    tps = _tipping_points()
    values = sorted(v for v in tps.values() if v is not None)
    n = len(values)

    def pct(p):
        return values[min(int(p / 100.0 * n), n - 1)] * 100

    by_id = {_seat_id(c): c for c in ALL_PHASE1_CONSTITUENCIES}
    ranked = sorted(((seat_id, tp) for seat_id, tp in tps.items() if tp is not None),
                    key=lambda kv: kv[1])
    most_contestable = [_summary(by_id[seat_id]) for seat_id, _ in ranked[:10]]
    safest = [_summary(by_id[seat_id]) for seat_id, _ in ranked[-10:]]
    return {
        "n_seats": len(tps), "n_with_tipping_point": n,
        "distribution_pct": {"min": values[0] * 100, "p10": pct(10),
                             "median": pct(50), "p90": pct(90),
                             "max": values[-1] * 100},
        "most_contestable": most_contestable, "safest_holds": safest,
    }


@app.get("/api/calibration")
def calibration():
    """Precomputed 2014->2019 real-swing-vs-tipping-point comparison --
    generated once by scripts/calibrate_campaign_magnitudes.py (a live
    network fetch of 2014 data), served from the static JSON it wrote, not
    re-fetched per request."""
    if not os.path.exists(_CALIBRATION_PATH):
        raise HTTPException(
            status_code=503,
            detail="calibration_2014_2019.json not found -- run "
                  "scripts/calibrate_campaign_magnitudes.py first")
    with open(_CALIBRATION_PATH) as f:
        return json.load(f)


_TOTAL_NATIONAL_ELECTORS = sum(c.electors for c in ALL_PHASE1_CONSTITUENCIES)


def _society_metrics(targeted, flipped):
    """Real, closed-form metrics over the seats a strategy TARGETED --
    "what does spending this budget cost/reach, societally" -- deliberately
    built only from fields already real and already validated (electors,
    state, tipping_point), not from a new mechanistic run. Wiring this to
    the intensity-sequence mechanism (simulate_level1_contact) would
    require inventing a budget-to-intensity-sequence mapping that doesn't
    exist anywhere in this codebase -- see FINDINGS.md#48 for why that
    was deliberately not attempted, the same discipline that keeps the
    persuasion mechanism and the forecast baseline as two separate
    questions rather than one merged (overclaiming) one."""
    if not targeted:
        return {"efficiency": None, "electors_in_scope": 0,
               "electors_in_scope_pct_national": 0.0, "n_states_touched": 0,
               "states_touched": [], "cost_mean_pct": None, "cost_dispersion_pct": 0.0}
    tps = _tipping_points()
    costs = [tps[_seat_id(c)] * 100.0 for c in targeted if tps[_seat_id(c)] is not None]
    electors_in_scope = sum(c.electors for c in targeted)
    states_touched = sorted({c.state for c in targeted})
    return {
        "efficiency": len(flipped) / len(targeted),
        "electors_in_scope": electors_in_scope,
        "electors_in_scope_pct_national": 100.0 * electors_in_scope / _TOTAL_NATIONAL_ELECTORS,
        "n_states_touched": len(states_touched),
        "states_touched": states_touched,
        "cost_mean_pct": statistics.fmean(costs) if costs else None,
        "cost_dispersion_pct": statistics.pstdev(costs) if len(costs) > 1 else 0.0,
    }


def _allocation_response(result):
    return {
        "n_targeted": len(result.targeted), "n_flipped": len(result.flipped),
        "budget_spent": result.budget_spent, "budget_remaining": result.budget_remaining,
        "flipped": [_summary(c) for c in result.flipped[:25]],
        "society": _society_metrics(result.targeted, result.flipped),
    }


@lru_cache(maxsize=1)
def _filter_facets():
    """Real, distinct values for every filterable seat covariate -- for the
    frontend's filter dropdowns, so options are never hand-typed/guessed."""
    return {
        "constituency_type": sorted({c.constituency_type for c in ALL_PHASE1_CONSTITUENCIES}),
        "sub_region": sorted({c.sub_region for c in ALL_PHASE1_CONSTITUENCIES
                             if c.sub_region}),
    }


@app.get("/api/strategist/filters")
def strategist_filters():
    """Real, distinct values this seat is filterable by -- see
    `_filter_facets`."""
    return _filter_facets()


def _filtered_constituencies(constituency_type, sub_region, incumbency):
    """Restrict the real 543-seat pool to those matching every given real
    covariate, before handing the subset to `greedy_allocation`/
    `even_allocation` (both already accept a `constituencies=` subset --
    no change to that module was needed). `incumbency`: "any" (default),
    "incumbent" (real 2019 winner was a real incumbent), or "challenger"
    (was not) -- real, per-seat `candidate_incumbent[0]`, not invented."""
    cs = ALL_PHASE1_CONSTITUENCIES
    if constituency_type:
        cs = [c for c in cs if c.constituency_type == constituency_type]
    if sub_region:
        cs = [c for c in cs if c.sub_region == sub_region]
    if incumbency == "incumbent":
        cs = [c for c in cs if c.candidate_incumbent[0]]
    elif incumbency == "challenger":
        cs = [c for c in cs if not c.candidate_incumbent[0]]
    return cs


@app.get("/api/strategist")
def strategist(budget_pct: float = 20.0, scope_max_tipping_pct: float = 20.0,
              constituency_type: Optional[str] = None,
              sub_region: Optional[str] = None,
              incumbency: str = "any"):
    """The national campaign-strategist agent (`attrition.election.
    strategist`): given one total budget, expressed in percent-of-
    electorate units (the same unit `tipping_point` already uses
    everywhere), compares fully-funding the cheapest real seats first
    (`greedy`) against spreading the identical budget evenly across every
    seat within `scope_max_tipping_pct` (`even`). A new resource-
    ALLOCATION decision layered on Level 0/1/2, not a re-application of
    Theorems 1-6 -- see strategist.py's module docstring.

    Optionally restricted to real seats matching `constituency_type`
    (GEN/SC/ST), `sub_region`, and/or `incumbency` ("any"/"incumbent"/
    "challenger") before allocating -- targeting by real, published
    covariates, not invented demographic/economic data (see this file's
    module docstring and FINDINGS.md#47 for why the latter isn't offered).
    """
    if incumbency not in ("any", "incumbent", "challenger"):
        raise HTTPException(status_code=400,
                            detail="incumbency must be 'any', 'incumbent', "
                                   "or 'challenger'")
    cs = _filtered_constituencies(constituency_type, sub_region, incumbency)
    greedy = greedy_allocation(budget_pct, constituencies=cs)
    even = even_allocation(budget_pct, scope_max_tipping_pct=scope_max_tipping_pct,
                           constituencies=cs)
    return {"budget_pct": budget_pct, "scope_max_tipping_pct": scope_max_tipping_pct,
           "n_seats_in_scope": len(cs),
           "filters": {"constituency_type": constituency_type,
                      "sub_region": sub_region, "incumbency": incumbency},
           "greedy": _allocation_response(greedy), "even": _allocation_response(even)}


@lru_cache(maxsize=1)
@app.get("/api/forecast")
def forecast():
    """The real, honestly-backtested uniform-national-swing forecasting
    baseline (`attrition.election.forecast`) -- a classical political-
    science technique, not the persuasion mechanism: predicts each real
    seat's 2019 winner from its real 2014 result plus a national swing
    estimated with that seat's own state held out of the estimate.
    Computed once per process (deterministic, no request-specific
    input)."""
    return leave_one_state_out_backtest()


@lru_cache(maxsize=1)
@app.get("/api/campaign-policy")
def campaign_policy():
    """A real trained campaign-contact policy (`attrition.election.
    train_gradient_bandit`) -- see FINDINGS.md#44 and examples/
    20_campaign_policy_checkpoint.py, which this endpoint reproduces in
    JSON form. No external campaign policy exists to transplant, so this
    trains one from scratch, in this repo's own engine, seeing only
    realised per-pull reward -- the same information Greedy sees, nothing
    about (p, e). Two stages: single-agent evaluation against Greedy/ECI on
    held-out seeds, then decentralising the frozen policy into m=3
    independent copies sharing one pool. Computed once per process
    (deterministic, no request-specific input) -- training takes ~1-2s."""
    v, p, e, intensity = derive_campaign_parameters()
    delta, horizon, m = 0.35, 9, 3

    def env_factory(seed):
        return ConsumableBandit(v, p, e, delta=delta, horizon=horizon, seed=seed)

    policy = train_gradient_bandit(env_factory, episodes=4000, alpha=0.15, seed=0)
    results = compare(env_factory, [Greedy(), ECI(), policy], seeds=200)

    planner = planner_value_simultaneous(v, p, e, delta, horizon, m)
    greedy_dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                                   rule="greedy")
    eci_dec = decentralised_value_simultaneous(v, p, e, delta, horizon, m,
                                               rule="eci")
    population = Population({f"campaign-{i}": policy for i in range(m)})
    pool = SimultaneousPool(v, p, e, delta=delta, horizon=horizon, n_agents=m, seed=0)
    sim = simulate_population_simultaneous(pool, population, rounds=horizon)

    return {
        "contact_intensity": [float(x) for x in intensity],
        "learned_preferences": [float(x) for x in policy.h],
        "favourite_arm": int(policy.h.argmax()),
        "single_agent": {
            "greedy": results["greedy"],
            "eci": results["eci"],
            "trained": results[policy.name],
        },
        "decentralised": {
            "n_agents": m,
            "planner_optimum": planner,
            "decentralised_greedy": greedy_dec,
            "decentralised_eci": eci_dec,
            "trained_system_value": sim["system_value"],
            "trained_system_regret": sim["system_regret"],
        },
    }


# --------------------------------------------------------- worlds/scenarios
# Everything below serves the unified frontend's non-election worlds
# (agent-ecosystem, clinical-amr, environment, cybersecurity, synthetic) --
# a generic scenario runner, the JSON form of `app.py`'s Streamlit sidebar,
# so the same capabilities exist in both surfaces rather than only one.

WORLD_PERSONAS = {
    "clinical-amr": PHARMA_PERSONAS,
    "election": CAMPAIGN_PERSONAS,
    "agent-ecosystem": AGENT_PERSONAS,
    "cybersecurity": CYBERSECURITY_PERSONAS,
    "environment": FISHERIES_PERSONAS,
}

POLICY_FACTORIES = {
    "greedy": lambda v, p, e: Greedy(),
    "eci": lambda v, p, e: ECI(),
    "seci": lambda v, p, e: SECI(),
    "conservative": lambda v, p, e: Conservative(e_bound=float(e.max())),
    "thompson": lambda v, p, e: ThompsonSampling(),
    "ucb": lambda v, p, e: UCB(),
}


def _scenario_summary(name, spec):
    return {"name": name, "world": spec.get("world", "custom"),
           "description": spec["description"], "phenomenon": spec["phenomenon"],
           "expected_behaviour": spec["expected_behaviour"], "agents": spec["agents"]}


@app.get("/api/worlds")
def list_worlds():
    """Every registered scenario grouped by world (`attrition.scenarios.
    worlds()`), plus each world's one-line meaning (`WORLDS`) -- the JSON
    form of `attrition list-worlds`."""
    grouped = worlds()
    return {
        w: {"description": WORLDS.get(w, "(no description registered)"),
           "scenarios": [_scenario_summary(sname, SCENARIOS[sname]) for sname in snames]}
        for w, snames in grouped.items()
    }


@app.get("/api/scenarios/{name}")
def scenario_detail(name: str):
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {name!r}")
    spec = SCENARIOS[name]
    v, p, e, kw = get_arrays(name)
    detail = _scenario_summary(name, spec)
    detail.update({"v": [float(x) for x in v], "p": [float(x) for x in p],
                   "e": [float(x) for x in e], "delta": kw["delta"],
                   "horizon": kw["horizon"]})
    return detail


class CompareRequest(BaseModel):
    policies: list[str] = ["greedy", "eci"]
    seeds: int = 20
    delta: Optional[float] = None
    horizon: Optional[int] = None


@app.post("/api/scenarios/{name}/compare")
def scenario_compare(name: str, req: CompareRequest):
    """Aggregate value/regret for one or more classical policies on this
    scenario, over matched seeded episodes -- the JSON form of `app.py`'s
    "Run simulation" button / `attrition.compare`."""
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {name!r}")
    v, p, e, kw = get_arrays(name)
    delta = req.delta if req.delta is not None else kw["delta"]
    horizon = req.horizon if req.horizon is not None else kw["horizon"]
    unknown = [pn for pn in req.policies if pn not in POLICY_FACTORIES]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unknown polic{'y' if len(unknown) == 1 else 'ies'} "
                                   f"{unknown}; available: {sorted(POLICY_FACTORIES)}")
    policies = [POLICY_FACTORIES[pn](v, p, e) for pn in req.policies]
    env_factory = lambda seed: ConsumableBandit(v, p, e, delta=delta,
                                                horizon=horizon, seed=seed)
    return compare(env_factory, policies, seeds=req.seeds)


def _persona_list(world):
    persona_set = WORLD_PERSONAS.get(world)
    if not persona_set:
        return []
    return [{"name": p.name, "role": p.role, "description": p.description,
            "risk_tolerance": p.risk_tolerance} for p in persona_set.values()]


@app.get("/api/scenarios/{name}/personas")
def scenario_personas(name: str):
    """Which persona set (if any) makes sense to run a real LLM population
    against, for this scenario's world -- empty for worlds with no
    registered persona set (synthetic; see FINDINGS.md #55 for
    environment/cybersecurity, which now have one)."""
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {name!r}")
    return _persona_list(SCENARIOS[name].get("world"))


class LLMPopulationRequest(BaseModel):
    provider: str  # "anthropic" or "groq"
    api_key: str
    model: Optional[str] = None
    persona_names: list[str]
    horizon: int = 8


_LLM_MODELS = {"anthropic": "claude-haiku-4-5-20251001", "groq": "openai/gpt-oss-20b"}


def _build_llm_client(provider, api_key, model):
    if provider == "anthropic":
        import anthropic
        sdk_client = anthropic.Anthropic(api_key=api_key)

        def call(system, user):
            msg = sdk_client.messages.create(
                model=model, max_tokens=300, system=system,
                messages=[{"role": "user", "content": user}])
            return msg.content[0].text
    elif provider == "groq":
        import groq
        sdk_client = groq.Groq(api_key=api_key)

        def call(system, user):
            kwargs = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
            completion = sdk_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                max_tokens=300, **kwargs)
            return completion.choices[0].message.content
    else:
        raise HTTPException(status_code=400,
                            detail=f"unknown provider {provider!r}; expected "
                                   f"'anthropic' or 'groq'")
    return CallableLLMClient(call)


@app.post("/api/scenarios/{name}/llm-population")
def scenario_llm_population(name: str, req: LLMPopulationRequest):
    """Run a real LLM-driven population against this scenario -- bring your
    own key (used only for this request, never logged or stored; still
    passes through this server process to place the call, the same
    disclosed tradeoff `app.py`'s equivalent section makes). JSON form of
    `app.py`'s "Real LLM population" section / `examples/04`, `13`, `19`."""
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {name!r}")
    spec = SCENARIOS[name]
    persona_set = WORLD_PERSONAS.get(spec.get("world"))
    if not persona_set:
        raise HTTPException(status_code=400,
                            detail=f"no persona set registered for world "
                                   f"{spec.get('world')!r}")
    unknown = [pn for pn in req.persona_names if pn not in persona_set]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unknown persona(s) {unknown}; available: "
                                   f"{sorted(persona_set)}")
    v, p, e, kw = get_arrays(name)
    model = req.model or _LLM_MODELS.get(req.provider)
    try:
        client = _build_llm_client(req.provider, req.api_key, model)
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"could not build client: {ex}")

    personas = [persona_set[pn] for pn in req.persona_names]
    population = Population.from_personas(personas, client=client)
    env = ConsumableBandit(v, p, e, delta=kw["delta"], horizon=req.horizon, seed=0)

    try:
        result = simulate_population(env, population)
    except Exception as ex:
        raise HTTPException(status_code=502,
                            detail=f"the API call failed before any decision "
                                   f"was logged: {ex}")

    per_persona = {}
    for pname, policy in population.members.items():
        log = getattr(policy, "log", [])
        genuine = sum(1 for row in log
                     if row.get("response") and _CHOICE.search(row["response"]))
        errors = {}
        for row in log:
            if row.get("error"):
                errors[row["error"]] = errors.get(row["error"], 0) + 1
        per_persona[pname] = {"genuine": genuine, "total": len(log), "errors": errors}

    return {"system_value": result["system_value"],
           "system_regret": result["system_regret"], "per_persona": per_persona}


@lru_cache(maxsize=1)
@app.get("/api/amr/reference-table")
def amr_reference_table():
    """The "unified five-policy table" (FINDINGS.md#26/#44,
    experiments/exp65_amr_five_policy_reference_table.py) -- planner/greedy/
    eci/lambda_weighted recomputed live from this repo's engine, plus the
    external abx_amr_simulator checkpoint numbers cited verbatim from
    FINDINGS.md#24.2/24.4 (NOT recomputed here -- see that module's
    docstring for exactly why the two halves are not on the same axis)."""
    repo = in_repo_half()
    return {"in_repo": repo, "external_siloed": EXTERNAL_SILOED,
           "external_shared": EXTERNAL_SHARED}
