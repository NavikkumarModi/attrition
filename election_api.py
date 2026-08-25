"""A thin FastAPI wrapper around `attrition.election`, for the dedicated
election-track frontend (`frontend/`) -- no new simulation logic lives
here, only JSON plumbing over functions already proven by `tests/
test_election.py` and `examples/12`-`18`.

Run:  uvicorn election_api:app --reload --port 8000
(pip install fastapi "uvicorn[standard]" -- not in requirements.txt, since
the core package and its Streamlit app (`app.py`) don't need a web server
to run the simulator itself; this is an additional, optional surface.)
"""
import json
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from attrition import (ConsumableBandit, Greedy, ECI, compare, Population,
                       SimultaneousPool, simulate_population_simultaneous,
                       planner_value_simultaneous, decentralised_value_simultaneous,
                       derive_campaign_parameters)
from attrition.election import (ALL_PHASE1_CONSTITUENCIES, build_constituency,
                                real_margin_pct, tipping_point, recompute_outcome,
                                simulate_level1_contact, simulate_level2_contact,
                                get_constituency, geojson_feature_collection,
                                greedy_allocation, even_allocation,
                                leave_one_state_out_backtest, train_gradient_bandit)

app = FastAPI(title="ATTRITION Election API")
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


class SimulateRequest(BaseModel):
    state: str
    name: str
    intensity_sequence: list[float] = [0.15, 0.2, 0.25, 0.3, 0.35]
    n_voters: int = 3000
    seed: int = 0
    include_level2: bool = True


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Level 1 (always) + Level 2 (optional) campaign simulation over one
    real constituency, plus the recomputed outcome -- the same three
    functions `examples/17_election_outcome.py` chains by hand."""
    c = _resolve(req.state, req.name)
    fractions, final = simulate_level1_contact(
        c, req.intensity_sequence, n_voters=req.n_voters, seed=req.seed)
    outcome = recompute_outcome(c, final)
    result = {
        "seat": _summary(c),
        "level1_switched_fraction_by_round": fractions,
        "level1_final_switched_fraction": final,
        "outcome": {
            "winner": outcome.winner, "winner_share_pct": outcome.winner_share_pct,
            "runner_up": outcome.runner_up, "runner_up_share_pct": outcome.runner_up_share_pct,
            "margin_pct": outcome.margin_pct, "flipped": outcome.flipped,
        },
    }
    if req.include_level2:
        result["level2_decisions"] = simulate_level2_contact(c, req.intensity_sequence)
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


def _allocation_response(result):
    by_id = {_seat_id(c): c for c in ALL_PHASE1_CONSTITUENCIES}
    return {
        "n_targeted": len(result.targeted), "n_flipped": len(result.flipped),
        "budget_spent": result.budget_spent, "budget_remaining": result.budget_remaining,
        "flipped": [_summary(c) for c in result.flipped[:25]],
    }


@app.get("/api/strategist")
def strategist(budget_pct: float = 20.0, scope_max_tipping_pct: float = 20.0):
    """The national campaign-strategist agent (`attrition.election.
    strategist`): given one total budget, expressed in percent-of-
    electorate units (the same unit `tipping_point` already uses
    everywhere), compares fully-funding the cheapest real seats first
    (`greedy`) against spreading the identical budget evenly across every
    seat within `scope_max_tipping_pct` (`even`). A new resource-
    ALLOCATION decision layered on Level 0/1/2, not a re-application of
    Theorems 1-6 -- see strategist.py's module docstring."""
    greedy = greedy_allocation(budget_pct)
    even = even_allocation(budget_pct, scope_max_tipping_pct=scope_max_tipping_pct)
    return {"budget_pct": budget_pct, "scope_max_tipping_pct": scope_max_tipping_pct,
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
