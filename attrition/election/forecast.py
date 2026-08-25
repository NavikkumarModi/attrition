"""A real, honestly-backtested election forecasting baseline -- a
genuinely different thing from the persuasion simulator
(`agents.py`/`outcome.py`), not another view of it.

The persuasion simulator asks: "if a campaign contacts voters this much,
what happens?" -- a mechanism question, deliberately not a forecast (see
FINDINGS.md #35/#36/#41). This module asks a different, classical
political-science question directly: "given one real prior election's
results, how well can a standard swing model predict the next one?" --
no campaign, no persuasion mechanism, no LLM. Just two real cross-
sections (2014, 2019 -- `lokdhaba.py`, `scripts/
fetch_election_data_historical.py`) and a textbook forecasting technique.

**Uniform national swing**, the same technique used throughout applied
political science (e.g. UK/Indian election-night forecasting): estimate
each party's average national vote-share change from the prior election
to the target one, then predict a seat's next result as
"prior share + that party's national swing." A party with too few real
seats to estimate a stable national swing (`MIN_SEATS_FOR_SWING`) is
assumed to swing zero, a stated simplification, not silently guessed.

**Backtested honestly, not just fit and reported:** `leave_one_state_out_
backtest` computes each state's swing estimate using every OTHER state's
real 2014->2019 transition, then predicts that state's own seats --
never using a state's own outcome to help predict itself. Compared
against the naive "incumbent retains, no swing at all" baseline, on the
same real, matched seats.

**What this does and does not establish.** It is a baseline, not a
claim that this is a good forecasting model -- no feature beyond prior
vote share and a national swing constant is used (no polling, no
demographics, no campaign spending). Reported accuracy is the honest
result of that real backtest, including its real failures (seats where a
new party entered, or a swing ran opposite to the national trend, are
seats this simple model gets wrong by construction, not smoothed over).
"""
from collections import defaultdict

from .lokdhaba import load_constituencies

__all__ = ["match_seats", "national_swing", "predict_seat_winner",
          "leave_one_state_out_backtest", "MIN_SEATS_FOR_SWING"]

# A party's national swing is only estimated from seats where it fielded
# a candidate in BOTH years -- below this many real matched seats, the
# estimate is too noisy to trust, so it's treated as zero swing (stated,
# not silently assumed accurate).
MIN_SEATS_FOR_SWING = 15


def _shares_by_party(row):
    return {c["party"]: c["vote_share_pct"] for c in row["candidates"]}


def match_seats(prior_year=2014, target_year=None):
    """Real seats present in both `prior_year` and `target_year`
    (`None` = the primary 2019 snapshot), matched by (state, constituency
    name) -- both real TCPD snapshots, same source, so names line up
    directly (no cross-source alias table needed here, unlike `geo.py`'s
    join against a different real source). Returns
    `(matched, unmatched_names)`: `matched` is a list of
    `(prior_row, target_row)` pairs; seats present in only one year (a
    genuine rename/reorganisation, not assumed to be a bug) are reported
    in `unmatched_names`, not silently dropped.
    """
    prior_rows = load_constituencies(year=prior_year)
    target_rows = load_constituencies(year=target_year)
    prior_by_key = defaultdict(list)
    for r in prior_rows:
        prior_by_key[(r["state"].strip().upper(), r["constituency"].strip().upper())].append(r)

    matched = []
    unmatched = []
    for t in target_rows:
        key = (t["state"].strip().upper(), t["constituency"].strip().upper())
        candidates = prior_by_key.get(key)
        if candidates and len(candidates) == 1:
            matched.append((candidates[0], t))
        else:
            unmatched.append((t["state"], t["constituency"]))
    return matched, unmatched


def national_swing(matched_pairs, min_seats=MIN_SEATS_FOR_SWING):
    """Each party's average real vote-share swing across `matched_pairs`
    (from `match_seats`), only over seats where the party fielded a real
    candidate in both years. Parties below `min_seats` real matched seats
    are omitted (callers must treat a missing party as zero swing, not an
    error) -- too few real observations to estimate a stable national
    number from."""
    deltas = defaultdict(list)
    for prior, target in matched_pairs:
        prior_shares = _shares_by_party(prior)
        target_shares = _shares_by_party(target)
        for party in prior_shares.keys() & target_shares.keys():
            if party == "IND":
                continue   # generic label spanning unrelated real people, not one party
            deltas[party].append(target_shares[party] - prior_shares[party])
    return {party: sum(vals) / len(vals) for party, vals in deltas.items()
           if len(vals) >= min_seats}


def predict_seat_winner(prior_row, swing):
    """Predicted 2019 (or target-year) winner for one seat: each party
    present in `prior_row`'s real candidates gets `prior_share +
    swing.get(party, 0.0)` (clipped to [0, 100]); the predicted winner is
    whichever party's predicted share is highest. A party absent from
    `prior_row` (a genuinely new entrant) cannot be predicted to win --
    a real, honest limitation of a swing-only model, not patched over."""
    best_party, best_share = None, -1.0
    for c in prior_row["candidates"]:
        predicted = min(100.0, max(0.0, c["vote_share_pct"] + swing.get(c["party"], 0.0)))
        if predicted > best_share:
            best_party, best_share = c["party"], predicted
    return best_party


def leave_one_state_out_backtest(prior_year=2014, target_year=None,
                                 min_seats=MIN_SEATS_FOR_SWING):
    """Honest backtest: for each real state, the national swing used to
    predict that state's seats is estimated from every OTHER state's real
    matched seats only -- a state's own 2019 result never leaks into the
    swing used to predict it. Compared, seat for seat, against the naive
    "incumbent retains" baseline (predicted winner = prior winner, no
    swing at all).

    Returns a dict: `n_seats` (matched and evaluated), `swing_model_
    accuracy`, `naive_baseline_accuracy`, `n_real_flips` (seats where the
    real winner changed prior->target), `swing_model_flip_recall`
    (fraction of those real flips the swing model correctly predicted --
    the naive baseline gets these all wrong by construction, so this is
    the real model's honest hit rate on the hard cases, not the easy
    incumbent-retention ones).
    """
    matched, _ = match_seats(prior_year, target_year)
    by_state = defaultdict(list)
    for prior, target in matched:
        by_state[prior["state"]].append((prior, target))

    n_seats = 0
    n_swing_correct = 0
    n_naive_correct = 0
    n_real_flips = 0
    n_flip_correct = 0
    for held_out_state in by_state:
        train_pairs = [pair for state, pairs in by_state.items()
                      if state != held_out_state for pair in pairs]
        swing = national_swing(train_pairs, min_seats=min_seats)
        for prior, target in by_state[held_out_state]:
            real_winner = target["candidates"][0]["party"]
            prior_winner = prior["candidates"][0]["party"]
            predicted = predict_seat_winner(prior, swing)

            n_seats += 1
            n_swing_correct += int(predicted == real_winner)
            n_naive_correct += int(prior_winner == real_winner)
            if prior_winner != real_winner:
                n_real_flips += 1
                n_flip_correct += int(predicted == real_winner)

    return {
        "n_seats": n_seats,
        "swing_model_accuracy": n_swing_correct / n_seats if n_seats else None,
        "naive_baseline_accuracy": n_naive_correct / n_seats if n_seats else None,
        "n_real_flips": n_real_flips,
        "swing_model_flip_recall": n_flip_correct / n_real_flips if n_real_flips else None,
    }
