import { useEffect, useState } from "react";
import {
  fetchStrategist, fetchStrategistFilters,
  type StrategistResult, type StrategistFacets, type StrategistFilters as Filters,
} from "./api";

export default function Strategist({
  onSelect,
}: {
  onSelect: (state: string, name: string) => void;
}) {
  const [budget, setBudget] = useState(30);
  const [result, setResult] = useState<StrategistResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [facets, setFacets] = useState<StrategistFacets | null>(null);
  const [constituencyType, setConstituencyType] = useState("");
  const [subRegion, setSubRegion] = useState("");
  const [incumbency, setIncumbency] = useState<Filters["incumbency"]>("any");

  useEffect(() => {
    fetchStrategistFilters().then(setFacets).catch(() => {});
  }, []);

  useEffect(() => {
    fetchStrategist(budget, 20, {
      constituency_type: constituencyType || undefined,
      sub_region: subRegion || undefined,
      incumbency,
    }).then(setResult).catch((e) => setError(String(e)));
  }, [budget, constituencyType, subRegion, incumbency]);

  return (
    <>
      <div className="card">
        <h2>National campaign strategist</h2>
        <p className="muted" style={{ fontSize: 13, maxWidth: 760 }}>
          One finite national budget, expressed in the same "percent of electorate" unit as every real
          tipping point on this page. Two policies spend it differently: <strong>greedy</strong> fully funds the
          cheapest real seats to flip first and stops when the budget runs out; <strong>even</strong> spreads the
          identical budget across every seat within reach (tipping point ≤ 20%) — a seat only flips if its
          equal share actually clears its own real tipping point. Not a re-run of this project's core theorems —
          a new, one-shot allocation decision across 543 independent real seats, using the same
          finite-resource lens.
        </p>
        <label className="muted" style={{ fontSize: 12 }}>
          National budget: {budget} (percentage-points of electorate-effort)
        </label>
        <input type="range" min={2} max={100} step={1} value={budget}
              onChange={(e) => setBudget(Number(e.target.value))} />
        {error && <p className="error-box" style={{ marginTop: 12 }}>{error}</p>}
      </div>

      <div className="card">
        <h2>Target by real seat covariates</h2>
        <p className="muted" style={{ fontSize: 12, marginBottom: 12 }}>
          Real, published per-seat facts only — reservation category, sub-region, and whether the real 2019
          winner was an incumbent. No per-voter demographic or economic data exists in the source (see
          FINDINGS.md #47), so targeting stops at what's real.
        </p>
        <div className="search-row">
          <select value={constituencyType} onChange={(e) => setConstituencyType(e.target.value)}>
            <option value="">All reservation categories</option>
            {facets?.constituency_type.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <select value={subRegion} onChange={(e) => setSubRegion(e.target.value)}>
            <option value="">All sub-regions</option>
            {facets?.sub_region.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <select value={incumbency} onChange={(e) => setIncumbency(e.target.value as Filters["incumbency"])}>
            <option value="any">Any incumbency</option>
            <option value="incumbent">Real winner was incumbent</option>
            <option value="challenger">Real winner was a challenger</option>
          </select>
        </div>
        {result && (
          <p className="muted" style={{ fontSize: 12 }}>
            {result.n_seats_in_scope} of 543 real seats match this filter.
          </p>
        )}
      </div>

      {result && (
        <div className="card">
          <h2>What this costs society, not just seats flipped</h2>
          <p className="muted" style={{ fontSize: 12, marginBottom: 12, maxWidth: 760 }}>
            Real, closed-form metrics over which seats each policy TARGETED — reach (electors, states touched) and
            efficiency (flipped ÷ targeted). Deliberately not re-derived from a mechanistic simulation run: doing
            that would require inventing a mapping from this budget unit to a contact-intensity sequence that
            doesn't exist anywhere in this project (see FINDINGS.md #48) — these numbers stay honest by staying
            inside the model that's actually validated.
          </p>
          <div className="grid cols-2">
            <div>
              <h3 style={{ marginTop: 0 }}>Greedy</h3>
              <div className="grid cols-2">
                <div className="stat">
                  <div className="value">{((result.greedy.society.efficiency ?? 0) * 100).toFixed(0)}%</div>
                  <div className="label">efficiency (flipped ÷ targeted)</div>
                </div>
                <div className="stat">
                  <div className="value">{result.greedy.society.n_states_touched}</div>
                  <div className="label">states touched</div>
                </div>
                <div className="stat">
                  <div className="value">{result.greedy.society.electors_in_scope_pct_national.toFixed(1)}%</div>
                  <div className="label">of national electorate in scope</div>
                </div>
                <div className="stat">
                  <div className="value">{result.greedy.society.cost_mean_pct?.toFixed(2) ?? "—"}%</div>
                  <div className="label">mean tipping-point cost of targeted seats</div>
                </div>
              </div>
            </div>
            <div>
              <h3 style={{ marginTop: 0 }}>Even</h3>
              <div className="grid cols-2">
                <div className="stat">
                  <div className="value">{((result.even.society.efficiency ?? 0) * 100).toFixed(0)}%</div>
                  <div className="label">efficiency (flipped ÷ targeted)</div>
                </div>
                <div className="stat">
                  <div className="value">{result.even.society.n_states_touched}</div>
                  <div className="label">states touched</div>
                </div>
                <div className="stat">
                  <div className="value">{result.even.society.electors_in_scope_pct_national.toFixed(1)}%</div>
                  <div className="label">of national electorate in scope</div>
                </div>
                <div className="stat">
                  <div className="value">{result.even.society.cost_mean_pct?.toFixed(2) ?? "—"}%</div>
                  <div className="label">mean tipping-point cost of targeted seats</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="grid cols-2">
          <div className="card">
            <h2>Greedy — cheapest seats first</h2>
            <div className="stat" style={{ marginBottom: 12 }}>
              <div className="value">{result.greedy.n_flipped}</div>
              <div className="label">real seats flipped, fully funded</div>
            </div>
            <p className="muted" style={{ fontSize: 12 }}>
              Spent {result.greedy.budget_spent.toFixed(1)} of {budget}, {result.greedy.budget_remaining.toFixed(1)} left over
              (too little to fund the next-cheapest seat).
            </p>
            <table className="seat-table">
              <thead><tr><th>Seat</th><th>State</th><th>Cost</th></tr></thead>
              <tbody>
                {result.greedy.flipped.map((s) => (
                  <tr key={s.id} onClick={() => onSelect(s.state, s.name)}>
                    <td>{s.name}</td><td>{s.state}</td><td>{s.tipping_point_pct?.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.greedy.n_flipped > result.greedy.flipped.length && (
              <p className="muted" style={{ fontSize: 12 }}>… and {result.greedy.n_flipped - result.greedy.flipped.length} more</p>
            )}
          </div>

          <div className="card">
            <h2>Even — spread across every reachable seat</h2>
            <div className="stat" style={{ marginBottom: 12 }}>
              <div className="value">{result.even.n_flipped}</div>
              <div className="label">of {result.even.n_targeted} targeted seats actually flip</div>
            </div>
            <p className="muted" style={{ fontSize: 12 }}>
              Same {budget}-point budget, split {result.even.n_targeted} ways — most seats get less than
              they need.
            </p>
            {result.even.flipped.length > 0 ? (
              <table className="seat-table">
                <thead><tr><th>Seat</th><th>State</th><th>Cost</th></tr></thead>
                <tbody>
                  {result.even.flipped.map((s) => (
                    <tr key={s.id} onClick={() => onSelect(s.state, s.name)}>
                      <td>{s.name}</td><td>{s.state}</td><td>{s.tipping_point_pct?.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">No seat's equal share clears its own tipping point at this budget.</p>
            )}
          </div>
        </div>
      )}
    </>
  );
}
