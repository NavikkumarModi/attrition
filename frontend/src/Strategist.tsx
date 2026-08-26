import { useEffect, useState } from "react";
import {
  fetchStrategist, fetchStrategistFilters,
  type StrategistResult, type StrategistFacets, type StrategistFilters as Filters,
} from "./api";
import Disclosure from "./Disclosure";

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
  const [povertyFilterOn, setPovertyFilterOn] = useState(false);
  const [minPovertyRate, setMinPovertyRate] = useState(0);
  const [maxPovertyRate, setMaxPovertyRate] = useState(60);
  const [useLiteracyEffect, setUseLiteracyEffect] = useState(false);

  useEffect(() => {
    fetchStrategistFilters().then(setFacets).catch(() => {});
  }, []);

  useEffect(() => {
    fetchStrategist(budget, 20, {
      constituency_type: constituencyType || undefined,
      sub_region: subRegion || undefined,
      incumbency,
      min_poverty_rate_pct: povertyFilterOn ? minPovertyRate : null,
      max_poverty_rate_pct: povertyFilterOn ? maxPovertyRate : null,
    }, useLiteracyEffect).then(setResult).catch((e) => setError(String(e)));
  }, [budget, constituencyType, subRegion, incumbency, povertyFilterOn,
      minPovertyRate, maxPovertyRate, useLiteracyEffect]);

  return (
    <>
      <div className="card">
        <h2>National campaign strategist</h2>
        <Disclosure summary="How this allocation works">
          One finite national budget, expressed in the same "percent of electorate" unit as every real
          tipping point on this page. Two policies spend it differently: <strong>greedy</strong> fully funds the
          cheapest real seats to flip first and stops when the budget runs out; <strong>even</strong> spreads the
          identical budget across every seat within reach (tipping point ≤ 20%) — a seat only flips if its
          equal share actually clears its own real tipping point. Not a re-run of this project's core theorems —
          a new, one-shot allocation decision across 543 independent real seats, using the same
          finite-resource lens.
        </Disclosure>
        <label className="muted" style={{ fontSize: 12 }}>
          National budget: {budget} (percentage-points of electorate-effort)
        </label>
        <input type="range" min={2} max={100} step={1} value={budget}
              onChange={(e) => setBudget(Number(e.target.value))} />

        <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6, marginTop: 14 }}>
          <input type="checkbox" checked={useLiteracyEffect}
                onChange={(e) => setUseLiteracyEffect(e.target.checked)} />
          Indicator-driven: rank and cost seats by real literacy-adjusted persuadability
        </label>
        {useLiteracyEffect && (
          <Disclosure summary="Why literacy drives this, and not other indicators">
            The same literature-checked mechanism as Constituency Lab's literacy control (FINDINGS.md #59):
            Zaller's (1992) Receive-Accept-Sample model predicts lower-literacy seats are more persuadable under
            high-intensity campaign contact — so this rescales each seat's real tipping-point cost by its real
            literacy-adjusted resistance, changing which seats greedy/even actually pick, not just how they're
            displayed. Income/poverty and road connectivity were checked against real causal literature and found
            to have no supported effect on persuadability — they're offered below as a real TARGETING filter, not
            as a cost driver. See FINDINGS.md #62.
          </Disclosure>
        )}
        {error && <p className="error-box" style={{ marginTop: 12 }}>{error}</p>}
      </div>

      <div className="card">
        <h2>Target by real seat covariates</h2>
        <Disclosure summary="What can be targeted, and why">
          Real, published per-seat facts — reservation category, sub-region, whether the real 2019 winner was an
          incumbent, and (new, FINDINGS.md #62) real rural poverty rate (2012 SECC, via SHRUG #57) — the first
          genuinely economic targeting criterion here. Still no per-voter demographic or economic data (FINDINGS.md
          #47) — every filter is a real, seat-level, published fact, never invented.
        </Disclosure>
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

        <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
          <input type="checkbox" checked={povertyFilterOn}
                onChange={(e) => setPovertyFilterOn(e.target.checked)} />
          Filter by real rural poverty rate (2012 SECC)
        </label>
        {povertyFilterOn && (
          <div className="round-slider-row" style={{ marginTop: 8 }}>
            <span className="n">{minPovertyRate}%</span>
            <input type="range" min={0} max={60} step={1} value={minPovertyRate}
                  onChange={(e) => setMinPovertyRate(Math.min(Number(e.target.value), maxPovertyRate))} />
            <span className="v">to</span>
            <input type="range" min={0} max={60} step={1} value={maxPovertyRate}
                  onChange={(e) => setMaxPovertyRate(Math.max(Number(e.target.value), minPovertyRate))} />
            <span className="v">{maxPovertyRate}%</span>
          </div>
        )}

        {result && (
          <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            {result.n_seats_in_scope} of 543 real seats match this filter.
          </p>
        )}
      </div>

      {result && (
        <div className="card">
          <h2>What this costs society, not just seats flipped</h2>
          <Disclosure summary="Why these are closed-form, not simulated">
            Real, closed-form metrics over which seats each policy TARGETED — reach (electors, states touched) and
            efficiency (flipped ÷ targeted). Deliberately not re-derived from a mechanistic simulation run: doing
            that would require inventing a mapping from this budget unit to a contact-intensity sequence that
            doesn't exist anywhere in this project (see FINDINGS.md #48) — these numbers stay honest by staying
            inside the model that's actually validated.
          </Disclosure>
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
                  <div className="label">
                    mean {result.use_literacy_effect ? "literacy-adjusted" : "tipping-point"} cost of targeted seats
                  </div>
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
                  <div className="label">
                    mean {result.use_literacy_effect ? "literacy-adjusted" : "tipping-point"} cost of targeted seats
                  </div>
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
              {result.use_literacy_effect && " Ranked by real literacy-adjusted cost — the Cost column below still shows each seat's real, unadjusted tipping point."}
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
