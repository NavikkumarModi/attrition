import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Cell,
} from "recharts";
import {
  fetchConstituencies, fetchNationalScan, fetchCampaignMap,
  type SeatSummary, type NationalScan, type CampaignMapResult,
} from "./api";
import IndiaMap from "./IndiaMap";
import Disclosure from "./Disclosure";
import {
  CHART_GRID, CHART_TICK, CHART_TOOLTIP_BG, CHART_TOOLTIP_BORDER, tippingPointColor, INTENSITY_PRESETS,
} from "./theme";

export default function Overview({ onSelect }: { onSelect: (state: string, name: string) => void }) {
  const [seats, setSeats] = useState<SeatSummary[] | null>(null);
  const [scan, setScan] = useState<NationalScan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [sortKey, setSortKey] = useState<keyof SeatSummary>("tipping_point_pct");

  const [simMode, setSimMode] = useState(false);
  const [simEffort, setSimEffort] = useState<keyof typeof INTENSITY_PRESETS>("Moderate");
  const [simScope, setSimScope] = useState<"nation" | "state" | "type">("nation");
  const [simState, setSimState] = useState("");
  const [simType, setSimType] = useState("GEN");
  const [simIncumbency, setSimIncumbency] = useState<"any" | "incumbent" | "challenger">("any");
  const [simPovertyOn, setSimPovertyOn] = useState(false);
  const [simMinPoverty, setSimMinPoverty] = useState(0);
  const [simMaxPoverty, setSimMaxPoverty] = useState(60);
  const [simUseLiteracy, setSimUseLiteracy] = useState(false);
  const [simResult, setSimResult] = useState<CampaignMapResult | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  useEffect(() => {
    fetchConstituencies().then(setSeats).catch((e) => setError(String(e)));
    fetchNationalScan().then(setScan).catch((e) => setError(String(e)));
  }, []);

  const states = useMemo(
    () => (seats ? Array.from(new Set(seats.map((s) => s.state))).sort() : []),
    [seats],
  );

  const constituencyTypes = useMemo(
    () => (seats ? Array.from(new Set(seats.map((s) => s.constituency_type))).sort() : ["GEN", "SC", "ST"]),
    [seats],
  );

  // Guided flow, same debounced-on-change pattern the single-seat
  // simulator and Strategist already use: pick an effort/scope and the
  // map recolours on its own.
  useEffect(() => {
    if (!simMode) { setSimResult(null); return; }
    const t = setTimeout(() => {
      setSimLoading(true);
      setSimError(null);
      fetchCampaignMap(INTENSITY_PRESETS[simEffort], {
        state: simScope === "state" && simState ? simState : null,
        constituency_type: simScope === "type" ? simType : null,
        incumbency: simIncumbency,
        min_poverty_rate_pct: simPovertyOn ? simMinPoverty : null,
        max_poverty_rate_pct: simPovertyOn ? simMaxPoverty : null,
        use_literacy_effect: simUseLiteracy,
      })
        .then(setSimResult)
        .catch((e) => setSimError(String(e)))
        .finally(() => setSimLoading(false));
    }, 400);
    return () => clearTimeout(t);
  }, [simMode, simEffort, simScope, simState, simType, simIncumbency,
      simPovertyOn, simMinPoverty, simMaxPoverty, simUseLiteracy]);

  const filtered = useMemo(() => {
    if (!seats) return [];
    return seats
      .filter((s) => (stateFilter ? s.state === stateFilter : true))
      .filter((s) => s.name.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => {
        const av = a[sortKey], bv = b[sortKey];
        if (av === null) return 1;
        if (bv === null) return -1;
        if (typeof av === "number" && typeof bv === "number") return av - bv;
        return String(av).localeCompare(String(bv));
      });
  }, [seats, query, stateFilter, sortKey]);

  if (error) return <div className="error-box">{error} — is election_api.py running on :8000?</div>;

  return (
    <>
      <div className="card">
        <h2>Campaign simulator</h2>
        <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={simMode} onChange={(e) => setSimMode(e.target.checked)} />
          Simulate a campaign and recolour the map by outcome, instead of real 2019 tipping points
        </label>
        {simMode && (
          <>
            <div className="grid cols-2" style={{ marginTop: 12 }}>
              <div>
                <label className="muted" style={{ fontSize: 12, display: "block", marginBottom: 6 }}>Effort</label>
                <div className="search-row">
                  {(Object.keys(INTENSITY_PRESETS) as (keyof typeof INTENSITY_PRESETS)[]).map((label) => (
                    <button
                      key={label}
                      className="run-btn"
                      style={{ padding: "8px 16px", opacity: simEffort === label ? 1 : 0.55 }}
                      onClick={() => setSimEffort(label)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="muted" style={{ fontSize: 12, display: "block", marginBottom: 6 }}>Scope</label>
                <select value={simScope} onChange={(e) => setSimScope(e.target.value as typeof simScope)}>
                  <option value="nation">Whole nation</option>
                  <option value="state">One state / UT</option>
                  <option value="type">Reservation category</option>
                </select>
                {simScope === "state" && (
                  <select style={{ marginTop: 6, width: "100%" }} value={simState}
                         onChange={(e) => setSimState(e.target.value)}>
                    <option value="">Choose a state…</option>
                    {states.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                )}
                {simScope === "type" && (
                  <select style={{ marginTop: 6, width: "100%" }} value={simType}
                         onChange={(e) => setSimType(e.target.value)}>
                    {constituencyTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                )}
              </div>
            </div>
            <div className="grid cols-2" style={{ marginTop: 12 }}>
              <div>
                <label className="muted" style={{ fontSize: 12, display: "block", marginBottom: 6 }}>Incumbency</label>
                <select value={simIncumbency} onChange={(e) => setSimIncumbency(e.target.value as typeof simIncumbency)}>
                  <option value="any">Any incumbency</option>
                  <option value="incumbent">Real winner was incumbent</option>
                  <option value="challenger">Real winner was a challenger</option>
                </select>
              </div>
              <div>
                <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="checkbox" checked={simUseLiteracy} onChange={(e) => setSimUseLiteracy(e.target.checked)} />
                  Literacy-adjusted persuasion (Zaller 1992 RAS model)
                </label>
              </div>
            </div>
            <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
              <input type="checkbox" checked={simPovertyOn} onChange={(e) => setSimPovertyOn(e.target.checked)} />
              Filter by real rural poverty rate (2012 SECC)
            </label>
            {simPovertyOn && (
              <div className="round-slider-row" style={{ marginTop: 8 }}>
                <span className="n">{simMinPoverty}%</span>
                <input type="range" min={0} max={60} step={1} value={simMinPoverty}
                      onChange={(e) => setSimMinPoverty(Math.min(Number(e.target.value), simMaxPoverty))} />
                <span className="v">to</span>
                <input type="range" min={0} max={60} step={1} value={simMaxPoverty}
                      onChange={(e) => setSimMaxPoverty(Math.max(Number(e.target.value), simMinPoverty))} />
                <span className="v">{simMaxPoverty}%</span>
              </div>
            )}
            <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
              {simLoading
                ? "Updating…"
                : simResult
                ? `${simResult.n_flipped} of ${simResult.n_seats} seats in scope flip at ${simEffort.toLowerCase()} effort.`
                : "Choose an effort and scope above."}
            </p>
            {simError && <p className="error-box" style={{ marginTop: 8 }}>{simError}</p>}
          </>
        )}
      </div>

      <IndiaMap onSelect={onSelect} simulation={simMode ? simResult : null} />

      {scan && (
        <div className="card">
          <div className="grid cols-4">
            <div className="stat"><div className="value">{scan.distribution_pct.min.toFixed(1)}%</div><div className="label">most contestable</div></div>
            <div className="stat"><div className="value">{scan.distribution_pct.median.toFixed(1)}%</div><div className="label">median seat</div></div>
            <div className="stat"><div className="value">{scan.distribution_pct.p90.toFixed(1)}%</div><div className="label">p90</div></div>
            <div className="stat"><div className="value">{scan.distribution_pct.max.toFixed(1)}%</div><div className="label">safest hold</div></div>
          </div>
          <Disclosure summary="What is a tipping point?">
            The fraction of the total electorate that would need to switch to flip each real 2019 seat,
            computed by <code>tipping_point</code> — no simulation, a closed-form search over the real
            reallocation rule.
          </Disclosure>
        </div>
      )}

      <div className="card">
        <h2>Margin vs. tipping point, every real seat</h2>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={CHART_GRID} />
            <XAxis type="number" dataKey="real_margin_pct" name="real margin" unit="pp"
                  tick={{ fill: CHART_TICK, fontSize: 11 }} />
            <YAxis type="number" dataKey="tipping_point_pct" name="tipping point" unit="%"
                  tick={{ fill: CHART_TICK, fontSize: 11 }} />
            <ZAxis range={[36, 36]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ background: CHART_TOOLTIP_BG, border: `1px solid ${CHART_TOOLTIP_BORDER}`, fontSize: 12 }}
              formatter={((value: unknown, key: string) => [
                typeof value === "number"
                  ? (key === "real_margin_pct" ? `${value.toFixed(2)}pp` : `${value.toFixed(2)}%`)
                  : "—",
                key,
              ]) as any}
              labelFormatter={() => ""}
            />
            <Scatter
              data={seats ?? []}
              onClick={(d: any) => onSelect(d.state, d.name)}
              cursor="pointer"
            >
              {(seats ?? []).map((s, i) => <Cell key={i} fill={tippingPointColor(s.tipping_point_pct)} />)}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <p className="muted" style={{ fontSize: 12 }}>
          <span className="legend-dot" style={{ background: tippingPointColor(1) }} /> tipping point &lt; 2%{"  "}
          <span className="legend-dot" style={{ background: tippingPointColor(5), marginLeft: 12 }} /> &lt; 8%{"  "}
          <span className="legend-dot" style={{ background: tippingPointColor(20), marginLeft: 12 }} /> &ge; 8% &mdash; click a point to open that seat
        </p>
      </div>

      <div className="card">
        <h2>All {seats?.length ?? "…"} real constituencies</h2>
        <div className="search-row">
          <input type="text" placeholder="Search constituency…" value={query} onChange={(e) => setQuery(e.target.value)} />
          <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
            <option value="">All states</option>
            {states.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div style={{ maxHeight: 480, overflow: "auto" }}>
          <table className="seat-table">
            <thead>
              <tr>
                <th onClick={() => setSortKey("name")}>Constituency</th>
                <th onClick={() => setSortKey("state")}>State</th>
                <th onClick={() => setSortKey("constituency_type")}>Type</th>
                <th onClick={() => setSortKey("winner_party")}>Winner (2019)</th>
                <th onClick={() => setSortKey("real_margin_pct")}>Margin</th>
                <th onClick={() => setSortKey("tipping_point_pct")}>Tipping pt.</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 200).map((s) => (
                <tr key={s.id} onClick={() => onSelect(s.state, s.name)}>
                  <td>{s.name}{s.note ? <span className="muted"> ({s.note})</span> : null}</td>
                  <td>{s.state}</td>
                  <td>{s.constituency_type}</td>
                  <td>{s.winner_party}</td>
                  <td>{s.real_margin_pct.toFixed(2)}pp</td>
                  <td>{s.tipping_point_pct?.toFixed(1) ?? "—"}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length > 200 && (
            <p className="muted" style={{ fontSize: 12 }}>Showing first 200 of {filtered.length} matches — narrow your search.</p>
          )}
        </div>
      </div>
    </>
  );
}
