import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Cell,
} from "recharts";
import { fetchConstituencies, fetchNationalScan, type SeatSummary, type NationalScan } from "./api";
import IndiaMap from "./IndiaMap";
import Disclosure from "./Disclosure";
import { CHART_GRID, CHART_TICK, CHART_TOOLTIP_BG, CHART_TOOLTIP_BORDER, tippingPointColor } from "./theme";

export default function Overview({ onSelect }: { onSelect: (state: string, name: string) => void }) {
  const [seats, setSeats] = useState<SeatSummary[] | null>(null);
  const [scan, setScan] = useState<NationalScan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [sortKey, setSortKey] = useState<keyof SeatSummary>("tipping_point_pct");

  useEffect(() => {
    fetchConstituencies().then(setSeats).catch((e) => setError(String(e)));
    fetchNationalScan().then(setScan).catch((e) => setError(String(e)));
  }, []);

  const states = useMemo(
    () => (seats ? Array.from(new Set(seats.map((s) => s.state))).sort() : []),
    [seats],
  );

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
      <IndiaMap onSelect={onSelect} />

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
              formatter={(value: number, key: string) => [key === "real_margin_pct" ? `${value.toFixed(2)}pp` : `${value?.toFixed?.(2)}%`, key]}
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
