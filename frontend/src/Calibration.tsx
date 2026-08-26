import { useEffect, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { fetchCalibration, type Calibration as CalibrationData } from "./api";
import Disclosure from "./Disclosure";
import { CHART_GRID, CHART_TICK, CHART_TOOLTIP_BG, CHART_TOOLTIP_BORDER } from "./theme";

export default function Calibration() {
  const [data, setData] = useState<CalibrationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCalibration().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">{error} — run scripts/calibrate_campaign_magnitudes.py once to generate attrition/data/calibration_2014_2019.json</div>;
  if (!data) return <div className="card muted">Loading…</div>;

  const chartData = [
    { metric: "median", "real 2014→2019 swing": data.real_swing_electorate_fraction_pct.median, "tipping point required": data.tipping_point_required_pct.median },
    { metric: "p90", "real 2014→2019 swing": data.real_swing_electorate_fraction_pct.p90, "tipping point required": data.tipping_point_required_pct.p90 },
  ];

  const ratio = (data.real_swing_electorate_fraction_pct.median / data.tipping_point_required_pct.median) * 100;

  return (
    <>
      <div className="card">
        <h2>Real historical swing vs. what the model requires to flip a seat</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={CHART_GRID} />
            <XAxis dataKey="metric" tick={{ fill: CHART_TICK, fontSize: 12 }} />
            <YAxis unit="%" tick={{ fill: CHART_TICK, fontSize: 11 }} />
            <Tooltip contentStyle={{ background: CHART_TOOLTIP_BG, border: `1px solid ${CHART_TOOLTIP_BORDER}`, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="real 2014→2019 swing" fill="#2563eb" radius={[4, 4, 0, 0]} />
            <Bar dataKey="tipping point required" fill="#d97706" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <Disclosure summary="Methodology">
          {data.prior_year}→{data.current_year}: same real 543 seats, same delimitation. For the{" "}
          {data.n_matched_same_party} seats where the same party won both times, this compares how much the
          winner's real vote share actually moved against this project's <code>tipping_point</code> requirement —
          a real, checkable order-of-magnitude anchor, not a fit (see FINDINGS.md #38).
        </Disclosure>
      </div>

      <div className="grid cols-4">
        <div className="card stat">
          <div className="value">{ratio.toFixed(0)}%</div>
          <div className="label">real median swing as a share of the median flip requirement</div>
        </div>
        <div className="card stat">
          <div className="value">{data.n_matched_same_party}</div>
          <div className="label">seats, same winning party both elections</div>
        </div>
        <div className="card stat">
          <div className="value">{data.n_flipped_party}</div>
          <div className="label">seats that flipped party outright, 2014→2019</div>
        </div>
        <div className="card stat">
          <div className="value">{data.n_missing_prior_year + data.n_ambiguous}</div>
          <div className="label">seats unmatched (renamed / ambiguous)</div>
        </div>
      </div>

      <div className="card">
        <Disclosure summary="Reading this">
          This does not calibrate any single campaign's persuasion contribution — a seat's real swing reflects
          national mood, local candidates and countless other factors, not one campaign's contact effort alone.
          What it does show: example campaign sequences elsewhere in this app (and in <code>examples/17</code>/
          <code>18</code>) can run an order of magnitude hotter than real historical swings — worth reading
          simulated switched-fraction numbers against this real distribution, not in isolation.
        </Disclosure>
      </div>
    </>
  );
}
