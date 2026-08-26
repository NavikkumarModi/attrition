import { useEffect, useState } from "react";
import { fetchForecast, type ForecastResult } from "./api";
import Disclosure from "./Disclosure";

export default function Forecast() {
  const [data, setData] = useState<ForecastResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchForecast().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <div className="card muted">Loading…</div>;

  const lift = (data.swing_model_accuracy - data.naive_baseline_accuracy) * 100;

  return (
    <>
      <div className="card">
        <h2>A real, backtested forecasting baseline — not the persuasion simulator</h2>
        <Disclosure summary="How this differs from the simulator">
          Different question from every other tab: no campaign, no contact intensity, no LLM.
          <strong> Uniform national swing</strong> — a classical political-science forecasting technique —
          estimates each party's average real vote-share change from 2014 to 2019, then predicts a seat's 2019
          winner as "2014 share + that party's national swing." Backtested honestly with leave-one-state-out
          cross-validation: each state's own real 2019 result is never used to estimate the swing that predicts
          it. Compared against the naive "incumbent retains" baseline on the same real, matched seats.
        </Disclosure>
      </div>

      <div className="grid cols-4">
        <div className="card stat">
          <div className="value">{(data.swing_model_accuracy * 100).toFixed(1)}%</div>
          <div className="label">swing model accuracy</div>
        </div>
        <div className="card stat">
          <div className="value">{(data.naive_baseline_accuracy * 100).toFixed(1)}%</div>
          <div className="label">naive "incumbent retains" baseline</div>
        </div>
        <div className="card stat">
          <div className="value">{lift >= 0 ? "+" : ""}{lift.toFixed(1)}pp</div>
          <div className="label">lift over the naive baseline</div>
        </div>
        <div className="card stat">
          <div className="value">{(data.swing_model_flip_recall * 100).toFixed(1)}%</div>
          <div className="label">of {data.n_real_flips} real 2014→2019 flips correctly caught</div>
        </div>
      </div>

      <div className="card">
        <h2>Reading this honestly</h2>
        <p className="muted" style={{ fontSize: 13 }}>
          {data.n_seats} real seats evaluated. The naive baseline is already strong ({(data.naive_baseline_accuracy * 100).toFixed(0)}%)
          because most seats don't change hands — a uniform swing model only improves on it by {lift.toFixed(1)} points.
          Far more telling: it correctly predicts only {(data.swing_model_flip_recall * 100).toFixed(0)}% of the seats
          that actually DID flip. 2019 was a large, geographically uneven wave — a single national swing number can't
          capture that unevenness. This is a real baseline honestly reported, not a claim that this project can
          predict elections — see FINDINGS.md for the full writeup.
        </p>
      </div>
    </>
  );
}
