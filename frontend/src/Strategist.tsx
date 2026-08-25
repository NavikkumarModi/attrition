import { useEffect, useState } from "react";
import { fetchStrategist, type StrategistResult } from "./api";

export default function Strategist({
  onSelect,
}: {
  onSelect: (state: string, name: string) => void;
}) {
  const [budget, setBudget] = useState(30);
  const [result, setResult] = useState<StrategistResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStrategist(budget).then(setResult).catch((e) => setError(String(e)));
  }, [budget]);

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
