import { useEffect, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Legend,
} from "recharts";
import { fetchConstituencies, simulate, type SeatSummary, type SimulateResult } from "./api";

const DEFAULT_SEQUENCE = [0.15, 0.2, 0.25, 0.3, 0.35];

export default function ConstituencyLab({
  initialState, initialName,
}: { initialState: string | null; initialName: string | null }) {
  const [seats, setSeats] = useState<SeatSummary[]>([]);
  const [state, setState] = useState(initialState ?? "Gujarat");
  const [name, setName] = useState(initialName ?? "AHMEDABAD EAST");
  const [sequence, setSequence] = useState<number[]>(DEFAULT_SEQUENCE);
  const [nVoters, setNVoters] = useState(3000);
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { fetchConstituencies().then(setSeats).catch(() => {}); }, []);

  useEffect(() => {
    if (initialState && initialName) { setState(initialState); setName(initialName); }
  }, [initialState, initialName]);

  const seat = seats.find((s) => s.state === state && s.name === name);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setResult(await simulate(state, name, sequence, nVoters));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartData = result
    ? result.level1_switched_fraction_by_round.map((f, i) => ({
        round: i + 1,
        switched: f * 100,
      }))
    : [];

  return (
    <>
      <div className="card">
        <h2>Run a campaign against one real seat</h2>
        <p className="muted" style={{ fontSize: 12, marginBottom: 12, maxWidth: 700 }}>
          Switched-fraction magnitudes are capped at 2% of the electorate — the upper edge of Kalla &amp;
          Broockman's (2018, <em>APSR</em>) real field-experiment confidence interval for campaign contact's
          persuasive effect on vote choice in a general election (their point estimate: statistically zero).
          Small numbers here are the honest result, not a bug — see FINDINGS.md #41.
        </p>
        <div className="grid cols-2">
          <div>
            <label className="muted" style={{ fontSize: 12 }}>Constituency</label>
            <select
              style={{ width: "100%", marginTop: 6 }}
              value={`${state}|${name}`}
              onChange={(e) => { const [st, nm] = e.target.value.split("|"); setState(st); setName(nm); setResult(null); }}
            >
              {seats.map((s) => (
                <option key={s.id} value={`${s.state}|${s.name}`}>
                  {s.name} — {s.state} (margin {s.real_margin_pct.toFixed(1)}pp)
                </option>
              ))}
            </select>
            {seat && (
              <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                Real 2019 result: <strong>{seat.winner_party}</strong> won by {seat.real_margin_pct.toFixed(2)}pp
                over {seat.runner_up_party} · tipping point ≈ {seat.tipping_point_pct?.toFixed(1)}% of the electorate ·{" "}
                winner: {seat.winner_sex === "F" ? "woman" : "man"}, {seat.winner_education?.toLowerCase()},{" "}
                {seat.winner_profession?.toLowerCase()}{seat.winner_incumbent ? ", incumbent" : ""}
                {seat.winner_no_terms ? `, serving term ${seat.winner_no_terms}` : ""}.
              </p>
            )}
          </div>
          <div>
            <label className="muted" style={{ fontSize: 12 }}>n_voters ({nVoters})</label>
            <input type="range" min={500} max={10000} step={500} value={nVoters}
                  onChange={(e) => setNVoters(Number(e.target.value))} />
            <label className="muted" style={{ fontSize: 12, display: "block", marginTop: 10 }}>
              Contact intensity by round
            </label>
            {sequence.map((v, i) => (
              <div className="round-slider-row" key={i}>
                <span className="n">round {i + 1}</span>
                <input type="range" min={0} max={1} step={0.05} value={v}
                      onChange={(e) => {
                        const next = [...sequence];
                        next[i] = Number(e.target.value);
                        setSequence(next);
                      }} />
                <span className="v">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
        <button className="run-btn" onClick={run} disabled={loading} style={{ marginTop: 10 }}>
          {loading ? "Running…" : "Run simulation"}
        </button>
        {error && <p className="error-box" style={{ marginTop: 12 }}>{error}</p>}
      </div>

      {result && (
        <div className="grid cols-2">
          <div className="card">
            <h2>Level 1: switched fraction by round</h2>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="#232938" />
                <XAxis dataKey="round" tick={{ fill: "#8b93a7", fontSize: 11 }} />
                <YAxis
                  unit="%" tick={{ fill: "#8b93a7", fontSize: 11 }}
                  domain={[0, (dataMax: number) => Math.ceil(Math.max(dataMax, result.seat.tipping_point_pct ?? 0) * 1.15)]}
                />
                <Tooltip contentStyle={{ background: "#171c28", border: "1px solid #232938", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine
                  y={(result.seat.tipping_point_pct ?? 0)}
                  stroke="#e0615e" strokeDasharray="4 4"
                  label={{ value: "tipping point", position: "insideTopRight", fill: "#e0615e", fontSize: 11 }}
                />
                <Line type="monotone" dataKey="switched" name="switched %" stroke="#5ec8e0" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2>Recomputed outcome</h2>
            <div style={{ marginBottom: 12 }}>
              <span className={`pill ${result.outcome.flipped ? "flip" : "hold"}`}>
                {result.outcome.flipped ? "SEAT FLIPS" : "SEAT HOLDS"}
              </span>
            </div>
            <p>
              <strong>{result.outcome.winner}</strong> wins with {result.outcome.winner_share_pct.toFixed(2)}%,
              margin {result.outcome.margin_pct.toFixed(2)}pp over {result.outcome.runner_up}.
            </p>
            <p className="muted" style={{ fontSize: 13 }}>
              Final switched fraction {(result.level1_final_switched_fraction * 100).toFixed(1)}% vs.
              a tipping point of {result.seat.tipping_point_pct?.toFixed(1)}%.
            </p>

            {result.level2_decisions && (
              <>
                <h2 style={{ marginTop: 18 }}>Level 2: persona decisions</h2>
                <table className="seat-table">
                  <thead><tr><th>Persona</th><th>Switches at round</th></tr></thead>
                  <tbody>
                    {Object.entries(result.level2_decisions).map(([persona, decisions]) => {
                      const round = decisions.findIndex((d) => d);
                      return (
                        <tr key={persona}>
                          <td>{persona}</td>
                          <td>{round === -1 ? "never" : round + 1}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
