import { useEffect, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Legend,
} from "recharts";
import {
  fetchConstituencies, simulate, fetchAssemblyRollup, fetchAssemblyRollupCoverage,
  fetchCensusRollup, fetchCensusRollupCoverage, fetchShrugRollup, fetchShrugRollupCoverage,
  type SeatSummary, type SimulateResult, type AssemblyRollupResponse, type AssemblyRollupCoverage,
  type CensusRollupResponse, type CensusRollupCoverage,
  type ShrugRollupResponse, type ShrugRollupCoverage,
} from "./api";
import Disclosure from "./Disclosure";
import { CHART_GRID, CHART_TICK, CHART_TOOLTIP_BG, CHART_TOOLTIP_BORDER } from "./theme";

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

  const [useNetwork, setUseNetwork] = useState(false);
  const [peerConversionProb, setPeerConversionProb] = useState(0.05);
  const [graphDensity, setGraphDensity] = useState(0.02);
  const [meanDegreeOn, setMeanDegreeOn] = useState(false);
  const [meanDegree, setMeanDegree] = useState(10);
  const [nSeeds, setNSeeds] = useState(1);
  const [incumbentOverride, setIncumbentOverride] = useState<"real" | "yes" | "no">("real");
  const [marginOverrideOn, setMarginOverrideOn] = useState(false);
  const [marginOverride, setMarginOverride] = useState(5);
  const [useDistrictGranularity, setUseDistrictGranularity] = useState(false);

  const [assemblyRollup, setAssemblyRollup] = useState<AssemblyRollupResponse | null>(null);
  const [assemblyCoverage, setAssemblyCoverage] = useState<AssemblyRollupCoverage | null>(null);
  const [censusRollup, setCensusRollup] = useState<CensusRollupResponse | null>(null);
  const [censusCoverage, setCensusCoverage] = useState<CensusRollupCoverage | null>(null);
  const [shrugRollup, setShrugRollup] = useState<ShrugRollupResponse | null>(null);
  const [shrugCoverage, setShrugCoverage] = useState<ShrugRollupCoverage | null>(null);

  useEffect(() => { fetchConstituencies().then(setSeats).catch(() => {}); }, []);
  useEffect(() => { fetchAssemblyRollupCoverage().then(setAssemblyCoverage).catch(() => {}); }, []);
  useEffect(() => { fetchCensusRollupCoverage().then(setCensusCoverage).catch(() => {}); }, []);
  useEffect(() => { fetchShrugRollupCoverage().then(setShrugCoverage).catch(() => {}); }, []);

  useEffect(() => {
    if (initialState && initialName) { setState(initialState); setName(initialName); }
  }, [initialState, initialName]);

  useEffect(() => {
    setAssemblyRollup(null);
    fetchAssemblyRollup(state, name).then(setAssemblyRollup).catch(() => {});
    setCensusRollup(null);
    fetchCensusRollup(state, name).then(setCensusRollup).catch(() => {});
    setShrugRollup(null);
    fetchShrugRollup(state, name).then(setShrugRollup).catch(() => {});
  }, [state, name]);

  const seat = seats.find((s) => s.state === state && s.name === name);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setResult(
        await simulate(state, name, sequence, nVoters, {
          n_seeds: nSeeds,
          use_network: useNetwork,
          peer_conversion_prob: peerConversionProb,
          graph_density: graphDensity,
          mean_degree: meanDegreeOn ? meanDegree : null,
          incumbent_override: incumbentOverride === "real" ? null : incumbentOverride === "yes",
          margin_override_pct: marginOverrideOn ? marginOverride : null,
          use_district_granularity: useDistrictGranularity,
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartData = result?.level1_switched_fraction_by_round
    ? result.level1_switched_fraction_by_round.map((f, i) => ({
        round: i + 1,
        switched: f * 100,
      }))
    : [];

  function addRound() {
    setSequence((s) => [...s, s[s.length - 1] ?? 0.2]);
  }
  function removeRound(i: number) {
    setSequence((s) => (s.length > 1 ? s.filter((_, idx) => idx !== i) : s));
  }

  return (
    <>
      <div className="card">
        <h2>Run a campaign against one real seat</h2>
        <Disclosure summary="Why are the numbers so small?">
          Switched-fraction magnitudes are capped at 2% of the electorate — the upper edge of Kalla &amp;
          Broockman's (2018, <em>APSR</em>) real field-experiment confidence interval for campaign contact's
          persuasive effect on vote choice in a general election (their point estimate: statistically zero).
          Small numbers here are the honest result, not a bug — see FINDINGS.md #41.
        </Disclosure>
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
            <label className="muted" style={{ fontSize: 12 }}>n_voters ({nVoters.toLocaleString()})</label>
            <input type="range" min={500} max={10000} step={500} value={Math.min(nVoters, 10000)}
                  onChange={(e) => setNVoters(Number(e.target.value))} />
            <div className="search-row" style={{ marginTop: 6 }}>
              {[10000, 100000, 500000, 1000000].map((n) => (
                <button key={n} className="run-btn" style={{ padding: "4px 10px", fontSize: 11 }}
                       onClick={() => setNVoters(n)}>
                  {n.toLocaleString()}
                </button>
              ))}
            </div>
            <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
              Real, benchmarked capacity (FINDINGS.md #52): up to 1,000,000 voters (~0.5s) without network
              propagation on; up to 500,000 with it on (graph construction dominates that path's cost).
            </p>
            <label className="muted" style={{ fontSize: 12, display: "block", marginTop: 10 }}>
              Contact intensity by round ({sequence.length} round{sequence.length !== 1 ? "s" : ""})
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
                <button className="run-btn" style={{ padding: "2px 8px", fontSize: 11 }}
                       onClick={() => removeRound(i)} disabled={sequence.length <= 1}>
                  ✕
                </button>
              </div>
            ))}
            <button className="run-btn" style={{ padding: "4px 10px", fontSize: 12, marginTop: 4 }}
                   onClick={addRound}>
              + add round
            </button>
          </div>
        </div>
        <button className="run-btn" onClick={run} disabled={loading} style={{ marginTop: 10 }}>
          {loading ? "Running…" : "Run simulation"}
        </button>
        {error && <p className="error-box" style={{ marginTop: 12 }}>{error}</p>}
      </div>

      <div className="card">
        <h2>Real state Assembly rollup for this seat</h2>
        <Disclosure summary="Source & methodology">
          A second real data source (TCPD's state Assembly election results, joined to this seat via DataMeet's
          real Assembly-Constituency-to-Parliamentary-Constituency crosswalk — an exact administrative fact, not an
          approximation) — not a new mechanism, real aggregate results over this seat's constituent state Assembly
          segments. State elections run on each state's own cycle, not a single national date, so the year used is
          not always 2019 — shown explicitly below. See FINDINGS.md #49.
        </Disclosure>
        {assemblyRollup?.rollup ? (
          <>
            <div className="grid cols-4">
              <div className="stat">
                <div className="value">{assemblyRollup.rollup.n_constituent_acs_matched}</div>
                <div className="label">constituent Assembly segments matched</div>
              </div>
              <div className="stat">
                <div className="value">{assemblyRollup.rollup.most_common_party}</div>
                <div className="label">
                  most common winner ({(assemblyRollup.rollup.most_common_party_fraction * 100).toFixed(0)}% of segments)
                </div>
              </div>
              <div className="stat">
                <div className="value">{(assemblyRollup.rollup.woman_winner_fraction * 100).toFixed(0)}%</div>
                <div className="label">segments with a real woman winner</div>
              </div>
              <div className="stat">
                <div className="value">{(assemblyRollup.rollup.incumbent_winner_fraction * 100).toFixed(0)}%</div>
                <div className="label">segments won by a real incumbent</div>
              </div>
            </div>
            <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
              Party diversity: {assemblyRollup.rollup.party_diversity} distinct real winning parties · mean turnout{" "}
              {assemblyRollup.rollup.mean_turnout_pct?.toFixed(1) ?? "—"}% · mean margin{" "}
              {assemblyRollup.rollup.mean_margin_pct?.toFixed(1) ?? "—"}pp · assembly year(s) used:{" "}
              {assemblyRollup.rollup.assembly_years_used.join(", ")}.
            </p>
          </>
        ) : (
          <p className="muted" style={{ fontSize: 13 }}>
            No constituent Assembly Constituency matched for this seat — a real, disclosed gap in the
            AC-to-PC join (
            {assemblyCoverage
              ? `${assemblyCoverage.n_real_seats_with_rollup} of ${assemblyCoverage.n_real_seats_total} real seats have rollup data`
              : "loading coverage…"}
            ), not a claim that this seat has no Assembly Constituencies.
          </p>
        )}
      </div>

      <div className="card">
        <h2>Real Census 2011 district demographics for this seat</h2>
        <Disclosure summary="Source & methodology">
          Real district-level population, households, and density (Census of India 2011, table A-1), joined via the
          same real AC-to-PC crosswalk. A seat spanning more than one district shows each district's real,
          unmodified figures — never summed into one population number, which would double-count people in a
          district shared with another seat. See FINDINGS.md #50.
        </Disclosure>
        {censusRollup?.rollup ? (
          <>
            <div className="grid cols-2">
              <div className="stat">
                <div className="value">{censusRollup.rollup.n_constituent_districts}</div>
                <div className="label">constituent real district(s)</div>
              </div>
              <div className="stat">
                <div className="value">{censusRollup.rollup.mean_population_density_per_sq_km.toFixed(0)}/km²</div>
                <div className="label">mean population density across constituent districts</div>
              </div>
            </div>
            <table className="seat-table" style={{ marginTop: 12 }}>
              <thead>
                <tr><th>District</th><th>Population (2011)</th><th>Households</th><th>Area (km²)</th><th>Density/km²</th></tr>
              </thead>
              <tbody>
                {censusRollup.rollup.constituent_districts.map((d) => (
                  <tr key={d.district}>
                    <td>{d.district}</td>
                    <td>{d.population_total.toLocaleString()}</td>
                    <td>{d.n_households.toLocaleString()}</td>
                    <td>{d.area_sq_km.toLocaleString()}</td>
                    <td>{d.population_density_per_sq_km.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p className="muted" style={{ fontSize: 13 }}>
            No constituent district matched for this seat — a real, disclosed gap in the district-to-PC join (
            {censusCoverage
              ? `${censusCoverage.n_real_seats_with_rollup} of ${censusCoverage.n_real_seats_total} real seats have rollup data`
              : "loading coverage…"}
            ).
          </p>
        )}
      </div>

      <div className="card">
        <h2>Real SHRUG socioeconomic indicators for this seat</h2>
        <Disclosure summary="Source & methodology">
          Real demographic (Census 2011), economic (2012 SECC consumption/poverty — the first genuinely economic
          indicator in this repo), satellite-derived wealth (2021 Facebook Relative Wealth Index), and infrastructure
          (PMGSY rural road connectivity) data, joined via the SAME real AC-to-PC crosswalk, restricted to
          post-2008-delimitation Assembly Constituencies specifically (SHRUG's own boundary vintage — a stricter,
          disclosed filter than the Assembly/Census rollups above use). Source:{" "}
          <a href="https://devdatalab.org/shrug_download/" target="_blank" rel="noreferrer">
            Development Data Lab, SHRUG v2.2
          </a>{" "}
          (CC BY-NC-SA 4.0). See FINDINGS.md #57.
        </Disclosure>
        {shrugRollup?.rollup ? (
          <>
            <div className="grid cols-2">
              <div className="stat">
                <div className="value">{shrugRollup.rollup.n_constituent_acs_matched}</div>
                <div className="label">constituent real Assembly Constituency/Constituencies</div>
              </div>
              <div className="stat">
                <div className="value">
                  {shrugRollup.rollup.population_total?.toLocaleString() ?? "—"}
                </div>
                <div className="label">total real population (Census 2011, summed across constituent ACs)</div>
              </div>
              <div className="stat">
                <div className="value">
                  {shrugRollup.rollup.secc_pov_rate_rural != null
                    ? `${(shrugRollup.rollup.secc_pov_rate_rural * 100).toFixed(1)}%`
                    : "—"}
                </div>
                <div className="label">rural poverty rate (2012 SECC, population-weighted)</div>
              </div>
              <div className="stat">
                <div className="value">
                  {shrugRollup.rollup.pmgsy_pct_villages_connected != null
                    ? `${(shrugRollup.rollup.pmgsy_pct_villages_connected * 100).toFixed(1)}%`
                    : "—"}
                </div>
                <div className="label">villages with a completed PMGSY road</div>
              </div>
            </div>
            <table className="seat-table" style={{ marginTop: 12 }}>
              <thead>
                <tr>
                  <th>Assembly Constituency</th><th>Population</th><th>Rural poverty %</th>
                  <th>Urban poverty %</th><th>Facebook wealth index</th>
                </tr>
              </thead>
              <tbody>
                {shrugRollup.rollup.constituent_acs.map((ac) => (
                  <tr key={ac.ac08_id}>
                    <td>{ac.ac_name}</td>
                    <td>{ac.population_total?.toLocaleString() ?? "—"}</td>
                    <td>{ac.secc_pov_rate_rural != null ? `${(ac.secc_pov_rate_rural * 100).toFixed(1)}%` : "—"}</td>
                    <td>{ac.secc_pov_rate_urban != null ? `${(ac.secc_pov_rate_urban * 100).toFixed(1)}%` : "—"}</td>
                    <td>{ac.facebook_mean_rwi != null ? ac.facebook_mean_rwi.toFixed(3) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted" style={{ fontSize: 11, marginTop: 8 }}>
              Rural and urban poverty rates are never combined into one figure — a constituency's missing segment
              (e.g. a fully rural AC with no urban SECC row) shows as "—", not zero or an average of the other
              segment. The wealth index is zero-centred with no fixed unit — useful for comparing constituencies
              against each other, not as an absolute rupee figure.
            </p>
          </>
        ) : (
          <p className="muted" style={{ fontSize: 13 }}>
            No constituent post-2008-delimitation Assembly Constituency matched for this seat — a real, disclosed
            gap in the join (
            {shrugCoverage
              ? `${shrugCoverage.n_real_seats_with_rollup} of ${shrugCoverage.n_real_seats_total} real seats have rollup data`
              : "loading coverage…"}
            ).
          </p>
        )}
      </div>

      <div className="card">
        <h2>Granular controls — real, already-causal factors only</h2>
        <Disclosure summary="What's real here, and what isn't">
          No per-voter demographic or economic data exists in the real TCPD source, so none is invented here (see
          FINDINGS.md #47). Every control below is either a mechanism already wired into the model (voter-to-voter
          propagation, seed variation) or a real, already-causal seat covariate (incumbency, margin) offered as a
          counterfactual.
        </Disclosure>
        <div className="grid cols-2">
          <div>
            <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={useNetwork} onChange={(e) => setUseNetwork(e.target.checked)} />
              Voter-to-voter network propagation
              {useDistrictGranularity && (
                <span className="muted" style={{ fontSize: 11 }}>(runs independently within each district — FINDINGS.md #53)</span>
              )}
            </label>
            {useNetwork && (
              <>
                <div className="round-slider-row">
                  <span>peer conversion prob</span>
                  <input type="range" min={0} max={0.3} step={0.01} value={peerConversionProb}
                        onChange={(e) => setPeerConversionProb(Number(e.target.value))} />
                  <span className="v">{peerConversionProb.toFixed(2)}</span>
                </div>
                <div className="round-slider-row">
                  <span>graph density</span>
                  <input type="range" min={0.005} max={0.1} step={0.005} value={graphDensity}
                        onChange={(e) => setGraphDensity(Number(e.target.value))}
                        disabled={meanDegreeOn} />
                  <span className="v">{graphDensity.toFixed(3)}</span>
                </div>
                <p className="muted" style={{ fontSize: 11 }}>
                  Above ~5,000 voters, graph density stops being used automatically — a fixed edge probability
                  implies a mean degree that grows with population size (a real correctness issue, not a style
                  choice — see FINDINGS.md #52). Set mean degree explicitly instead:
                </p>
                <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="checkbox" checked={meanDegreeOn} onChange={(e) => setMeanDegreeOn(e.target.checked)} />
                  Set mean degree explicitly (real neighbors per voter, population-independent)
                </label>
                {meanDegreeOn && (
                  <div className="round-slider-row">
                    <span>mean degree</span>
                    <input type="range" min={1} max={100} step={1} value={meanDegree}
                          onChange={(e) => setMeanDegree(Number(e.target.value))} />
                    <span className="v">{meanDegree}</span>
                  </div>
                )}
              </>
            )}

            <label className="muted" style={{ fontSize: 12, display: "block", marginTop: 14 }}>
              Flip-probability sweep (seeds: {nSeeds})
            </label>
            <input type="range" min={1} max={50} value={nSeeds}
                  onChange={(e) => setNSeeds(Number(e.target.value))} />
            <p className="muted" style={{ fontSize: 11 }}>
              {nSeeds === 1
                ? "one deterministic run"
                : `${nSeeds} seeds — reports what fraction actually flip, not one draw`}
            </p>
          </div>

          <div>
            <label className="muted" style={{ fontSize: 12 }}>Incumbency (real for this seat: shown in the card above)</label>
            <select style={{ width: "100%", marginTop: 6 }} value={incumbentOverride}
                   onChange={(e) => setIncumbentOverride(e.target.value as any)}>
              <option value="real">Use real 2019 incumbency</option>
              <option value="yes">Counterfactual: winner IS incumbent</option>
              <option value="no">Counterfactual: winner is NOT incumbent</option>
            </select>

            <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6, marginTop: 14 }}>
              <input type="checkbox" checked={marginOverrideOn} onChange={(e) => setMarginOverrideOn(e.target.checked)} />
              Override real margin (counterfactual: closer/safer seat)
            </label>
            {marginOverrideOn && (
              <div className="round-slider-row">
                <span>margin</span>
                <input type="range" min={0.1} max={40} step={0.1} value={marginOverride}
                      onChange={(e) => setMarginOverride(Number(e.target.value))} />
                <span className="v">{marginOverride.toFixed(1)}pp</span>
              </div>
            )}

            <label className="muted" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6, marginTop: 14 }}>
              <input type="checkbox" checked={useDistrictGranularity}
                    onChange={(e) => setUseDistrictGranularity(e.target.checked)} />
              Population granularity: split voters across real constituent districts
            </label>
          </div>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>Agents in this run</h2>
          <div className="grid cols-4">
            <div className="stat">
              <div className="value">{nVoters.toLocaleString()}</div>
              <div className="label">Level 1 statistical voter-agents</div>
            </div>
            <div className="stat">
              <div className="value">{result.level2_decisions ? Object.keys(result.level2_decisions).length : 0}</div>
              <div className="label">Level 2 LLM-driven voter personas</div>
            </div>
            <div className="stat">
              <div className="value">{result.network ? nVoters.toLocaleString() : "off"}</div>
              <div className="label">voters in the peer-propagation graph</div>
            </div>
            <div className="stat">
              <div className="value">{result.flip_probability?.n_seeds ?? 1}</div>
              <div className="label">independent seeds simulated this run</div>
            </div>
          </div>
        </div>
      )}

      {result?.district_granularity && (
        <div className="card">
          <h2>Population granularity: split by real constituent district</h2>
          <p className="muted" style={{ fontSize: 12, marginBottom: 12 }}>
            Sub-populations differ only in real population SIZE, not persuadability — there is no real per-district
            margin data to derive a differentiated resistance distribution from. See FINDINGS.md #51.
          </p>
          <table className="seat-table">
            <thead>
              <tr>
                <th>District</th><th>Voters</th>
                {result.district_granularity.network_within_district && <th>Direct only %</th>}
                <th>Final switched %</th>
              </tr>
            </thead>
            <tbody>
              {result.district_granularity.per_district.map((d, i) => (
                <tr key={i}>
                  <td>{d.district ?? "(no rollup — whole seat as one pool)"}</td>
                  <td>{d.n_voters.toLocaleString()}</td>
                  {result.district_granularity!.network_within_district && (
                    <td>{((d.direct_only_final_switched_fraction ?? 0) * 100).toFixed(2)}%</td>
                  )}
                  <td>{(d.final_switched_fraction * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          {result.district_granularity.network_within_district && (
            <p className="muted" style={{ fontSize: 11, marginTop: 8 }}>
              Network propagation runs independently within each district (no cross-district edges — a real, stated
              scoping choice, not a claim that real social ties respect district lines). See FINDINGS.md #53.
            </p>
          )}
          <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
            Combined (exact population-weighted mean):{" "}
            {(result.district_granularity.combined_final_switched_fraction * 100).toFixed(2)}%
          </p>
        </div>
      )}

      {result && (
        <div className="grid cols-2">
          <div className="card">
            <h2>Level 1: switched fraction by round</h2>
            {result.level1_switched_fraction_by_round ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke={CHART_GRID} />
                  <XAxis dataKey="round" tick={{ fill: CHART_TICK, fontSize: 11 }} />
                  <YAxis
                    unit="%" tick={{ fill: CHART_TICK, fontSize: 11 }}
                    domain={[0, (dataMax: number) => Math.ceil(Math.max(dataMax, result.seat.tipping_point_pct ?? 0) * 1.15)]}
                  />
                  <Tooltip contentStyle={{ background: CHART_TOOLTIP_BG, border: `1px solid ${CHART_TOOLTIP_BORDER}`, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <ReferenceLine
                    y={(result.seat.tipping_point_pct ?? 0)}
                    stroke="#dc2626" strokeDasharray="4 4"
                    label={{ value: "tipping point", position: "insideTopRight", fill: "#dc2626", fontSize: 11 }}
                  />
                  <Line type="monotone" dataKey="switched" name="switched %" stroke="#2563eb" strokeWidth={2} dot />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="muted" style={{ fontSize: 13 }}>
                No per-round breakdown for population-granular runs — see the district table above for the final
                per-district and combined figures.
              </p>
            )}
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

            {result.network && (
              <>
                <h2 style={{ marginTop: 18 }}>Voter-to-voter network effect</h2>
                <p className="muted" style={{ fontSize: 13 }}>
                  Direct contact alone: {(result.network.direct_only_final_switched_fraction * 100).toFixed(2)}% ·
                  with peer propagation: {(result.network.final_switched_fraction * 100).toFixed(2)}% — the
                  network added{" "}
                  {((result.network.final_switched_fraction - result.network.direct_only_final_switched_fraction) * 100).toFixed(2)}
                  pp on top of direct contact, at peer_conversion_prob={result.network.peer_conversion_prob}, graph:{" "}
                  {result.network.graph_method === "random_by_degree"
                    ? `mean degree ${result.network.graph_param} (large-population generator, FINDINGS.md #52)`
                    : `density ${result.network.graph_param} (exact Erdos-Renyi)`}.
                </p>
              </>
            )}

            {result.flip_probability && (
              <>
                <h2 style={{ marginTop: 18 }}>Flip probability ({result.flip_probability.n_seeds} seeds)</h2>
                <div className="grid cols-2">
                  <div className="card stat">
                    <div className="value">{(result.flip_probability.flip_rate * 100).toFixed(0)}%</div>
                    <div className="label">of seeds this seat actually flips</div>
                  </div>
                  <div className="card stat">
                    <div className="value">
                      {(result.flip_probability.final_switched_fraction_min * 100).toFixed(2)}–
                      {(result.flip_probability.final_switched_fraction_max * 100).toFixed(2)}%
                    </div>
                    <div className="label">switched-fraction range across seeds</div>
                  </div>
                </div>
              </>
            )}

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
