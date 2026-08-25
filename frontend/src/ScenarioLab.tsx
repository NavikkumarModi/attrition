import { useEffect, useMemo, useState } from "react";
import {
  fetchWorlds,
  fetchScenario,
  fetchScenarioCompare,
  fetchScenarioPersonas,
  runScenarioLLMPopulation,
  type WorldsResult,
  type ScenarioDetail,
  type PolicyStats,
  type PersonaSummary,
  type LLMPopulationResult,
} from "./api";

const POLICIES = ["greedy", "eci", "conservative", "thompson", "ucb", "seci"];
const LLM_MODELS: Record<string, string> = {
  anthropic: "claude-haiku-4-5-20251001",
  groq: "openai/gpt-oss-20b",
};

export default function ScenarioLab({ world }: { world: string | string[] }) {
  const worldKeys = Array.isArray(world) ? world : [world];
  const [worlds, setWorlds] = useState<WorldsResult | null>(null);
  const [scenarioName, setScenarioName] = useState<string | null>(null);
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [chosenPolicies, setChosenPolicies] = useState<string[]>(["greedy", "eci"]);
  const [seeds, setSeeds] = useState(20);
  const [compareResult, setCompareResult] = useState<Record<string, PolicyStats> | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const [personas, setPersonas] = useState<PersonaSummary[] | null>(null);
  const [chosenPersonas, setChosenPersonas] = useState<string[]>([]);
  const [provider, setProvider] = useState<"none" | "anthropic" | "groq">("none");
  const [apiKey, setApiKey] = useState("");
  const [llmHorizon, setLlmHorizon] = useState(8);
  const [llmResult, setLlmResult] = useState<LLMPopulationResult | null>(null);
  const [llmError, setLlmError] = useState<string | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);

  useEffect(() => {
    fetchWorlds().then(setWorlds).catch((e) => setError(String(e)));
  }, []);

  const scenarios = useMemo(
    () => worldKeys.flatMap((w) => worlds?.[w]?.scenarios ?? []),
    [worlds, world],
  );
  const description = worldKeys.map((w) => worlds?.[w]?.description).filter(Boolean).join(" ");

  useEffect(() => {
    setScenarioName(scenarios[0]?.name ?? null);
    setCompareResult(null);
    setLlmResult(null);
  }, [scenarios]);

  useEffect(() => {
    if (!scenarioName) return;
    setDetail(null);
    setCompareResult(null);
    setLlmResult(null);
    fetchScenario(scenarioName).then(setDetail).catch((e) => setError(String(e)));
    fetchScenarioPersonas(scenarioName).then((ps) => {
      setPersonas(ps);
      setChosenPersonas(ps.slice(0, 2).map((p) => p.name));
    }).catch((e) => setError(String(e)));
  }, [scenarioName]);

  if (error) return <div className="error-box">{error} — is election_api.py running on :8000?</div>;
  if (!worlds) return <div className="card muted">Loading…</div>;
  if (scenarios.length === 0) return <div className="card muted">No scenarios registered in this world.</div>;

  async function runCompare() {
    if (!scenarioName || chosenPolicies.length === 0) return;
    setCompareLoading(true);
    try {
      setCompareResult(await fetchScenarioCompare(scenarioName, chosenPolicies, seeds));
    } catch (e) {
      setError(String(e));
    } finally {
      setCompareLoading(false);
    }
  }

  async function runLLMPopulation() {
    if (!scenarioName || provider === "none" || !apiKey || chosenPersonas.length === 0) return;
    setLlmLoading(true);
    setLlmError(null);
    try {
      setLlmResult(
        await runScenarioLLMPopulation(scenarioName, {
          provider,
          api_key: apiKey,
          model: LLM_MODELS[provider],
          persona_names: chosenPersonas,
          horizon: llmHorizon,
        }),
      );
    } catch (e) {
      setLlmError(String(e));
    } finally {
      setLlmLoading(false);
    }
  }

  return (
    <>
      <div className="card">
        <h2>{description}</h2>
        <div className="search-row">
          <select value={scenarioName ?? ""} onChange={(e) => setScenarioName(e.target.value)}>
            {scenarios.map((s) => (
              <option key={s.name} value={s.name}>{s.name}{worldKeys.length > 1 ? ` (${s.world})` : ""}</option>
            ))}
          </select>
        </div>
        {detail && (
          <>
            <p className="muted" style={{ fontSize: 13, maxWidth: 760 }}>{detail.description}</p>
            <p className="muted" style={{ fontSize: 12 }}>
              <strong>phenomenon:</strong> {detail.phenomenon}
            </p>
            <p className="muted" style={{ fontSize: 12 }}>
              <strong>expected (from theory):</strong> {detail.expected_behaviour}
            </p>
            <p className="muted" style={{ fontSize: 12 }}>
              {detail.v.length} arms · {detail.agents} agent{detail.agents !== 1 ? "s" : ""} · delta={detail.delta} · horizon={detail.horizon}
            </p>
          </>
        )}
      </div>

      <div className="card">
        <h2>Compare classical policies</h2>
        <div className="search-row" style={{ flexWrap: "wrap" }}>
          {POLICIES.map((p) => (
            <label key={p} className="muted" style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 4 }}>
              <input
                type="checkbox"
                checked={chosenPolicies.includes(p)}
                onChange={(e) =>
                  setChosenPolicies((cur) =>
                    e.target.checked ? [...cur, p] : cur.filter((x) => x !== p),
                  )
                }
              />
              {p}
            </label>
          ))}
        </div>
        <div className="round-slider-row">
          <span>seeds</span>
          <input type="range" min={1} max={200} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} />
          <span className="v">{seeds}</span>
        </div>
        <button className="run-btn" disabled={chosenPolicies.length === 0 || compareLoading} onClick={runCompare}>
          {compareLoading ? "Running…" : "Run simulation"}
        </button>

        {compareResult && (
          <div className="grid cols-4" style={{ marginTop: 16 }}>
            {Object.entries(compareResult).map(([name, stats]) => (
              <div key={name} className="card stat">
                <div className="value">{stats.value.toFixed(3)}</div>
                <div className="label">{name} (regret {stats.regret.toFixed(3)})</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Real LLM population</h2>
        {!personas || personas.length === 0 ? (
          <p className="muted" style={{ fontSize: 13 }}>
            No persona set is registered for the <code>{worldKeys.join(", ")}</code> world(s) yet — see{" "}
            <code>attrition.persona</code> and <code>election_api.py</code>'s <code>WORLD_PERSONAS</code>.
          </p>
        ) : (
          <>
            <p className="muted" style={{ fontSize: 13 }}>
              Bring your own key — used only for this request, never logged or stored. Runs on this
              scenario's own v/p/e/delta.
            </p>
            <div className="search-row">
              <select value={provider} onChange={(e) => setProvider(e.target.value as any)}>
                <option value="none">No provider</option>
                <option value="anthropic">Anthropic</option>
                <option value="groq">Groq</option>
              </select>
              {provider !== "none" && (
                <input
                  type="password"
                  placeholder="sk-... / gsk_..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  style={{ minWidth: 220 }}
                />
              )}
            </div>
            {provider !== "none" && (
              <>
                <div className="search-row" style={{ flexWrap: "wrap" }}>
                  {personas.map((p) => (
                    <label key={p.name} className="muted" style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 4 }}>
                      <input
                        type="checkbox"
                        checked={chosenPersonas.includes(p.name)}
                        onChange={(e) =>
                          setChosenPersonas((cur) =>
                            e.target.checked ? [...cur, p.name] : cur.filter((x) => x !== p.name),
                          )
                        }
                      />
                      {p.name}
                    </label>
                  ))}
                </div>
                <div className="round-slider-row">
                  <span>shared pull budget</span>
                  <input type="range" min={1} max={15} value={llmHorizon}
                        onChange={(e) => setLlmHorizon(Number(e.target.value))} />
                  <span className="v">{llmHorizon}</span>
                </div>
                <button
                  className="run-btn"
                  disabled={!apiKey || chosenPersonas.length === 0 || llmLoading}
                  onClick={runLLMPopulation}
                >
                  {llmLoading ? "Running…" : "Run real LLM population"}
                </button>
              </>
            )}
            {llmError && <p className="error-box" style={{ marginTop: 12 }}>{llmError}</p>}
            {llmResult && (
              <div style={{ marginTop: 16 }}>
                <div className="grid cols-2">
                  <div className="card stat">
                    <div className="value">{llmResult.system_value.toFixed(3)}</div>
                    <div className="label">system value</div>
                  </div>
                  <div className="card stat">
                    <div className="value">{llmResult.system_regret.toFixed(3)}</div>
                    <div className="label">system regret</div>
                  </div>
                </div>
                {Object.entries(llmResult.per_persona).map(([name, log]) => (
                  <p key={name} className="muted" style={{ fontSize: 12 }}>
                    <strong>{name}</strong>: {log.genuine}/{log.total} calls produced a genuine parseable
                    decision — the rest fell back to Greedy.
                    {Object.keys(log.errors).length > 0 && (
                      <> {Object.entries(log.errors).map(([msg, count]) => ` (${count}x ${msg.slice(0, 80)})`)}</>
                    )}
                  </p>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
