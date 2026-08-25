import { useEffect, useState } from "react";
import { fetchCampaignPolicy, type CampaignPolicyResult } from "./api";

export default function PolicyCheckpoint() {
  const [data, setData] = useState<CampaignPolicyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCampaignPolicy().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <div className="card muted">Training a policy from scratch (~1-2s)…</div>;

  const { single_agent, decentralised } = data;
  const belowGreedy = decentralised.trained_system_value < decentralised.decentralised_greedy;

  return (
    <>
      <div className="card">
        <h2>A real trained campaign policy — not hand-derived</h2>
        <p className="muted" style={{ fontSize: 13, maxWidth: 780 }}>
          No external campaign policy exists to transplant (unlike AMR's real PPO checkpoint — see FINDINGS.md#24.4),
          so this trains one from scratch: the gradient-bandit method, seeing only realised per-pull reward on the{" "}
          <code>campaign-persuasion</code> arms — the same information Greedy sees, nothing about the true
          destruction probability or externality. Trained fresh once per server process.
        </p>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Learned preference by contact intensity</h3>
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end", height: 120, padding: "8px 0" }}>
          {data.contact_intensity.map((intensity, i) => {
            const pref = data.learned_preferences[i];
            const max = Math.max(...data.learned_preferences);
            const min = Math.min(...data.learned_preferences);
            const height = ((pref - min) / (max - min || 1)) * 90 + 10;
            const isFav = i === data.favourite_arm;
            return (
              <div key={i} style={{ textAlign: "center", flex: 1 }}>
                <div
                  style={{
                    height: `${height}px`,
                    background: isFav ? "var(--accent, #6ea8fe)" : "var(--border)",
                    borderRadius: 4,
                  }}
                  title={`preference ${pref.toFixed(2)}`}
                />
                <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>{intensity.toFixed(2)}</div>
              </div>
            );
          })}
        </div>
        <p className="muted" style={{ fontSize: 12 }}>
          Learned favourite: contact intensity {data.contact_intensity[data.favourite_arm].toFixed(2)} (arm{" "}
          {data.favourite_arm}) — the highest-value arm, same as greedy would pick.
        </p>
      </div>

      <h3>Single agent, 200 held-out seeds</h3>
      <div className="grid cols-4">
        <div className="card stat">
          <div className="value">{single_agent.greedy.value.toFixed(3)}</div>
          <div className="label">greedy value (regret {single_agent.greedy.regret.toFixed(3)})</div>
        </div>
        <div className="card stat">
          <div className="value">{single_agent.eci.value.toFixed(3)}</div>
          <div className="label">eci value (regret {single_agent.eci.regret.toFixed(3)})</div>
        </div>
        <div className="card stat">
          <div className="value">{single_agent.trained.value.toFixed(3)}</div>
          <div className="label">trained-policy value (regret {single_agent.trained.regret.toFixed(3)})</div>
        </div>
        <div className="card stat">
          <div className="value">
            {single_agent.trained.regret < single_agent.eci.regret / 2 ? "greedy-like" : "eci-like"}
          </div>
          <div className="label">which baseline it resembles</div>
        </div>
      </div>
      <p className="muted" style={{ fontSize: 13 }}>
        The trained policy lands close to greedy's near-zero regret, not ECI's — the predicted result, not a
        surprise one: realised per-pull reward already folds the externality into a burden term common to every
        available arm in a round, so nothing in the training signal tells the policy a cheap-to-the-pool arm from
        an expensive one.
      </p>

      <h3>Decentralised — {decentralised.n_agents} frozen copies sharing one pool</h3>
      <div className="grid cols-4">
        <div className="card stat">
          <div className="value">{decentralised.planner_optimum.toFixed(2)}</div>
          <div className="label">planner (optimum)</div>
        </div>
        <div className="card stat">
          <div className="value">{decentralised.decentralised_greedy.toFixed(2)}</div>
          <div className="label">decentralised greedy</div>
        </div>
        <div className="card stat">
          <div className="value">{decentralised.decentralised_eci.toFixed(2)}</div>
          <div className="label">decentralised eci</div>
        </div>
        <div className="card stat">
          <div className="value">{decentralised.trained_system_value.toFixed(2)}</div>
          <div className="label">trained policy × {decentralised.n_agents}</div>
        </div>
      </div>
      <p className="muted" style={{ fontSize: 13 }}>
        {belowGreedy
          ? "This run lands BELOW even decentralised greedy: identical copies of the same trained policy share a " +
            "strong preference for the same arm, so collision compounds on top of the externality-blindness shown " +
            "above — a real cost of decentralising a policy that was never trained to expect company."
          : "This run holds up better than decentralised greedy — the coordination-failure risk above is a real " +
            "mechanism either way, not this run's headline."}{" "}
        See FINDINGS.md#44 and <code>examples/20_campaign_policy_checkpoint.py</code> for the full writeup.
      </p>
    </>
  );
}
