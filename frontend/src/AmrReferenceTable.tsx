import { useEffect, useState } from "react";
import { fetchAmrReferenceTable, type AmrReferenceTable as AmrTable } from "./api";
import Disclosure from "./Disclosure";

export default function AmrReferenceTable() {
  const [data, setData] = useState<AmrTable | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAmrReferenceTable().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <div className="card muted">Loading…</div>;

  const { in_repo, external_siloed, external_shared } = data;
  const poaGreedy = in_repo.planner / in_repo.greedy;
  const poaEci = in_repo.planner / in_repo.eci;

  return (
    <>
      <div className="card">
        <h2>The unified five-policy table — not on one axis, on purpose</h2>
        <Disclosure summary="Why these numbers aren't directly comparable">
          FINDINGS.md#26 recorded this as open rather than forced: this repo's engine (planner, greedy, eci,
          lambda-weighted) and <code>abx_amr_simulator</code>'s own environment (their trained PPO checkpoint) are
          not on the same axis — different environments, different value semantics, different agent-count and
          coordination models. Putting them side by side does not create common ground; it puts the
          incomparability in view instead of a footnote. Read down each half separately; read across only for
          the qualitative shape. See FINDINGS.md#44 and{" "}
          <code>experiments/exp65_amr_five_policy_reference_table.py</code>.
        </Disclosure>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Half 1 — this repo's engine (m=3, horizon=9, delta=0.35)</h3>
        <div className="grid cols-4">
          <div className="card stat">
            <div className="value">{in_repo.planner.toFixed(2)}</div>
            <div className="label">planner (optimum)</div>
          </div>
          <div className="card stat">
            <div className="value">{in_repo.greedy.toFixed(2)}</div>
            <div className="label">decentralised greedy (PoA {poaGreedy.toFixed(2)})</div>
          </div>
          <div className="card stat">
            <div className="value">{in_repo.eci.toFixed(2)}</div>
            <div className="label">decentralised eci (PoA {poaEci.toFixed(2)})</div>
          </div>
          <div className="card stat">
            <div className="value">{in_repo["lambda_weighted(lam=1)"].toFixed(2)}</div>
            <div className="label">lambda_weighted(lam=1)</div>
          </div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Half 2a — siloed (single centralised policy, FINDINGS.md#24.2)</h3>
          {Object.entries(external_siloed).map(([k, v]) => (
            <p key={k} className="muted" style={{ fontSize: 13 }}>
              <strong>{k}:</strong> {v.toFixed(3)}
            </p>
          ))}
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Half 2b — shared (m=3 decentralised copies, FINDINGS.md#24.4)</h3>
          {Object.entries(external_shared).map(([k, v]) => (
            <p key={k} className="muted" style={{ fontSize: 13 }}>
              <strong>{k}:</strong> {v.toFixed(3)}
            </p>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Qualitative reading (not a numeric comparison)</h3>
        <p className="muted" style={{ fontSize: 13 }}>
          Half 1: eci/lambda_weighted recovers {((in_repo.eci / in_repo.planner) * 100).toFixed(1)}% of the planner
          ceiling; greedy recovers only {((in_repo.greedy / in_repo.planner) * 100).toFixed(1)}%. Half 2:
          decentralising the trained checkpoint moves antibiotic A's resistance from{" "}
          {(external_siloed.final_amr_a * 100).toFixed(1)}% (siloed) to{" "}
          {(external_shared.final_amr_a * 100).toFixed(1)}% (shared) — most of the way back to the
          no-stewardship level. Both halves show the same shape: a single-decision-maker fix recovers real value
          alone, and decentralising into non-coordinating copies erodes a large fraction of that recovery.
        </p>
      </div>
    </>
  );
}
