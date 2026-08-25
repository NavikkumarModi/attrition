import { useState } from "react";
import Overview from "./Overview";
import ConstituencyLab from "./ConstituencyLab";
import Calibration from "./Calibration";
import Strategist from "./Strategist";
import Forecast from "./Forecast";
import PolicyCheckpoint from "./PolicyCheckpoint";
import ScenarioLab from "./ScenarioLab";
import AmrReferenceTable from "./AmrReferenceTable";

type World = "election" | "clinical-amr" | "agent-ecosystem" | "other";
type ElectionTab = "overview" | "lab" | "strategist" | "calibration" | "forecast" | "policy";
type AmrTab = "lab" | "reference";

const WORLD_LABELS: Record<World, string> = {
  election: "Election",
  "clinical-amr": "Clinical / AMR",
  "agent-ecosystem": "Agent ecosystem",
  other: "Other domains",
};

export default function App() {
  const [world, setWorld] = useState<World>("election");
  const [electionTab, setElectionTab] = useState<ElectionTab>("overview");
  const [amrTab, setAmrTab] = useState<AmrTab>("lab");
  const [selected, setSelected] = useState<{ state: string; name: string } | null>(null);

  function openSeat(state: string, name: string) {
    setSelected({ state, name });
    setWorld("election");
    setElectionTab("lab");
  }

  return (
    <>
      <header className="app-header">
        <div>
          <h1>ATTRITION</h1>
          <div className="subtitle">one engine, many worlds — bandits with consumable action sets</div>
        </div>
        <div className="subtitle">no-regret is not no-harm</div>
      </header>

      <nav className="tabs">
        {(Object.keys(WORLD_LABELS) as World[]).map((w) => (
          <button key={w} className={world === w ? "active" : ""} onClick={() => setWorld(w)}>
            {WORLD_LABELS[w]}
          </button>
        ))}
      </nav>

      {world === "election" && (
        <>
          <nav className="tabs" style={{ marginBottom: 20 }}>
            <button className={electionTab === "overview" ? "active" : ""} onClick={() => setElectionTab("overview")}>National overview</button>
            <button className={electionTab === "lab" ? "active" : ""} onClick={() => setElectionTab("lab")}>Constituency lab</button>
            <button className={electionTab === "strategist" ? "active" : ""} onClick={() => setElectionTab("strategist")}>Strategist</button>
            <button className={electionTab === "forecast" ? "active" : ""} onClick={() => setElectionTab("forecast")}>Forecast</button>
            <button className={electionTab === "policy" ? "active" : ""} onClick={() => setElectionTab("policy")}>Trained policy</button>
            <button className={electionTab === "calibration" ? "active" : ""} onClick={() => setElectionTab("calibration")}>Calibration</button>
          </nav>
          {electionTab === "overview" && <Overview onSelect={openSeat} />}
          {electionTab === "lab" && <ConstituencyLab initialState={selected?.state ?? null} initialName={selected?.name ?? null} />}
          {electionTab === "strategist" && <Strategist onSelect={openSeat} />}
          {electionTab === "forecast" && <Forecast />}
          {electionTab === "policy" && <PolicyCheckpoint />}
          {electionTab === "calibration" && <Calibration />}
        </>
      )}

      {world === "clinical-amr" && (
        <>
          <nav className="tabs" style={{ marginBottom: 20 }}>
            <button className={amrTab === "lab" ? "active" : ""} onClick={() => setAmrTab("lab")}>Scenario lab</button>
            <button className={amrTab === "reference" ? "active" : ""} onClick={() => setAmrTab("reference")}>Reference table</button>
          </nav>
          {amrTab === "lab" && <ScenarioLab world="clinical-amr" />}
          {amrTab === "reference" && <AmrReferenceTable />}
        </>
      )}

      {world === "agent-ecosystem" && <ScenarioLab world="agent-ecosystem" />}

      {world === "other" && <ScenarioLab world={["environment", "cybersecurity", "synthetic"]} />}
    </>
  );
}
