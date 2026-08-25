import { useState } from "react";
import Overview from "./Overview";
import ConstituencyLab from "./ConstituencyLab";
import Calibration from "./Calibration";
import Strategist from "./Strategist";
import Forecast from "./Forecast";

type Tab = "overview" | "lab" | "strategist" | "calibration" | "forecast";

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [selected, setSelected] = useState<{ state: string; name: string } | null>(null);

  function openSeat(state: string, name: string) {
    setSelected({ state, name });
    setTab("lab");
  }

  return (
    <>
      <header className="app-header">
        <div>
          <h1>ATTRITION — Election Lab</h1>
          <div className="subtitle">543 real 2019 Lok Sabha constituencies · attrition.election</div>
        </div>
        <div className="subtitle">a persuasion-mechanism simulator, plus one real backtested forecast</div>
      </header>

      <nav className="tabs">
        <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")}>National overview</button>
        <button className={tab === "lab" ? "active" : ""} onClick={() => setTab("lab")}>Constituency lab</button>
        <button className={tab === "strategist" ? "active" : ""} onClick={() => setTab("strategist")}>Strategist</button>
        <button className={tab === "forecast" ? "active" : ""} onClick={() => setTab("forecast")}>Forecast</button>
        <button className={tab === "calibration" ? "active" : ""} onClick={() => setTab("calibration")}>Calibration</button>
      </nav>

      {tab === "overview" && <Overview onSelect={openSeat} />}
      {tab === "lab" && <ConstituencyLab initialState={selected?.state ?? null} initialName={selected?.name ?? null} />}
      {tab === "strategist" && <Strategist onSelect={openSeat} />}
      {tab === "forecast" && <Forecast />}
      {tab === "calibration" && <Calibration />}
    </>
  );
}
