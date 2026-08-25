import { useEffect, useMemo, useState } from "react";
import { geoMercator, geoPath } from "d3-geo";
import type { FeatureCollection, Geometry } from "geojson";
import { fetchConstituenciesGeoJSON, type SeatFeature, type SeatSummary } from "./api";

const WIDTH = 720;
const HEIGHT = 780;

function colorFor(seat: SeatSummary) {
  if (seat.tipping_point_pct === null) return "#333";
  if (seat.tipping_point_pct < 2) return "#e0615e";
  if (seat.tipping_point_pct < 8) return "#f2a65a";
  if (seat.tipping_point_pct < 15) return "#e5d36a";
  return "#5ec8e0";
}

export default function IndiaMap({ onSelect }: { onSelect: (state: string, name: string) => void }) {
  const [fc, setFc] = useState<FeatureCollection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<SeatSummary | null>(null);
  const [mouse, setMouse] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  useEffect(() => {
    fetchConstituenciesGeoJSON()
      .then((data) => setFc(data as unknown as FeatureCollection))
      .catch((e) => setError(String(e)));
  }, []);

  const { path, features } = useMemo(() => {
    if (!fc) return { path: null, features: [] as SeatFeature[] };
    const projection = geoMercator().fitSize([WIDTH, HEIGHT], fc as any);
    const generator = geoPath(projection);
    return { path: generator, features: fc.features as unknown as SeatFeature[] };
  }, [fc]);

  if (error) return <div className="error-box">{error} — is election_api.py running on :8000?</div>;
  if (!fc || !path) return <div className="card muted">Loading map…</div>;

  return (
    <div className="card" style={{ position: "relative" }}>
      <h2>All 543 real constituencies, by tipping point</h2>
      <div
        onMouseMove={(e) => setMouse({ x: e.clientX, y: e.clientY })}
        style={{ position: "relative" }}
      >
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ width: "100%", height: "auto", maxHeight: 640 }}>
          {features.map((f) => {
            const d = path(f.geometry as unknown as Geometry);
            if (!d) return null;
            const seat = f.properties;
            const isHover = hover?.id === seat.id;
            return (
              <path
                key={seat.id}
                d={d}
                fill={colorFor(seat)}
                stroke={isHover ? "#e7ebf3" : "#0b0e14"}
                strokeWidth={isHover ? 1.2 : 0.3}
                opacity={isHover ? 1 : 0.92}
                onMouseEnter={() => setHover(seat)}
                onMouseLeave={() => setHover((h) => (h?.id === seat.id ? null : h))}
                onClick={() => onSelect(seat.state, seat.name)}
                style={{ cursor: "pointer" }}
              />
            );
          })}
        </svg>
        {hover && (
          <div
            style={{
              position: "fixed", left: mouse.x + 14, top: mouse.y + 14,
              background: "#171c28", border: "1px solid #232938", borderRadius: 8,
              padding: "8px 12px", fontSize: 12, pointerEvents: "none", zIndex: 50,
              maxWidth: 240,
            }}
          >
            <strong>{hover.name}</strong> — {hover.state}
            <div className="muted">
              {hover.winner_party} won by {hover.real_margin_pct.toFixed(2)}pp
            </div>
            <div className="muted">
              tipping point: {hover.tipping_point_pct?.toFixed(1) ?? "—"}%
            </div>
          </div>
        )}
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
        <span className="legend-dot" style={{ background: "#e0615e" }} /> &lt; 2%{"  "}
        <span className="legend-dot" style={{ background: "#f2a65a", marginLeft: 10 }} /> &lt; 8%{"  "}
        <span className="legend-dot" style={{ background: "#e5d36a", marginLeft: 10 }} /> &lt; 15%{"  "}
        <span className="legend-dot" style={{ background: "#5ec8e0", marginLeft: 10 }} /> &ge; 15%
        {"   ·   "}fraction of the electorate that would need to switch to flip the seat · click to open ·
        boundaries: DataMeet (CC BY 4.0)
      </p>
    </div>
  );
}
