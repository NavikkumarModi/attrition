import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, ZoomControl, useMap } from "react-leaflet";
import type { Layer, GeoJSON as LeafletGeoJSON, LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";
import type { FeatureCollection } from "geojson";
import { fetchConstituenciesGeoJSON, type SeatFeature, type SeatSummary } from "./api";
import Disclosure from "./Disclosure";

const BUCKET_COLORS = { none: "#c7cbd1", contestable: "#dc2626", competitive: "#d97706",
                        close: "#eab308", safe: "#2563eb" };

function colorFor(seat: SeatSummary) {
  if (seat.tipping_point_pct === null) return BUCKET_COLORS.none;
  if (seat.tipping_point_pct < 2) return BUCKET_COLORS.contestable;
  if (seat.tipping_point_pct < 8) return BUCKET_COLORS.competitive;
  if (seat.tipping_point_pct < 15) return BUCKET_COLORS.close;
  return BUCKET_COLORS.safe;
}

// India's real extent (approximate bounding box), used to frame the map
// on load before the GeoJSON layer's own bounds are available, and as a
// "reset view" target.
const INDIA_BOUNDS: LatLngBoundsExpression = [[6.5, 68.0], [35.7, 97.5]];

function FitToData({ fc }: { fc: FeatureCollection | null }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (fc && !fitted.current) {
      fitted.current = true;
      // Deferred: react-leaflet's GeoJSON layer computes bounds from the
      // data itself, but this component doesn't hold that layer instance
      // -- fitting to India's real extent is equivalent for this dataset
      // (543 seats spanning the whole country) and avoids a fragile
      // cross-component ref handoff for one-time framing.
      map.fitBounds(INDIA_BOUNDS, { padding: [16, 16] });
    }
  }, [fc, map]);
  return null;
}

export default function IndiaMap({ onSelect }: { onSelect: (state: string, name: string) => void }) {
  const [fc, setFc] = useState<FeatureCollection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<SeatSummary | null>(null);
  const [query, setQuery] = useState("");
  const geoJsonRef = useRef<LeafletGeoJSON | null>(null);

  useEffect(() => {
    fetchConstituenciesGeoJSON()
      .then((data) => setFc(data as unknown as FeatureCollection))
      .catch((e) => setError(String(e)));
  }, []);

  const features = useMemo(
    () => (fc ? (fc.features as unknown as SeatFeature[]) : []),
    [fc],
  );

  const matches = useMemo(() => {
    if (query.trim().length < 2) return [];
    const q = query.toLowerCase();
    return features
      .filter((f) => f.properties.name.toLowerCase().includes(q)
        || f.properties.state.toLowerCase().includes(q))
      .slice(0, 8);
  }, [features, query]);

  if (error) return <div className="error-box">{error} — is election_api.py running on :8000?</div>;

  return (
    <div className="map-hero">
      {!fc && <div className="map-loading">Loading map…</div>}
      <MapContainer
        center={[22.5, 80]}
        zoom={4.5}
        style={{ width: "100%", height: "100%" }}
        zoomControl={false}
        attributionControl={true}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          subdomains="abc"
          maxZoom={19}
        />
        <FitToData fc={fc} />
        <ZoomControl position="topright" />
        {fc && (
          <GeoJSON
            ref={geoJsonRef}
            data={fc as any}
            style={(feature: any) => {
              const seat = feature.properties as SeatSummary;
              const isHover = hover?.id === seat.id;
              return {
                fillColor: colorFor(seat),
                fillOpacity: isHover ? 1 : 0.85,
                color: isHover ? "#1a1d24" : "#ffffff",
                weight: isHover ? 1.6 : 0.6,
              };
            }}
            onEachFeature={(feature: any, layer: Layer) => {
              const seat = feature.properties as SeatSummary;
              layer.on({
                mouseover: () => setHover(seat),
                mouseout: () => setHover((h) => (h?.id === seat.id ? null : h)),
                click: () => onSelect(seat.state, seat.name),
              });
            }}
          />
        )}
      </MapContainer>

      <div className="map-overlay map-overlay-top">
        <div className="map-search">
          <input
            type="text"
            placeholder="Search a constituency or state…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {matches.length > 0 && (
            <div className="map-search-results">
              {matches.map((f) => (
                <button
                  key={f.properties.id}
                  onClick={() => { onSelect(f.properties.state, f.properties.name); setQuery(""); }}
                >
                  {f.properties.name} <span className="muted">— {f.properties.state}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="map-overlay map-overlay-legend">
        <span className="legend-dot" style={{ background: BUCKET_COLORS.contestable }} /> &lt; 2%{"  "}
        <span className="legend-dot" style={{ background: BUCKET_COLORS.competitive, marginLeft: 8 }} /> &lt; 8%{"  "}
        <span className="legend-dot" style={{ background: BUCKET_COLORS.close, marginLeft: 8 }} /> &lt; 15%{"  "}
        <span className="legend-dot" style={{ background: BUCKET_COLORS.safe, marginLeft: 8 }} /> &ge; 15%
      </div>

      {hover && (
        <div className="map-overlay map-overlay-hover">
          <strong>{hover.name}</strong> — {hover.state}
          <div className="muted">{hover.winner_party} won by {hover.real_margin_pct.toFixed(2)}pp</div>
          <div className="muted">tipping point: {hover.tipping_point_pct?.toFixed(1) ?? "—"}%</div>
        </div>
      )}

      <div className="map-caption">
        <span>All 543 real constituencies, by tipping point — fraction of the electorate that would
          need to switch to flip the seat. Click a seat to open it.</span>
        <Disclosure summary="Map source & methodology">
          Boundaries: DataMeet (CC BY 4.0). Basemap: standard OpenStreetMap tiles.
          Tipping point is the fraction of the total electorate that would need to switch to flip the
          real 2019 result — a closed-form search over the real reallocation rule, not a simulation.
        </Disclosure>
      </div>
    </div>
  );
}
