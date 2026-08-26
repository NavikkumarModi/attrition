import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, ZoomControl, useMap } from "react-leaflet";
import type { Layer, GeoJSON as LeafletGeoJSON, LatLngBoundsExpression, PathOptions } from "leaflet";
import "leaflet/dist/leaflet.css";
import type { FeatureCollection } from "geojson";
import { fetchConstituenciesGeoJSON, type SeatFeature, type SeatSummary, type CampaignMapResult } from "./api";
import Disclosure from "./Disclosure";

const BUCKET_COLORS = { none: "#c7cbd1", contestable: "#dc2626", competitive: "#d97706",
                        close: "#eab308", safe: "#2563eb" };
const SIM_COLORS = { flip: "#dc2626", hold: "#2563eb", outOfScope: "#d8dbe0" };

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

export default function IndiaMap({
  onSelect, simulation,
}: {
  onSelect: (state: string, name: string) => void;
  // When set (FINDINGS.md#64), the map recolours by SIMULATED outcome
  // (flip/hold at the given campaign effort/scope) instead of the
  // static real-2019 tipping-point buckets -- the same
  // simulate_level1_contact mechanism the single-seat lab uses, just
  // run once per seat in scope.
  simulation?: CampaignMapResult | null;
}) {
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

  const simIndex = useMemo(
    () => (simulation ? new Map(simulation.seats.map((s) => [s.id, s])) : null),
    [simulation],
  );

  const styleFor = (seat: SeatSummary, isHover: boolean): PathOptions => {
    if (simIndex) {
      const sim = simIndex.get(seat.id);
      const fill = !sim ? SIM_COLORS.outOfScope : sim.flipped ? SIM_COLORS.flip : SIM_COLORS.hold;
      return {
        fillColor: fill,
        fillOpacity: isHover ? 1 : (sim ? 0.85 : 0.35),
        color: isHover ? "#1a1d24" : "#ffffff",
        weight: isHover ? 1.6 : 0.6,
      };
    }
    return {
      fillColor: colorFor(seat),
      fillOpacity: isHover ? 1 : 0.85,
      color: isHover ? "#1a1d24" : "#ffffff",
      weight: isHover ? 1.6 : 0.6,
    };
  };

  // react-leaflet's <GeoJSON> only applies its `style` prop once, at
  // mount -- it does not re-run the style function when `simulation` or
  // `hover` change on a later render. Recolouring imperatively via the
  // underlying Leaflet layer's own setStyle (through the ref) keeps the
  // user's current pan/zoom instead of remounting the whole layer every
  // time an effort/scope control changes.
  useEffect(() => {
    geoJsonRef.current?.setStyle((feature: any) => {
      const seat = feature.properties as SeatSummary;
      return styleFor(seat, hover?.id === seat.id);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simIndex, hover]);

  if (error) return <div className="error-box">{error} — is election_api.py running on :8000?</div>;

  const simSeat = simIndex && hover ? simIndex.get(hover.id) : null;

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
            style={(feature: any) => styleFor(feature.properties as SeatSummary, false)}
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
        {simIndex ? (
          <>
            <span className="legend-dot" style={{ background: SIM_COLORS.flip }} /> flips{"  "}
            <span className="legend-dot" style={{ background: SIM_COLORS.hold, marginLeft: 8 }} /> holds{"  "}
            <span className="legend-dot" style={{ background: SIM_COLORS.outOfScope, marginLeft: 8 }} /> out of scope
          </>
        ) : (
          <>
            <span className="legend-dot" style={{ background: BUCKET_COLORS.contestable }} /> &lt; 2%{"  "}
            <span className="legend-dot" style={{ background: BUCKET_COLORS.competitive, marginLeft: 8 }} /> &lt; 8%{"  "}
            <span className="legend-dot" style={{ background: BUCKET_COLORS.close, marginLeft: 8 }} /> &lt; 15%{"  "}
            <span className="legend-dot" style={{ background: BUCKET_COLORS.safe, marginLeft: 8 }} /> &ge; 15%
          </>
        )}
      </div>

      {hover && (
        <div className="map-overlay map-overlay-hover">
          <strong>{hover.name}</strong> — {hover.state}
          {simIndex ? (
            simSeat ? (
              <>
                <div className="muted">
                  simulated: <strong>{simSeat.flipped ? "FLIPS" : "HOLDS"}</strong> — {simSeat.winner} wins,
                  margin {simSeat.margin_pct.toFixed(1)}pp
                </div>
                <div className="muted">switched fraction: {(simSeat.final_switched_fraction * 100).toFixed(2)}%</div>
              </>
            ) : (
              <div className="muted">not in this simulation's scope</div>
            )
          ) : (
            <>
              <div className="muted">{hover.winner_party} won by {hover.real_margin_pct.toFixed(2)}pp</div>
              <div className="muted">tipping point: {hover.tipping_point_pct?.toFixed(1) ?? "—"}%</div>
            </>
          )}
        </div>
      )}

      <div className="map-caption">
        {simIndex ? (
          <span>
            Simulated campaign result — {simulation!.n_flipped} of {simulation!.n_seats} seats in scope flip.
            Click a seat to open it in Constituency Lab.
          </span>
        ) : (
          <span>All 543 real constituencies, by tipping point — fraction of the electorate that would
            need to switch to flip the seat. Click a seat to open it.</span>
        )}
        <Disclosure summary="Map source & methodology">
          Boundaries: DataMeet (CC BY 4.0). Basemap: standard OpenStreetMap tiles.
          {simIndex ? (
            <> Colours here are a real, simulated outcome (`simulate_level1_contact`/`recompute_outcome`, the
              SAME mechanism the single-seat Constituency Lab uses) run once per seat in scope at one uniform
              campaign-effort level — not a claim about what will actually happen. See FINDINGS.md #64.</>
          ) : (
            <> Tipping point is the fraction of the total electorate that would need to switch to flip the
              real 2019 result — a closed-form search over the real reallocation rule, not a simulation.</>
          )}
        </Disclosure>
      </div>
    </div>
  );
}
