// Shared chart-styling constants for recharts components, which need
// literal color strings (SVG props, not CSS custom properties) -- kept
// in one place instead of duplicated across every chart-owning file, and
// updated together with index.css's :root tokens when the theme changes.
export const CHART_GRID = "#e2e5eb";
export const CHART_TICK = "#6b7280";
export const CHART_TOOLTIP_BG = "#ffffff";
export const CHART_TOOLTIP_BORDER = "#e2e5eb";
export const MAP_STROKE_DEFAULT = "#ffffff";
export const MAP_STROKE_HOVER = "#1a1d24";

// Tipping-point contestability buckets, shared by IndiaMap and Overview's
// scatter plot so a seat is colored identically in both views.
export const TIPPING_POINT_COLORS = {
  contestable: "#dc2626",  // < 2%
  competitive: "#d97706",  // < 8%
  safe: "#2563eb",         // >= 8%
} as const;

export function tippingPointColor(pct: number | null): string {
  if (pct === null) return "#9aa1ad";
  if (pct < 2) return TIPPING_POINT_COLORS.contestable;
  if (pct < 8) return TIPPING_POINT_COLORS.competitive;
  return TIPPING_POINT_COLORS.safe;
}

// Simple-mode campaign-effort presets -- one plain choice instead of a
// technical multi-round intensity sequence (FINDINGS.md#63/#64). Shared
// between ConstituencyLab's single-seat simulator and Overview's
// map-wide campaign simulator so "Heavy" means the same real
// intensity_sequence in both places.
export const INTENSITY_PRESETS: Record<string, number[]> = {
  Light: [0.1, 0.1, 0.15, 0.15, 0.15],
  Moderate: [0.15, 0.2, 0.25, 0.3, 0.35],
  Heavy: [0.3, 0.4, 0.5, 0.6, 0.7],
};
