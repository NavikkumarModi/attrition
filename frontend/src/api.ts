export const API_BASE = "http://localhost:8000";

export interface SeatSummary {
  id: string;
  state: string;
  name: string;
  note: string;
  constituency_type: "GEN" | "SC" | "ST";
  sub_region: string | null;
  electors: number;
  turnout_pct: number;
  real_margin_pct: number;
  tipping_point_pct: number | null;
  winner_party: string;
  winner_share_pct: number;
  runner_up_party: string;
  runner_up_share_pct: number;
  winner_sex: string | null;
  winner_education: string | null;
  winner_profession: string | null;
  winner_incumbent: boolean | null;
  winner_no_terms: number | null;
  winner_turncoat: boolean | null;
}

export interface NationalScan {
  n_seats: number;
  n_with_tipping_point: number;
  distribution_pct: { min: number; p10: number; median: number; p90: number; max: number };
  most_contestable: SeatSummary[];
  safest_holds: SeatSummary[];
}

export interface Calibration {
  prior_year: number;
  current_year: number;
  n_seats: number;
  n_matched_same_party: number;
  n_flipped_party: number;
  n_missing_prior_year: number;
  n_ambiguous: number;
  real_swing_electorate_fraction_pct: { min: number; p10: number; median: number; p90: number; max: number };
  tipping_point_required_pct: { median: number; p90: number };
}

export interface SimulateResult {
  seat: SeatSummary;
  level1_switched_fraction_by_round: number[];
  level1_final_switched_fraction: number;
  outcome: {
    winner: string;
    winner_share_pct: number;
    runner_up: string;
    runner_up_share_pct: number;
    margin_pct: number;
    flipped: boolean;
  };
  level2_decisions?: Record<string, boolean[]>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

export const fetchConstituencies = () => getJSON<SeatSummary[]>("/api/constituencies");

export interface SeatFeature {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: SeatSummary & { _boundary_state: string; _boundary_pc_name: string };
}
export interface SeatFeatureCollection {
  type: "FeatureCollection";
  features: SeatFeature[];
}
export const fetchConstituenciesGeoJSON = () => getJSON<SeatFeatureCollection>("/api/constituencies.geojson");

export interface AllocationSide {
  n_targeted: number;
  n_flipped: number;
  budget_spent: number;
  budget_remaining: number;
  flipped: SeatSummary[];
}
export interface StrategistResult {
  budget_pct: number;
  scope_max_tipping_pct: number;
  greedy: AllocationSide;
  even: AllocationSide;
}
export const fetchStrategist = (budgetPct: number, scopeMaxTippingPct = 20) =>
  getJSON<StrategistResult>(`/api/strategist?budget_pct=${budgetPct}&scope_max_tipping_pct=${scopeMaxTippingPct}`);

export interface ForecastResult {
  n_seats: number;
  swing_model_accuracy: number;
  naive_baseline_accuracy: number;
  n_real_flips: number;
  swing_model_flip_recall: number;
}
export const fetchForecast = () => getJSON<ForecastResult>("/api/forecast");
export const fetchNationalScan = () => getJSON<NationalScan>("/api/national-scan");
export const fetchCalibration = () => getJSON<Calibration>("/api/calibration");

export interface PolicyStats {
  value: number;
  value_se: number;
  regret: number;
  arms_left: number;
}
export interface CampaignPolicyResult {
  contact_intensity: number[];
  learned_preferences: number[];
  favourite_arm: number;
  single_agent: { greedy: PolicyStats; eci: PolicyStats; trained: PolicyStats };
  decentralised: {
    n_agents: number;
    planner_optimum: number;
    decentralised_greedy: number;
    decentralised_eci: number;
    trained_system_value: number;
    trained_system_regret: number;
  };
}
export const fetchCampaignPolicy = () =>
  getJSON<CampaignPolicyResult>("/api/campaign-policy");

export async function simulate(
  state: string,
  name: string,
  intensity_sequence: number[],
  n_voters: number,
): Promise<SimulateResult> {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state, name, intensity_sequence, n_voters, include_level2: true }),
  });
  if (!res.ok) throw new Error(`simulate: ${res.status} ${await res.text()}`);
  return res.json();
}
