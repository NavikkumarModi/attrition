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

export interface SimulateSeatSummary {
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
  winner_incumbent: boolean | null;
}
export interface FlipProbability {
  n_seeds: number;
  flip_rate: number;
  final_switched_fraction_mean: number;
  final_switched_fraction_min: number;
  final_switched_fraction_max: number;
}
export interface NetworkEffect {
  final_switched_fraction: number;
  direct_only_final_switched_fraction: number;
  peer_conversion_prob: number;
  graph_density: number;
  graph_method: "random" | "random_by_degree";
  graph_param: number;
}
export interface DistrictGranularityDistrict {
  state: string;
  district: string | null;
  n_voters: number;
  final_switched_fraction: number;
  direct_only_final_switched_fraction?: number;
}
export interface DistrictGranularity {
  per_district: DistrictGranularityDistrict[];
  combined_final_switched_fraction: number;
  network_within_district: boolean;
}
export interface SimulateResult {
  seat: SimulateSeatSummary;
  level1_switched_fraction_by_round: number[] | null;
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
  network?: NetworkEffect;
  flip_probability?: FlipProbability;
  district_granularity?: DistrictGranularity;
}
export interface SimulateOptions {
  seed?: number;
  n_seeds?: number;
  use_network?: boolean;
  peer_conversion_prob?: number;
  graph_density?: number;
  incumbent_override?: boolean | null;
  margin_override_pct?: number | null;
  use_district_granularity?: boolean;
  mean_degree?: number | null;
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

export interface SocietyMetrics {
  efficiency: number | null;
  electors_in_scope: number;
  electors_in_scope_pct_national: number;
  n_states_touched: number;
  states_touched: string[];
  cost_mean_pct: number | null;
  cost_dispersion_pct: number;
}
export interface AllocationSide {
  n_targeted: number;
  n_flipped: number;
  budget_spent: number;
  budget_remaining: number;
  flipped: SeatSummary[];
  society: SocietyMetrics;
}
export interface StrategistFilters {
  constituency_type?: string;
  sub_region?: string;
  incumbency?: "any" | "incumbent" | "challenger";
}
export interface StrategistResult {
  budget_pct: number;
  scope_max_tipping_pct: number;
  n_seats_in_scope: number;
  filters: StrategistFilters;
  greedy: AllocationSide;
  even: AllocationSide;
}
export const fetchStrategist = (
  budgetPct: number,
  scopeMaxTippingPct = 20,
  filters: StrategistFilters = {},
) => {
  const params = new URLSearchParams({
    budget_pct: String(budgetPct),
    scope_max_tipping_pct: String(scopeMaxTippingPct),
  });
  if (filters.constituency_type) params.set("constituency_type", filters.constituency_type);
  if (filters.sub_region) params.set("sub_region", filters.sub_region);
  if (filters.incumbency && filters.incumbency !== "any") params.set("incumbency", filters.incumbency);
  return getJSON<StrategistResult>(`/api/strategist?${params.toString()}`);
};

export interface StrategistFacets {
  constituency_type: string[];
  sub_region: string[];
}
export const fetchStrategistFilters = () => getJSON<StrategistFacets>("/api/strategist/filters");

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

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

export interface ScenarioSummary {
  name: string;
  world: string;
  description: string;
  phenomenon: string;
  expected_behaviour: string;
  agents: number;
}
export interface WorldsResult {
  [world: string]: { description: string; scenarios: ScenarioSummary[] };
}
export const fetchWorlds = () => getJSON<WorldsResult>("/api/worlds");

export interface ScenarioDetail extends ScenarioSummary {
  v: number[];
  p: number[];
  e: number[];
  delta: number;
  horizon: number;
}
export const fetchScenario = (name: string) =>
  getJSON<ScenarioDetail>(`/api/scenarios/${encodeURIComponent(name)}`);

export const fetchScenarioCompare = (
  name: string,
  policies: string[],
  seeds = 20,
) =>
  postJSON<Record<string, PolicyStats>>(
    `/api/scenarios/${encodeURIComponent(name)}/compare`,
    { policies, seeds },
  );

export interface PersonaSummary {
  name: string;
  role: string;
  description: string;
  risk_tolerance: number;
}
export const fetchScenarioPersonas = (name: string) =>
  getJSON<PersonaSummary[]>(`/api/scenarios/${encodeURIComponent(name)}/personas`);

export interface LLMPopulationResult {
  system_value: number;
  system_regret: number;
  per_persona: Record<string, { genuine: number; total: number; errors: Record<string, number> }>;
}
export const runScenarioLLMPopulation = (
  name: string,
  req: { provider: string; api_key: string; model?: string; persona_names: string[]; horizon?: number },
) =>
  postJSON<LLMPopulationResult>(
    `/api/scenarios/${encodeURIComponent(name)}/llm-population`,
    req,
  );

export interface AmrReferenceTable {
  in_repo: {
    planner: number;
    greedy: number;
    eci: number;
    "lambda_weighted(lam=1)": number;
  };
  external_siloed: Record<string, number>;
  external_shared: Record<string, number>;
}
export const fetchAmrReferenceTable = () =>
  getJSON<AmrReferenceTable>("/api/amr/reference-table");

export interface AssemblyRollup {
  n_constituent_acs_matched: number;
  party_diversity: number;
  most_common_party: string;
  most_common_party_fraction: number;
  mean_turnout_pct: number | null;
  mean_margin_pct: number | null;
  woman_winner_fraction: number;
  incumbent_winner_fraction: number;
  assembly_years_used: number[];
}
export interface AssemblyRollupResponse {
  seat: { state: string; name: string };
  rollup: AssemblyRollup | null;
}
export const fetchAssemblyRollup = (state: string, name: string) =>
  getJSON<AssemblyRollupResponse>(
    `/api/constituencies/${encodeURIComponent(state)}/${encodeURIComponent(name)}/assembly-rollup`,
  );

export interface AssemblyRollupCoverage {
  n_ac_rows_total: number;
  n_ac_rows_matched_to_crosswalk: number;
  n_real_seats_total: number;
  n_real_seats_with_rollup: number;
  year_by_state: Record<string, { year: number; gap_from_2019: number }>;
}
export const fetchAssemblyRollupCoverage = () =>
  getJSON<AssemblyRollupCoverage>("/api/assembly-rollup/coverage");

export interface CensusDistrict {
  state: string;
  district: string;
  n_villages_inhabited: number;
  n_villages_uninhabited: number;
  n_towns: number;
  n_households: number;
  population_total: number;
  population_male: number;
  population_female: number;
  area_sq_km: number;
  population_density_per_sq_km: number;
}
export interface CensusRollup {
  n_constituent_districts: number;
  constituent_districts: CensusDistrict[];
  mean_population_density_per_sq_km: number;
  mean_households_per_district: number;
}
export interface CensusRollupResponse {
  seat: { state: string; name: string };
  rollup: CensusRollup | null;
}
export const fetchCensusRollup = (state: string, name: string) =>
  getJSON<CensusRollupResponse>(
    `/api/constituencies/${encodeURIComponent(state)}/${encodeURIComponent(name)}/census-rollup`,
  );

export interface CensusRollupCoverage {
  n_districts_total: number;
  n_real_seats_total: number;
  n_real_seats_with_rollup: number;
}
export const fetchCensusRollupCoverage = () =>
  getJSON<CensusRollupCoverage>("/api/census-rollup/coverage");

export async function simulate(
  state: string,
  name: string,
  intensity_sequence: number[],
  n_voters: number,
  options: SimulateOptions = {},
): Promise<SimulateResult> {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state, name, intensity_sequence, n_voters, include_level2: true, ...options,
    }),
  });
  if (!res.ok) throw new Error(`simulate: ${res.status} ${await res.text()}`);
  return res.json();
}
