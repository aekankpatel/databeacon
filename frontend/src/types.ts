export interface Dataset {
  id: string;
  name: string;
  table: string;
  filename: string;
  created_at: string;
  row_count: number;
  column_count: number;
  columns: { name: string; type: string }[];
}

export interface ColumnProfile {
  name: string;
  type: string;
  role: string;
  missing: number;
  missing_pct: number;
  distinct: number;
  stats?: Record<string, number | string | null>;
  top_values?: { value: string; count: number }[];
  outliers_iqr?: number;
}

export interface Profile {
  table: string;
  row_count: number;
  column_count: number;
  columns: ColumnProfile[];
  strong_correlations: { a: string; b: string; pearson_r: number }[];
  warnings: string[];
  sample_rows: Record<string, unknown>[];
  summary: string;
}

export interface TimelineEvent {
  id: string;
  type: "node" | "llm" | "tool" | "decision" | "guardrail" | "discovery" | "error";
  name: string;
  status: "running" | "ok" | "error" | "warning";
  started_at: string;
  duration_ms: number;
  input?: unknown;
  output?: unknown;
  error?: string;
  sql?: string;
  server?: string;
  tokens?: { input_tokens: number; output_tokens: number };
  evidence_id?: string;
  origin?: "agent" | "system" | "fallback";
}

export interface Verification {
  status: "verified" | "partially_verified" | "unverified" | "qualitative";
  checked_numbers: number;
  supported: number;
  supported_values: string[];
  unsupported_values: string[];
  missing_evidence_ids: string[];
}

export interface Finding {
  title: string;
  statement: string;
  evidence_ids: string[];
  confidence: string;
  verification?: Verification;
}

export interface Chart {
  chart_id: string;
  title: string;
  chart_type: string;
  figure: { data: unknown[]; layout: Record<string, unknown> };
  evidence_id?: string;
  sql?: string;
  origin?: "agent" | "system" | "fallback";
}

export interface Evidence {
  id: string;
  origin?: "agent" | "system" | "fallback";
  tool: string;
  server: string;
  args: Record<string, unknown>;
  ok: boolean;
  error: string | null;
  elapsed_ms: number;
  iteration: number;
  purpose: string;
  sql: string | null;
  summary: string;
  preview: { columns: string[]; rows: unknown[][] } | null;
  data?: unknown;
}

export interface Plan {
  objective: string;
  key_metric: string;
  time_scope: string;
  hypotheses: { id: string; statement: string; approach: string }[];
  steps: string[];
}

export interface Reflection {
  iteration: number;
  sufficient: boolean;
  confidence: number;
  hypothesis_status: Record<string, string>;
  gaps: string[];
  next_actions: string[];
}

export interface RunMetrics {
  latency_s: number;
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  tool_calls: number;
  tool_errors: number;
  sql_queries: number;
}

export interface Answer {
  primary_entity: string | null;
  dimension: string | null;
  metric_column: string | null;
  direction: string | null;
  current_period: string[] | null;
  previous_period: string[] | null;
  key_value: number | null;
  evidence_ids: string[];
}

export interface PeriodInfo {
  start: string;
  end: string;
  label: string;
  complete: boolean;
  coverage_ratio: number;
}

export interface TaskSpec {
  question_type: string;
  metric_column: string | null;
  aggregation: string | null;
  group_by: string[];
  answer_target: string;
  date_column: string | null;
  current: PeriodInfo | null;
  previous: PeriodInfo | null;
  assumptions: string[];
  issues: string[];
}

export interface Alignment {
  status: "aligned" | "misaligned";
  checks: Record<string, "pass" | "fail" | "n/a" | "unverified">;
  issues: string[];
}

export interface Clarification {
  question: string;
  options: string[];
  phrase: string | null;
}

export interface RunResult {
  spec?: TaskSpec | null;
  alignment?: Alignment | null;
  needs_clarification?: Clarification | null;
  plan?: Plan;
  reflections?: Reflection[];
  synthesis?: {
    title: string;
    summary: string;
    answer?: Answer;
    findings: Finding[];
    limitations: string[];
    recommendations: string[];
    summary_verification?: Verification;
  };
  verification?: {
    findings_total: number;
    findings_verified: number;
    numeric_claims: number;
    numeric_claims_supported: number;
    hallucination_rate: number;
    unsupported: { finding: string; values: string[] }[];
  };
  charts?: Chart[];
  evidence?: Evidence[];
  timeline: TimelineEvent[];
  metrics: RunMetrics;
  report?: { ok: boolean; path: string | null; error: string | null };
  errors?: string[];
  stop_reason?: string;
  iterations?: number;
}

export interface Run {
  id: string;
  dataset_id: string;
  question: string;
  status: string;
  created_at: string;
  finished_at: string | null;
  result: RunResult | null;
}

export interface RunSummary {
  id: string;
  dataset_id: string;
  question: string;
  status: string;
  created_at: string;
}

export interface Catalog {
  tools: { name: string; server: string; description: string; input_schema: Record<string, unknown> }[];
  resources: { server: string; uri: string; name: string; description: string; template?: boolean }[];
  prompts: { server: string; name: string; description: string; arguments: string[] }[];
}

export interface Health {
  status: string;
  version: string;
  llm: { provider: string; model: string; configured: boolean };
  mcp_transport: string;
  public_mode?: boolean;
  access_code_required?: boolean;
  limits?: {
    max_upload_mb: number;
    max_rows: number;
    max_columns: number;
    max_datasets_per_session: number;
    runs_per_hour: number;
    runs_per_day: number;
    data_ttl_hours: number;
  };
}
