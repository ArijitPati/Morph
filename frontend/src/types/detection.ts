export interface DetectionResult {
  verdict: "REAL" | "FAKE";
  risk_score: number;
  real_probability: number;
  fake_probability: number;
  model_version: string;
  duration: number;
  feature_count: number;
}

export interface WindowResult {
  window_index: number;
  window: number;
  start_sec: number;
  end_sec: number;
  duration: number;
  chunk_duration: number;
  is_partial: boolean;
  label: number;
  label_str: "REAL" | "FAKE";
  confidence: number;
  real_probability: number;
  fake_probability: number;
  real_prob: number;
  fake_prob: number;
  model_version: string;
}

export interface AggregationResult {
  total_windows: number;
  n_fake: number;
  n_real: number;
  pct_fake: number;
  pct_real: number;
  mean_fake_prob: number;
  median_fake_prob: number;
  max_fake_prob: number;
  min_fake_prob: number;
  std_fake_prob: number;
  final_fake_prob: number;
  final_real_prob: number;
  aggregated_label: number;
  aggregated_label_str: "REAL" | "FAKE";
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  risk_score: number;
  confidence: number;
}

export interface WindowedDetectionResult {
  total_duration: number;
  duration: number;
  window_sec: number;
  hop_sec: number;
  model_version: string;
  windows: WindowResult[];
  aggregation: AggregationResult;
  full_result: DetectionResult | null;
  verdict: "REAL" | "FAKE";
  risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  real_probability: number;
  fake_probability: number;
  feature_count: number;
}

export interface DetectionRequest {
  file: File;
}

export interface DetectionHistoryEntry {
  id: string;
  timestamp: string;
  filename: string;
  result: DetectionResult;
}

export interface SystemStatus {
  operational: boolean;
  model_version: string;
  uptime: number;
  total_analyzed: number;
  threats_detected: number;
}
