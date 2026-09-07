export interface DetectionResult {
  verdict: "REAL" | "FAKE";
  risk_score: number;
  real_probability: number;
  fake_probability: number;
  model_version: string;
  duration: number;
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
