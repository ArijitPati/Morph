export interface WebSocketMessage {
  type: "audio_chunk" | "detection_result" | "aggregation_update" | "status_update" | "error";
  payload: unknown;
}

export interface AudioChunkMessage {
  type: "audio_chunk";
  payload: {
    data: string; // base64-encoded audio chunk
    sample_rate: number;
    timestamp: number;
  };
}

export interface DetectionResultMessage {
  type: "detection_result";
  payload: {
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
    fake_probability_legacy?: number;
  };
}

export interface AggregationUpdateMessage {
  type: "aggregation_update";
  payload: {
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
  };
}

export interface StatusUpdateMessage {
  type: "status_update";
  payload: {
    status: "connected" | "analyzing" | "idle" | "error";
    message?: string;
  };
}

export interface ErrorMessage {
  type: "error";
  payload: {
    code: string;
    message: string;
  };
}

export type WSMessage =
  | AudioChunkMessage
  | DetectionResultMessage
  | AggregationUpdateMessage
  | StatusUpdateMessage
  | ErrorMessage;

export interface WebSocketConfig {
  url: string;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}
