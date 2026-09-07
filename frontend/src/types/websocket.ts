export interface WebSocketMessage {
  type: "audio_chunk" | "detection_result" | "status_update" | "error";
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
    label: number;
    label_str: "REAL" | "FAKE";
    confidence: number;
    real_probability: number;
    fake_probability: number;
    chunk_duration: number;
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
  | StatusUpdateMessage
  | ErrorMessage;

export interface WebSocketConfig {
  url: string;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}
