import type { WSMessage, WebSocketConfig } from "@/types/websocket";

type MessageHandler = (message: WSMessage) => void;
type StatusHandler = (status: "connected" | "disconnected" | "reconnecting") => void;

export class MorphWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectCount = 0;
  private messageHandler: MessageHandler | null = null;
  private statusHandler: StatusHandler | null = null;
  private config: Required<WebSocketConfig>;

  constructor(config: WebSocketConfig) {
    this.config = {
      url: config.url,
      reconnectAttempts: config.reconnectAttempts ?? 5,
      reconnectInterval: config.reconnectInterval ?? 2000,
    };
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.config.url);

    this.ws.onopen = () => {
      this.reconnectCount = 0;
      this.statusHandler?.("connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WSMessage;
        this.messageHandler?.(message);
      } catch {
        console.error("Failed to parse WebSocket message");
      }
    };

    this.ws.onclose = () => {
      this.statusHandler?.("disconnected");
      this.attemptReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectCount = this.config.reconnectAttempts;
    this.ws?.close();
    this.ws = null;
  }

  send(data: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  onMessage(handler: MessageHandler) {
    this.messageHandler = handler;
  }

  onStatusChange(handler: StatusHandler) {
    this.statusHandler = handler;
  }

  private attemptReconnect() {
    if (this.reconnectCount >= this.config.reconnectAttempts) return;

    this.reconnectCount++;
    this.statusHandler?.("reconnecting");

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, this.config.reconnectInterval * this.reconnectCount);
  }
}
