import type { SignalingIncoming, SignalingOutgoing } from "@/types/signaling";
import { getApiBaseUrl } from "@/services/websocket";

/** WebSocket URL for the call-signaling relay (mirrors /api/signaling/ws). */
export function getSignalingSocketUrl(): string {
  return `${getApiBaseUrl().replace(/^http/, "ws")}/api/signaling/ws`;
}

type IncomingHandler = (message: SignalingIncoming) => void;
type StatusHandler = (
  status: "connecting" | "connected" | "disconnected",
) => void;
type ErrorHandler = (message: string) => void;

// Outbound frames queued while the socket is still opening are tiny
// (join/offer/answer/ICE JSON) — cap the buffer so a dead backend can't
// grow memory without bound.
const MAX_QUEUED_MESSAGES = 50;

export class SignalingSocket {
  readonly url: string;
  private ws: WebSocket | null = null;
  private queue: string[] = [];
  private everOpened = false;
  private incomingHandler: IncomingHandler | null = null;
  private statusHandler: StatusHandler | null = null;
  private errorHandler: ErrorHandler | null = null;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    if (
      this.ws?.readyState === WebSocket.OPEN ||
      this.ws?.readyState === WebSocket.CONNECTING
    )
      return;
    // getSignalingSocketUrl() is evaluated at click time (not cached), so
    // this always reflects the CURRENT page host — but a baked-in
    // NEXT_PUBLIC_API_URL would override it. Log both so a hotspot network
    // switch that changes the LAN IP is debuggable from the console.
    console.info(
      `[signaling] connecting → ${this.url} ` +
        `(page host: ${window.location.host}, ` +
        `NEXT_PUBLIC_API_URL: ${process.env.NEXT_PUBLIC_API_URL ?? "(unset)"})`,
    );
    this.statusHandler?.("connecting");

    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => {
      this.everOpened = true;
      const flushed = this.queue.splice(0);
      for (const frame of flushed) this.ws?.send(frame);
      console.info(
        `[signaling] open → ${this.url} (flushed ${flushed.length} queued message(s))`,
      );
      this.statusHandler?.("connected");
    };
    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as SignalingIncoming;
        this.incomingHandler?.(message);
      } catch (err) {
        console.warn("[signaling] WS parse failed — skipping frame", err);
      }
    };
    this.ws.onerror = () => {
      // onclose surfaces the user-visible error; keep at warn level to avoid console-error noise
      console.warn(`[signaling] socket error → ${this.url}`);
    };
    this.ws.onclose = () => {
      this.ws = null;
      if (!this.everOpened) {
        // Never established: backend unreachable (wrong host/port, backend
        // down, hotspot client isolation). Surface visibly — and keep the
        // queue so a retry flushes it.
        this.errorHandler?.(
          `Signaling unreachable at ${this.url} — is the backend running on port 8000 and reachable from this host?`,
        );
      }
      this.statusHandler?.("disconnected");
    };
  }

  disconnect() {
    this.queue = [];
    try {
      this.ws?.close();
    } finally {
      this.ws = null;
    }
  }

  send(message: SignalingOutgoing) {
    const frame = JSON.stringify(message);
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(frame);
      return;
    }
    // Socket still opening (or not yet created): queue and flush on onopen
    // instead of silently dropping. This is the join-before-open race.
    if (this.ws == null) this.connect();
    if (this.queue.length >= MAX_QUEUED_MESSAGES) this.queue.shift();
    this.queue.push(frame);
    console.info(
      `[signaling] queued "${message.type}" (socket not open yet, queue=${this.queue.length})`,
    );
  }

  onMessage(handler: IncomingHandler) {
    this.incomingHandler = handler;
  }

  onStatusChange(handler: StatusHandler) {
    this.statusHandler = handler;
  }

  onError(handler: ErrorHandler) {
    this.errorHandler = handler;
  }
}
