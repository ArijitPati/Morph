"use client";

/**
 * Minimal dashboard shared state — single source of truth is backend-processed risk.
 * Call page publishes processed risk (fake_probability → risk%) without recalculating.
 * Dashboard subscribes via BroadcastChannel + localStorage fallback so Live risk /
 * Active Calls update without refresh and without polling.
 *
 * No duplicate EWMA/aggregation — dashboard displays exactly the risk already
 * used by Direct Live Call & Room Call RiskTimeline.
 */

import type { WindowResult, AggregationResult } from "@/types/detection";

export type DashboardRiskLevel = "Low" | "Suspicious" | "High" | "Critical";

export interface DashboardCall {
  id: string; // room code / session id
  risk: number; // 0-100 processed risk (round(fake*100))
  riskLevel: DashboardRiskLevel;
  status: "connected" | "monitoring" | "ended" | "terminated";
  terminated: boolean;
  activeThreshold: number | null; // 70|75|80|85|90
  reason: string | null;
  windows: WindowResult[];
  aggregation: AggregationResult | null;
  lastUpdate: number; // epoch ms
}

export interface DashboardAlert {
  id: string;
  callId: string;
  threshold: number; // 70|75|80|85|90
  risk: number;
  reason: string;
  title: string;
  timestamp: number;
  terminated?: boolean;
}

const CHANNEL = "morph-dashboard";
const STORAGE_KEY = "morph:dashboard:calls";
const ALERT_KEY = "morph:dashboard:alerts";

function getChannel(): BroadcastChannel | null {
  if (typeof window === "undefined") return null;
  try {
    return new BroadcastChannel(CHANNEL);
  } catch {
    return null;
  }
}

function readStorage<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key: string, value: unknown) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

// Publish a call update (upsert)
export function publishCall(call: DashboardCall) {
  const calls = readStorage<Record<string, DashboardCall>>(STORAGE_KEY, {});
  calls[call.id] = call;
  writeStorage(STORAGE_KEY, calls);
  const ch = getChannel();
  ch?.postMessage({ type: "call", call });
  ch?.close();
  // also dispatch storage event via localStorage write already does for other tabs,
  // but for same-tab dashboard we rely on channel; also dispatch custom event
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("morph-dashboard-call", { detail: call }));
  }
}

export function removeCall(id: string) {
  const calls = readStorage<Record<string, DashboardCall>>(STORAGE_KEY, {});
  delete calls[id];
  writeStorage(STORAGE_KEY, calls);
  const ch = getChannel();
  ch?.postMessage({ type: "remove", id });
  ch?.close();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("morph-dashboard-remove", { detail: id }));
  }
}

export function publishAlert(alert: DashboardAlert) {
  const alerts = readStorage<DashboardAlert[]>(ALERT_KEY, []);
  alerts.unshift(alert);
  // keep last 20
  const trimmed = alerts.slice(0, 20);
  writeStorage(ALERT_KEY, trimmed);
  const ch = getChannel();
  ch?.postMessage({ type: "alert", alert });
  ch?.close();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("morph-dashboard-alert", { detail: alert }));
  }
}

export function getCalls(): DashboardCall[] {
  const calls = readStorage<Record<string, DashboardCall>>(STORAGE_KEY, {});
  return Object.values(calls).sort((a, b) => b.lastUpdate - a.lastUpdate);
}

export function getAlerts(): DashboardAlert[] {
  return readStorage<DashboardAlert[]>(ALERT_KEY, []);
}

export function clearDashboardStore() {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(ALERT_KEY);
  }
}

// Hook helper for dashboard page
export function subscribeToDashboard(
  onCall: (calls: DashboardCall[]) => void,
  onAlert: (alerts: DashboardAlert[]) => void
): () => void {
  if (typeof window === "undefined") return () => {};
  const ch = getChannel();
  const handler = (e: MessageEvent) => {
    const data = e.data as { type: string; call?: DashboardCall; id?: string; alert?: DashboardAlert };
    if (data.type === "call" || data.type === "remove") {
      onCall(getCalls());
    }
    if (data.type === "alert") {
      onAlert(getAlerts());
    }
  };
  ch?.addEventListener("message", handler);

  const onStorage = (ev: StorageEvent) => {
    if (ev.key === STORAGE_KEY) onCall(getCalls());
    if (ev.key === ALERT_KEY) onAlert(getAlerts());
  };
  window.addEventListener("storage", onStorage);

  const onCustomCall = () => onCall(getCalls());
  const onCustomAlert = () => onAlert(getAlerts());
  window.addEventListener("morph-dashboard-call", onCustomCall as EventListener);
  window.addEventListener("morph-dashboard-remove", onCustomCall as EventListener);
  window.addEventListener("morph-dashboard-alert", onCustomAlert as EventListener);

  // also poll fallback in case channel missed
  const iv = setInterval(() => {
    onCall(getCalls());
    onAlert(getAlerts());
  }, 5000);

  return () => {
    ch?.removeEventListener("message", handler);
    ch?.close();
    window.removeEventListener("storage", onStorage);
    window.removeEventListener("morph-dashboard-call", onCustomCall as EventListener);
    window.removeEventListener("morph-dashboard-remove", onCustomCall as EventListener);
    window.removeEventListener("morph-dashboard-alert", onCustomAlert as EventListener);
    clearInterval(iv);
  };
}

export function riskLevelFromRisk(risk: number): DashboardRiskLevel {
  if (risk >= 90) return "Critical";
  if (risk >= 70) return "High";
  if (risk >= 50) return "Suspicious";
  return "Low";
}
