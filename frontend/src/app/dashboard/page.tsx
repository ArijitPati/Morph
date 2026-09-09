"use client";

import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  Shield,
  Cpu,
  Radio,
  Wifi,
  Phone,
  AlertTriangle,
  Activity,
  Bell,
  Clock,
  ExternalLink,
  ShieldAlert,
  CheckCircle,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PageContainer } from "@/components/layout/PageContainer";
import { RiskTimeline } from "@/components/detection/RiskTimeline";
import { getApiBaseUrl } from "@/services/websocket";
import {
  getCalls,
  getAlerts,
  subscribeToDashboard,
  riskLevelFromRisk,
  type DashboardCall,
  type DashboardAlert,
} from "@/lib/dashboardStore";
import { toast } from "sonner";

type HealthState = {
  status: "online" | "offline" | "loading";
  engine: string;
  model: string;
  featureCount: number;
};

const riskColor = (level: string) => {
  switch (level) {
    case "Critical":
      return "text-danger";
    case "High":
      return "text-warning";
    case "Suspicious":
      return "text-warning/80";
    default:
      return "text-primary";
  }
};

const riskBg = (level: string) => {
  switch (level) {
    case "Critical":
      return "bg-danger/10 border-danger/30";
    case "High":
      return "bg-warning/10 border-warning/30";
    case "Suspicious":
      return "bg-warning/5 border-warning/20";
    default:
      return "bg-primary/10 border-primary/20";
  }
};

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthState>({ status: "loading", engine: "—", model: "—", featureCount: 0 });
  const [wsStatus, setWsStatus] = useState<"online" | "offline" | "checking">("checking");
  const [calls, setCalls] = useState<DashboardCall[]>(() => getCalls());
  const [alerts, setAlerts] = useState<DashboardAlert[]>(() => getAlerts());
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Health polling — single source of truth: /api/health
  useEffect(() => {
    let cancelled = false;
    const fetchHealth = async () => {
      try {
        const base = getApiBaseUrl();
        const res = await fetch(`${base}/api/health`, { cache: "no-store" });
        if (!res.ok) throw new Error("health failed");
        const data = await res.json();
        if (cancelled) return;
        setHealth({
          status: "online",
          engine: data.engine ?? "loaded",
          model: data.model_version ?? "—",
          featureCount: data.feature_count ?? 0,
        });
      } catch {
        if (!cancelled) setHealth({ status: "offline", engine: "offline", model: "offline", featureCount: 0 });
      }
    };
    fetchHealth();
    const iv = setInterval(fetchHealth, 8000);
    return () => {
      cancelled = true;
      clearInterval(iv);
    };
  }, []);

  // WebSocket check — lightweight test of /api/detection/ws reachability
  useEffect(() => {
    if (health.status !== "online") {
      setWsStatus("offline");
      return;
    }
    let ws: WebSocket | null = null;
    let done = false;
    try {
      const base = getApiBaseUrl().replace(/^http/, "ws");
      ws = new WebSocket(`${base}/api/detection/ws`);
      const t = setTimeout(() => {
        if (!done) {
          setWsStatus("offline");
          try { ws?.close(); } catch {}
        }
      }, 4000);
      ws.onopen = () => {
        if (done) return;
        done = true;
        clearTimeout(t);
        setWsStatus("online");
        try { ws?.close(); } catch {}
      };
      ws.onerror = () => {
        if (done) return;
        done = true;
        clearTimeout(t);
        setWsStatus("offline");
      };
      ws.onclose = () => {
        if (!done) {
          done = true;
          clearTimeout(t);
          setWsStatus("offline");
        }
      };
    } catch {
      setWsStatus("offline");
    }
    return () => {
      done = true;
      try { ws?.close(); } catch {}
    };
  }, [health.status]);

  // Subscribe to live calls / alerts published by Room Call (processed risk, no recalculation)
  useEffect(() => {
    setCalls(getCalls());
    setAlerts(getAlerts());
    return subscribeToDashboard(
      (c) => setCalls(c),
      (a) => setAlerts(a)
    );
  }, []);

  const selectedCall = selectedId ? calls.find((c) => c.id === selectedId) ?? null : null;
  // Or if alert selected, find its call
  const activeCall = calls[0] ?? null; // most recent for Live Risk
  const overallRisk = activeCall?.risk ?? 0;
  const overallLevel = activeCall ? activeCall.riskLevel : "Low";
  const overallWindows = activeCall?.windows ?? [];
  const overallAgg = activeCall?.aggregation ?? null;

  const handleVerify = useCallback(() => {
    toast.info("End this call and contact the organization using the official number from their website/card/statement. Do not use a number provided by the caller.", { duration: 8000 });
  }, []);
  const handleMfa = useCallback(() => {
    toast.info("Require independent verification through an approved MFA/authentication channel before completing the requested action.", { duration: 8000 });
  }, []);
  const handleReport = useCallback(() => {
    toast.info("Report noted — please escalate to your supervisor / security team with risk score and transcript.", { duration: 8000 });
  }, []);
  const handleBlock = useCallback(() => {
    toast.info("Contact blocked — flagged as high-risk for future calls.", { duration: 8000 });
  }, []);

  const isBackendOffline = health.status === "offline";

  return (
    <>
      <Navbar />
      <PageContainer>
        <div className="space-y-6">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <h1 className="text-2xl font-bold tracking-tight">Security Dashboard</h1>
            <p className="mt-1 text-sm text-muted-foreground">Operational monitoring — backend-processed risk, no duplicate calculation.</p>
          </motion.div>

          {/* 1. SYSTEM STATUS */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.05 }} className="rounded-xl border border-border bg-card p-5">
            <h2 className="text-sm font-semibold mb-4">System Status</h2>
            {isBackendOffline ? (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-3 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-danger" />
                <p className="text-sm font-medium text-danger">Backend Offline</p>
                <span className="text-xs text-danger/70">— FastAPI :8000 unreachable. Check LAN HTTPS and host.</span>
              </div>
            ) : (
              <div className="grid gap-4 sm:grid-cols-4">
                <div className="flex items-center gap-3 rounded-lg bg-background/50 px-4 py-3">
                  <Wifi className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Backend</p>
                    <p className="text-sm font-medium capitalize">{health.status}</p>
                  </div>
                  <div className={`h-2 w-2 rounded-full ${health.status === "online" ? "bg-primary" : "bg-muted-foreground/50"}`} />
                </div>
                <div className="flex items-center gap-3 rounded-lg bg-background/50 px-4 py-3">
                  <Radio className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Detection Engine</p>
                    <p className="text-sm font-medium capitalize">{health.engine}</p>
                  </div>
                  <div className={`h-2 w-2 rounded-full ${health.engine !== "offline" && health.status === "online" ? "bg-primary" : "bg-muted-foreground/50"}`} />
                </div>
                <div className="flex items-center gap-3 rounded-lg bg-background/50 px-4 py-3">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">WebSocket</p>
                    <p className="text-sm font-medium capitalize">{wsStatus}</p>
                  </div>
                  <div className={`h-2 w-2 rounded-full ${wsStatus === "online" ? "bg-primary" : wsStatus === "checking" ? "bg-warning" : "bg-muted-foreground/50"}`} />
                </div>
                <div className="flex items-center gap-3 rounded-lg bg-background/50 px-4 py-3">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Model</p>
                    <p className="text-sm font-medium truncate">{health.model}</p>
                  </div>
                  <div className={`h-2 w-2 rounded-full ${health.status === "online" ? "bg-primary" : "bg-muted-foreground/50"}`} />
                </div>
              </div>
            )}
          </motion.div>

          {/* 2. LIVE RISK */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }} className="rounded-xl border border-border bg-card p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold flex items-center gap-2"><Shield className="h-4 w-4 text-primary" />Live Risk</h2>
              <span className="text-xs text-muted-foreground">Active calls: {calls.length}</span>
            </div>
            {isBackendOffline ? (
              <div className="rounded-lg border border-dashed border-border py-8 text-center">
                <p className="text-sm text-muted-foreground">Backend Offline — live risk unavailable</p>
              </div>
            ) : !activeCall ? (
              <div className="rounded-lg border border-dashed border-border py-8 text-center">
                <p className="text-sm text-muted-foreground">No active calls — risk appears when a Room/WebRTC call is live</p>
                <p className="text-xs text-muted-foreground/60 mt-1">Start a call from <Link href="/call" className="underline">Live Call</Link></p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className={`rounded-lg border px-4 py-3 ${riskBg(overallLevel)}`}>
                    <p className="text-[11px] text-muted-foreground uppercase">Current Risk</p>
                    <p className={`text-2xl font-bold ${riskColor(overallLevel)}`}>{overallRisk}%</p>
                    <p className="text-xs text-muted-foreground">{overallLevel}</p>
                  </div>
                  <div className="rounded-lg bg-background/50 border px-4 py-3">
                    <p className="text-[11px] text-muted-foreground uppercase">Risk Level</p>
                    <p className={`text-sm font-semibold ${riskColor(overallLevel)}`}>{overallLevel}</p>
                    <p className="text-xs text-muted-foreground mt-1">{overallAgg ? `${overallAgg.n_fake}/${overallAgg.total_windows} windows FAKE` : "—"}</p>
                  </div>
                  <div className="rounded-lg bg-background/50 border px-4 py-3">
                    <p className="text-[11px] text-muted-foreground uppercase">Session</p>
                    <p className="text-sm font-mono font-medium truncate">{activeCall.id}</p>
                    <p className="text-xs text-muted-foreground capitalize">{activeCall.status}{activeCall.terminated ? " · terminated" : ""}</p>
                  </div>
                </div>
                {overallWindows.length > 0 && (
                  <div className="rounded-lg border border-border bg-background/30 p-3">
                    <p className="text-[11px] text-muted-foreground uppercase mb-2">Live risk timeline (same data as Room Call)</p>
                    <RiskTimeline windows={overallWindows} />
                  </div>
                )}
              </div>
            )}
          </motion.div>

          {/* 3. ACTIVE CALLS */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.15 }} className="rounded-xl border border-border bg-card p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold">Active Calls</h2>
              <span className="text-xs text-muted-foreground">{calls.length} active</span>
            </div>
            {calls.length === 0 ? (
              <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-12">
                <Phone className="h-8 w-8 text-muted-foreground/30 mb-3" />
                <p className="text-sm text-muted-foreground">No active calls</p>
                <p className="text-xs text-muted-foreground/60 mt-1">Room/WebRTC calls appear here live without refresh</p>
              </div>
            ) : (
              <div className="space-y-2">
                {calls.map((c) => (
                  <div key={c.id} className={`flex items-center justify-between rounded-lg border px-4 py-3 ${c.terminated ? "border-danger/30 bg-danger/10" : "border-border bg-background/50"}`}>
                    <div className="min-w-0">
                      <p className="text-sm font-mono font-medium truncate">{c.id}</p>
                      <p className="text-xs text-muted-foreground">{c.terminated ? "Terminated at 90%" : c.status} · {c.reason ?? "monitoring"}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <p className={`text-sm font-bold ${riskColor(c.riskLevel)}`}>{c.risk}%</p>
                        <p className={`text-[11px] ${riskColor(c.riskLevel)}`}>{c.riskLevel}</p>
                      </div>
                      <button onClick={() => setSelectedId(c.id)} className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium hover:bg-secondary">
                        View <ExternalLink className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>

          {/* 4. RECENT SECURITY ALERTS */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }} className="rounded-xl border border-border bg-card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Bell className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold">Recent Security Alerts</h2>
            </div>
            {alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-12">
                <Shield className="h-8 w-8 text-muted-foreground/30 mb-3" />
                <p className="text-sm text-muted-foreground">No alerts yet</p>
                <p className="text-xs text-muted-foreground/60 mt-1 max-w-xs text-center">Thresholds 70/75/80/85 trigger warnings; 90 terminates and appears here.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {alerts.map((a) => (
                  <div key={a.id} className={`rounded-lg border px-4 py-3 flex items-start gap-3 ${a.terminated ? "border-danger/30 bg-danger/10" : a.threshold >= 80 ? "border-warning/30 bg-warning/10" : "border-warning/20 bg-card"}`}>
                    <div className={`rounded-md p-1.5 ${a.terminated ? "bg-danger/20" : "bg-warning/15"}`}>
                      {a.terminated ? <ShieldAlert className="h-4 w-4 text-danger" /> : <AlertTriangle className="h-4 w-4 text-warning" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium">{a.terminated ? "🛑 Critical — Call terminated" : `⚠️ ${a.title}`} — {a.risk}%</p>
                      <p className="text-xs text-muted-foreground truncate">{a.reason}</p>
                      <p className="text-[11px] text-muted-foreground/60">{a.callId} · {new Date(a.timestamp).toLocaleTimeString()} · threshold {a.threshold}%{a.terminated ? " · terminated" : ""}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>

          {/* 5. INCIDENT ACTIONS (when viewing a call) */}
          {selectedCall && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-xl border border-border bg-card p-5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold">Incident Actions — {selectedCall.id}</h2>
                <button onClick={() => setSelectedId(null)} className="text-xs text-muted-foreground hover:text-foreground">Close</button>
              </div>
              <div className="flex items-center gap-2 mb-3">
                <span className={`text-sm font-bold ${riskColor(selectedCall.riskLevel)}`}>{selectedCall.risk}% · {selectedCall.riskLevel}</span>
                <span className="text-xs text-muted-foreground">{selectedCall.reason ?? ""}</span>
                {selectedCall.terminated && <span className="text-xs font-medium text-danger flex items-center gap-1"><ShieldAlert className="h-3 w-3" /> Terminated at 90%</span>}
              </div>
              {!selectedCall.terminated && selectedCall.risk >= 70 ? (
                <div className="flex flex-wrap gap-2">
                  <button onClick={handleVerify} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">Verify Through Official Number</button>
                  <button onClick={handleMfa} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">Request MFA</button>
                  <button onClick={handleReport} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">Report / Escalate</button>
                </div>
              ) : selectedCall.terminated ? (
                <div className="flex flex-wrap gap-2">
                  <button onClick={handleBlock} className="inline-flex items-center gap-1.5 rounded-lg bg-danger px-3 py-1.5 text-xs font-semibold text-white hover:bg-danger/90">Block</button>
                  <button onClick={handleReport} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">Report</button>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No action required — risk below 70%. Continue monitoring.</p>
              )}
              {selectedCall.windows.length > 0 && (
                <div className="mt-4 rounded-lg border border-border bg-background/30 p-3">
                  <p className="text-[11px] text-muted-foreground uppercase mb-2">Risk timeline for this call</p>
                  <RiskTimeline windows={selectedCall.windows} />
                </div>
              )}
            </motion.div>
          )}

          {/* Quick Actions — keep minimal */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.25 }} className="rounded-xl border border-border bg-card p-5">
            <h2 className="text-sm font-semibold mb-3">Quick Actions</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              <Link href="/call" className="flex items-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-3 hover:border-primary/20 hover:bg-secondary">
                <Phone className="h-4 w-4 text-primary" />
                <div>
                  <p className="text-sm font-medium">Start Live Call</p>
                  <p className="text-xs text-muted-foreground">Begin real-time voice analysis</p>
                </div>
              </Link>
              <Link href="/detection" className="flex items-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-3 hover:border-primary/20 hover:bg-secondary">
                <Shield className="h-4 w-4 text-primary" />
                <div>
                  <p className="text-sm font-medium">Analyze Voice</p>
                  <p className="text-xs text-muted-foreground">Upload a recording for detection</p>
                </div>
              </Link>
            </div>
          </motion.div>
        </div>
      </PageContainer>
      <Footer />
    </>
  );
}
