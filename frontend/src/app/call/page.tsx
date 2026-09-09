"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  MicOff,
  Phone,
  PhoneOff,
  Shield,
  User,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PageContainer } from "@/components/layout/PageContainer";
import { useWebRTC } from "@/hooks/useWebRTC";
import { useLiveDetection } from "@/hooks/useLiveDetection";
import { RiskTimeline } from "@/components/detection/RiskTimeline";
import { useDetectionSocket } from "@/hooks/useDetectionSocket";
import { useTemporalDetection } from "@/hooks/useTemporalDetection";
import { getDetectionSocketUrl } from "@/services/websocket";
import {
  SignalingSocket,
  getSignalingSocketUrl,
} from "@/services/signaling";
import type { SignalingIncoming } from "@/types/signaling";
import type { DetectionResultMessage } from "@/types/websocket";
import type { WindowResult, AggregationResult } from "@/types/detection";
import { toast } from "sonner";
import { publishCall, removeCall, publishAlert, riskLevelFromRisk } from "@/lib/dashboardStore";

type CallStatus =
  | "idle"
  | "requesting"
  | "connecting"
  | "connected"
  | "monitoring"
  | "ended"
  | "error";

const statusLabels: Record<CallStatus, string> = {
  idle: "Idle",
  requesting: "Requesting microphone",
  connecting: "Connecting",
  connected: "Connected",
  monitoring: "Monitoring",
  ended: "Call ended",
  error: "Error",
};

const statusColors: Record<CallStatus, string> = {
  idle: "bg-muted-foreground/40",
  requesting: "bg-warning",
  connecting: "bg-info",
  connected: "bg-primary",
  monitoring: "bg-primary",
  ended: "bg-muted-foreground/40",
  error: "bg-danger",
};

// ─── Room-call pre-transaction warning config (70/75/80/85/90) ─────────
type WarningThreshold = 70 | 75 | 80 | 85 | 90;
const ROOM_WARNING_CONFIG: Record<WarningThreshold, { icon: string; title: string; desc: string }> = {
  70: { icon: "⚠️", title: "Suspicious call detected", desc: "Verify the caller before sharing sensitive information." },
  75: { icon: "🚨", title: "Elevated Risk", desc: "Potential impersonation detected. Do not share sensitive information." },
  80: { icon: "🚨", title: "High Risk Call", desc: "Do not share OTP, passwords, or financial information." },
  85: { icon: "🚨", title: "Critical Warning", desc: "Strong signs of impersonation/fraud detected. Verify the caller through an official channel before continuing." },
  90: { icon: "🛑", title: "Critical risk detected", desc: "Morph has ended the call to prevent potential fraud." },
};

function deriveWarningReason(result: DetectionResultMessage["payload"] | null, risk: number): string {
  if (!result) return "Potential voice impersonation detected.";
  const p: unknown = result as unknown;
  // Use only signals already produced by backend — do not invent
  const obj = p as Record<string, unknown>;
  // Semantic/OTP signals if backend ever provides them (forward-compatible)
  const semantic = obj["semantic"] as Record<string, unknown> | undefined;
  const transcript = (obj["transcript"] as string) || (semantic?.["transcript"] as string);
  const otpFlag = obj["otp_detected"] ?? semantic?.["otp_detected"] ?? obj["credential_request"];
  const financialFlag = obj["financial_manipulation"] ?? semantic?.["financial_manipulation"];
  const urgencyFlag = obj["urgency_detected"] ?? semantic?.["urgency_detected"];

  // Surface semantic reason when available
  if (otpFlag) return "Sensitive request detected: OTP / credential request.";
  if (financialFlag) return "Financial manipulation detected.";
  if (urgencyFlag) return "Urgency / pressure tactics detected.";
  if (transcript && typeof transcript === "string" && transcript.trim().length > 0) {
    return `Transcript flag: "${transcript.slice(0, 80)}"`;
  }
  // Default acoustic reason
  if (risk >= 85) return "Strong signs of impersonation/fraud detected.";
  if (risk >= 70) return "Potential voice impersonation detected.";
  return "Potential voice impersonation detected.";
}

// ─── Alert Modal ───────────────────────────────────────────────

interface AlertModalProps {
  open: boolean;
  riskScore: number | null;
  onEndCall: () => void;
  onContinue: () => void;
}

function AlertModal({ open, riskScore, onEndCall, onContinue }: AlertModalProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 8 }}
            className="w-full max-w-sm rounded-xl border border-danger/30 bg-card p-6 shadow-xl"
          >
            <div className="flex items-start gap-3 mb-4">
              <div className="rounded-lg bg-danger/10 p-2">
                <AlertTriangle className="h-5 w-5 text-danger" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">
                  Potential Synthetic Voice Detected
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  The Morph engine has flagged this voice stream.
                </p>
              </div>
            </div>

            {riskScore !== null && (
              <div className="mb-4 rounded-lg bg-background/50 px-4 py-3">
                <p className="text-xs text-muted-foreground">Risk Score</p>
                <p className="text-xl font-bold text-danger">{riskScore}%</p>
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={onEndCall}
                className="flex-1 inline-flex items-center justify-center gap-2 rounded-lg bg-danger px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-danger/90"
              >
                <PhoneOff className="h-4 w-4" />
                End Call
              </button>
              <button
                onClick={onContinue}
                className="flex-1 inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold transition-colors hover:bg-secondary"
              >
                Continue Call
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ─── Room Warning Card (pre-transaction / secondary verification) ───────
// State-aware: 70–89 shows verification actions (no Block); 90 shows only Block+Report (no End Call)
interface RoomWarningCardProps {
  threshold: WarningThreshold;
  risk: number;
  reason: string;
  onVerify: () => void;
  onRequestMfa: () => void;
  onEscalate: () => void;
  onBlock: () => void;
}

function RoomWarningCard({ threshold, risk, reason, onVerify, onRequestMfa, onEscalate, onBlock }: RoomWarningCardProps) {
  const cfg = ROOM_WARNING_CONFIG[threshold];
  const isTerminated = threshold === 90;
  return (
    <div className={`rounded-xl border p-4 ${isTerminated ? "border-danger/40 bg-danger/10" : threshold >= 80 ? "border-warning/30 bg-warning/10" : "border-warning/20 bg-card"}`}>
      <div className="flex items-start gap-3">
        <div className={`rounded-lg p-2 ${isTerminated ? "bg-danger/20" : "bg-warning/15"}`}>
          <AlertTriangle className={`h-5 w-5 ${isTerminated ? "text-danger" : "text-warning"}`} />
        </div>
        <div className="flex-1">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            {cfg.icon} {cfg.title} — {risk}%
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">{cfg.desc}</p>
          <p className="mt-2 text-xs font-medium text-foreground">{reason}</p>
          {threshold < 90 && <p className="mt-1 text-[11px] text-muted-foreground/70">Current risk {risk}% (threshold {threshold}%)</p>}
          {isTerminated && <p className="mt-2 text-xs font-medium text-danger">Morph has automatically ended the call to prevent potential fraud. Recommended next steps below.</p>}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {isTerminated ? (
          <>
            <button onClick={onBlock} className="inline-flex items-center gap-1.5 rounded-lg bg-danger px-3 py-1.5 text-xs font-semibold text-white hover:bg-danger/90">
              Block
            </button>
            <button onClick={onEscalate} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">
              Report
            </button>
          </>
        ) : (
          <>
            <button onClick={onVerify} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">
              Verify Through Official Number
            </button>
            <button onClick={onRequestMfa} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">
              Request MFA
            </button>
            <button onClick={onEscalate} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:bg-secondary">
              Report / Escalate
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Participant Card ──────────────────────────────────────────

interface ParticipantCardProps {
  label: string;
  isMuted: boolean;
  isActive: boolean;
}

function ParticipantCard({ label, isMuted, isActive }: ParticipantCardProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border bg-card p-6 transition-colors ${
        isActive ? "border-primary/20" : "border-border"
      }`}
    >
      <div
        className={`flex h-16 w-16 items-center justify-center rounded-full transition-colors ${
          isActive ? "bg-primary/10" : "bg-secondary"
        }`}
      >
        <User
          className={`h-7 w-7 transition-colors ${
            isActive ? "text-primary" : "text-muted-foreground"
          }`}
        />
      </div>
      <p className="mt-3 text-sm font-medium">{label}</p>
      <div className="mt-2 flex items-center gap-1.5">
        {isMuted ? (
          <MicOff className="h-3 w-3 text-muted-foreground" />
        ) : (
          <Mic
            className={`h-3 w-3 ${
              isActive ? "text-primary" : "text-muted-foreground"
            }`}
          />
        )}
        <span className="text-[10px] text-muted-foreground">
          {isMuted ? "Muted" : "Audio"}
        </span>
      </div>
    </div>
  );
}

// ─── Monitoring Panel ──────────────────────────────────────────
// Combined: preserves Mayukh's windowed/aggregation/RiskTimeline + Akasdip's result/isAnalyzing

interface MonitoringPanelProps {
  status: CallStatus;
  windows: import("@/types/detection").WindowResult[];
  aggregation: import("@/types/detection").AggregationResult | null;
  isStreaming: boolean;
  result: DetectionResultMessage["payload"] | null;
  isAnalyzing: boolean;
}

function MonitoringPanel({ status, windows, aggregation, isStreaming, result, isAnalyzing }: MonitoringPanelProps) {
  const monitoringActive = status === "connected" || status === "monitoring" || isStreaming || isAnalyzing;
  const riskScore = result ? Math.round(result.fake_probability * 100) : null;

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Shield className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold">Morph Monitoring</h3>
      </div>

      <div className="space-y-3">
        {/* Status */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Status</span>
          <span
            className={`text-xs font-medium ${
              monitoringActive ? "text-primary" : "text-muted-foreground"
            }`}
          >
            {isAnalyzing ? "Analyzing" : monitoringActive ? "Active" : "Waiting"}
          </span>
        </div>

        {/* Verdict — combined: prefer live streaming result, fallback to aggregation */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Verdict</span>
          <span
            className={`text-xs font-medium ${
              (result?.label_str === "FAKE" || aggregation?.aggregated_label_str === "FAKE") ? "text-danger" : "text-primary"
            }`}
          >
            {result ? result.label_str : aggregation ? aggregation.aggregated_label_str : "—"}
          </span>
        </div>

        {/* Risk Score — combined: aggregation first (windowed), then per-window result */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Risk Score</span>
          {aggregation ? (
            <span
              className={`text-xs font-bold ${
                aggregation.risk_level === "HIGH"
                  ? "text-danger"
                  : aggregation.risk_level === "MEDIUM"
                    ? "text-warning"
                    : "text-primary"
              }`}
            >
              {aggregation.risk_score}% · {aggregation.risk_level}
            </span>
          ) : riskScore !== null ? (
            <span
              className={`text-xs font-medium ${
                riskScore >= 50 ? "text-danger" : "text-muted-foreground/50"
              }`}
            >
              {riskScore}%
            </span>
          ) : (
            <span className="text-xs font-medium text-muted-foreground/50">—</span>
          )}
        </div>

        {aggregation && (
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div className="rounded bg-background/50 px-2 py-1.5">
              <p className="text-[10px] text-muted-foreground">Mean</p>
              <p className="font-semibold">{(aggregation.mean_fake_prob * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded bg-background/50 px-2 py-1.5">
              <p className="text-[10px] text-muted-foreground">Max</p>
              <p className="font-semibold">{(aggregation.max_fake_prob * 100).toFixed(1)}%</p>
            </div>
          </div>
        )}

        {/* Timeline — combined: RiskTimeline for windowed, fallback to single result view */}
        {windows.length > 0 ? (
          <RiskTimeline windows={windows} />
        ) : result ? (
          <div className="rounded-lg border border-dashed border-border bg-background/30 py-6 text-center">
            <div className="space-y-1">
              <p className="text-xs font-semibold">
                {result.label_str === "FAKE" ? "SYNTHETIC VOICE" : "REAL VOICE"}
              </p>
              <p className="text-[10px] text-muted-foreground/60">
                chunk {result.chunk_duration.toFixed(1)}s · "
                {result.label_str === "FAKE"
                  ? (result.fake_probability * 100).toFixed(1)
                  : (result.real_probability * 100).toFixed(1)}
                % confidence
              </p>
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-border bg-background/30 py-6 text-center">
            <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">
              Detection timeline
            </p>
            <p className="text-[10px] text-muted-foreground/60 mt-0.5">
              {monitoringActive ? "Listening… 4 s windows" : "Appears during active monitoring"}
            </p>
          </div>
        )}

        {windows.length > 0 && aggregation && (
          <p className="text-[10px] text-muted-foreground/60">
            {aggregation.n_fake}/{aggregation.total_windows} windows FAKE · {windows.length} × 4 s windows
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Room-call aggregation helper (mirrors ai_engine/inference/aggregation.py) ─
function computeRoomAggregation(windows: WindowResult[]): AggregationResult | null {
  if (!windows.length) return null;
  const probs = windows.map((w) => w.fake_probability);
  const n = probs.length;
  const n_fake = probs.filter((p) => p >= 0.5).length;
  const n_real = n - n_fake;
  const mean = probs.reduce((a, b) => a + b, 0) / n;
  const sorted = [...probs].sort((a, b) => a - b);
  const median = n % 2 === 1 ? sorted[Math.floor(n / 2)] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const max = Math.max(...probs);
  const min = Math.min(...probs);
  const std = n > 1 ? Math.sqrt(probs.reduce((s, p) => s + (p - mean) ** 2, 0) / n) : 0;
  const final_fake = mean;
  const final_real = 1 - final_fake;
  const label = final_fake >= 0.5 ? 1 : 0;
  const risk_level = final_fake >= 0.65 ? "HIGH" : final_fake >= 0.35 ? "MEDIUM" : "LOW";
  return {
    total_windows: n,
    n_fake,
    n_real,
    pct_fake: Math.round((n_fake / n) * 10000) / 100,
    pct_real: Math.round((n_real / n) * 10000) / 100,
    mean_fake_prob: Math.round(mean * 1e6) / 1e6,
    median_fake_prob: Math.round(median * 1e6) / 1e6,
    max_fake_prob: Math.round(max * 1e6) / 1e6,
    min_fake_prob: Math.round(min * 1e6) / 1e6,
    std_fake_prob: Math.round(std * 1e6) / 1e6,
    final_fake_prob: Math.round(final_fake * 1e6) / 1e6,
    final_real_prob: Math.round(final_real * 1e6) / 1e6,
    aggregated_label: label,
    aggregated_label_str: label === 1 ? "FAKE" : "REAL",
    risk_level: risk_level as AggregationResult["risk_level"],
    risk_score: Math.round(final_fake * 10000) / 100,
    confidence: Math.round(Math.abs(final_fake - 0.5) * 2 * 10000) / 10000,
  };
}

// ─── Main Page ─────────────────────────────────────────────────

export default function CallPage() {
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [alertOpen, setAlertOpen] = useState(false);
  const [alertScore, setAlertScore] = useState<number | null>(null);
  const [riskScore, setRiskScore] = useState<number | null>(null);
  const [lastResult, setLastResult] =
    useState<DetectionResultMessage["payload"] | null>(null);

  // ── Room-call live graph (reuses RiskTimeline) ─────────────────
  const [roomWindows, setRoomWindows] = useState<WindowResult[]>([]);
  const [roomAggregation, setRoomAggregation] = useState<AggregationResult | null>(null);
  const [morphTerminated, setMorphTerminated] = useState(false);
  const [terminationReason, setTerminationReason] = useState<string | null>(null);
  // Pre-transaction warning UI — current threshold/risk/reason (does not block graph)
  const [activeWarningThreshold, setActiveWarningThreshold] = useState<WarningThreshold | null>(null);
  const [activeWarningRisk, setActiveWarningRisk] = useState<number | null>(null);
  const [activeWarningReason, setActiveWarningReason] = useState<string | null>(null);
  // Progressive thresholds: 70→75→80→85→90(hard stop). Set ensures each fires once per call.
  const triggeredThresholdsRef = useRef<Set<number>>(new Set());
  const hasRemoteAudioRefStable = useRef(false);
  const callStatusRef = useRef<CallStatus>("idle");
  const roomWindowsRef = useRef<WindowResult[]>([]);
  const morphTerminatedRef = useRef(false);
  // Keep refs in sync with state for synchronous handleDetection access
  useEffect(() => {
    roomWindowsRef.current = roomWindows;
  }, [roomWindows]);
  useEffect(() => {
    morphTerminatedRef.current = morphTerminated;
  }, [morphTerminated]);

  // ── Two-browser signaling state ──────────────────────────────
  const [roomInput, setRoomInput] = useState("");
  const [roomCode, setRoomCode] = useState<string | null>(null);
  const [peerCount, setPeerCount] = useState(0);
  const [role, setRole] = useState<"caller" | "callee" | null>(null);
  const [signalingStatus, setSignalingStatus] = useState<
    "idle" | "connecting" | "connected" | "disconnected"
  >("idle");
  const [signalingUrl, setSignalingUrl] = useState<string>("");
  const signalingRef = useRef<SignalingSocket | null>(null);
  const roomRef = useRef<string>("");
  const roleRef = useRef<"caller" | "callee" | null>(null);
  const offerSentRef = useRef(false);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);

  // Secondary verification guidance actions (no backend workflow)
  const handleVerifyGuidance = useCallback(() => {
    toast.info("End this call and contact the organization using the official number from their website/card/statement. Do not use a number provided by the caller.", { duration: 8000 });
  }, []);
  const handleMfaGuidance = useCallback(() => {
    toast.info("Require independent verification through an approved MFA/authentication channel before completing the requested action.", { duration: 8000 });
  }, []);
  const handleEscalateGuidance = useCallback(() => {
    toast.info("Escalation noted — report this call to your supervisor / security team with risk score and any transcript.", { duration: 8000 });
  }, []);
  const handleBlockGuidance = useCallback(() => {
    toast.info("Contact blocked — this number has been flagged as high-risk for future calls.", { duration: 8000 });
  }, []);

  const temporal = useTemporalDetection();

  // Defer WS URL to client effect to avoid hydration mismatch (server localhost vs LAN IP)
  const [detectionUrl, setDetectionUrl] = useState("");
  useEffect(() => {
    setDetectionUrl(getDetectionSocketUrl());
  }, []);

  useEffect(() => {
    callStatusRef.current = callStatus;
  }, [callStatus]);

  const handleDetection = useCallback(
    (result: DetectionResultMessage["payload"]) => {
      setLastResult(result);
      const risk = Math.round(result.fake_probability * 100);
      setRiskScore(risk);

      // Determine if this is a connected Room Call (must be remote peer, not host mic)
      const isRoomCall = !!roomRef.current && hasRemoteAudioRefStable.current;

      if (isRoomCall) {
        // ── Build Room-call WindowResult for RiskTimeline (reuse existing component) ──
        // Convert lightweight detection_result into full WindowResult shape.
        // Once remote is connected, host mic is already detached in useWebRTC (attachDetectionStream remote),
        // so results here are exclusively remote peer audio.
        const wIdx = roomWindowsRef.current.length;
        const hop = 1.0;
        const winSec = 4.0;
        const start = wIdx * hop;
        const win: WindowResult = {
          window_index: wIdx,
          window: wIdx + 1,
          start_sec: start,
          end_sec: start + winSec,
          duration: (result as unknown as { chunk_duration?: number }).chunk_duration ?? winSec,
          chunk_duration: (result as unknown as { chunk_duration?: number }).chunk_duration ?? winSec,
          is_partial: (result as unknown as { is_partial?: boolean }).is_partial ?? false,
          label: (result as unknown as { label?: number }).label ?? (result.label_str === "FAKE" ? 1 : 0),
          label_str: result.label_str,
          confidence: result.confidence,
          real_probability: result.real_probability,
          fake_probability: result.fake_probability,
          real_prob: (result as unknown as { real_prob?: number }).real_prob ?? result.real_probability,
          fake_prob: (result as unknown as { fake_prob?: number }).fake_prob ?? result.fake_probability,
          model_version: (result as unknown as { model_version?: string }).model_version ?? "w2v2_aasist",
        };
        const nextWindows = [...roomWindowsRef.current, win];
        roomWindowsRef.current = nextWindows;
        setRoomWindows(nextWindows);
        const agg = computeRoomAggregation(nextWindows);
        setRoomAggregation(agg);

        // ── Progressive thresholds: 70→75→80→85→90 (hard stop) ──
        // Each threshold fires at most once per call. Jump 68→82 triggers 70,75,80.
        // 90 terminates immediately.
        const reason = deriveWarningReason(result, risk);
        if (risk >= 90 && !triggeredThresholdsRef.current.has(90)) {
          triggeredThresholdsRef.current.add(90);
          setAlertScore(90);
          setActiveWarningThreshold(90);
          setActiveWarningRisk(risk);
          setActiveWarningReason(reason);
          const cfg90 = ROOM_WARNING_CONFIG[90];
          toast.error(`${cfg90.icon} ${cfg90.title} — ${risk}%`, { description: `${reason} ${cfg90.desc}`, duration: 8000 });
          morphTerminatedRef.current = true;
          setMorphTerminated(true);
          setTerminationReason(`Morph ended the call — risk reached critical ${risk}% (90% threshold)`);
          setAlertOpen(false);
          // Publish to dashboard — processed risk, no recalculation
          publishCall({
            id: roomRef.current,
            risk,
            riskLevel: riskLevelFromRisk(risk),
            status: "terminated",
            terminated: true,
            activeThreshold: 90,
            reason,
            windows: nextWindows,
            aggregation: agg,
            lastUpdate: Date.now(),
          });
          publishAlert({
            id: `alert-${Date.now()}-90`,
            callId: roomRef.current,
            threshold: 90,
            risk,
            reason,
            title: cfg90.title,
            timestamp: Date.now(),
            terminated: true,
          });
          setTimeout(() => {
            window.dispatchEvent(new CustomEvent("morph-terminate-90"));
          }, 0);
          return;
        }
        let firedThisChunk = false;
        for (const th of [70, 75, 80, 85] as WarningThreshold[]) {
          if (risk >= th && !triggeredThresholdsRef.current.has(th)) {
            triggeredThresholdsRef.current.add(th);
            setActiveWarningThreshold(th);
            setActiveWarningRisk(risk);
            setActiveWarningReason(reason);
            const cfg = ROOM_WARNING_CONFIG[th];
            toast.warning(`${cfg.icon} ${cfg.title} — ${risk}%`, { description: `${reason} ${cfg.desc}`, duration: 5000 });
            setAlertScore(risk);
            setAlertOpen(true);
            publishAlert({
              id: `alert-${Date.now()}-${th}`,
              callId: roomRef.current,
              threshold: th,
              risk,
              reason,
              title: cfg.title,
              timestamp: Date.now(),
            });
            firedThisChunk = true;
          }
        }
        // Always publish latest processed risk to dashboard (live risk / active call)
        const activeTh = triggeredThresholdsRef.current.size ? Math.max(...Array.from(triggeredThresholdsRef.current).filter((t) => t !== 90) as number[]) as WarningThreshold | null : null;
        // Use highest triggered threshold that is <= risk, or null if <70
        const currentTh = risk >= 70 ? (risk >= 85 ? 85 : risk >= 80 ? 80 : risk >= 75 ? 75 : 70) : null;
        publishCall({
          id: roomRef.current,
          risk,
          riskLevel: riskLevelFromRisk(risk),
          status: hasRemoteAudioRefStable.current ? "monitoring" : "connected",
          terminated: false,
          activeThreshold: activeTh ?? currentTh,
          reason,
          windows: nextWindows,
          aggregation: agg,
          lastUpdate: Date.now(),
        });
        // Avoid duplicate alert spam — if we just fired, active warning already set
        if (firedThisChunk) return;
        // Even if no new threshold, still keep dashboard live risk updated (already published)
        return;
      }

      // Non-room path (direct live or pre-remote fallback): preserve existing temporal behavior
      const decision = temporal.process(result.fake_probability);
      if (decision.enteredFake) {
        setAlertOpen(true);
      }
    },
    [temporal],
  );

  const detection = useDetectionSocket({
    url: detectionUrl,
    onDetection: handleDetection,
    onError: (message) => toast.error(`Detection error: ${message}`),
  });

  const {
    localStream,
    remoteStream,
    isConnected,
    hasRemoteAudio,
    isMuted,
    start,
    stop,
    toggleMute,
    createOffer,
    acceptOfferAndCreateAnswer,
    acceptAnswer,
    addIceCandidate,
  } = useWebRTC({
    onAudioChunk: (chunk, sampleRate) =>
      detection.sendAudioChunk(chunk, sampleRate),
    onIceCandidate: (candidate) => {
      const room = roomRef.current;
      if (room) {
        signalingRef.current?.send({
          type: "ice-candidate",
          payload: { room, candidate },
        });
      }
    },
    onRemoteTrack: () => setCallStatus("connected"),
  });

  const live = useLiveDetection({
    windowSec: 4.0,
    aggregation: "mean",
    onAggregation: (agg) => {
      if (agg.risk_level === "HIGH" && agg.risk_score >= 65) {
        setAlertScore(agg.risk_score);
        setAlertOpen(true);
      }
    },
  });

  // Play the remote party's real audio once ontrack fires.
  useEffect(() => {
    const el = remoteAudioRef.current;
    if (el && remoteStream) {
      el.srcObject = remoteStream;
      void el.play().catch(() => {});
    }
  }, [remoteStream, hasRemoteAudio]);

  // A live peer connection outranks the detection-only "monitoring" label.
  useEffect(() => {
    if (isConnected) setCallStatus("connected");
  }, [isConnected]);

  // Keep hasRemoteAudio ref in sync for stable handleDetection closure (must be after useWebRTC)
  useEffect(() => {
    hasRemoteAudioRefStable.current = hasRemoteAudio;
  }, [hasRemoteAudio]);

  // Latest RTC methods for the signaling callback (avoids stale closures).
  const rtcRef = useRef({
    start,
    createOffer,
    acceptOfferAndCreateAnswer,
    acceptAnswer,
    addIceCandidate,
  });
  rtcRef.current = {
    start,
    createOffer,
    acceptOfferAndCreateAnswer,
    acceptAnswer,
    addIceCandidate,
  };

  const handleSignalingMessage = useCallback(async (msg: SignalingIncoming) => {
    const room = roomRef.current;
    try {
      switch (msg.type) {
        case "joined":
          setPeerCount(msg.payload.peers);
          break;
        case "peer-joined": {
          setPeerCount(msg.payload.peers);
          // Caller creates the offer once the callee arrives.
          if (roleRef.current === "caller" && !offerSentRef.current) {
            offerSentRef.current = true;
            const sdp = await rtcRef.current.createOffer();
            signalingRef.current?.send({ type: "offer", payload: { room, sdp } });
            setCallStatus("connecting");
          }
          break;
        }
        case "offer": {
          // Callee answers. Local media was started before joining.
          const sdp = await rtcRef.current.acceptOfferAndCreateAnswer(
            msg.payload.sdp,
          );
          signalingRef.current?.send({ type: "answer", payload: { room, sdp } });
          setCallStatus("connecting");
          break;
        }
        case "answer":
          await rtcRef.current.acceptAnswer(msg.payload.sdp);
          break;
        case "ice-candidate":
          await rtcRef.current.addIceCandidate(msg.payload.candidate);
          break;
        case "peer-left":
          setPeerCount(msg.payload.peers);
          toast.info("Peer left the room.");
          break;
        case "error":
          if (msg.payload.code === "NO_PEER") break; // waiting — not an error
          toast.error(`Signaling: ${msg.payload.message}`);
          break;
      }
    } catch {
      toast.error("Failed to negotiate the call. Try re-joining the room.");
    }
  }, []);

  const ensureSignaling = useCallback(() => {
    if (!signalingRef.current) {
      // Computed here at click time from window.location — always the
      // current page host, never a stale cached value.
      const url = getSignalingSocketUrl();
      setSignalingUrl(url);
      const socket = new SignalingSocket(url);
      socket.onMessage(handleSignalingMessage);
      socket.onStatusChange((status) => setSignalingStatus(status));
      // Connection failures must be visible, not console-only.
      socket.onError((message) => toast.error(message, { duration: 8000 }));
      signalingRef.current = socket;
    }
    signalingRef.current.connect();
    return signalingRef.current;
  }, [handleSignalingMessage]);

  const startLocalAndJoin = useCallback(
    async (code: string, asRole: "caller" | "callee") => {
      const room = code.trim().toUpperCase();
      if (!room) {
        toast.error("Enter a room code first.");
        return;
      }
      try {
        setCallStatus("requesting");
        // Reset Room-call graph and progressive thresholds for new call
        setRoomWindows([]);
        setRoomAggregation(null);
        roomWindowsRef.current = [];
        triggeredThresholdsRef.current.clear();
        morphTerminatedRef.current = false;
        setMorphTerminated(false);
        setTerminationReason(null);
        setActiveWarningThreshold(null);
        setActiveWarningRisk(null);
        setActiveWarningReason(null);
        setAlertOpen(false);
        setAlertScore(null);
        await rtcRef.current.start();
        setCallStatus("connecting");
        detection.connect();
        roomRef.current = room;
        roleRef.current = asRole;
        offerSentRef.current = false;
        setRoomCode(room);
        setRole(asRole);
        setPeerCount(0);
        ensureSignaling().send({ type: "join", payload: { room } });
        // Publish initial active call to dashboard (backend-processed risk still 0)
        publishCall({
          id: room,
          risk: 0,
          riskLevel: "Low",
          status: "connected",
          terminated: false,
          activeThreshold: null,
          reason: null,
          windows: [],
          aggregation: null,
          lastUpdate: Date.now(),
        });
      } catch {
        setCallStatus("error");
        toast.error("Could not access microphone.");
      }
    },
    [ensureSignaling, detection],
  );

  const handleHostCall = useCallback(() => {
    const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    const code = Array.from(
      { length: 6 },
      () => alphabet[Math.floor(Math.random() * alphabet.length)],
    ).join("");
    setRoomInput(code);
    void startLocalAndJoin(code, "caller");
  }, [startLocalAndJoin]);

  const handleJoinCall = useCallback(() => {
    void startLocalAndJoin(roomInput, "callee");
  }, [startLocalAndJoin, roomInput]);

  // Once the mic is live and the detection socket is up, start monitoring.
  useEffect(() => {
    if (detection.isConnected && (callStatus === "connecting" || callStatus === "requesting")) {
      setCallStatus("monitoring");
    }
  }, [detection.isConnected, callStatus]);

  const handleStartCall = useCallback(async () => {
    try {
      setCallStatus("requesting");
      await start();
      setCallStatus("connecting");

      // Start real-time streaming (Morph windowed inference via /ws/detect)
      try {
        await live.startStreaming();
        setCallStatus("monitoring");
        toast.success("Live monitoring active — 4 s windows");
      } catch {
        toast.info("Microphone access granted. Live detection failed to connect.");
        setCallStatus("connected");
      }
      // Also connect detection socket for temporal path (Akasdip live analysis)
      try {
        detection.connect();
      } catch {}
    } catch {
      setCallStatus("error");
      toast.error("Could not access microphone.");
    }
  }, [start, live, detection]);

  const handleEndCall = useCallback(() => {
    const wasMorphTerminated = morphTerminatedRef.current;
    const reason = terminationReason;
    const endedId = roomRef.current;
    if (endedId) {
      signalingRef.current?.send({
        type: "leave",
        payload: { room: endedId },
      });
    }
    signalingRef.current?.disconnect();
    signalingRef.current = null;
    detection.disconnect();
    live.stopStreaming();
    stop();
    temporal.reset();
    // Remove active call from dashboard (alerts remain for terminated)
    if (endedId) removeCall(endedId);
    // Reset Room-call graph (preserve termination UI until next call)
    setRoomWindows([]);
    setRoomAggregation(null);
    roomWindowsRef.current = [];
    if (!wasMorphTerminated) {
      triggeredThresholdsRef.current.clear();
      setTerminationReason(null);
      setMorphTerminated(false);
      morphTerminatedRef.current = false;
      setActiveWarningThreshold(null);
      setActiveWarningRisk(null);
      setActiveWarningReason(null);
    }
    setLastResult(null);
    setRiskScore(null);
    setRoomCode(null);
    setPeerCount(0);
    setRole(null);
    setSignalingStatus("idle");
    roomRef.current = "";
    roleRef.current = null;
    offerSentRef.current = false;
    setCallStatus("ended");
    setAlertOpen(false);
    setAlertScore(null);
    if (wasMorphTerminated && reason) {
      // Already toasted in handleDetection; keep ended state visible
      toast.error(reason, { duration: 8000 });
    } else {
      toast.info("Call ended.");
    }
  }, [stop, detection, live, temporal, terminationReason]);

  // Hard-stop at 90%: handleDetection dispatches this event, we then perform full teardown
  useEffect(() => {
    const onMorphTerminate = () => handleEndCall();
    window.addEventListener("morph-terminate-90", onMorphTerminate as EventListener);
    return () => window.removeEventListener("morph-terminate-90", onMorphTerminate as EventListener);
  }, [handleEndCall]);

  const handleToggleMute = useCallback(() => {
    toggleMute();
  }, [toggleMute]);

  const isIdleOrEnded = callStatus === "idle" || callStatus === "ended" || callStatus === "error";
  const isActive = callStatus === "requesting" || callStatus === "connecting" || callStatus === "connected" || callStatus === "monitoring";

  return (
    <>
      <Navbar />
      <PageContainer>
        <div className="space-y-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="text-2xl font-bold tracking-tight">Live Call</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Real-time voice communication with Morph monitoring
            </p>
          </motion.div>

          <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
            {/* Left: Call area */}
            <div className="space-y-4">
              {/* Status bar */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.05 }}
                className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2.5"
              >
                <div
                  className={`h-2 w-2 rounded-full ${statusColors[callStatus]}`}
                />
                <span className="text-xs font-medium text-muted-foreground">
                  {statusLabels[callStatus]}
                </span>
                {isActive && callStatus !== "connected" && callStatus !== "monitoring" && (
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                )}
              </motion.div>

              {/* Participant cards */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.1 }}
                className="grid gap-4 sm:grid-cols-2"
              >
                <ParticipantCard
                  label="You"
                  isMuted={isMuted}
                  isActive={!!localStream}
                />
                <ParticipantCard
                  label={
                    roomCode
                      ? `Remote Caller · ${roomCode} · ${peerCount} in room`
                      : "Remote Caller"
                  }
                  isMuted={false}
                  isActive={isConnected || hasRemoteAudio}
                />
              </motion.div>
              {/* Hidden sink — plays the peer's real audio on call connect */}
              <audio ref={remoteAudioRef} autoPlay playsInline className="hidden" />

              {/* Room controls — two laptops join the same code */}
              {isIdleOrEnded && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.12 }}
                  className="rounded-xl border border-border bg-card p-4"
                >
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    WEB-RTC ROOM CALL
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    One laptop hosts a room, the other joins with the same
                    code. {role ? `You are the ${role}.` : ""}
                  </p>
                  <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                    <input
                      value={roomInput}
                      onChange={(e) =>
                        setRoomInput(e.target.value.toUpperCase().slice(0, 16))
                      }
                      placeholder="ROOM CODE (e.g. KQ7X2P)"
                      className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono uppercase tracking-widest placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/40"
                    />
                    <button
                      onClick={handleJoinCall}
                      disabled={!roomInput.trim()}
                      className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-secondary px-4 py-2 text-sm font-semibold transition-colors hover:bg-secondary/80 disabled:opacity-50"
                    >
                      <Phone className="h-4 w-4" />
                      Join Room
                    </button>
                    <button
                      onClick={handleHostCall}
                      className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                      <Phone className="h-4 w-4" />
                      Host New Room
                    </button>
                  </div>
                  {roomCode && (
                    <p className="mt-2 text-xs text-muted-foreground">
                      Share code{" "}
                      <span className="font-mono font-bold text-foreground">
                        {roomCode}
                      </span>{" "}
                      · {peerCount} peer{peerCount === 1 ? "" : "s"} in room
                      {hasRemoteAudio ? " · remote audio live" : ""}
                    </p>
                  )}
                  {/* Signaling connection — visible so a broken backend
                      link fails loudly instead of silently. */}
                  {signalingStatus !== "idle" && (
                    <p className="mt-2 text-xs text-muted-foreground">
                      <span
                        className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle ${
                          signalingStatus === "connected"
                            ? "bg-primary"
                            : signalingStatus === "connecting"
                              ? "bg-warning"
                              : "bg-danger"
                        }`}
                      />
                      Signaling: {signalingStatus}
                      {signalingUrl && (
                        <span className="ml-1 font-mono break-all">
                          {signalingUrl}
                        </span>
                      )}
                      {signalingStatus === "disconnected" && roomCode && (
                        <span className="ml-1">
                          — backend unreachable. Check the host/IP and port
                          8000, then re-join.
                        </span>
                      )}
                    </p>
                  )}
                  <div className="mt-3 flex items-center justify-center">
                    <button
                      onClick={handleStartCall}
                      className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                      <Phone className="h-4 w-4" />
                      Start Direct Call (Live Windowed)
                    </button>
                  </div>
                </motion.div>
              )}

              {/* Controls */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.15 }}
                className="flex items-center justify-center gap-3"
              >
                {isIdleOrEnded ? (
                  <p className="text-center text-xs text-muted-foreground">
                    Host or join a room above to place a call.
                  </p>
                ) : (
                  <>
                    <button
                      onClick={handleToggleMute}
                      className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-semibold transition-colors ${
                        isMuted
                          ? "border-danger/30 bg-danger/10 text-danger hover:bg-danger/20"
                          : "border-border bg-card hover:bg-secondary"
                      }`}
                    >
                      {isMuted ? (
                        <MicOff className="h-4 w-4" />
                      ) : (
                        <Mic className="h-4 w-4" />
                      )}
                      {isMuted ? "Unmute" : "Mute"}
                    </button>
                    <button
                      onClick={handleEndCall}
                      className="inline-flex items-center gap-2 rounded-lg bg-danger px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-danger/90"
                    >
                      <PhoneOff className="h-4 w-4" />
                      End Call
                    </button>
                  </>
                )}
              </motion.div>
            </div>

            {/* Right: Monitoring panel — Room Call reuses same RiskTimeline graph, fed by remote peer */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              {/* Pre-transaction warnings — visible during live call, do not block graph */}
              {activeWarningThreshold && activeWarningRisk !== null && activeWarningReason && (
                <div className="mb-3">
                  <RoomWarningCard
                    threshold={activeWarningThreshold}
                    risk={activeWarningRisk}
                    reason={activeWarningReason}
                    onVerify={handleVerifyGuidance}
                    onRequestMfa={handleMfaGuidance}
                    onEscalate={handleEscalateGuidance}
                    onBlock={handleBlockGuidance}
                  />
                </div>
              )}
              {/* Fallback termination banner if activeWarning not set (edge) */}
              {morphTerminated && !activeWarningThreshold && terminationReason && (
                <div className="mb-3 rounded-lg border border-danger/30 bg-danger/10 px-4 py-3">
                  <p className="text-xs font-semibold text-danger">🛑 {terminationReason}</p>
                  <p className="mt-1 text-[11px] text-danger/80">Morph automatically ended the WebRTC call and closed audio/detection resources.</p>
                </div>
              )}
              <MonitoringPanel
                status={morphTerminated ? "ended" : callStatus}
                windows={roomCode ? roomWindows : live.windows}
                aggregation={roomCode ? roomAggregation : live.aggregation}
                isStreaming={roomCode ? hasRemoteAudio : live.isStreaming}
                result={lastResult}
                isAnalyzing={detection.isAnalyzing}
              />
              {roomCode && hasRemoteAudio && (
                <p className="mt-2 text-[11px] text-muted-foreground/60">Graph tracks remote peer audio only (host mic detached after WebRTC connected).</p>
              )}
              {roomCode && activeWarningThreshold && (
                <p className="mt-2 text-[11px] text-muted-foreground/60">Warning at {activeWarningThreshold}% (current {activeWarningRisk}%) — {deriveWarningReason(lastResult, activeWarningRisk ?? 0)}</p>
              )}
              {!roomCode && live.windows.length > 0 && (
                <p className="mt-2 text-[11px] text-muted-foreground/60">Direct live graph (local mic) — Room Call graph above reuses same RiskTimeline component.</p>
              )}
            </motion.div>
          </div>
        </div>
      </PageContainer>
      <Footer />

      {/* Alert Modal — combined: fires on temporal, live aggregation, or Room progressive thresholds */}
      <AlertModal
        open={alertOpen}
        riskScore={alertScore ?? riskScore ?? roomAggregation?.risk_score ?? live.aggregation?.risk_score ?? null}
        onEndCall={handleEndCall}
        onContinue={() => setAlertOpen(false)}
      />
    </>
  );
}
