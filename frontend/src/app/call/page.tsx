"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Mic,
  MicOff,
  Phone,
  PhoneOff,
  Shield,
  User,
  Loader2,
  AlertTriangle,
  X,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PageContainer } from "@/components/layout/PageContainer";
import { useWebRTC } from "@/hooks/useWebRTC";
import { useLiveDetection } from "@/hooks/useLiveDetection";
import { RiskTimeline } from "@/components/detection/RiskTimeline";
import { toast } from "sonner";

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

interface MonitoringPanelProps {
  status: CallStatus;
  windows: import("@/types/detection").WindowResult[];
  aggregation: import("@/types/detection").AggregationResult | null;
  isStreaming: boolean;
}

function MonitoringPanel({ status, windows, aggregation, isStreaming }: MonitoringPanelProps) {
  const monitoringActive = status === "connected" || status === "monitoring" || isStreaming;

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
            {monitoringActive ? "Active" : "Waiting"}
          </span>
        </div>

        {/* Risk Score */}
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

        {/* Timeline */}
        {windows.length > 0 ? (
          <RiskTimeline windows={windows} />
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

// ─── Main Page ─────────────────────────────────────────────────

export default function CallPage() {
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [alertOpen, setAlertOpen] = useState(false);
  const [alertScore, setAlertScore] = useState<number | null>(null);

  const {
    localStream,
    isConnected,
    isMuted,
    start,
    stop,
    toggleMute,
  } = useWebRTC();

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

  const handleStartCall = useCallback(async () => {
    try {
      setCallStatus("requesting");
      await start();
      setCallStatus("connecting");

      // Start real-time streaming (Morph windowed inference)
      try {
        await live.startStreaming();
        setCallStatus("monitoring");
        toast.success("Live monitoring active — 4 s windows");
      } catch {
        toast.info("Microphone access granted. Live detection failed to connect.");
        setCallStatus("connected");
      }
    } catch {
      setCallStatus("error");
      toast.error("Could not access microphone.");
    }
  }, [start, live]);

  const handleEndCall = useCallback(() => {
    live.stopStreaming();
    stop();
    setCallStatus("ended");
    setAlertOpen(false);
    setAlertScore(null);
    toast.info("Call ended.");
  }, [stop, live]);

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
                  label="Remote Caller"
                  isMuted={false}
                  isActive={isConnected}
                />
              </motion.div>

              {/* Controls */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.15 }}
                className="flex items-center justify-center gap-3"
              >
                {isIdleOrEnded ? (
                  <button
                    onClick={handleStartCall}
                    className="inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                  >
                    <Phone className="h-4 w-4" />
                    Start Call
                  </button>
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

            {/* Right: Monitoring panel */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              <MonitoringPanel
                status={callStatus}
                windows={live.windows}
                aggregation={live.aggregation}
                isStreaming={live.isStreaming}
              />
            </motion.div>
          </div>
        </div>
      </PageContainer>
      <Footer />

      {/* Alert Modal — real-time high-risk trigger */}
      <AlertModal
        open={alertOpen}
        riskScore={alertScore ?? live.aggregation?.risk_score ?? null}
        onEndCall={handleEndCall}
        onContinue={() => setAlertOpen(false)}
      />
    </>
  );
}
