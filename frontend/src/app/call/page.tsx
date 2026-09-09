"use client";

import { useState, useCallback, useEffect, useRef } from "react";
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
import { useDetectionSocket } from "@/hooks/useDetectionSocket";
import { useTemporalDetection } from "@/hooks/useTemporalDetection";
import { getDetectionSocketUrl } from "@/services/websocket";
import {
  SignalingSocket,
  getSignalingSocketUrl,
} from "@/services/signaling";
import type { SignalingIncoming } from "@/types/signaling";
import type { DetectionResultMessage } from "@/types/websocket";
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
  result: DetectionResultMessage["payload"] | null;
  isAnalyzing: boolean;
}

function MonitoringPanel({ status, result, isAnalyzing }: MonitoringPanelProps) {
  const monitoringActive = status === "connected" || status === "monitoring";
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

        {/* Verdict */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Verdict</span>
          <span
            className={`text-xs font-medium ${
              result?.label_str === "FAKE" ? "text-danger" : "text-primary"
            }`}
          >
            {result ? result.label_str : "—"}
          </span>
        </div>

        {/* Risk Score */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Risk Score</span>
          <span
            className={`text-xs font-medium ${
              riskScore !== null && riskScore >= 50
                ? "text-danger"
                : "text-muted-foreground/50"
            }`}
          >
            {riskScore !== null ? `${riskScore}%` : "—"}
          </span>
        </div>

        {/* Timeline */}
        <div className="rounded-lg border border-dashed border-border bg-background/30 py-6 text-center">
          {result ? (
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
          ) : (
            <>
              <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">
                Detection timeline
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                {isAnalyzing ? "Analyzing audio…" : "Appears during active monitoring"}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────

export default function CallPage() {
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [alertOpen, setAlertOpen] = useState(false);
  const [riskScore, setRiskScore] = useState<number | null>(null);
  const [lastResult, setLastResult] =
    useState<DetectionResultMessage["payload"] | null>(null);

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

  const temporal = useTemporalDetection();

  const handleDetection = useCallback(
    (result: DetectionResultMessage["payload"]) => {
      setLastResult(result);
      // Risk Score tracks every raw window (pre-smoothing behavior).
      setRiskScore(Math.round(result.fake_probability * 100));
      // Temporal decision layer (EWMA + hysteresis + consecutive-window
      // gate) controls ONLY whether the alert opens: exactly once per
      // REAL → FAKE transition, never once per FAKE window.
      const decision = temporal.process(result.fake_probability);
      if (decision.enteredFake) {
        setAlertOpen(true);
      }
    },
    [temporal],
  );

  const detection = useDetectionSocket({
    url: getDetectionSocketUrl(),
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
      } catch {
        setCallStatus("error");
        toast.error("Could not access microphone.");
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ensureSignaling],
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

  const handleEndCall = useCallback(() => {
    if (roomRef.current) {
      signalingRef.current?.send({
        type: "leave",
        payload: { room: roomRef.current },
      });
    }
    signalingRef.current?.disconnect();
    signalingRef.current = null;
    detection.disconnect();
    stop();
    temporal.reset();
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
    toast.info("Call ended.");
  }, [stop, detection, temporal]);

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
                    Two-laptop call
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

            {/* Right: Monitoring panel */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              <MonitoringPanel
                status={callStatus}
                result={lastResult}
                isAnalyzing={detection.isAnalyzing}
              />
            </motion.div>
          </div>
        </div>
      </PageContainer>
      <Footer />

      {/* Alert Modal — fires once per REAL → FAKE temporal transition */}
      <AlertModal
        open={alertOpen}
        riskScore={riskScore}
        onEndCall={handleEndCall}
        onContinue={() => setAlertOpen(false)}
      />
    </>
  );
}
