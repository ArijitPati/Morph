"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Mic,
  MicOff,
  Upload,
  Play,
  Pause,
  Square,
  Loader2,
  Shield,
  AlertTriangle,
  FileAudio,
  X,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PageContainer } from "@/components/layout/PageContainer";
import type { DetectionResult } from "@/types/detection";
import { upload, ApiError } from "@/services/api";
import { toast } from "sonner";

type AnalysisStatus = "waiting" | "ready" | "analyzing" | "complete" | "error";

// ─── Waveform Canvas ───────────────────────────────────────────

function WaveformCanvas({
  audioBuffer,
  isPlaying,
  progress,
}: {
  audioBuffer: AudioBuffer | null;
  isPlaying: boolean;
  progress: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioBuffer) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / w);
    const amp = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Draw waveform
    for (let i = 0; i < w; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[i * step + j];
        if (datum !== undefined) {
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }
      }

      const played = i / w < progress;
      ctx.fillStyle = played
        ? "rgba(6, 214, 160, 0.8)"
        : "rgba(148, 163, 184, 0.35)";

      const y1 = (1 + min) * amp;
      const y2 = (1 + max) * amp;
      ctx.fillRect(i, y1, 1, Math.max(1, y2 - y1));
    }

    // Playhead
    if (progress > 0 && progress < 1) {
      const x = progress * w;
      ctx.fillStyle = "rgba(6, 214, 160, 1)";
      ctx.fillRect(x, 0, 1.5, h);
    }
  }, [audioBuffer, progress]);

  // Idle state — flat line
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || audioBuffer) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  }, [audioBuffer]);

  return (
    <canvas
      ref={canvasRef}
      className="h-24 w-full rounded-lg bg-background/50"
    />
  );
}

// ─── Main Page ─────────────────────────────────────────────────

export default function DetectionPage() {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState<string>("");
  const [duration, setDuration] = useState<number | null>(null);

  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Audio playback
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);

  // Analysis
  const [analysisStatus, setAnalysisStatus] =
    useState<AnalysisStatus>("waiting");
  const [result, setResult] = useState<DetectionResult | null>(null);

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Recording ────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setRecordedBlob(blob);
        const url = URL.createObjectURL(blob);
        setRecordedUrl(url);
        setFilename(`recording-${Date.now()}.webm`);
        setAnalysisStatus("ready");
        stream.getTracks().forEach((t) => t.stop());

        // Decode audio for waveform display
        const ctx = new AudioContext();
        audioContextRef.current = ctx;
        blob.arrayBuffer().then((buf) => {
          ctx.decodeAudioData(buf).then((decoded) => {
            setAudioBuffer(decoded);
            setDuration(decoded.duration);
          }).catch(() => {
            // WebM/Opus may not decode in all browsers — waveform won't show
            // but analysis still works via backend
          });
        });
      };

      mediaRecorder.start();
      setIsRecording(true);
      setAnalysisStatus("waiting");
      setResult(null);
      setRecordedBlob(null);
      setRecordedUrl(null);
      setAudioBuffer(null);
      setDuration(null);
      toast.info("Recording started.");
    } catch {
      toast.error("Could not access microphone.");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    toast.info("Recording stopped.");
  }, []);

  // ── File Upload ──────────────────────────────────────────

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setSelectedFile(file);
      setFilename(file.name);
      setRecordedBlob(null);
      setRecordedUrl(null);
      setAudioBuffer(null);
      setResult(null);
      setAnalysisStatus("ready");

      // Decode audio for waveform
      const url = URL.createObjectURL(file);
      setRecordedUrl(url);

      const ctx = new AudioContext();
      audioContextRef.current = ctx;
      file.arrayBuffer().then((buf) => {
        ctx.decodeAudioData(buf).then((decoded) => {
          setAudioBuffer(decoded);
          setDuration(decoded.duration);
        }).catch(() => {
          // WebM/Opus may not decode in all browsers — waveform won't show
          // but analysis still works via backend
        });
      });
    },
    [],
  );

  // ── Playback ─────────────────────────────────────────────

  const togglePlayback = useCallback(() => {
    if (!recordedUrl) return;

    if (isPlaying) {
      audioElementRef.current?.pause();
      cancelAnimationFrame(animFrameRef.current);
      setIsPlaying(false);
      return;
    }

    const audio = new Audio(recordedUrl);
    audioElementRef.current = audio;

    audio.onended = () => {
      setIsPlaying(false);
      setPlayProgress(0);
      cancelAnimationFrame(animFrameRef.current);
    };

    audio.play();
    setIsPlaying(true);

    const tick = () => {
      if (audio.currentTime && audio.duration) {
        setPlayProgress(audio.currentTime / audio.duration);
      }
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
  }, [recordedUrl, isPlaying]);

  // ── Analyze ──────────────────────────────────────────────

  const handleAnalyze = useCallback(async () => {
    const audioFile = recordedBlob
      ? new File([recordedBlob], filename || "recording.webm", {
          type: recordedBlob.type || "audio/webm",
        })
      : selectedFile;

    if (!audioFile) {
      toast.error("No audio file to analyze.");
      return;
    }

    setAnalysisStatus("analyzing");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", audioFile);

      const res = await upload<DetectionResult>(
        "/api/detection/analyze",
        formData,
      );

      setResult(res);
      setAnalysisStatus("complete");
      toast.success(`Analysis complete — ${res.verdict}`);
    } catch (err) {
      setAnalysisStatus("error");
      if (err instanceof ApiError) {
        toast.error(`Backend error (${err.status}): ${err.message}`);
      } else {
        toast.error("Could not reach the detection backend.");
      }
    }
  }, [recordedBlob, selectedFile, filename]);

  // ── Cleanup ──────────────────────────────────────────────

  const handleClear = useCallback(() => {
    setRecordedBlob(null);
    setRecordedUrl(null);
    setSelectedFile(null);
    setAudioBuffer(null);
    setFilename("");
    setDuration(null);
    setAnalysisStatus("waiting");
    setResult(null);
    setIsPlaying(false);
    setPlayProgress(0);
    cancelAnimationFrame(animFrameRef.current);
    audioElementRef.current?.pause();
    audioElementRef.current = null;
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  // ── Derived ──────────────────────────────────────────────

  const hasAudio = !!recordedUrl;
  const canAnalyze = hasAudio && analysisStatus === "ready";

  // ── Format duration ──────────────────────────────────────

  const fmtDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <>
      <Navbar />
      <PageContainer>
        <div className="space-y-6 max-w-3xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="text-2xl font-bold tracking-tight">
              Voice Detection
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Analyze a voice sample using Morph&apos;s synthetic voice
              detection engine.
            </p>
          </motion.div>

          {/* Audio Input Card */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.05 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h2 className="text-sm font-semibold mb-4">Audio Input</h2>

            <div className="flex flex-wrap gap-3">
              {/* Record */}
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                >
                  <Mic className="h-4 w-4" />
                  Record Audio
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="inline-flex items-center gap-2 rounded-lg bg-danger px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-danger/90"
                >
                  <Square className="h-4 w-4" />
                  Stop Recording
                </button>
              )}

              {/* Upload */}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold transition-colors hover:bg-secondary"
              >
                <Upload className="h-4 w-4" />
                Upload Audio
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={handleFileSelect}
              />

              {/* Clear */}
              {hasAudio && (
                <button
                  onClick={handleClear}
                  className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* File info */}
            {filename && (
              <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                <FileAudio className="h-3.5 w-3.5" />
                <span>{filename}</span>
                {duration !== null && (
                  <span className="text-muted-foreground/60">
                    · {fmtDuration(duration)}
                  </span>
                )}
                {isRecording && (
                  <span className="flex items-center gap-1 text-danger">
                    <span className="h-1.5 w-1.5 rounded-full bg-danger animate-pulse" />
                    Recording
                  </span>
                )}
              </div>
            )}
          </motion.div>

          {/* Waveform */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold">Waveform</h2>
              {hasAudio && (
                <button
                  onClick={togglePlayback}
                  className="inline-flex items-center gap-1.5 rounded-md bg-background px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary"
                >
                  {isPlaying ? (
                    <Pause className="h-3 w-3" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}
                  {isPlaying ? "Pause" : "Play"}
                </button>
              )}
            </div>
            <WaveformCanvas
              audioBuffer={audioBuffer}
              isPlaying={isPlaying}
              progress={playProgress}
            />
            {!hasAudio && (
              <p className="mt-2 text-xs text-muted-foreground/60 text-center">
                Record or upload an audio sample to see the waveform.
              </p>
            )}
          </motion.div>

          {/* Analysis Section */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.15 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold">Analysis</h2>
              {canAnalyze && (
                <button
                  onClick={handleAnalyze}
                  className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-1.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
                >
                  <Shield className="h-3.5 w-3.5" />
                  Run Detection
                </button>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg bg-background/50 px-4 py-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Model
                </p>
                <p className="text-sm font-medium mt-0.5">Morph V2</p>
              </div>
              <div className="rounded-lg bg-background/50 px-4 py-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Features
                </p>
                <p className="text-sm font-medium mt-0.5">132</p>
              </div>
              <div className="rounded-lg bg-background/50 px-4 py-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Status
                </p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  {analysisStatus === "analyzing" && (
                    <Loader2 className="h-3 w-3 animate-spin text-primary" />
                  )}
                  <p className="text-sm font-medium capitalize">
                    {analysisStatus}
                  </p>
                </div>
              </div>
              <div className="rounded-lg bg-background/50 px-4 py-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Risk Score
                </p>
                <p className="text-sm font-medium mt-0.5">
                  {result ? `${result.risk_score.toFixed(1)}%` : "—"}
                </p>
              </div>
            </div>
          </motion.div>

          {/* Result Card — prepared for backend */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h2 className="text-sm font-semibold mb-4">Result</h2>

            {!result ? (
              <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-10">
                <Shield className="h-8 w-8 text-muted-foreground/30 mb-3" />
                <p className="text-sm text-muted-foreground">
                  {analysisStatus === "waiting"
                    ? "Record or upload an audio sample to begin."
                    : analysisStatus === "analyzing"
                      ? "Analysis in progress..."
                      : "Ready for analysis."}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Verdict */}
                <div
                  className={`flex items-center gap-3 rounded-lg px-4 py-3 ${
                    result.verdict === "REAL"
                      ? "bg-primary/10 border border-primary/20"
                      : "bg-danger/10 border border-danger/20"
                  }`}
                >
                  {result.verdict === "REAL" ? (
                    <Shield className="h-5 w-5 text-primary" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-danger" />
                  )}
                  <div>
                    <p className="text-sm font-semibold">
                      {result.verdict === "REAL"
                        ? "REAL VOICE"
                        : "SYNTHETIC VOICE"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Risk Score: {result.risk_score.toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Details */}
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-lg bg-background/50 px-3 py-2">
                    <p className="text-[10px] text-muted-foreground uppercase">
                      Real Prob
                    </p>
                    <p className="text-sm font-medium">
                      {(result.real_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-lg bg-background/50 px-3 py-2">
                    <p className="text-[10px] text-muted-foreground uppercase">
                      Fake Prob
                    </p>
                    <p className="text-sm font-medium">
                      {(result.fake_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-lg bg-background/50 px-3 py-2">
                    <p className="text-[10px] text-muted-foreground uppercase">
                      Duration
                    </p>
                    <p className="text-sm font-medium">
                      {fmtDuration(result.duration)}
                    </p>
                  </div>
                  <div className="rounded-lg bg-background/50 px-3 py-2">
                    <p className="text-[10px] text-muted-foreground uppercase">
                      Model
                    </p>
                    <p className="text-sm font-medium uppercase">
                      {result.model_version}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>

          {/* Risk Timeline — empty state */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.25 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h2 className="text-sm font-semibold mb-3">Risk Timeline</h2>
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-8">
              <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">
                Window-by-window analysis
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                Appears after backend analysis
              </p>
            </div>
          </motion.div>
        </div>
      </PageContainer>
      <Footer />
    </>
  );
}
