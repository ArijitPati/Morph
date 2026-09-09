"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { WindowResult, AggregationResult } from "@/types/detection";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010";
const WS_BASE = API_BASE.replace(/^http/, "ws");

export interface UseLiveDetectionOptions {
  windowSec?: number;
  hopSec?: number;
  aggregation?: "mean" | "median" | "max" | "vote";
  autoConnect?: boolean;
  onWindow?: (w: WindowResult) => void;
  onAggregation?: (agg: AggregationResult) => void;
}

export interface UseLiveDetectionReturn {
  isConnected: boolean;
  isStreaming: boolean;
  isSupported: boolean;
  windows: WindowResult[];
  aggregation: AggregationResult | null;
  error: string | null;
  startStreaming: () => Promise<void>;
  stopStreaming: () => void;
  flush: () => void;
  reset: () => void;
}

function float32ToBase64(buffer: Float32Array): string {
  const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export function useLiveDetection(options: UseLiveDetectionOptions = {}): UseLiveDetectionReturn {
  const {
    windowSec = 4.0,
    hopSec,
    aggregation = "mean",
    onWindow,
    onAggregation,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [windows, setWindows] = useState<WindowResult[]>([]);
  const [aggregationRes, setAggregationRes] = useState<AggregationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const isSupported = typeof window !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  const reset = useCallback(() => {
    setWindows([]);
    setAggregationRes(null);
    setError(null);
  }, []);

  const connectWs = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      const hop = hopSec ?? windowSec;
      const url = `${WS_BASE}/ws/detect?window_sec=${windowSec}&hop_sec=${hop}&aggregation=${aggregation}`;
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        wsRef.current = ws;
        resolve(ws);
      };
      ws.onerror = () => {
        reject(new Error("WebSocket connection failed"));
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "detection_result") {
            const w = msg.payload as WindowResult;
            setWindows((prev) => [...prev, w]);
            onWindow?.(w);
          } else if (msg.type === "aggregation_update") {
            const agg = msg.payload as AggregationResult;
            setAggregationRes(agg);
            onAggregation?.(agg);
          } else if (msg.type === "error") {
            setError(msg.payload?.message ?? "Unknown error");
          }
        } catch {
          // ignore parse errors
        }
      };
      ws.onclose = () => {
        setIsConnected(false);
      };
    });
  }, [windowSec, hopSec, aggregation, onWindow, onAggregation]);

  const startStreaming = useCallback(async () => {
    if (isStreaming) return;
    setError(null);

    // Connect WS first
    let ws: WebSocket;
    try {
      ws = await connectWs();
    } catch (e) {
      setError((e as Error).message);
      return;
    }

    // Get microphone
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;
    } catch {
      setError("Could not access microphone.");
      ws.close();
      return;
    }

    // AudioContext at 16 kHz
    const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({
      sampleRate: 16000,
    });
    audioContextRef.current = ctx;

    // Some browsers ignore sampleRate hint — info only (server resamples)
    if (ctx.sampleRate !== 16000) {
      console.info(`[live-detect] AudioContext sampleRate ${ctx.sampleRate} != 16000, will resample on server`);
    }

    const source = ctx.createMediaStreamSource(stream);
    // ScriptProcessorNode deprecated but retained for compat (see useWebRTC.ts); one-time browser warning is expected.
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      // Copy and encode as Float32 base64
      const copy = new Float32Array(input.length);
      copy.set(input);
      const b64 = float32ToBase64(copy);
      ws.send(
        JSON.stringify({
          type: "audio_chunk",
          payload: {
            data: b64,
            sample_rate: ctx.sampleRate,
            encoding: "pcm_f32",
            timestamp: Date.now(),
          },
        }),
      );
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    setIsStreaming(true);
  }, [isStreaming, connectWs]);

  const stopStreaming = useCallback(() => {
    // Disconnect audio graph
    try {
      processorRef.current?.disconnect();
      processorRef.current = null;
    } catch {}
    try {
      audioContextRef.current?.close();
      audioContextRef.current = null;
    } catch {}
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    // Close WS
    try {
      wsRef.current?.close();
    } catch {}
    wsRef.current = null;

    setIsStreaming(false);
    setIsConnected(false);
  }, []);

  const flush = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "flush", payload: {} }));
    }
  }, []);

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  return {
    isConnected,
    isStreaming,
    isSupported,
    windows,
    aggregation: aggregationRes,
    error,
    startStreaming,
    stopStreaming,
    flush,
    reset,
  };
}
