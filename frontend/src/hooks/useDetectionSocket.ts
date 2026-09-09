"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MorphWebSocket } from "@/services/websocket";
import type { WSMessage, DetectionResultMessage } from "@/types/websocket";

interface UseDetectionSocketOptions {
  url: string;
  onDetection?: (result: DetectionResultMessage["payload"]) => void;
  onError?: (error: string) => void;
}

interface UseDetectionSocketReturn {
  isConnected: boolean;
  isAnalyzing: boolean;
  lastResult: DetectionResultMessage["payload"] | null;
  connect: () => void;
  disconnect: () => void;
  sendAudioChunk: (chunk: ArrayBuffer, sampleRate: number) => void;
}

export function useDetectionSocket({
  url,
  onDetection,
  onError,
}: UseDetectionSocketOptions): UseDetectionSocketReturn {
const [isConnected, setIsConnected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastResult, setLastResult] =
    useState<DetectionResultMessage["payload"] | null>(null);

  const wsRef = useRef<MorphWebSocket | null>(null);
  const isConnectedRef = useRef(false);

  const handleMessage = useCallback(
    (message: WSMessage) => {
      switch (message.type) {
        case "detection_result": {
          const result = message.payload as DetectionResultMessage["payload"];
          setLastResult(result);
          setIsAnalyzing(false);
          onDetection?.(result);
          break;
        }
        case "status_update": {
          const status = message.payload as { status: string };
          setIsAnalyzing(status.status === "analyzing");
          break;
        }
        case "error": {
          const err = message.payload as { message: string };
          setIsAnalyzing(false);
          onError?.(err.message);
          break;
        }
      }
    },
    [onDetection, onError],
  );

  const connect = useCallback(() => {
    if (wsRef.current) return;

    const ws = new MorphWebSocket({ url });
    ws.onMessage(handleMessage);
    ws.onStatusChange((status) => {
      isConnectedRef.current = status === "connected";
      setIsConnected(status === "connected");
    });
    ws.connect();
    wsRef.current = ws;
  }, [url, handleMessage]);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
    wsRef.current = null;
    isConnectedRef.current = false;
    setIsConnected(false);
    setIsAnalyzing(false);
  }, []);

  const sendAudioChunk = useCallback(
    (chunk: ArrayBuffer, sampleRate: number) => {
      if (!wsRef.current || !isConnectedRef.current) return;

      setIsAnalyzing(true);

      // Convert ArrayBuffer to base64
      const bytes = new Uint8Array(chunk);
      let binary = "";
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64 = btoa(binary);

      wsRef.current.send({
        type: "audio_chunk",
        payload: {
          data: base64,
          sample_rate: sampleRate,
          timestamp: Date.now(),
        },
      });
    },
    [],
  );

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    isAnalyzing,
    lastResult,
    connect,
    disconnect,
    sendAudioChunk,
  };
}
