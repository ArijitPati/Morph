"use client";

import { useCallback, useRef, useState } from "react";

interface UseWebRTCOptions {
  iceServers?: RTCIceServer[];
}

interface UseWebRTCReturn {
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  isConnected: boolean;
  isMuted: boolean;
  start: () => Promise<void>;
  stop: () => void;
  toggleMute: () => void;
}

export function useWebRTC(options?: UseWebRTCOptions): UseWebRTCReturn {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  const peerConnection = useRef<RTCPeerConnection | null>(null);

  const iceServers = options?.iceServers ?? [
    { urls: "stun:stun.l.google.com:19302" },
  ];

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 16000,
        },
      });

      setLocalStream(stream);

      const pc = new RTCPeerConnection({ iceServers });
      peerConnection.current = pc;

      const remote = new MediaStream();
      setRemoteStream(remote);

      stream.getTracks().forEach((track) => {
        pc.addTrack(track, stream);
      });

      pc.ontrack = (event) => {
        event.streams[0].getTracks().forEach((track) => {
          remote.addTrack(track);
        });
      };

      pc.oniceconnectionstatechange = () => {
        setIsConnected(pc.iceConnectionState === "connected");
      };

      // Signaling (SDP exchange with backend) will be implemented
      // when the FastAPI backend is built.
    } catch (err) {
      console.error("Failed to start WebRTC:", err);
      throw err;
    }
  }, [iceServers]);

  const stop = useCallback(() => {
    peerConnection.current?.close();
    peerConnection.current = null;

    localStream?.getTracks().forEach((track) => track.stop());

    setLocalStream(null);
    setRemoteStream(null);
    setIsConnected(false);
    setIsMuted(false);
  }, [localStream]);

  const toggleMute = useCallback(() => {
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsMuted(!audioTrack.enabled);
      }
    }
  }, [localStream]);

  return {
    localStream,
    remoteStream,
    isConnected,
    isMuted,
    start,
    stop,
    toggleMute,
  };
}
