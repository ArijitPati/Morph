"use client";

import { useCallback, useRef, useState } from "react";

interface UseWebRTCOptions {
  iceServers?: RTCIceServer[];
  /**
   * Called with every captured PCM chunk (mono, raw float32 little-endian)
   * from the detection tap. `sampleRate` is the AudioContext rate (16000
   * when the browser honours it, otherwise the native rate and the backend
   * resamples).
   *
   * The tap follows the REMOTE peer's audio (the caller being verified).
   *Until a remote audio track arrives via ontrack, it falls back to the
   * local mic as a demo loopback.
   */
  onAudioChunk?: (chunk: ArrayBuffer, sampleRate: number) => void;
  /** Emitted for each local ICE candidate — forward via signaling. */
  onIceCandidate?: (candidate: RTCIceCandidateInit) => void;
  /** Fires when ontrack delivers the remote party's real audio stream. */
  onRemoteTrack?: (stream: MediaStream) => void;
}

interface UseWebRTCReturn {
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  isConnected: boolean;
  hasRemoteAudio: boolean;
  isMuted: boolean;
  start: () => Promise<void>;
  stop: () => void;
  toggleMute: () => void;
  createOffer: () => Promise<RTCSessionDescriptionInit>;
  acceptOfferAndCreateAnswer: (
    offer: RTCSessionDescriptionInit,
  ) => Promise<RTCSessionDescriptionInit>;
  acceptAnswer: (answer: RTCSessionDescriptionInit) => Promise<void>;
  addIceCandidate: (candidate: RTCIceCandidateInit) => Promise<void>;
}

const PROCESSOR_BUFFER_SIZE = 4096;

/**
 * Copy mono Float32 samples as raw little-endian float32 PCM bytes.
 *
 * Full precision is sent deliberately: the V2 model is measurably
 * sensitive to 16-bit quantization (several real samples flip to FAKE
 * when round-tripped through int16), so the wire format preserves float32
 * exactly as the feature extractor consumes it.
 */
function float32ToPcmBytes(input: Float32Array): ArrayBuffer {
  return input.slice().buffer;
}

export function useWebRTC(options?: UseWebRTCOptions): UseWebRTCReturn {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [hasRemoteAudio, setHasRemoteAudio] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  const peerConnection = useRef<RTCPeerConnection | null>(null);
  const pendingCandidates = useRef<RTCIceCandidateInit[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const detectionStreamRef = useRef<MediaStream | null>(null);
  const attachDetectionRef = useRef<
    ((target: MediaStream, label: string) => void) | null
  >(null);

  const onAudioChunkRef = useRef(options?.onAudioChunk);
  onAudioChunkRef.current = options?.onAudioChunk;
  const onIceCandidateRef = useRef(options?.onIceCandidate);
  onIceCandidateRef.current = options?.onIceCandidate;
  const onRemoteTrackRef = useRef(options?.onRemoteTrack);
  onRemoteTrackRef.current = options?.onRemoteTrack;

  const iceServers = options?.iceServers ?? [
    { urls: "stun:stun.l.google.com:19302" },
  ];

  const start = useCallback(async () => {
    // Idempotent — joining/hosting twice must not orphan mic tracks or PCs.
    if (peerConnection.current) return;
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

      // Capture loop — taps a MediaStream as raw PCM. The AudioContext is
      // forced to 16000 Hz so the browser's continuous resampler (not a
      // per-chunk resample) delivers the model's native rate directly; the
      // backend then needs no resampling. If the browser can't honour 16k,
      // we send its actual rate and the backend falls back to resampling.
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      let ctx: AudioContext;
      try {
        ctx = new Ctx({ sampleRate: 16000 });
      } catch {
        // Fallback: browser refused 16 kHz — capture natively and let the
        // backend resample.
        ctx = new Ctx();
      }
      const processor = ctx.createScriptProcessor(
        PROCESSOR_BUFFER_SIZE,
        1,
        1,
      );

      processor.onaudioprocess = (event) => {
        const channel = event.inputBuffer.getChannelData(0);
        onAudioChunkRef.current?.(
          float32ToPcmBytes(channel),
          ctx.sampleRate,
        );
      };

      // A ScriptProcessorNode only fires while connected to the graph; a
      // zero-gain stub keeps it running without audible monitoring.
      const sink = ctx.createGain();
      sink.gain.value = 0;
      processor.connect(sink);
      sink.connect(ctx.destination);

      audioContextRef.current = ctx;
      processorRef.current = processor;

      // Detection tap — (re)points the capture graph at a MediaStream.
      // Primary source is the REMOTE peer (the caller being verified, wired
      // up in pc.ontrack below). Until a remote track arrives, tap the local
      // mic so the demo loopback still produces windows.
      const attachDetectionStream = (target: MediaStream, label: string) => {
        if (detectionStreamRef.current === target) return;
        sourceNodeRef.current?.disconnect();
        const next = ctx.createMediaStreamSource(target);
        next.connect(processor);
        sourceNodeRef.current = next;
        detectionStreamRef.current = target;
        console.info(`[detection] tap → ${label}`);
      };
      attachDetectionRef.current = attachDetectionStream;
      attachDetectionStream(stream, "local mic (demo fallback)");

      const pc = new RTCPeerConnection({ iceServers });
      peerConnection.current = pc;
      pendingCandidates.current = [];

      const remote = new MediaStream();
      setRemoteStream(remote);
      setHasRemoteAudio(false);

      stream.getTracks().forEach((track) => {
        pc.addTrack(track, stream);
      });

      pc.onicecandidate = (event) => {
        if (event.candidate) {
          onIceCandidateRef.current?.(event.candidate.toJSON());
        }
      };

      pc.ontrack = (event) => {
        event.streams[0].getTracks().forEach((track) => {
          remote.addTrack(track);
        });
        // Remote audio arrived — switch the detection tap from the local
        // demo fallback to the actual caller being verified.
        if (event.track.kind === "audio") {
          setHasRemoteAudio(true);
          attachDetectionStream(remote, "remote peer");
          onRemoteTrackRef.current?.(remote);
          console.info("[webrtc] ontrack: remote audio live");
        }
      };

      pc.oniceconnectionstatechange = () => {
        console.info("[webrtc] ICE connection state:", pc.iceConnectionState);
        setIsConnected(
          pc.iceConnectionState === "connected" ||
            pc.iceConnectionState === "completed",
        );
      };

      pc.onconnectionstatechange = () => {
        console.info("[webrtc] peer connection state:", pc.connectionState);
      };
    } catch (err) {
      console.error("Failed to start WebRTC:", err);
      throw err;
    }
  }, [iceServers]);

  const flushPendingCandidates = useCallback(async () => {
    const pc = peerConnection.current;
    if (!pc || !pc.remoteDescription) return;
    const queued = pendingCandidates.current.splice(0);
    for (const init of queued) {
      try {
        await pc.addIceCandidate(new RTCIceCandidate(init));
      } catch (err) {
        console.warn("[webrtc] dropping queued ICE candidate:", err);
      }
    }
  }, []);

  /** Caller side: create an SDP offer after local media is live. */
  const createOffer = useCallback(async () => {
    const pc = peerConnection.current;
    if (!pc) throw new Error("PeerConnection not started — call start() first");
    const offer = await pc.createOffer({ offerToReceiveAudio: true });
    await pc.setLocalDescription(offer);
    return pc.localDescription?.toJSON() ?? offer;
  }, []);

  /** Callee side: accept the caller's offer, return an answer to signal back. */
  const acceptOfferAndCreateAnswer = useCallback(
    async (offer: RTCSessionDescriptionInit) => {
      const pc = peerConnection.current;
      if (!pc)
        throw new Error("PeerConnection not started — call start() first");
      await pc.setRemoteDescription(new RTCSessionDescription(offer));
      await flushPendingCandidates();
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      return pc.localDescription?.toJSON() ?? answer;
    },
    [flushPendingCandidates],
  );

  /** Caller side: apply the callee's answer. */
  const acceptAnswer = useCallback(
    async (answer: RTCSessionDescriptionInit) => {
      const pc = peerConnection.current;
      if (!pc)
        throw new Error("PeerConnection not started — call start() first");
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
      await flushPendingCandidates();
    },
    [flushPendingCandidates],
  );

  /** Add a remote ICE candidate (queued if remote description isn't set yet). */
  const addIceCandidate = useCallback(
    async (candidate: RTCIceCandidateInit) => {
      const pc = peerConnection.current;
      if (!pc) return;
      if (!pc.remoteDescription) {
        pendingCandidates.current.push(candidate);
        return;
      }
      try {
        await pc.addIceCandidate(new RTCIceCandidate(candidate));
      } catch (err) {
        console.warn("[webrtc] addIceCandidate failed:", err);
      }
    },
    [],
  );

  const stop = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    void audioContextRef.current?.close();
    audioContextRef.current = null;
    detectionStreamRef.current = null;

    peerConnection.current?.close();
    peerConnection.current = null;
    pendingCandidates.current = [];
    attachDetectionRef.current = null;

    localStream?.getTracks().forEach((track) => track.stop());

    setLocalStream(null);
    setRemoteStream(null);
    setIsConnected(false);
    setHasRemoteAudio(false);
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
    hasRemoteAudio,
    isMuted,
    start,
    stop,
    toggleMute,
    createOffer,
    acceptOfferAndCreateAnswer,
    acceptAnswer,
    addIceCandidate,
  };
}