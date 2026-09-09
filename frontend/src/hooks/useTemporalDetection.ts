"use client";

import { useCallback, useMemo, useRef } from "react";
import { TemporalDetector, type TemporalDecision } from "@/lib/temporalDetection";

interface UseTemporalDetectionReturn {
  /** Feed one window's raw fake probability; returns the temporal decision. */
  process: (rawFakeProbability: number) => TemporalDecision;
  /** Clear all temporal state — call when the call/monitoring ends. */
  reset: () => void;
}

/**
 * React binding for the temporal detection state machine.
 *
 * The detector lives in a ref (never in render state), so repeated renders
 * or handler re-registrations can never fork or duplicate its state — each
 * received detection result is processed exactly once, in order.
 */
export function useTemporalDetection(): UseTemporalDetectionReturn {
  const detectorRef = useRef<TemporalDetector | null>(null);
  if (!detectorRef.current) {
    detectorRef.current = new TemporalDetector();
  }

  const process = useCallback(
    (rawFakeProbability: number): TemporalDecision =>
      detectorRef.current!.process(rawFakeProbability),
    [],
  );
  const reset = useCallback(() => {
    detectorRef.current!.reset();
  }, []);

  return useMemo(() => ({ process, reset }), [process, reset]);
}
