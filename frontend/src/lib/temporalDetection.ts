/**
 * Temporal decision layer for live voice-clone detection.
 *
 * Sits AFTER the existing per-window detection result is received and
 * decides when a REAL → FAKE transition deserves a user alert. Operates
 * only on the raw `fake_probability` of each completed window — it never
 * touches the model, audio, WebRTC, signaling, or the detection protocol.
 *
 * Decision pipeline per window:
 *   1. EWMA smoothing of the raw fake probability (alpha = 0.4).
 *   2. Entry gate: 3 consecutive raw windows > 0.75 AND smoothed >= 0.82.
 *   3. Hysteresis: once FAKE, stay FAKE until smoothed <= 0.55.
 *   4. `enteredFake` is true exactly once per REAL → FAKE transition.
 */

export const TEMPORAL_ALPHA = 0.4;
export const FAKE_ENTER_THRESHOLD = 0.82;
export const FAKE_EXIT_THRESHOLD = 0.55;
export const ENTRY_RAW_THRESHOLD = 0.75;
export const ENTRY_CONSECUTIVE_WINDOWS = 3;

export type TemporalState = "REAL" | "FAKE";

export interface TemporalDecision {
  /** Current EWMA-smoothed fake probability. */
  smoothedProbability: number;
  /** Latched temporal state (independent of any UI alert state). */
  state: TemporalState;
  /** True only on the single processing call that transitions REAL → FAKE. */
  enteredFake: boolean;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export class TemporalDetector {
  private smoothed: number | null = null;
  private state: TemporalState = "REAL";
  private consecutiveHigh = 0;

  /**
   * Feed one window's raw fake probability through the decision layer.
   * Deterministic and side-effect free apart from the detector's own
   * state — safe to call exactly once per received detection result.
   */
  process(rawFakeProbability: number): TemporalDecision {
    const raw = clamp01(rawFakeProbability);
    this.smoothed =
      this.smoothed === null
        ? raw
        : TEMPORAL_ALPHA * raw + (1 - TEMPORAL_ALPHA) * this.smoothed;

    if (this.state === "REAL") {
      this.consecutiveHigh = raw > ENTRY_RAW_THRESHOLD ? this.consecutiveHigh + 1 : 0;
      if (
        this.consecutiveHigh >= ENTRY_CONSECUTIVE_WINDOWS &&
        this.smoothed >= FAKE_ENTER_THRESHOLD
      ) {
        this.state = "FAKE";
        return { smoothedProbability: this.smoothed, state: "FAKE", enteredFake: true };
      }
      return { smoothedProbability: this.smoothed, state: "REAL", enteredFake: false };
    }

    // FAKE state — hysteresis: only a smoothed score at or below the exit
    // threshold returns to REAL. Values between the thresholds hold FAKE.
    if (this.smoothed <= FAKE_EXIT_THRESHOLD) {
      this.state = "REAL";
      this.consecutiveHigh = 0;
      return { smoothedProbability: this.smoothed, state: "REAL", enteredFake: false };
    }
    return { smoothedProbability: this.smoothed, state: "FAKE", enteredFake: false };
  }

  /** Drop all temporal state (call ended / monitoring stopped). */
  reset(): void {
    this.smoothed = null;
    this.state = "REAL";
    this.consecutiveHigh = 0;
  }

  getState(): TemporalState {
    return this.state;
  }

  getSmoothed(): number | null {
    return this.smoothed;
  }
}
