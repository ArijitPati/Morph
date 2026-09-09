/**
 * Unit tests for the temporal detection state machine (cases A–H).
 *
 * No test runner is vendored in this repo, so this file is intentionally
 * dependency-free (only `node:assert/strict`) and is executed by compiling
 * it alongside `temporalDetection.ts` with the project's TypeScript and
 * running the emitted JS with node:
 *
 *   npx tsc src/lib/temporalDetection.ts src/lib/temporalDetection.test.ts \
 *     --outDir <tmp> --module commonjs --target es2020 --skipLibCheck
 *   node <tmp>/temporalDetection.test.js
 */

import { strict as assert } from "node:assert";
import { TemporalDetector, type TemporalState } from "./temporalDetection";

function feed(detector: TemporalDetector, raws: number[]) {
  return raws.map((raw) => detector.process(raw));
}

function countEntries(decisions: { enteredFake: boolean }[]): number {
  return decisions.filter((d) => d.enteredFake).length;
}

let passed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    passed += 1;
    console.log(`ok - ${name}`);
  } catch (err) {
    console.error(`FAIL - ${name}`);
    throw err;
  }
}

// A. Single high prediction does NOT immediately alert.
test("A: single 0.95 → no alert", () => {
  const d = new TemporalDetector();
  const [decision] = feed(d, [0.95]);
  assert.equal(decision.enteredFake, false);
  assert.equal(decision.state, "REAL");
  assert.equal(decision.smoothedProbability, 0.95);
});

// B. Three consecutive raw windows > 0.75 plus EWMA evidence enters FAKE.
test("B: 0.9 x3 → FAKE on third window", () => {
  const d = new TemporalDetector();
  const decisions = feed(d, [0.9, 0.9, 0.9]);
  assert.equal(decisions[0].enteredFake, false);
  assert.equal(decisions[1].enteredFake, false);
  assert.equal(decisions[2].enteredFake, true);
  assert.equal(decisions[2].state, "FAKE");
});

// B2. Counter resets when raw evidence breaks before three in a row.
test("B2: broken streak resets the consecutive counter", () => {
  const d = new TemporalDetector();
  const decisions = feed(d, [0.9, 0.9, 0.2, 0.9, 0.9, 0.9]);
  assert.equal(countEntries(decisions), 1);
  assert.equal(decisions[5].enteredFake, true);
});

// C. Sustained FAKE produces exactly ONE transition.
test("C: six high windows → exactly one enteredFake", () => {
  const d = new TemporalDetector();
  const decisions = feed(d, [0.9, 0.92, 0.94, 0.91, 0.93, 0.95]);
  assert.equal(countEntries(decisions), 1);
  assert.equal(decisions[2].enteredFake, true);
  assert.equal(d.getState(), "FAKE");
});

// D. Hysteresis: mid-band values do not exit FAKE.
test("D: smoothed 0.55–0.82 holds FAKE", () => {
  const d = new TemporalDetector();
  feed(d, [0.95, 0.95, 0.95]);
  assert.equal(d.getState(), "FAKE");
  const decisions = feed(d, [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]);
  for (const decision of decisions) {
    assert.equal(decision.state, "FAKE");
    assert.equal(decision.enteredFake, false);
  }
});

// E. Sustained low values eventually return to REAL (not instantly).
test("E: 0.1 stream exits FAKE, but not on the first window", () => {
  const d = new TemporalDetector();
  feed(d, [0.95, 0.95, 0.95]);
  const first = d.process(0.1);
  assert.equal(first.state, "FAKE"); // 0.4*0.1+0.6*0.95 = 0.61 > 0.55
  let state: TemporalState = first.state;
  for (let i = 0; i < 20 && state !== "REAL"; i++) {
    state = d.process(0.1).state;
  }
  assert.equal(state, "REAL");
});

// F. After REAL, a later sustained sequence triggers exactly one new alert.
test("F: re-entry after exit produces exactly one new transition", () => {
  const d = new TemporalDetector();
  const first = feed(d, [0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
  assert.equal(countEntries(first), 1);
  assert.equal(d.getState(), "REAL");
  const second = feed(d, [0.93, 0.93, 0.93, 0.93]);
  assert.equal(countEntries(second), 1);
  assert.equal(d.getState(), "FAKE");
});

// G. Closing the modal (no detector call) does not reset detection state.
test("G: no detector interaction preserves FAKE without re-alerting", () => {
  const d = new TemporalDetector();
  feed(d, [0.95, 0.95, 0.95]);
  // Simulate "Continue Call": UI-only close, detector untouched.
  const next = d.process(0.93);
  assert.equal(d.getState(), "FAKE");
  assert.equal(next.enteredFake, false);
});

// H. reset() clears all temporal state.
test("H: reset returns to pristine REAL", () => {
  const d = new TemporalDetector();
  feed(d, [0.95, 0.95, 0.95]);
  assert.equal(d.getState(), "FAKE");
  d.reset();
  assert.equal(d.getState(), "REAL");
  assert.equal(d.getSmoothed(), null);
  const [decision] = feed(d, [0.95]);
  assert.equal(decision.enteredFake, false);
  assert.equal(decision.state, "REAL");
});

console.log(`\n${passed} tests passed.`);
