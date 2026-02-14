import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  GjrGarch,
  calibrateGarch,
  calibrateEgarch,
  calibrateGjrGarch,
  EXPECTED_ABS_NORMAL,
} from '../src/index.js';

// ── helpers ──────────────────────────────────────────────────

function makePrices(n: number, seed = 12345): number[] {
  const prices = [100];
  let state = seed;
  for (let i = 1; i < n; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    const r = ((state / 0x7fffffff) - 0.5) * 0.04;
    prices.push(prices[i - 1] * Math.exp(r));
  }
  return prices;
}

function lcg(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function seededRandn(rng: () => number) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function generateEgarchData(
  n: number, omega: number, alpha: number, gamma: number, beta: number, seed = 42,
): number[] {
  const rng = lcg(seed);
  let logVar = omega / (1 - beta);
  let v = Math.exp(logVar);
  const prices = [100];
  for (let i = 0; i < n; i++) {
    const z = seededRandn(rng);
    prices.push(prices[prices.length - 1] * Math.exp(Math.sqrt(v) * z));
    logVar = omega + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL) + gamma * z + beta * logVar;
    v = Math.exp(logVar);
  }
  return prices;
}

// ── Regression snapshots ────────────────────────────────────
// Deterministic inputs → exact outputs. Guards against silent regressions.
// Values updated for Student-t MLE.

describe('Regression snapshots', () => {
  const prices = makePrices(100);

  it('GARCH params on makePrices(100)', () => {
    const r = calibrateGarch(prices);

    // Student-t MLE exact snapshot values
    expect(r.params.omega).toBeCloseTo(1.0885186560315161e-8, 12);
    expect(r.params.alpha).toBeCloseTo(1.1992837185889667e-10, 14);
    expect(r.params.beta).toBeCloseTo(0.9998999998095188, 6);
    expect(r.params.df).toBeCloseTo(33.77, 0);
    expect(r.params.persistence).toBeLessThan(1);
    expect(r.diagnostics.logLikelihood).toBeCloseTo(309.2177927685958, 2);
    expect(r.diagnostics.converged).toBe(true);
  });

  it('EGARCH params on makePrices(100)', () => {
    const r = calibrateEgarch(prices);

    // Student-t MLE exact snapshot values
    expect(r.params.omega).toBeCloseTo(-17.154545344556958, 2);
    expect(r.params.alpha).toBeCloseTo(-0.186113131218345, 4);
    expect(r.params.gamma).toBeCloseTo(0.21493459570073092, 4);
    expect(r.params.beta).toBeCloseTo(-0.8745672428204935, 4);
    expect(r.params.df).toBeCloseTo(100, 0);
    expect(r.diagnostics.logLikelihood).toBeCloseTo(313.4950425741637, 2);
    expect(r.diagnostics.converged).toBe(true);
  });
});

// ── Cross-model consistency ─────────────────────────────────

describe('Cross-model consistency', () => {
  it('GARCH and EGARCH: similar unconditional variance on same data', () => {
    const prices = makePrices(500);
    const gUV = calibrateGarch(prices).params.unconditionalVariance;
    const eUV = calibrateEgarch(prices).params.unconditionalVariance;

    const ratio = gUV / eUV;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });

  it('GARCH and GJR-GARCH: similar unconditional variance on same data', () => {
    const prices = makePrices(500);
    const gUV = calibrateGarch(prices).params.unconditionalVariance;
    const gjrUV = calibrateGjrGarch(prices).params.unconditionalVariance;

    const ratio = gUV / gjrUV;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });

  it('forecast annualized vol converges to params.annualizedVol', () => {
    const model = new Garch(makePrices(200));
    const result = model.fit();
    const { alpha, beta } = result.params;
    const persistence = alpha + beta;

    const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);

    const relErr = Math.abs(fc.annualized[steps - 1] - result.params.annualizedVol)
      / result.params.annualizedVol;

    expect(relErr).toBeLessThan(0.01);
  });

  it('EGARCH forecast annualized vol converges to params.annualizedVol', () => {
    const prices = generateEgarchData(500, -0.1, 0.1, -0.05, 0.9, 42);
    const model = new Egarch(prices);
    const result = model.fit();
    const persistence = Math.abs(result.params.beta);

    const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);

    const relErr = Math.abs(fc.annualized[steps - 1] - result.params.annualizedVol)
      / result.params.annualizedVol;

    expect(relErr).toBeLessThan(0.01);
  });

  it('GJR-GARCH forecast annualized vol converges to params.annualizedVol', () => {
    const model = new GjrGarch(makePrices(200));
    const result = model.fit();
    const { alpha, gamma, beta } = result.params;
    const persistence = alpha + gamma / 2 + beta;

    const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);

    const relErr = Math.abs(fc.annualized[steps - 1] - result.params.annualizedVol)
      / result.params.annualizedVol;

    expect(relErr).toBeLessThan(0.01);
  });
});

// ── Property-based (fuzz) ───────────────────────────────────
// For any valid prices: invariants must hold.

describe('Property-based invariants', () => {
  const seeds = [1, 42, 123, 999, 7777];

  it('GARCH: persistence < 1, ω > 0, LL finite for diverse seeds', () => {
    for (const seed of seeds) {
      const result = calibrateGarch(makePrices(100, seed));

      expect(result.params.persistence).toBeLessThan(1);
      expect(result.params.omega).toBeGreaterThan(0);
      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    }
  });

  it('GARCH: all conditional variances > 0 for diverse seeds', () => {
    for (const seed of seeds) {
      const prices = makePrices(100, seed);
      const model = new Garch(prices);
      const result = model.fit();
      const variance = model.getVarianceSeries(result.params);

      for (const v of variance) {
        expect(v).toBeGreaterThan(0);
      }
    }
  });

  it('EGARCH: LL finite, variances positive for diverse seeds', () => {
    for (const seed of seeds) {
      const prices = makePrices(100, seed);
      const model = new Egarch(prices);
      const result = model.fit();

      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);

      const variance = model.getVarianceSeries(result.params);
      for (const v of variance) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('GARCH forecast converges for diverse seeds', () => {
    for (const seed of seeds) {
      const prices = makePrices(100, seed);
      const model = new Garch(prices);
      const result = model.fit();
      const { omega, alpha, beta } = result.params;
      const persistence = alpha + beta;

      // Skip seeds where persistence is too close to 1 (unconditional variance unstable)
      if (persistence > 0.999) continue;

      const unconditional = omega / (1 - persistence);
      const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
      const fc = model.forecast(result.params, steps);

      const relErr = Math.abs(fc.variance[steps - 1] - unconditional) / unconditional;
      expect(relErr).toBeLessThan(0.1);
    }
  });
});
