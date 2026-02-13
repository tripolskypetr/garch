import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  calibrateGarch,
  calibrateEgarch,
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

describe('Regression snapshots', () => {
  const prices = makePrices(100);

  it('GARCH params on makePrices(100)', () => {
    const r = calibrateGarch(prices);

    expect(r.params.omega).toBeCloseTo(7.605233309723745e-6, 10);
    expect(r.params.alpha).toBeCloseTo(1.2288592727602956e-10, 10);
    expect(r.params.beta).toBeCloseTo(0.9313297809134129, 8);
    expect(r.diagnostics.logLikelihood).toBeCloseTo(400.9767762200383, 4);
  });

  it('EGARCH params on makePrices(100)', () => {
    const r = calibrateEgarch(prices);

    expect(r.params.omega).toBeCloseTo(-17.143867066920855, 4);
    expect(r.params.alpha).toBeCloseTo(-0.19227942475252052, 6);
    expect(r.params.gamma).toBeCloseTo(0.214065630362474, 6);
    expect(r.params.beta).toBeCloseTo(-0.8711189887965891, 6);
    expect(r.diagnostics.logLikelihood).toBeCloseTo(404.7371866985035, 4);
  });
});

// ── Cross-model consistency ─────────────────────────────────

describe('Cross-model consistency', () => {
  it('GARCH and EGARCH: similar unconditional variance on same data', () => {
    const prices = makePrices(500);
    const gUV = calibrateGarch(prices).params.unconditionalVariance;
    const eUV = calibrateEgarch(prices).params.unconditionalVariance;

    // Within same order of magnitude
    const ratio = gUV / eUV;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });

  it('forecast annualized vol converges to params.annualizedVol', () => {
    const model = new Garch(makePrices(200));
    const result = model.fit();
    const fc = model.forecast(result.params, 500);

    const relErr = Math.abs(fc.annualized[499] - result.params.annualizedVol)
      / result.params.annualizedVol;

    expect(relErr).toBeLessThan(0.01);
  });

  it('EGARCH forecast annualized vol converges to params.annualizedVol', () => {
    const prices = generateEgarchData(500, -0.1, 0.1, -0.05, 0.9, 42);
    const model = new Egarch(prices);
    const result = model.fit();
    const fc = model.forecast(result.params, 500);

    const relErr = Math.abs(fc.annualized[499] - result.params.annualizedVol)
      / result.params.annualizedVol;

    expect(relErr).toBeLessThan(0.05);
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
      const fc = model.forecast(result.params, 100);
      const { omega, alpha, beta } = result.params;
      const unconditional = omega / (1 - alpha - beta);

      const relErr = Math.abs(fc.variance[99] - unconditional) / unconditional;
      expect(relErr).toBeLessThan(0.1);
    }
  });
});
