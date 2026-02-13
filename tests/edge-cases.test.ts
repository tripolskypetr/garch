import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  sampleVarianceWithMean,
  EXPECTED_ABS_NORMAL,
  type Candle,
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

function makeCandles(n: number, seed = 42): Candle[] {
  const rng = (() => {
    let state = seed;
    return () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
  })();

  const candles: Candle[] = [];
  let close = 100;
  for (let i = 0; i < n; i++) {
    const r = (rng() - 0.5) * 0.04;
    const newClose = close * Math.exp(r);
    candles.push({
      open: close,
      high: Math.max(close, newClose) * (1 + rng() * 0.005),
      low: Math.min(close, newClose) * (1 - rng() * 0.005),
      close: newClose,
      volume: Math.round(rng() * 10000),
    });
    close = newClose;
  }
  return candles;
}

function garchParams(omega: number, alpha: number, beta: number) {
  const persistence = alpha + beta;
  return {
    omega, alpha, beta, persistence,
    unconditionalVariance: omega / (1 - persistence),
    annualizedVol: Math.sqrt((omega / (1 - persistence)) * 252) * 100,
  };
}

function egarchParams(omega: number, alpha: number, gamma: number, beta: number) {
  return {
    omega, alpha, gamma, beta,
    persistence: beta,
    unconditionalVariance: Math.exp(omega / (1 - beta)),
    annualizedVol: Math.sqrt(Math.exp(omega / (1 - beta)) * 252) * 100,
    leverageEffect: gamma,
  };
}

// ── Candle end-to-end ───────────────────────────────────────

describe('Candle[] end-to-end', () => {
  it('GARCH: construct → fit → forecast', () => {
    const candles = makeCandles(200);
    const model = new Garch(candles);
    const result = model.fit();
    const fc = model.forecast(result.params, 10);

    expect(result.params.persistence).toBeLessThan(1);
    expect(result.diagnostics.converged).toBe(true);
    expect(fc.variance).toHaveLength(10);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });

  it('EGARCH: construct → fit → forecast', () => {
    const candles = makeCandles(200);
    const model = new Egarch(candles);
    const result = model.fit();
    const fc = model.forecast(result.params, 10);

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });

  it('candle returns use only close prices (OHLV ignored)', () => {
    const candles1: Candle[] = [
      { open: 99, high: 102, low: 98, close: 100, volume: 1000 },
      { open: 100, high: 112, low: 99, close: 110, volume: 1200 },
    ];
    const candles2: Candle[] = [
      { open: 50, high: 200, low: 10, close: 100, volume: 5 },
      { open: 80, high: 150, low: 50, close: 110, volume: 99999 },
    ];

    const r1 = calculateReturns(candles1);
    const r2 = calculateReturns(candles2);

    expect(r1[0]).toBe(r2[0]);
  });
});

// ── Egarch accessors ────────────────────────────────────────

describe('Egarch accessors', () => {
  it('getReturns() matches calculateReturnsFromPrices', () => {
    const prices = makePrices(55);
    const model = new Egarch(prices);
    const returns = model.getReturns();
    const expected = calculateReturnsFromPrices(prices);

    expect(returns).toHaveLength(expected.length);
    for (let i = 0; i < returns.length; i++) {
      expect(returns[i]).toBe(expected[i]);
    }
  });

  it('getInitialVariance() equals sampleVariance(returns)', () => {
    const prices = makePrices(55);
    const model = new Egarch(prices);

    expect(model.getInitialVariance()).toBe(sampleVariance(model.getReturns()));
  });
});

// ── forecast(params, 0) ─────────────────────────────────────

describe('forecast edge cases', () => {
  it('forecast(params, 0) still returns one step', () => {
    const model = new Garch(makePrices(55));
    const result = model.fit();
    const fc0 = model.forecast(result.params, 0);
    const fc1 = model.forecast(result.params, 1);

    // Implementation: one-step push is unconditional, so steps=0 ≡ steps=1
    expect(fc0.variance).toHaveLength(1);
    expect(fc0.variance[0]).toBe(fc1.variance[0]);
  });

  it('EGARCH forecast(params, 0) still returns one step', () => {
    const model = new Egarch(makePrices(55));
    const result = model.fit();
    const fc0 = model.forecast(result.params, 0);

    expect(fc0.variance).toHaveLength(1);
    expect(fc0.variance[0]).toBeGreaterThan(0);
  });
});

// ── Scalar utils edge cases ─────────────────────────────────

describe('Scalar utils edge cases', () => {
  it('sampleVariance with single element = x²', () => {
    expect(sampleVariance([0.05])).toBeCloseTo(0.05 ** 2, 14);
    expect(sampleVariance([-0.03])).toBeCloseTo(0.03 ** 2, 14);
  });

  it('sampleVarianceWithMean with 2 elements (Bessel n−1 = 1)', () => {
    const returns = [0.01, 0.03];
    const mean = 0.02;
    const expected = ((0.01 - mean) ** 2 + (0.03 - mean) ** 2) / 1;

    expect(sampleVarianceWithMean(returns)).toBeCloseTo(expected, 14);
  });

  it('calculateReturnsFromPrices with exactly 2 prices → 1 return', () => {
    const returns = calculateReturnsFromPrices([100, 110]);

    expect(returns).toHaveLength(1);
    expect(returns[0]).toBeCloseTo(Math.log(1.1), 14);
  });
});

// ── Degenerate GARCH params ─────────────────────────────────

describe('Degenerate GARCH params', () => {
  it('α = 0 (pure AR(1)): σ²ₜ = ω + β·σ²ₜ₋₁', () => {
    const model = new Garch(makePrices(55));
    const params = garchParams(0.00001, 0, 0.9);
    const variance = model.getVarianceSeries(params);
    const returns = model.getReturns();

    for (let i = 1; i < returns.length; i++) {
      const expected = 0.00001 + 0.9 * variance[i - 1];
      expect(variance[i]).toBeCloseTo(expected, 12);
    }
  });

  it('β = 0 (pure ARCH(1)): σ²ₜ = ω + α·ε²ₜ₋₁, no memory', () => {
    const model = new Garch(makePrices(55));
    const params = garchParams(0.0001, 0.3, 0);
    const variance = model.getVarianceSeries(params);
    const returns = model.getReturns();

    for (let i = 1; i < returns.length; i++) {
      const expected = 0.0001 + 0.3 * returns[i - 1] ** 2;
      expect(variance[i]).toBeCloseTo(expected, 12);
    }
  });

  it('β = 0: variance[i] independent of variance[i−2]', () => {
    const model = new Garch(makePrices(55));
    const returns = model.getReturns();
    const omega = 0.0001, alpha = 0.3;
    const params = garchParams(omega, alpha, 0);
    const variance = model.getVarianceSeries(params);

    // Changing variance[5] should NOT affect variance[7]
    // (each step depends only on previous return, not previous variance)
    for (let i = 2; i < returns.length; i++) {
      const fromScratch = omega + alpha * returns[i - 1] ** 2;
      expect(variance[i]).toBeCloseTo(fromScratch, 12);
    }
  });
});

// ── Degenerate EGARCH params ────────────────────────────────

describe('Degenerate EGARCH params', () => {
  it('positive γ (inverse leverage): valid variances', () => {
    const model = new Egarch(makePrices(55));
    const params = egarchParams(-0.1, 0.1, 0.05, 0.95);
    const variance = model.getVarianceSeries(params);

    for (const v of variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('γ = 0, α = 0: pure AR(1) in log-variance', () => {
    const model = new Egarch(makePrices(55));
    const params = egarchParams(-0.5, 0, 0, 0.95);
    const variance = model.getVarianceSeries(params);
    const initVar = model.getInitialVariance();

    let logVar = Math.log(initVar);
    for (let i = 1; i < model.getReturns().length; i++) {
      logVar = -0.5 + 0.95 * logVar;
      logVar = Math.max(-50, Math.min(50, logVar));
      expect(variance[i]).toBeCloseTo(Math.exp(logVar), 10);
    }
  });
});
