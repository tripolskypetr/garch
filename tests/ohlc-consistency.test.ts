import { describe, it, expect } from 'vitest';
import { validateCandles, yangZhangVariance } from '../src/utils.js';
import { Garch } from '../src/garch.js';
import { Egarch } from '../src/egarch.js';
import { GjrGarch } from '../src/gjr-garch.js';
import { HarRv } from '../src/har.js';
import { predict, backtestStats } from '../src/predict.js';
import type { Candle } from '../src/types.js';

// ── Deterministic synthetic GARCH(1,1) candles ──────────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function synthCandles(
  n: number,
  seed: number,
  opts: { omega?: number; alpha?: number; beta?: number; driftFrac?: number } = {},
): Candle[] {
  const { omega = 4e-6, alpha = 0.08, beta = 0.88, driftFrac = 0 } = opts;
  const rng = mulberry32(seed);
  let v = omega / (1 - alpha - beta);
  let price = 100;
  const candles: Candle[] = [];
  for (let i = 0; i < n; i++) {
    const r = driftFrac * Math.sqrt(v) + Math.sqrt(v) * gaussian(rng);
    const open = price;
    price = price * Math.exp(r);
    const close = price;
    const high = Math.max(open, close) * Math.exp(Math.abs(gaussian(rng)) * Math.sqrt(v) * 0.5);
    const low = Math.min(open, close) * Math.exp(-Math.abs(gaussian(rng)) * Math.sqrt(v) * 0.5);
    candles.push({ open, high, low, close, volume: 1000 });
    v = omega + alpha * r * r + beta * v;
  }
  return candles;
}

const good: Candle = { open: 100, high: 101, low: 99, close: 100.5, volume: 1 };

// ── validateCandles: open/close must lie inside [low, high] ─────
// Range-based estimators (Parkinson, Garman-Klass, Yang-Zhang) assume OHLC
// consistency; corrupted feeds (close > high) used to pass validation and
// silently distort every variance estimate.

describe('validateCandles OHLC body consistency', () => {
  it('rejects close above high', () => {
    expect(() => validateCandles([{ ...good, close: 102 }])).toThrow(/outside \[low, high\]/);
  });

  it('rejects close below low', () => {
    expect(() => validateCandles([{ ...good, close: 98 }])).toThrow(/outside \[low, high\]/);
  });

  it('rejects open above high', () => {
    expect(() => validateCandles([{ ...good, open: 101.5 }])).toThrow(/outside \[low, high\]/);
  });

  it('rejects open below low', () => {
    expect(() => validateCandles([{ ...good, open: 98.5 }])).toThrow(/outside \[low, high\]/);
  });

  it('reports the offending candle index', () => {
    expect(() => validateCandles([good, { ...good, close: 102 }])).toThrow(/candle 1/);
  });

  it('accepts exact equality (close === high, marubozu)', () => {
    expect(() => validateCandles([{ open: 99, high: 101, low: 99, close: 101, volume: 1 }])).not.toThrow();
  });

  it('accepts doji with H=L=O=C', () => {
    expect(() => validateCandles([{ open: 100, high: 100, low: 100, close: 100, volume: 1 }])).not.toThrow();
  });

  it('tolerates float-rounding dust below relative 1e-9', () => {
    const dusty: Candle = { open: 100, high: 100.5 * (1 - 1e-12), low: 99, close: 100.5, volume: 1 };
    expect(() => validateCandles([dusty])).not.toThrow();
  });

  it('model constructors reject corrupted candles', () => {
    const candles = synthCandles(300, 42);
    candles[150] = { ...candles[150], close: candles[150].high * 1.02 };
    expect(() => new Garch(candles)).toThrow(/outside \[low, high\]/);
    expect(() => new Egarch(candles)).toThrow(/outside \[low, high\]/);
    expect(() => new GjrGarch(candles)).toThrow(/outside \[low, high\]/);
    expect(() => new HarRv(candles)).toThrow(/outside \[low, high\]/);
  });

  it('predict rejects corrupted candles instead of silently mispricing the corridor', () => {
    const candles = synthCandles(300, 42);
    candles[150] = { ...candles[150], close: candles[150].high * 1.02 };
    expect(() => predict(candles, '1h')).toThrow(/outside \[low, high\]/);
  });
});

// ── Degenerate (constant-price) data must not produce ±Infinity ─
// EGARCH used to return omega = -Infinity (ln of zero initial variance)
// with converged: true. The variance floor keeps the fit finite; predict
// still degrades gracefully with reliable: false.

describe('constant-price degenerate data stays finite', () => {
  const flat: Candle[] = Array.from({ length: 300 }, () => ({
    open: 100, high: 100, low: 100, close: 100, volume: 1,
  }));

  it('Egarch fit returns finite params (regression: omega was -Infinity)', () => {
    const model = new Egarch(flat);
    expect(model.getInitialVariance()).toBeGreaterThan(0);
    const fit = model.fit();
    expect(Number.isFinite(fit.params.omega)).toBe(true);
    expect(Number.isFinite(fit.params.beta)).toBe(true);
    expect(Number.isFinite(fit.params.unconditionalVariance)).toBe(true);
  });

  it('Garch fit returns finite params and positive initial variance', () => {
    const model = new Garch(flat);
    expect(model.getInitialVariance()).toBeGreaterThan(0);
    const fit = model.fit();
    expect(Number.isFinite(fit.params.omega)).toBe(true);
    expect(Number.isFinite(fit.params.unconditionalVariance)).toBe(true);
  });

  it('GjrGarch fit returns finite params', () => {
    const fit = new GjrGarch(flat).fit();
    expect(Number.isFinite(fit.params.omega)).toBe(true);
  });

  it('predict on flat market degrades gracefully: zero-width band, reliable=false', () => {
    const p = predict(flat, '1h');
    expect(Number.isFinite(p.upperPrice)).toBe(true);
    expect(Number.isFinite(p.lowerPrice)).toBe(true);
    expect(p.reliable).toBe(false);
  });

  it('yangZhangVariance of flat candles is 0 (floor applied in models, not in the estimator)', () => {
    expect(yangZhangVariance(flat)).toBe(0);
  });
});

// ── Corridor behavior on a trending market ──────────────────────
// The band is symmetric around the current price (no drift term). Overall
// coverage is rescued by the empirical |z| calibration (drift inflates the
// standardized residuals, widening the band), but misses skew to the trend
// side. This test pins the overall walk-forward coverage; the asymmetry is
// a documented modeling limitation, not a bug.

describe('walk-forward coverage under drift', () => {
  it('68% corridor keeps sane aggregate coverage on strongly trending markets (drift = 0.5σ/bar)', () => {
    // Single-seed hit rates at n=38 swing ±15% from binomial noise alone;
    // aggregate over three seeds (n=114, 2σ ≈ ±8.7%) for a stable bound.
    let hits = 0;
    let total = 0;
    for (const seed of [321, 777, 12345]) {
      const candles = synthCandles(600, seed, { driftFrac: 0.5 });
      const stats = backtestStats(candles, '1h', 0.6827, { stride: 4 });
      hits += stats.hits;
      total += stats.total;
    }
    const coverage = (hits / total) * 100;
    expect(total).toBeGreaterThanOrEqual(90);
    expect(coverage, `coverage=${coverage.toFixed(1)}% (n=${total})`).toBeGreaterThanOrEqual(55);
    expect(coverage, `coverage=${coverage.toFixed(1)}% (n=${total})`).toBeLessThanOrEqual(82);
  }, 300_000);
});
