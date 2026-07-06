import { describe, it, expect } from 'vitest';
import { Garch } from '../src/garch.js';
import { GjrGarch } from '../src/gjr-garch.js';
import { Egarch } from '../src/egarch.js';
import { HarRv } from '../src/har.js';
import { NoVaS } from '../src/novas.js';
import { predict } from '../src/predict.js';
import { profileStudentTDf } from '../src/utils.js';
import type { Candle } from '../src/types.js';

/**
 * Every calibration must be scale-equivariant: multiplying every return by
 * k (a stablecoin pair, an FX minor, a shorter candle interval) must give
 * the same shape parameters (α/β/γ/df/lag weights) and scale the level
 * parameters (ω, β₀, annualizedVol) by the appropriate power of k.
 *
 * The models fit in normalized space (returns scaled to unit variance
 * internally), so this holds by construction — the optimizer follows the
 * same path at any data scale. Residual tolerances only absorb float
 * rounding from the price→return conversion.
 */

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Simulate GARCH(1,1) returns with Gaussian innovations. */
function simGarchReturns(
  n: number,
  omega: number,
  alpha: number,
  beta: number,
  seed: number,
): number[] {
  const rng = mulberry32(seed);
  let v = omega / (1 - alpha - beta);
  let r = Math.sqrt(v) * randn(rng);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    v = omega + alpha * r * r + beta * v;
    r = Math.sqrt(v) * randn(rng);
    out.push(r);
  }
  return out;
}

function pricesFromReturns(returns: number[], p0 = 100): number[] {
  const prices = [p0];
  for (const r of returns) {
    prices.push(prices[prices.length - 1] * Math.exp(r));
  }
  return prices;
}

/** Deterministic OHLC candles from a return series (wick = |r|/2 in log space). */
function candlesFromReturns(returns: number[], seed: number, p0 = 100): Candle[] {
  const rng = mulberry32(seed);
  const candles: Candle[] = [{ open: p0, high: p0, low: p0, close: p0, volume: 1 }];
  let close = p0;
  for (const r of returns) {
    const open = close;
    close = open * Math.exp(r);
    const wick = (Math.abs(r) / 2) * (0.5 + rng());
    const high = Math.max(open, close) * Math.exp(wick);
    const low = Math.min(open, close) * Math.exp(-wick);
    candles.push({ open, high, low, close, volume: 1 });
  }
  return candles;
}

// Unconditional per-bar variance (2e-4)² = 4e-8 — a normal crypto/equity level
const N = 800;
const TRUE_ALPHA = 0.08;
const TRUE_BETA = 0.88;
const TRUE_OMEGA = 4e-8 * (1 - TRUE_ALPHA - TRUE_BETA);
const BASE_RETURNS = simGarchReturns(N, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 42);

// From stablecoin dust (std ~2e-10 per bar) to extreme vol (std ~20% per bar)
const SCALES = [1e-6, 1e-3, 1e3];

function relErr(a: number, b: number): number {
  return Math.abs(a / b - 1);
}

describe.each(SCALES)('scale equivariance at k = %s', (K: number) => {
  const TINY_RETURNS = BASE_RETURNS.map(r => r * K);

  it('Garch: same alpha/beta/df, vol scales by k', () => {
    const base = new Garch(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new Garch(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.alpha - base.params.alpha)).toBeLessThan(1e-3);
    expect(Math.abs(tiny.params.beta - base.params.beta)).toBeLessThan(1e-3);
    expect(Math.abs(tiny.params.df - base.params.df)).toBeLessThan(0.5);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(1e-3);
    expect(relErr(tiny.params.omega, base.params.omega * K * K)).toBeLessThan(1e-2);
    expect(tiny.diagnostics.logLikelihood).toBeGreaterThan(-1e9);
  });

  it('GjrGarch: same alpha/gamma/beta/df, vol scales by k', () => {
    const base = new GjrGarch(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new GjrGarch(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.alpha - base.params.alpha)).toBeLessThan(1e-3);
    expect(Math.abs(tiny.params.beta - base.params.beta)).toBeLessThan(1e-3);
    expect(Math.abs(tiny.params.gamma - base.params.gamma)).toBeLessThan(1e-3);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(1e-3);
    expect(tiny.diagnostics.logLikelihood).toBeGreaterThan(-1e9);
  });

  it('Egarch: same alpha/gamma/beta/df, vol scales by k', () => {
    const base = new Egarch(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new Egarch(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.alpha - base.params.alpha)).toBeLessThan(1e-3);
    expect(Math.abs(tiny.params.beta - base.params.beta)).toBeLessThan(1e-3);
    expect(Math.abs(tiny.params.gamma - base.params.gamma)).toBeLessThan(1e-3);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(1e-2);
    expect(tiny.diagnostics.logLikelihood).toBeGreaterThan(-1e9);
  });

  it('HarRv: same slopes/r2/df, beta0 scales by k²', () => {
    const base = new HarRv(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new HarRv(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.betaShort - base.params.betaShort)).toBeLessThan(1e-6);
    expect(Math.abs(tiny.params.betaMedium - base.params.betaMedium)).toBeLessThan(1e-6);
    expect(Math.abs(tiny.params.betaLong - base.params.betaLong)).toBeLessThan(1e-6);
    expect(Math.abs(tiny.params.r2 - base.params.r2)).toBeLessThan(1e-6);
    expect(Math.abs(tiny.params.df - base.params.df)).toBeLessThan(0.1);
    expect(relErr(tiny.params.beta0, base.params.beta0 * K * K)).toBeLessThan(1e-6);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(1e-4);
  });

  it('NoVaS: forecast scales by k², df invariant', () => {
    const baseModel = new NoVaS(pricesFromReturns(BASE_RETURNS));
    const tinyModel = new NoVaS(pricesFromReturns(TINY_RETURNS));
    const base = baseModel.fit();
    const tiny = tinyModel.fit();

    // The D² minimum is a flat manifold (weights are not identified — that
    // is why stage 2 exists), so float noise from the price→return
    // conversion can land the optimizer on a different manifold point at
    // extreme scales. Assert what users consume: the stage-2 rescaled
    // forecast, which the manifold indeterminacy mostly cancels out of.
    expect(Math.abs(tiny.params.df - base.params.df)).toBeLessThan(0.5);
    expect(Math.abs(tiny.params.persistence - base.params.persistence)).toBeLessThan(0.1);

    const fBase = baseModel.forecast(base.params, 5).variance;
    const fTiny = tinyModel.forecast(tiny.params, 5).variance;
    for (let h = 0; h < 5; h++) {
      expect(relErr(fTiny[h], fBase[h] * K * K)).toBeLessThan(0.05);
    }
  });
});

describe('scale equivariance end-to-end (predict)', () => {
  it('same model selected, corridor width scales by k', () => {
    const K = 1e-3;
    const base = predict(candlesFromReturns(BASE_RETURNS, 7), '1h');
    const tiny = predict(candlesFromReturns(BASE_RETURNS.map(r => r * K), 7), '1h');

    expect(tiny.modelType).toBe(base.modelType);
    expect(Math.abs(tiny.zScore - base.zScore)).toBeLessThan(1e-2);
    expect(relErr(tiny.sigma, base.sigma * K)).toBeLessThan(1e-2);
    // movePercent ≈ z·σ·100 in the small-move limit
    expect(relErr(tiny.movePercent, base.movePercent * K)).toBeLessThan(2e-2);
    expect(tiny.reliable).toBe(base.reliable);
  });
});

describe('df profiling is scale-free', () => {
  it('profileStudentTDf: df invariant to rescaling returns and variances', () => {
    const vs: number[] = [];
    let v = TRUE_OMEGA / (1 - TRUE_ALPHA - TRUE_BETA);
    for (let i = 0; i < BASE_RETURNS.length; i++) {
      vs.push(v);
      v = TRUE_OMEGA + TRUE_ALPHA * BASE_RETURNS[i] ** 2 + TRUE_BETA * v;
    }

    const dfBase = profileStudentTDf(BASE_RETURNS, vs);
    for (const k of [1e-6, 1e-3, 1e3]) {
      const dfScaled = profileStudentTDf(
        BASE_RETURNS.map(r => r * k),
        vs.map(x => x * k * k),
      );
      // NLL(df) shifts by an additive constant under rescaling → same argmin
      expect(dfScaled).toBeCloseTo(dfBase, 6);
    }
  });
});
