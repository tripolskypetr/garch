import { describe, it, expect } from 'vitest';
import { Garch } from '../src/garch.js';
import { GjrGarch } from '../src/gjr-garch.js';
import { Egarch } from '../src/egarch.js';
import { profileStudentTDf } from '../src/utils.js';

/**
 * MLE fits must be scale-equivariant: multiplying every return by k
 * (a stablecoin pair, an FX minor, a shorter candle interval) must give
 * the same alpha/beta/df and scale omega by k² / annualizedVol by k.
 *
 * Absolute floors like `omega <= 1e-12` or `variance <= 1e-12` inside the
 * likelihood silently reject ALL feasible parameters once the per-bar
 * return std drops below ~1e-6, so the optimizer "converges" on the
 * penalty plateau and returns the untouched initial guess.
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

// Unconditional per-bar variance (2e-4)² = 4e-8 — a normal crypto/equity level
const N = 800;
const TRUE_ALPHA = 0.08;
const TRUE_BETA = 0.88;
const TRUE_OMEGA = 4e-8 * (1 - TRUE_ALPHA - TRUE_BETA);
const BASE_RETURNS = simGarchReturns(N, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, 42);

// k = 1e-3 puts the per-bar std at ~2e-7 — the stablecoin/low-vol regime
const K = 1e-3;
const TINY_RETURNS = BASE_RETURNS.map(r => r * K);

function relErr(a: number, b: number): number {
  return Math.abs(a / b - 1);
}

describe('scale invariance of MLE calibration (low-volatility series)', () => {
  it('Garch: alpha/beta/df match the normal-scale fit, vol scales by k', () => {
    const base = new Garch(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new Garch(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.alpha - base.params.alpha)).toBeLessThan(0.02);
    expect(Math.abs(tiny.params.beta - base.params.beta)).toBeLessThan(0.02);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(0.05);
    expect(relErr(tiny.params.omega, base.params.omega * K * K)).toBeLessThan(0.25);
    expect(Number.isFinite(tiny.diagnostics.logLikelihood)).toBe(true);
    // A plateau exit reports fx = 1e10 → logLikelihood = -1e10
    expect(tiny.diagnostics.logLikelihood).toBeGreaterThan(-1e9);
  });

  it('GjrGarch: params match the normal-scale fit, vol scales by k', () => {
    const base = new GjrGarch(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new GjrGarch(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.alpha - base.params.alpha)).toBeLessThan(0.02);
    expect(Math.abs(tiny.params.beta - base.params.beta)).toBeLessThan(0.02);
    expect(Math.abs(tiny.params.gamma - base.params.gamma)).toBeLessThan(0.02);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(0.05);
    expect(tiny.diagnostics.logLikelihood).toBeGreaterThan(-1e9);
  });

  it('Egarch: params match the normal-scale fit, vol scales by k', () => {
    const base = new Egarch(pricesFromReturns(BASE_RETURNS)).fit();
    const tiny = new Egarch(pricesFromReturns(TINY_RETURNS)).fit();

    expect(Math.abs(tiny.params.alpha - base.params.alpha)).toBeLessThan(0.05);
    expect(Math.abs(tiny.params.beta - base.params.beta)).toBeLessThan(0.05);
    expect(relErr(tiny.params.annualizedVol, base.params.annualizedVol * K)).toBeLessThan(0.10);
    expect(tiny.diagnostics.logLikelihood).toBeGreaterThan(-1e9);
  });

  it('profileStudentTDf: df is invariant to rescaling returns and variances', () => {
    // Variance series from the true GARCH recursion
    const vs: number[] = [];
    let v = TRUE_OMEGA / (1 - TRUE_ALPHA - TRUE_BETA);
    for (let i = 0; i < BASE_RETURNS.length; i++) {
      vs.push(v);
      v = TRUE_OMEGA + TRUE_ALPHA * BASE_RETURNS[i] ** 2 + TRUE_BETA * v;
    }

    const dfBase = profileStudentTDf(BASE_RETURNS, vs);
    const dfTiny = profileStudentTDf(TINY_RETURNS, vs.map(x => x * K * K));

    // NLL(df) shifts by an additive constant under rescaling → same argmin
    expect(dfTiny).toBeCloseTo(dfBase, 6);
  });
});
