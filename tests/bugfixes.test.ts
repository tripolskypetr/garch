import { describe, it, expect } from 'vitest';
import { nelderMead } from '../src/optimizer.js';
import { Egarch } from '../src/egarch.js';
import { backtest, predict, type CandleInterval } from '../src/predict.js';
import type { Candle } from '../src/types.js';

// ── Deterministic candle generator (GARCH-like synthetic data) ──

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): () => number {
  return () => {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

function makeCandles(n: number, seed = 42): Candle[] {
  const rand = mulberry32(seed);
  const norm = gaussian(rand);
  const candles: Candle[] = [];
  let price = 100;
  let variance = 4e-4; // ~2% per period
  const omega = 4e-6;
  const alpha = 0.08;
  const beta = 0.9;

  for (let i = 0; i < n; i++) {
    const z = norm();
    const r = Math.sqrt(variance) * z;
    variance = omega + alpha * r * r + beta * variance;

    const open = price;
    const close = price * Math.exp(r);
    const wick = Math.sqrt(variance) * (0.3 + 0.7 * rand());
    const high = Math.max(open, close) * Math.exp(wick);
    const low = Math.min(open, close) * Math.exp(-wick);
    candles.push({ open, high, low, close, volume: 1000 + 1000 * rand(), timestamp: i });
    price = close;
  }
  return candles;
}

// ── Bug 1: nelderMead must return the best vertex on maxIter exit ──
//
// Rejected trial points are never strictly better than the best vertex,
// so min over all evaluations == best vertex ever constructed. If the
// simplex is not re-sorted before returning on maxIter exit, the best
// point found in the final iteration sits at index n and is lost.
describe('nelderMead maxIter exit', () => {
  it('returns the minimum over all evaluated vertices for any maxIter', () => {
    for (let maxIter = 1; maxIter <= 80; maxIter++) {
      const evals: number[] = [];
      const rosenbrock = (x: number[]) => {
        const v = 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2;
        evals.push(v);
        return v;
      };
      const res = nelderMead(rosenbrock, [-1.2, 1], { maxIter, tol: 1e-12 });
      const minEval = Math.min(...evals);
      expect(res.fx, `maxIter=${maxIter}`).toBeLessThanOrEqual(minEval + 1e-15);
    }
  });

  it('fx is non-increasing as maxIter grows (quadratic)', () => {
    let prev = Infinity;
    for (let maxIter = 1; maxIter <= 40; maxIter++) {
      const fn = (x: number[]) => (x[0] - 3) ** 2 + (x[1] + 2) ** 2 + x[0] * x[1] * 0.1;
      const res = nelderMead(fn, [10, 10], { maxIter, tol: 0 });
      expect(res.fx, `maxIter=${maxIter}`).toBeLessThanOrEqual(prev + 1e-12);
      prev = res.fx;
    }
  });
});

// ── Bug 2: backtest with exactly MIN_CANDLES has an empty test window ──
//
// window = max(MIN_CANDLES, floor(0.75·n)) can swallow the whole series;
// then total === 0 and hits/total is NaN. The docstring promises a throw
// when there is not enough data — a silent `false` misreports the model
// as failing the backtest.
describe('backtest with empty test window', () => {
  it('throws (not silently false) when no candle is left for testing', () => {
    const candles = makeCandles(200); // MIN_CANDLES for 1h is exactly 200
    expect(() => backtest(candles, '1h' as CandleInterval, 0.6827, 1)).toThrow(/candles/i);
  });

  it('still works when there is a real test window', () => {
    const candles = makeCandles(280);
    const result = backtest(candles, '1h' as CandleInterval, 0.6827, 1);
    expect(typeof result).toBe('boolean');
  }, 600_000);
});

// ── Bug 3: EGARCH forecast must clamp log-variance like fit does ──
//
// fit() and getVarianceSeries() clamp ln(σ²) to [-50, 50]; forecast()
// did not, so extreme (non-stationary) params overflow exp() to Infinity.
describe('Egarch forecast clamping', () => {
  it('never returns Infinity even with extreme parameters', () => {
    const candles = makeCandles(300);
    const model = new Egarch(candles);
    const params = {
      omega: 60, // extreme: forces ln(σ²) > 50 immediately
      alpha: 0.1,
      gamma: -0.05,
      beta: 0.99,
      persistence: 0.99,
      unconditionalVariance: 1,
      annualizedVol: 100,
      leverageEffect: -0.05,
      df: 5,
    };
    const fc = model.forecast(params, 10);
    for (const v of fc.variance) {
      expect(Number.isFinite(v)).toBe(true);
    }
    // matches the same clamp used in fit/getVarianceSeries
    expect(Math.max(...fc.variance)).toBeLessThanOrEqual(Math.exp(50));
  });
});

// ── Sanity: predict on synthetic GARCH data stays reasonable ──
describe('predict sanity on synthetic data', () => {
  it('produces a finite, ordered corridor', () => {
    const candles = makeCandles(400, 7);
    const res = predict(candles, '1h' as CandleInterval);
    expect(Number.isFinite(res.sigma)).toBe(true);
    expect(res.sigma).toBeGreaterThan(0);
    expect(res.lowerPrice).toBeLessThan(res.currentPrice);
    expect(res.upperPrice).toBeGreaterThan(res.currentPrice);
    // synthetic vol is ~2%/period; corridor must be in a sane range
    expect(res.sigma).toBeLessThan(0.5);
  }, 600_000);
});
