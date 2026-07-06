import { describe, it, expect } from 'vitest';
import { empiricalQuantile, validateCandles, studentTProbit } from '../src/utils.js';
import { nelderMead, nelderMeadMultiStart } from '../src/optimizer.js';
import {
  selectHarLagCandidates,
  adaptiveNovasLags,
  predict,
  predictRange,
  backtest,
  backtestStats,
  type CandleInterval,
} from '../src/predict.js';
import type { Candle } from '../src/types.js';

// ── empiricalQuantile ──────────────────────────────────────────

describe('empiricalQuantile', () => {
  it('matches exact quantiles on a linear ramp', () => {
    const xs = Array.from({ length: 101 }, (_, i) => i); // 0..100
    expect(empiricalQuantile(xs, 0)).toBe(0);
    expect(empiricalQuantile(xs, 0.5)).toBe(50);
    expect(empiricalQuantile(xs, 0.95)).toBe(95);
    expect(empiricalQuantile(xs, 1)).toBe(100);
  });

  it('interpolates between order statistics', () => {
    expect(empiricalQuantile([0, 10], 0.25)).toBeCloseTo(2.5, 10);
    expect(empiricalQuantile([1, 2, 3], 0.5)).toBe(2);
  });

  it('handles degenerate inputs', () => {
    expect(empiricalQuantile([7], 0.9)).toBe(7);
    expect(Number.isNaN(empiricalQuantile([], 0.5))).toBe(true);
  });
});

// ── validateCandles ────────────────────────────────────────────

describe('validateCandles', () => {
  const good: Candle = { open: 100, high: 101, low: 99, close: 100.5, volume: 1 };

  it('accepts consistent candles', () => {
    expect(() => validateCandles([good, good])).not.toThrow();
  });

  it('rejects high < low', () => {
    expect(() => validateCandles([good, { ...good, high: 98 }])).toThrow(/high.*low/i);
  });

  it('rejects non-positive and non-finite prices', () => {
    expect(() => validateCandles([{ ...good, low: 0 }])).toThrow(/Invalid OHLC/);
    expect(() => validateCandles([{ ...good, open: -1 }])).toThrow(/Invalid OHLC/);
    expect(() => validateCandles([{ ...good, close: NaN }])).toThrow(/Invalid OHLC/);
    expect(() => validateCandles([{ ...good, high: Infinity }])).toThrow(/Invalid OHLC/);
  });
});

// ── HAR lag candidates ─────────────────────────────────────────

describe('selectHarLagCandidates', () => {
  it('always returns at least one feasible triple', () => {
    for (const n of [149, 199, 299, 459, 999, 1499]) {
      for (const ppy of [525_600, 35_040, 8_760, 2_190, 1_095]) {
        const candidates = selectHarLagCandidates(n, ppy);
        expect(candidates.length).toBeGreaterThanOrEqual(1);
        for (const [s, m, l] of candidates) {
          expect(s).toBeLessThan(m);
          expect(m).toBeLessThan(l);
          expect(l).toBeLessThanOrEqual(Math.floor(Math.min(n / 5, n - 31)));
        }
      }
    }
  });

  it('offers a day/week triple for 1h data when the sample supports it', () => {
    // 1h: 24 bars/day, 168 bars/week — needs n ≥ 840
    const candidates = selectHarLagCandidates(999, 8_760);
    expect(candidates).toContainEqual([1, 24, 168]);
    expect(candidates).toContainEqual([1, 5, 22]);
  });

  it('caps the long horizon by sample size instead of dropping the daily structure', () => {
    // 4h: 6 bars/day, 42 bars/week; n=299 → maxLong=59 ≥ 42 — full weekly fits
    expect(selectHarLagCandidates(299, 2_190)).toContainEqual([1, 6, 42]);
    // 1h with n=459: weekly 168 > maxLong=91 → capped long horizon
    expect(selectHarLagCandidates(459, 8_760)).toContainEqual([1, 24, 91]);
  });

  it('deduplicates candidates', () => {
    for (const n of [199, 459, 1499]) {
      const candidates = selectHarLagCandidates(n, 8_760);
      const keys = candidates.map(c => c.join(','));
      expect(new Set(keys).size).toBe(keys.length);
    }
  });
});

// ── NoVaS adaptive lag order ───────────────────────────────────

describe('adaptiveNovasLags', () => {
  it('matches the old default at n ≈ 500 and stays in [5, 20]', () => {
    expect(adaptiveNovasLags(500)).toBe(10);
    expect(adaptiveNovasLags(149)).toBeGreaterThanOrEqual(5);
    expect(adaptiveNovasLags(149)).toBeLessThanOrEqual(10);
    expect(adaptiveNovasLags(50)).toBe(5);
    expect(adaptiveNovasLags(10_000)).toBe(20);
  });

  it('is monotone in sample size', () => {
    let prev = 0;
    for (const n of [100, 200, 400, 800, 1600, 3200]) {
      const p = adaptiveNovasLags(n);
      expect(p).toBeGreaterThanOrEqual(prev);
      prev = p;
    }
  });
});

// ── Adaptive multi-start restarts ──────────────────────────────

describe('nelderMeadMultiStart adaptive restarts', () => {
  // Rastrigin-like multimodal function: many local minima
  const rastrigin = (x: number[]) =>
    10 * x.length + x.reduce((s, v) => s + v * v - 10 * Math.cos(2 * Math.PI * v), 0);

  it('is never worse than a single start', () => {
    const single = nelderMead(rastrigin, [2.5, -1.5], { maxIter: 500, tol: 1e-10 });
    const multi = nelderMeadMultiStart(rastrigin, [2.5, -1.5], { maxIter: 500, tol: 1e-10, restarts: 3 });
    expect(multi.fx).toBeLessThanOrEqual(single.fx + 1e-12);
  });

  it('stays deterministic', () => {
    const a = nelderMeadMultiStart(rastrigin, [2.5, -1.5], { maxIter: 500, tol: 1e-10, restarts: 3 });
    const b = nelderMeadMultiStart(rastrigin, [2.5, -1.5], { maxIter: 500, tol: 1e-10, restarts: 3 });
    expect(a.fx).toBe(b.fx);
    expect(a.x).toEqual(b.x);
  });
});

// ── Blended corridor quantile behaves sensibly end-to-end ──────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCandles(n: number, seed: number): Candle[] {
  const rand = mulberry32(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const u1 = Math.max(rand(), 1e-12);
    const r = 0.01 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rand());
    const open = price;
    const close = price * Math.exp(r);
    const wick = 0.01 * (0.4 + 0.6 * rand());
    candles.push({
      open,
      high: Math.max(open, close) * Math.exp(wick),
      low: Math.min(open, close) * Math.exp(-wick),
      close,
      volume: 1000,
    });
    price = close;
  }
  return candles;
}

describe('predictRange input validation', () => {
  it('rejects steps < 1 and non-finite steps', () => {
    const candles = makeCandles(300, 9);
    expect(() => predictRange(candles, '4h' as CandleInterval, 0)).toThrow(/steps/);
    expect(() => predictRange(candles, '4h' as CandleInterval, -3)).toThrow(/steps/);
    expect(() => predictRange(candles, '4h' as CandleInterval, NaN)).toThrow(/steps/);
  });
});

describe('backtestStats stride', () => {
  it('auto-stride caps refits at ~100 and explicit stride subsamples accordingly', () => {
    const candles = makeCandles(300, 77);
    // window = max(200, 225) = 225 → testSpan = 74 → auto stride 1
    const full = backtestStats(candles, '4h' as CandleInterval, 0.9);
    expect(full.total).toBe(74);

    const strided = backtestStats(candles, '4h' as CandleInterval, 0.9, { stride: 5 });
    expect(strided.total).toBe(Math.ceil(74 / 5));
    expect(strided.hitRate).toBeCloseTo((strided.hits / strided.total) * 100, 10);
  }, 900_000);
});

describe('backtest requiredPercent=100 semantics', () => {
  it('is decided by the actual hit rate, not short-circuited', () => {
    const candles = makeCandles(300, 77);
    const stats = backtestStats(candles, '4h' as CandleInterval, 0.6827);
    const expected = stats.hitRate >= 100;
    expect(backtest(candles, '4h' as CandleInterval, 0.6827, 100)).toBe(expected);
    // above 100 is impossible by construction
    expect(backtest(candles, '4h' as CandleInterval, 0.6827, 101)).toBe(false);
  }, 900_000);
});

describe('corridor zScore', () => {
  it('is exposed, positive, and increases with confidence', () => {
    const candles = makeCandles(400, 123);
    const p68 = predict(candles, '1h' as CandleInterval, null, 0.6827);
    const p95 = predict(candles, '1h' as CandleInterval, null, 0.95);
    const p99 = predict(candles, '1h' as CandleInterval, null, 0.99);
    expect(p68.zScore).toBeGreaterThan(0);
    expect(p95.zScore).toBeGreaterThan(p68.zScore);
    expect(p99.zScore).toBeGreaterThan(p95.zScore);
  }, 600_000);

  it('bands reconstruct exactly from zScore and sigma', () => {
    const candles = makeCandles(300, 321);
    const res = predict(candles, '4h' as CandleInterval, null, 0.9);
    expect(res.upperPrice).toBeCloseTo(res.currentPrice * Math.exp(res.zScore * res.sigma), 10);
    expect(res.lowerPrice).toBeCloseTo(res.currentPrice * Math.exp(-res.zScore * res.sigma), 10);
  }, 600_000);

  it('on Gaussian data the 68% multiplier lands near 1.0 (not the fat-tail t value)', () => {
    // Empirical blend should keep z close to the true Gaussian quantile
    const candles = makeCandles(1000, 55);
    const res = predict(candles, '1h' as CandleInterval, null, 0.6827);
    expect(res.zScore).toBeGreaterThan(0.8);
    expect(res.zScore).toBeLessThan(1.2);
    // and far-tail stays anchored by the model half
    const res99 = predict(candles, '1h' as CandleInterval, null, 0.999);
    expect(res99.zScore).toBeGreaterThan(studentTProbit(0.999, 100) * 0.7);
  }, 600_000);
});
