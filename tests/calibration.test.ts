import { describe, it, expect } from 'vitest';
import { studentTCdf, studentTProbit, probit } from '../src/utils.js';
import { predict, backtestStats, type CandleInterval } from '../src/predict.js';
import type { Candle } from '../src/types.js';

// ── Deterministic RNG ──────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianPair(rand: () => number): [number, number] {
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

/**
 * Standardized Student-t(df) sample (unit variance): normal / sqrt(chi2/df),
 * scaled by sqrt((df-2)/df).
 */
function makeStudentT(rand: () => number, df: number): () => number {
  return () => {
    const [z] = gaussianPair(rand);
    let chi2 = 0;
    // df assumed integer here (test generator only)
    for (let i = 0; i < df; i += 2) {
      const u = Math.max(rand(), 1e-12);
      chi2 += -2 * Math.log(u); // Exp(1/2) pair ~ chi2(2)
    }
    const t = z / Math.sqrt(chi2 / df);
    return t * Math.sqrt((df - 2) / df);
  };
}

/**
 * Fat-tailed GJR-GARCH price series with realistic wicks.
 * True DGP: σ²ₜ = ω + (α + γ·I(r<0))·r²ₜ₋₁ + β·σ²ₜ₋₁, z ~ standardized t(5).
 */
function makeFatTailedCandles(n: number, seed: number, wickScale = 1): Candle[] {
  const rand = mulberry32(seed);
  const tSample = makeStudentT(rand, 5);
  const candles: Candle[] = [];
  let price = 100;

  const omega = 5e-6;
  const alpha = 0.05;
  const gamma = 0.08;
  const beta = 0.88;
  let variance = omega / (1 - alpha - gamma / 2 - beta);

  for (let i = 0; i < n; i++) {
    const z = tSample();
    const r = Math.sqrt(variance) * z;
    variance = omega + (alpha + (r < 0 ? gamma : 0)) * r * r + beta * variance;

    const open = price;
    const close = price * Math.exp(r);
    // Wick size tied to per-period vol; wickScale < 1 compresses ranges
    const wick = Math.sqrt(variance) * (0.4 + 0.6 * rand()) * wickScale;
    const high = Math.max(open, close) * Math.exp(wick);
    const low = Math.min(open, close) * Math.exp(-wick);
    candles.push({ open, high, low, close, volume: 1000 + 1000 * rand(), timestamp: i });
    price = close;
  }
  return candles;
}

// ── Student-t quantile correctness ─────────────────────────────

describe('studentTProbit', () => {
  it('matches table values (standardized t quantiles)', () => {
    // raw t_{0.975,5} = 2.570582, standardized × sqrt(3/5)
    expect(studentTProbit(0.95, 5)).toBeCloseTo(2.570582 * Math.sqrt(3 / 5), 4);
    // raw t_{0.975,10} = 2.228139, standardized × sqrt(8/10)
    expect(studentTProbit(0.95, 10)).toBeCloseTo(2.228139 * Math.sqrt(8 / 10), 4);
    // raw t_{0.995,3} = 5.840909, standardized × sqrt(1/3)
    expect(studentTProbit(0.99, 3)).toBeCloseTo(5.840909 * Math.sqrt(1 / 3), 4);
    // raw t_{0.95,4} = 2.131847, standardized × sqrt(2/4)
    expect(studentTProbit(0.9, 4)).toBeCloseTo(2.131847 * Math.sqrt(2 / 4), 4);
  });

  it('converges to probit for large df', () => {
    for (const conf of [0.6827, 0.9, 0.95, 0.99]) {
      // t(150) is genuinely still ~0.5-1.5% off Gaussian in the far tail
      expect(Math.abs(studentTProbit(conf, 150) - probit(conf))).toBeLessThan(0.02);
      expect(studentTProbit(conf, 1000)).toBeCloseTo(probit(conf), 6); // probit fallback
    }
  });

  it('fat tails: narrower center, wider tails than Gaussian', () => {
    // Variance is 1 in both cases, so quantiles must cross
    expect(studentTProbit(0.6827, 5)).toBeLessThan(probit(0.6827));
    expect(studentTProbit(0.99, 5)).toBeGreaterThan(probit(0.99));
  });

  it('CDF round-trip', () => {
    for (const df of [3, 5, 8, 20, 50]) {
      for (const conf of [0.5, 0.6827, 0.9, 0.95, 0.99]) {
        const z = studentTProbit(conf, df);
        const raw = z / Math.sqrt((df - 2) / df);
        expect(2 * studentTCdf(raw, df) - 1).toBeCloseTo(conf, 8);
      }
    }
  });

  it('is monotone in confidence', () => {
    let prev = 0;
    for (let conf = 0.05; conf < 1; conf += 0.05) {
      const z = studentTProbit(conf, 6);
      expect(z).toBeGreaterThan(prev);
      prev = z;
    }
  });
});

// ── Walk-forward calibration on fat-tailed data ────────────────
//
// The real acceptance criterion for the tool: the corridor at confidence c
// must contain the next close ~c·100% of the time, out of sample, on data
// with Student-t(5) tails and leverage — exactly what real markets look like.
//
// Coverage at 3 confidence levels is derived from a single fit per step by
// reconstructing bands from sigma and the reported df.
describe('walk-forward corridor calibration (t(5) + leverage DGP)', () => {
  it('empirical coverage tracks nominal confidence', () => {
    const candles = makeFatTailedCandles(460, 20260706);
    const window = 300;
    const confidences = [0.6827, 0.9, 0.99];
    const hits = [0, 0, 0];
    let total = 0;

    for (let i = window; i < candles.length - 1; i++) {
      const slice = candles.slice(i - window, i + 1);
      const price = slice[slice.length - 1].close;
      const res = predict(slice, '1h' as CandleInterval, price, confidences[0]);
      const actual = candles[i + 1].close;
      const actualZ = Math.abs(Math.log(actual / price)) / res.sigma;

      for (let k = 0; k < confidences.length; k++) {
        const zk = studentTProbit(confidences[k], res.df);
        if (actualZ <= zk) hits[k]++;
      }
      total++;
    }

    expect(total).toBeGreaterThanOrEqual(150);
    const coverage = hits.map(h => (h / total) * 100);

    // n≈159, binomial SE: ~3.7pp at 68%, ~2.4pp at 90%, ~0.8pp at 99%.
    // Bounds are ~2.5σ — catches systematic miscalibration, not noise.
    expect(coverage[0], `68% band coverage=${coverage[0].toFixed(1)}%`).toBeGreaterThanOrEqual(59);
    expect(coverage[0], `68% band coverage=${coverage[0].toFixed(1)}%`).toBeLessThanOrEqual(78);
    expect(coverage[1], `90% band coverage=${coverage[1].toFixed(1)}%`).toBeGreaterThanOrEqual(84);
    expect(coverage[1], `90% band coverage=${coverage[1].toFixed(1)}%`).toBeLessThanOrEqual(96.5);
    expect(coverage[2], `99% band coverage=${coverage[2].toFixed(1)}%`).toBeGreaterThanOrEqual(96);
  }, 900_000);
});

// ── Scale robustness: compressed wicks ─────────────────────────
//
// When candle ranges are much smaller than close-to-close moves (gaps,
// illiquid books), Parkinson RV massively underestimates return variance.
// RV-based models (HAR/NoVaS) then win QLIKE while forecasting a variance
// several times too small — without the scale correction the corridor
// collapses and coverage drops to a fraction of nominal.
describe('corridor survives RV/return scale mismatch', () => {
  it('keeps ~68% coverage when wicks are 4x compressed', () => {
    const candles = makeFatTailedCandles(460, 777, 0.25);
    const window = 300;
    let hits = 0;
    let total = 0;

    for (let i = window; i < candles.length - 1; i += 2) {
      const slice = candles.slice(i - window, i + 1);
      const price = slice[slice.length - 1].close;
      const res = predict(slice, '1h' as CandleInterval, price, 0.6827);
      const actual = candles[i + 1].close;
      if (actual >= res.lowerPrice && actual <= res.upperPrice) hits++;
      total++;
    }

    const coverage = (hits / total) * 100;
    // Uncorrected RV-scale corridors give coverage far below 50% here
    expect(coverage, `coverage=${coverage.toFixed(1)}% (n=${total})`).toBeGreaterThanOrEqual(55);
  }, 900_000);
});

// ── backtestStats API ──────────────────────────────────────────

describe('backtestStats', () => {
  it('reports hit counts consistent with backtest', () => {
    const candles = makeFatTailedCandles(280, 4242);
    const stats = backtestStats(candles, '1h' as CandleInterval, 0.9);
    expect(stats.total).toBeGreaterThan(0);
    expect(stats.hits).toBeGreaterThanOrEqual(0);
    expect(stats.hits).toBeLessThanOrEqual(stats.total);
    expect(stats.hitRate).toBeCloseTo((stats.hits / stats.total) * 100, 10);
  }, 900_000);
});
