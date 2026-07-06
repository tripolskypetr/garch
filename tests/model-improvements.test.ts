import { describe, it, expect } from 'vitest';
import { GjrGarch, calibrateGjrGarch } from '../src/gjr-garch.js';
import { Egarch } from '../src/egarch.js';
import {
  predict,
  computeSeasonality,
  deseasonalizeCandles,
  type CandleInterval,
} from '../src/predict.js';
import { validateCandles } from '../src/utils.js';
import type { Candle } from '../src/types.js';

// ── helpers ──────────────────────────────────────────────────

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

/** GJR-GARCH(1,1) returns: v = ω + (α + γ·I(r<0))·r² + β·v. */
function simGjrReturns(
  n: number,
  omega: number,
  alpha: number,
  gamma: number,
  beta: number,
  seed: number,
): number[] {
  const rng = mulberry32(seed);
  let v = omega / (1 - alpha - gamma / 2 - beta);
  let r = Math.sqrt(v) * randn(rng);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    v = omega + (alpha + (r < 0 ? gamma : 0)) * r * r + beta * v;
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

/**
 * Hourly candles with GARCH(1,1) base variance multiplied by a diurnal
 * profile f(hourOfDay) = 1 + amp·sin(2π·hod/24), plus timestamps.
 * Returns candles and the true factor per hour bucket.
 */
function seasonalHourlyCandles(
  n: number,
  amp: number,
  seed: number,
): { candles: Candle[]; trueFactor: number[] } {
  const rng = mulberry32(seed);
  const omega = 4e-4 * (1 - 0.08 - 0.88);
  const alpha = 0.08;
  const beta = 0.88;
  const trueFactor = Array.from({ length: 24 }, (_, h) => 1 + amp * Math.sin((2 * Math.PI * h) / 24));

  const t0 = Date.UTC(2026, 0, 1);
  const hourMs = 3_600_000;
  const intraSteps = 12; // intra-bar walk → realistic Parkinson range noise
  let close = 100;
  const candles: Candle[] = [{ open: 100, high: 100, low: 100, close: 100, volume: 1, timestamp: t0 }];
  let v = omega / (1 - alpha - beta);
  let rPrev = Math.sqrt(v) * randn(rng);

  for (let i = 1; i <= n; i++) {
    v = omega + alpha * rPrev * rPrev + beta * v;
    const hod = Math.floor(((t0 + i * hourMs) % 86_400_000) / hourMs);
    const f = trueFactor[hod];
    const sigmaBar = Math.sqrt(v * f);

    let px = 0;
    let hi = 0;
    let lo = 0;
    for (let k = 0; k < intraSteps; k++) {
      px += (sigmaBar * randn(rng)) / Math.sqrt(intraSteps);
      if (px > hi) hi = px;
      if (px < lo) lo = px;
    }
    rPrev = px / Math.sqrt(f); // base-variance shock feeds the GARCH recursion

    const open = close;
    close = open * Math.exp(px);
    candles.push({
      open,
      high: open * Math.exp(hi),
      low: open * Math.exp(lo),
      close,
      volume: 1,
      timestamp: t0 + i * hourMs,
    });
  }
  return { candles, trueFactor };
}

// ── GJR inverted leverage ────────────────────────────────────

describe('GJR-GARCH inverted leverage (γ < 0)', () => {
  it('recovers negative gamma when pumps drive volatility harder than dumps', () => {
    // γ = −0.08: response to positive returns is α = 0.12, to negative
    // returns α + γ = 0.04. Before the γ ≥ 0 constraint was lifted, this
    // DGP was silently censored to γ = 0.
    const returns = simGjrReturns(2500, 4e-4 * 0.07, 0.12, -0.08, 0.85, 99);
    const result = calibrateGjrGarch(pricesFromReturns(returns));

    expect(result.params.gamma).toBeLessThan(0);
    expect(Math.abs(result.params.gamma - -0.08)).toBeLessThan(0.06);
    expect(result.params.alpha + result.params.gamma).toBeGreaterThanOrEqual(0);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('still recovers positive gamma on classic-leverage data', () => {
    const returns = simGjrReturns(2500, 4e-4 * 0.06, 0.04, 0.10, 0.85, 7);
    const result = calibrateGjrGarch(pricesFromReturns(returns));

    expect(result.params.gamma).toBeGreaterThan(0.03);
    expect(result.params.alpha).toBeGreaterThanOrEqual(0);
  });
});

// ── EGARCH multi-step drift with RV magnitude ────────────────

describe('EGARCH multi-step forecast drift (RV magnitude)', () => {
  const { candles } = seasonalHourlyCandles(700, 0, 4242);

  it('magnitudeDrift is nonzero for candle input and exactly zero for prices', () => {
    const model = new Egarch(candles);
    const fit = model.fit();
    const mbar = model.magnitudeDrift(fit.params);
    expect(Number.isFinite(mbar)).toBe(true);
    // E[√(RV/σ²)] ≠ E[|z|] in general (the sign depends on the range
    // estimator's bias) — the offset the old forecast dropped entirely
    expect(mbar).not.toBe(0);
    expect(Math.abs(mbar)).toBeLessThan(0.5);

    const prices = candles.map(c => c.close);
    const pModel = new Egarch(prices);
    expect(pModel.magnitudeDrift(pModel.fit().params)).toBe(0);
  });

  it('long-run forecast converges to the drift-corrected level, near the in-sample mean', () => {
    const model = new Egarch(candles);
    const fit = model.fit();
    const { omega, alpha, beta } = fit.params;
    const mbar = model.magnitudeDrift(fit.params);

    const fc = model.forecast(fit.params, 600);
    const logLongRun = Math.log(fc.variance[599]);
    const expected = (omega + alpha * mbar) / (1 - beta);
    expect(logLongRun).toBeCloseTo(expected, 3);

    // Level consistency: the fitted dynamics' mean variance and the
    // forecast's long-run level must agree (the old forecast sat
    // systematically below it by exp(−α·m̄/(1−β)))
    const series = model.getVarianceSeries(fit.params);
    const meanLog = series.reduce((s, v) => s + Math.log(v), 0) / series.length;
    expect(Math.abs(logLongRun - meanLog)).toBeLessThan(0.8);
  });
});

// ── Seasonality estimation ───────────────────────────────────

describe('intraday seasonality', () => {
  it('recovers a strong diurnal profile (shape and amplitude)', () => {
    const { candles, trueFactor } = seasonalHourlyCandles(900, 0.75, 11);
    const season = computeSeasonality(candles, '1h');
    expect(season).not.toBeNull();

    const f = season!.factors;
    expect(f).toHaveLength(24);
    expect(Math.max(...f) / Math.min(...f)).toBeGreaterThan(2);

    // Shape: correlation with the true profile
    const mf = f.reduce((s, v) => s + v, 0) / 24;
    const mt = trueFactor.reduce((s, v) => s + v, 0) / 24;
    let num = 0;
    let df2 = 0;
    let dt2 = 0;
    for (let b = 0; b < 24; b++) {
      num += (f[b] - mf) * (trueFactor[b] - mt);
      df2 += (f[b] - mf) ** 2;
      dt2 += (trueFactor[b] - mt) ** 2;
    }
    expect(num / Math.sqrt(df2 * dt2)).toBeGreaterThan(0.8);
  });

  it('significance gate: returns null on non-seasonal GARCH data', () => {
    const { candles } = seasonalHourlyCandles(900, 0, 12);
    expect(computeSeasonality(candles, '1h')).toBeNull();
  });

  it('deseasonalized candles are valid OHLC with a flat profile', () => {
    const { candles } = seasonalHourlyCandles(900, 0.75, 13);
    const season = computeSeasonality(candles, '1h');
    expect(season).not.toBeNull();

    const flat = deseasonalizeCandles(candles, season!);
    expect(() => validateCandles(flat)).not.toThrow();
    expect(flat).toHaveLength(candles.length);

    // Re-estimating on the deseasonalized series finds nothing significant
    // (or at most a residual profile far weaker than the original)
    const residual = computeSeasonality(flat, '1h');
    if (residual) {
      const ratio = Math.max(...residual.factors) / Math.min(...residual.factors);
      const origRatio = Math.max(...season!.factors) / Math.min(...season!.factors);
      expect(ratio).toBeLessThan(origRatio / 2);
    }
  });
});

// ── Walk-forward conditional coverage under seasonality ──────
//
// The point of deseasonalization: without it the corridor is calibrated to
// the *average* diurnal variance, so it systematically overshoots in quiet
// hours and undershoots in active ones even when total coverage looks fine.
// Both hour groups must be near nominal, not just their average.

describe('conditional coverage with a strong diurnal profile', () => {
  it('68% corridor covers ≈ nominal in both high-vol and low-vol hours', () => {
    const { candles, trueFactor } = seasonalHourlyCandles(560, 0.75, 21);
    const window = 400;
    const confidence = 0.6827;

    let hitsHigh = 0;
    let nHigh = 0;
    let hitsLow = 0;
    let nLow = 0;

    for (let i = window; i < candles.length - 1; i += 2) {
      const slice = candles.slice(i - window, i + 1);
      const price = slice[slice.length - 1].close;
      const res = predict(slice, '1h' as CandleInterval, price, confidence);
      const actual = candles[i + 1].close;
      const hit = actual >= res.lowerPrice && actual <= res.upperPrice;

      const hod = Math.floor(((candles[i + 1].timestamp!) % 86_400_000) / 3_600_000);
      if (trueFactor[hod] >= 1) {
        nHigh++;
        if (hit) hitsHigh++;
      } else {
        nLow++;
        if (hit) hitsLow++;
      }
    }

    const covHigh = (hitsHigh / nHigh) * 100;
    const covLow = (hitsLow / nLow) * 100;

    // ~40 points per group → binomial SE ≈ 7.4pp; bounds ≈ 2σ.
    // Without deseasonalization the low-vol group sits near ~90% and the
    // high-vol group near ~57% on this DGP.
    expect(nHigh + nLow).toBeGreaterThanOrEqual(70);
    expect(covHigh, `high-vol hours coverage=${covHigh.toFixed(1)}% (n=${nHigh})`).toBeGreaterThanOrEqual(53);
    expect(covHigh, `high-vol hours coverage=${covHigh.toFixed(1)}% (n=${nHigh})`).toBeLessThanOrEqual(84);
    expect(covLow, `low-vol hours coverage=${covLow.toFixed(1)}% (n=${nLow})`).toBeGreaterThanOrEqual(53);
    expect(covLow, `low-vol hours coverage=${covLow.toFixed(1)}% (n=${nLow})`).toBeLessThanOrEqual(84);
  }, 900_000);
});

// ── Determinism ──────────────────────────────────────────────

describe('prediction determinism', () => {
  it('same candles produce bit-identical predictions (seeded simulation)', () => {
    const { candles } = seasonalHourlyCandles(300, 0.75, 31);
    const a = predict(candles, '1h');
    const b = predict(candles, '1h');
    expect(b.sigma).toBe(a.sigma);
    expect(b.zScore).toBe(a.zScore);
    expect(b.upperPrice).toBe(a.upperPrice);
    expect(b.lowerPrice).toBe(a.lowerPrice);
    expect(b.modelType).toBe(a.modelType);
  }, 120_000);
});
