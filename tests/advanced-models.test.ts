import { describe, it, expect } from 'vitest';
import { Garch } from '../src/garch.js';
import { HarRv } from '../src/har.js';
import { RealizedGarch, calibrateRealizedGarch } from '../src/realized-garch.js';
import { predict, createPredictor, type CandleInterval } from '../src/predict.js';
import { qlike, perCandleParkinson, calculateReturns } from '../src/utils.js';
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

/**
 * Candle whose Parkinson RV equals a prescribed value: the log-range is set
 * to √(rv·4ln2) around the body (body wider than the range clamps to body).
 */
function candleWithRv(open: number, close: number, rv: number, volume = 1, timestamp?: number): Candle {
  const body = Math.abs(Math.log(close / open));
  const hl = Math.max(Math.sqrt(rv * 4 * Math.LN2), body);
  const excess = hl - body;
  const high = Math.max(open, close) * Math.exp(excess / 2);
  const low = Math.min(open, close) * Math.exp(-excess / 2);
  return { open, high, low, close, volume, timestamp };
}

/**
 * Simulate the Realized GARCH DGP exactly (φ = 1) and encode the simulated
 * RV into the candle ranges so Parkinson recovers it.
 */
function simRealizedGarchCandles(
  n: number,
  p: { omega: number; beta: number; gamma: number; xi: number; tau1: number; tau2: number; sigmaU: number },
  seed: number,
): Candle[] {
  const rng = mulberry32(seed);
  const uncondLog = (p.omega + p.gamma * p.xi) / (1 - p.beta - p.gamma);
  let lnv = uncondLog;
  let lnrvPrev = p.xi + lnv;
  let close = 100;
  const candles: Candle[] = [{ open: 100, high: 100, low: 100, close: 100, volume: 1 }];

  for (let i = 0; i < n; i++) {
    lnv = p.omega + p.beta * lnv + p.gamma * lnrvPrev;
    const z = randn(rng);
    const r = Math.exp(lnv / 2) * z;
    const lnrv = p.xi + lnv + p.tau1 * z + p.tau2 * (z * z - 1) + p.sigmaU * randn(rng);
    lnrvPrev = lnrv;

    const open = close;
    close = open * Math.exp(r);
    candles.push(candleWithRv(open, close, Math.exp(lnrv)));
  }
  return candles;
}

/** GARCH(1,1) prices with an upward variance regime break at breakAt·n. */
function regimeBreakPrices(n: number, breakAt: number, seed: number): {
  prices: number[];
  postBreakVar: number;
} {
  const rng = mulberry32(seed);
  const alpha = 0.08;
  const beta = 0.85;
  const uncond1 = 4e-4;
  const uncond2 = 16e-4; // 4× variance after the break
  let v = uncond1;
  let r = Math.sqrt(v) * randn(rng);
  const prices = [100];
  for (let i = 0; i < n; i++) {
    const uncond = i < n * breakAt ? uncond1 : uncond2;
    const omega = uncond * (1 - alpha - beta);
    v = omega + alpha * r * r + beta * v;
    r = Math.sqrt(v) * randn(rng);
    prices.push(prices[prices.length - 1] * Math.exp(r));
  }
  return { prices, postBreakVar: uncond2 };
}

/** Hourly candles with negatively skewed innovations (crashes heavier than rallies). */
function skewedCandles(n: number, seed: number): Candle[] {
  const rng = mulberry32(seed);
  const omega = 4e-4 * 0.04;
  const alpha = 0.08;
  const beta = 0.88;
  const t0 = Date.UTC(2026, 0, 1);
  const intraSteps = 12;
  // Scale mixture on sign: negative shocks 1.5×, positive 0.75× — then
  // center and normalize to unit variance
  const drawSkewed = (): number => {
    const g = randn(rng);
    const raw = g < 0 ? g * 1.5 : g * 0.75;
    return (raw + 0.2992) / 1.1837; // moments matched numerically for this mixture
  };

  let close = 100;
  const candles: Candle[] = [{ open: 100, high: 100, low: 100, close: 100, volume: 1, timestamp: t0 }];
  let v = omega / (1 - alpha - beta);
  let rPrev = Math.sqrt(v) * drawSkewed();

  for (let i = 1; i <= n; i++) {
    v = omega + alpha * rPrev * rPrev + beta * v;
    const sigmaBar = Math.sqrt(v);
    let px = 0;
    let hi = 0;
    let lo = 0;
    for (let k = 0; k < intraSteps; k++) {
      px += (sigmaBar * drawSkewed()) / Math.sqrt(intraSteps);
      if (px > hi) hi = px;
      if (px < lo) lo = px;
    }
    rPrev = px;
    const open = close;
    close = open * Math.exp(px);
    candles.push({
      open,
      high: open * Math.exp(hi),
      low: open * Math.exp(lo),
      close,
      volume: 1,
      timestamp: t0 + i * 3_600_000,
    });
  }
  return candles;
}

// ── Realized GARCH ───────────────────────────────────────────

describe('Realized GARCH', () => {
  const TRUE = { omega: -0.35, beta: 0.55, gamma: 0.35, xi: -0.3, tau1: -0.1, tau2: 0.1, sigmaU: 0.4 };
  // uncond log var target ≈ ln(4e-4): ω = (1−β−γ)·L − γ·ξ with L = ln(4e-4)
  TRUE.omega = (1 - TRUE.beta - TRUE.gamma) * Math.log(4e-4) - TRUE.gamma * TRUE.xi;
  const candles = simRealizedGarchCandles(1500, TRUE, 314);

  it('recovers the DGP parameters from candles', () => {
    const result = calibrateRealizedGarch(candles);
    const p = result.params;

    expect(Math.abs(p.beta - TRUE.beta)).toBeLessThan(0.2);
    expect(Math.abs(p.gamma - TRUE.gamma)).toBeLessThan(0.2);
    expect(Math.abs(p.persistence - (TRUE.beta + TRUE.gamma))).toBeLessThan(0.1);
    expect(Math.abs(p.tau1 - TRUE.tau1)).toBeLessThan(0.1);
    expect(Math.abs(p.sigmaU - TRUE.sigmaU)).toBeLessThan(0.2);
    // Unconditional level within a factor of 2 of the truth
    expect(Math.abs(Math.log(p.unconditionalVariance / 4e-4))).toBeLessThan(Math.log(2));
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });

  it('beats plain GARCH out-of-sample on Realized GARCH data', () => {
    const nTrain = Math.floor(candles.length * 0.75);
    const train = candles.slice(0, nTrain);
    const returns = calculateReturns(candles);
    const rv = perCandleParkinson(candles, returns);
    const evalStart = nTrain - 1;

    const rgTrain = new RealizedGarch(train).fit();
    const rgFull = new RealizedGarch(candles);
    const rgScore = qlike(rgFull.getVarianceSeries(rgTrain.params).slice(evalStart), rv.slice(evalStart));

    const gTrain = new Garch(train).fit();
    const gFull = new Garch(candles);
    const gScore = qlike(gFull.getVarianceSeries(gTrain.params).slice(evalStart), rv.slice(evalStart));

    expect(rgScore).toBeLessThan(gScore);
  });

  it('forecast converges to the unconditional level', () => {
    const result = calibrateRealizedGarch(candles);
    const model = new RealizedGarch(candles);
    const fc = model.forecast(result.params, 500);
    expect(fc.variance[499] / result.params.unconditionalVariance).toBeCloseTo(1, 2);
    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── log-HAR ──────────────────────────────────────────────────

describe('log-HAR', () => {
  it('fits, reports the log spec, and stays positive without level clamps', () => {
    const candles = simRealizedGarchCandles(500, { omega: -0.35, beta: 0.5, gamma: 0.3, xi: -0.3, tau1: 0, tau2: 0, sigmaU: 0.5 }, 5);
    const model = new HarRv(candles, { logSpec: true });
    const fit = model.fit();

    expect(fit.params.logSpec).toBe(true);
    expect(fit.params.residualLogVar).toBeGreaterThan(0);
    expect(fit.params.unconditionalVariance).toBeGreaterThan(0);

    const series = model.getVarianceSeries(fit.params);
    for (const v of series) {
      expect(v).toBeGreaterThan(1e-18); // never the 1e-20 negative-prediction clamp
      expect(Number.isFinite(v)).toBe(true);
    }
    const fc = model.forecast(fit.params, 10);
    for (const v of fc.variance) expect(v).toBeGreaterThan(0);
  });

  it('coefficients are far more stable under a single extreme RV print than the level spec', () => {
    // A 100× RV outlier is a leverage point for the level OLS: it drags the
    // slopes (attenuation), changing the fitted persistence everywhere. In
    // log space the same print is an ordinary residual.
    const build = (withSpike: boolean): Candle[] => {
      const rng = mulberry32(77);
      const candles: Candle[] = [{ open: 100, high: 100, low: 100, close: 100, volume: 1 }];
      let close = 100;
      for (let i = 0; i < 400; i++) {
        const r = 0.01 * randn(rng);
        const open = close;
        close = open * Math.exp(r);
        const rv = withSpike && i === 200 ? 1e-2 : 1e-4 * (0.5 + rng());
        candles.push(candleWithRv(open, close, rv));
      }
      return candles;
    };
    const clean = build(false);
    const spiked = build(true);

    const shift = (logSpec: boolean): number => {
      const pClean = new HarRv(clean, { logSpec }).fit().params.persistence;
      const pSpiked = new HarRv(spiked, { logSpec }).fit().params.persistence;
      return Math.abs(pSpiked - pClean);
    };

    expect(shift(true)).toBeLessThan(shift(false));
  });
});

// ── Exponential forgetting ───────────────────────────────────

describe('exponential forgetting in the likelihood', () => {
  it('tracks an upward variance regime break that the unweighted fit averages away', () => {
    const { prices, postBreakVar } = regimeBreakPrices(900, 0.6, 2026);

    const plain = new Garch(prices).fit();
    const adaptive = new Garch(prices).fit({ forgetting: 0.99 });

    const distPlain = Math.abs(Math.log(plain.params.unconditionalVariance / postBreakVar));
    const distAdaptive = Math.abs(Math.log(adaptive.params.unconditionalVariance / postBreakVar));

    // The λ-weighted likelihood anchors the level to the recent regime
    expect(distAdaptive).toBeLessThan(distPlain);
    expect(adaptive.params.unconditionalVariance).toBeGreaterThan(plain.params.unconditionalVariance);
  });

  it('λ = 1 reproduces the unweighted fit exactly', () => {
    const { prices } = regimeBreakPrices(400, 0.5, 7);
    const a = new Garch(prices).fit();
    const b = new Garch(prices).fit({ forgetting: 1 });
    expect(b.params.omega).toBe(a.params.omega);
    expect(b.params.alpha).toBe(a.params.alpha);
    expect(b.params.beta).toBe(a.params.beta);
  });
});

// ── Asymmetric corridor ──────────────────────────────────────

describe('asymmetric corridor (per-tail calibration)', () => {
  const candles = skewedCandles(700, 90210);

  it('widens the lower band on negatively skewed data', () => {
    const res = predict(candles, '1h' as CandleInterval, null, 0.95);
    expect(res.zScoreUp).toBeGreaterThan(0);
    expect(res.zScoreDown).toBeGreaterThan(0);
    // Left tail is 2× heavier by construction — the corridor must see it
    expect(res.zScoreDown).toBeGreaterThan(res.zScoreUp * 1.03);
    expect(res.zScore).toBeCloseTo((res.zScoreUp + res.zScoreDown) / 2, 12);
  });

  it('walk-forward: both tail exceedance rates stay near nominal (5% each at 90%)', () => {
    const window = 450;
    const confidence = 0.90;
    let upExceed = 0;
    let downExceed = 0;
    let total = 0;

    const predictor = createPredictor('1h');

    for (let i = window; i < candles.length - 1; i += 2) {
      const slice = candles.slice(i - window, i + 1);
      const price = slice[slice.length - 1].close;
      const res = predictor.predict(slice, price, confidence);
      const actual = candles[i + 1].close;
      if (actual > res.upperPrice) upExceed++;
      if (actual < res.lowerPrice) downExceed++;
      total++;
    }

    // ~125 points, 5% nominal per tail → binomial SE ≈ 1.9pp; bounds ≈ 3σ.
    // A symmetric corridor on this DGP puts ~8-9% in the lower tail and
    // ~1-2% in the upper.
    expect(total).toBeGreaterThanOrEqual(100);
    const upRate = (upExceed / total) * 100;
    const downRate = (downExceed / total) * 100;
    expect(upRate, `upper tail exceedance=${upRate.toFixed(1)}%`).toBeLessThanOrEqual(11);
    expect(downRate, `lower tail exceedance=${downRate.toFixed(1)}%`).toBeLessThanOrEqual(11);
    expect(upRate + downRate, 'total exceedance').toBeGreaterThanOrEqual(3);
    expect(upRate + downRate, 'total exceedance').toBeLessThanOrEqual(17);
  }, 900_000);
});

// ── Warm start ───────────────────────────────────────────────

describe('warm-started rolling predictions', () => {
  it('createPredictor matches cold predict closely and runs faster on rolling windows', () => {
    const candles = skewedCandles(560, 41);
    const window = 450;
    const predictor = createPredictor('1h');

    const coldStart = Date.now();
    const cold: Array<{ sigma: number; z: number }> = [];
    for (let i = window; i < candles.length - 1; i += 20) {
      const slice = candles.slice(i - window, i + 1);
      const r = predict(slice, '1h' as CandleInterval, null, 0.9);
      cold.push({ sigma: r.sigma, z: r.zScore });
    }
    const coldMs = Date.now() - coldStart;

    const warmStart = Date.now();
    const warm: Array<{ sigma: number; z: number }> = [];
    for (let i = window; i < candles.length - 1; i += 20) {
      const slice = candles.slice(i - window, i + 1);
      const r = predictor.predict(slice, null, 0.9);
      warm.push({ sigma: r.sigma, z: r.zScore });
    }
    const warmMs = Date.now() - warmStart;

    for (let k = 0; k < cold.length; k++) {
      expect(Math.abs(warm[k].sigma / cold[k].sigma - 1), `sigma at window ${k}`).toBeLessThan(0.1);
      expect(Math.abs(warm[k].z / cold[k].z - 1), `zScore at window ${k}`).toBeLessThan(0.1);
    }
    // The point of warm starts; generous margin to stay CI-safe
    expect(warmMs).toBeLessThan(coldMs);
  }, 900_000);
});
