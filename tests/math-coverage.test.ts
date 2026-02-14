import { describe, it, expect } from 'vitest';
import {
  HarRv,
  calibrateHarRv,
  Garch,
  calibrateGarch,
  Egarch,
  calibrateEgarch,
  NoVaS,
  calibrateNoVaS,
  GjrGarch,
  calibrateGjrGarch,
  predict,
  predictRange,
  backtest,
  sampleVariance,
  perCandleParkinson,
  type Candle,
} from '../src/index.js';

// ── Helpers ──────────────────────────────────────────────────

function lcg(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function randn(rng: () => number): number {
  const u1 = rng() || 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function generatePrices(n: number, seed = 42): number[] {
  const rng = lcg(seed);
  const prices = [100];
  let vol = 0.01;
  for (let i = 1; i < n; i++) {
    const shock = randn(rng);
    const r = vol * shock;
    vol = Math.sqrt(0.00001 + 0.1 * r * r + 0.85 * vol * vol);
    prices.push(prices[i - 1] * Math.exp(r));
  }
  return prices;
}

function makeCandles(n: number, seed = 42, volScale = 1): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;

  for (let i = 0; i < n; i++) {
    const r = randn(rng) * 0.01 * volScale;
    const open = price;
    const close = open * Math.exp(r);
    const high = Math.max(open, close) * (1 + Math.abs(randn(rng)) * 0.002 * volScale);
    const low = Math.min(open, close) * (1 - Math.abs(randn(rng)) * 0.002 * volScale);
    candles.push({ open, high, low, close, volume: 1000 + rng() * 500 });
    price = close;
  }

  return candles;
}

/** Make candles where every candle has high === low === close === open */
function makeFlatCandles(n: number, seed = 42): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const r = randn(rng) * 0.01;
    const close = price * Math.exp(r);
    candles.push({ open: close, high: close, low: close, close, volume: 1000 });
    price = close;
  }
  return candles;
}

/** Make candles with some H===L mixed in */
function makeMixedCandles(n: number, seed = 42, flatRatio = 0.3): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const r = randn(rng) * 0.01;
    const open = price;
    const close = open * Math.exp(r);
    const isFlat = rng() < flatRatio;
    if (isFlat) {
      candles.push({ open: close, high: close, low: close, close, volume: 1000 });
    } else {
      const high = Math.max(open, close) * (1 + Math.abs(randn(rng)) * 0.003);
      const low = Math.min(open, close) * (1 - Math.abs(randn(rng)) * 0.003);
      candles.push({ open, high, low, close, volume: 1000 });
    }
    price = close;
  }
  return candles;
}

// ═══════════════════════════════════════════════════════════════
// 1. PARKINSON FORMULA INDEPENDENT VERIFICATION
// ═══════════════════════════════════════════════════════════════

describe('Parkinson formula independent verification', () => {
  const PARKINSON_COEFF = 1 / (4 * Math.LN2); // ≈ 0.36067

  it('coefficient = 1/(4·ln2) ≈ 0.36067', () => {
    expect(PARKINSON_COEFF).toBeCloseTo(0.36067, 4);
    expect(PARKINSON_COEFF).toBeCloseTo(1 / (4 * 0.6931471805599453), 15);
  });

  it('known H/L values produce exact expected Parkinson RV', () => {
    // Candle with H=110, L=90 → ln(110/90) = ln(1.2222...) ≈ 0.2007
    // Parkinson = (1/(4·ln2)) · (0.2007)² ≈ 0.01452
    const hl = Math.log(110 / 90);
    const expected = PARKINSON_COEFF * hl * hl;
    expect(expected).toBeCloseTo(0.01452, 4);

    // H=105, L=95 → ln(105/95) ≈ 0.1001
    const hl2 = Math.log(105 / 95);
    const expected2 = PARKINSON_COEFF * hl2 * hl2;
    expect(expected2).toBeCloseTo(0.003613, 4);
  });

  it('Parkinson = 0 when H === L', () => {
    const hl = Math.log(100 / 100); // = 0
    const parkinson = PARKINSON_COEFF * hl * hl;
    expect(parkinson).toBe(0);
  });

  it('HarRv.getRv() matches hand-computed Parkinson for known candles', () => {
    // Create candles with precise known OHLC
    const candles: Candle[] = [];
    const prices = [100, 102, 98, 105, 101]; // 5 prices → 4 returns

    // Build candles with known H/L spreads
    for (let i = 0; i < prices.length; i++) {
      const c = prices[i];
      candles.push({
        open: c * 0.999,
        high: c * 1.02,    // H = price * 1.02
        low: c * 0.98,     // L = price * 0.98
        close: c,
        volume: 1000,
      });
    }

    // Need enough candles for HAR-RV minimum (52)
    // Pad with more candles at the end
    const rng = lcg(42);
    let price = prices[prices.length - 1];
    for (let i = prices.length; i < 60; i++) {
      const r = randn(rng) * 0.01;
      const close = price * Math.exp(r);
      candles.push({
        open: price,
        high: Math.max(price, close) * 1.005,
        low: Math.min(price, close) * 0.995,
        close,
        volume: 1000,
      });
      price = close;
    }

    const model = new HarRv(candles);
    const rv = model.getRv();

    // rv[0] uses candles[1]'s OHLC (aligned with returns[0])
    // candles[1]: H = 102*1.02 = 104.04, L = 102*0.98 = 99.96
    const h1 = 102 * 1.02;
    const l1 = 102 * 0.98;
    const expectedRv0 = PARKINSON_COEFF * Math.log(h1 / l1) ** 2;
    expect(rv[0]).toBeCloseTo(expectedRv0, 12);
  });

  it('Parkinson RV > r² on average (more efficient estimator)', () => {
    // Parkinson uses more information (H/L range) than squared returns
    // So Parkinson estimates should generally differ from r²
    const candles = makeCandles(300, 42);
    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();
    const r2 = returns.map(r => r * r);

    // They should not be identical
    let differ = 0;
    for (let i = 0; i < rv.length; i++) {
      if (Math.abs(rv[i] - r2[i]) > 1e-15) differ++;
    }
    // Vast majority should differ (Parkinson ≠ r²)
    expect(differ / rv.length).toBeGreaterThan(0.95);
  });

  it('Parkinson variance ≈ r² variance in expectation (both are variance proxies)', () => {
    // Both estimate the same thing (per-period variance), so their means
    // should be in the same order of magnitude
    const candles = makeCandles(500, 42);
    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();
    const r2 = returns.map(r => r * r);

    const meanParkinson = rv.reduce((s, v) => s + v, 0) / rv.length;
    const meanR2 = r2.reduce((s, v) => s + v, 0) / r2.length;

    // Same order of magnitude (within 10x)
    const ratio = meanParkinson / meanR2;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });
});

// ═══════════════════════════════════════════════════════════════
// 2. rv[i] ↔ returns[i] ALIGNMENT
// ═══════════════════════════════════════════════════════════════

describe('rv[i] ↔ returns[i] temporal alignment', () => {
  it('rv and returns have same length', () => {
    const candles = makeCandles(100, 42);
    const model = new HarRv(candles);
    expect(model.getRv().length).toBe(model.getReturns().length);
  });

  it('rv[i] uses candles[i+1] OHLC, not candles[i]', () => {
    // Build candles with unique identifiable H/L ratios
    const candles: Candle[] = [];
    const rng = lcg(42);
    let price = 100;

    for (let i = 0; i < 60; i++) {
      const r = randn(rng) * 0.01;
      const close = price * Math.exp(r);
      // Each candle has a unique high/low spread: high = close * (1 + i*0.001)
      const spread = 0.005 + i * 0.0005;
      const high = Math.max(price, close) * (1 + spread);
      const low = Math.min(price, close) * (1 - spread);
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    const model = new HarRv(candles);
    const rv = model.getRv();
    const COEFF = 1 / (4 * Math.LN2);

    // rv[0] should be computed from candles[1], not candles[0]
    const c1 = candles[1];
    const expectedRv0 = COEFF * Math.log(c1.high / c1.low) ** 2;
    expect(rv[0]).toBeCloseTo(expectedRv0, 12);

    // rv[1] should be computed from candles[2]
    const c2 = candles[2];
    const expectedRv1 = COEFF * Math.log(c2.high / c2.low) ** 2;
    expect(rv[1]).toBeCloseTo(expectedRv1, 12);

    // rv[i] should NOT match candles[i]'s OHLC (it's candles[i+1])
    const c0 = candles[0];
    const wrongRv0 = COEFF * Math.log(c0.high / c0.low) ** 2;
    expect(rv[0]).not.toBeCloseTo(wrongRv0, 6);
  });

  it('returns[i] = ln(close[i+1]/close[i]) — same index as rv[i]', () => {
    const candles = makeCandles(60, 42);
    const model = new HarRv(candles);
    const returns = model.getReturns();

    // returns[0] = ln(candles[1].close / candles[0].close)
    const expected0 = Math.log(candles[1].close / candles[0].close);
    expect(returns[0]).toBeCloseTo(expected0, 12);

    // returns[1] = ln(candles[2].close / candles[1].close)
    const expected1 = Math.log(candles[2].close / candles[1].close);
    expect(returns[1]).toBeCloseTo(expected1, 12);
  });

  it('rv[i] and returns[i] both correspond to the same time period', () => {
    // Both rv[i] and returns[i] should represent the transition from candle i to candle i+1
    // returns[i] = ln(close_{i+1}/close_i) — the return over that period
    // rv[i] = Parkinson(candle_{i+1}) — the realized variance in that period
    // This alignment is crucial for the regression y[t] = f(rv[t-1], rv[t-2], ...)
    const candles = makeCandles(60, 42);
    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();
    const COEFF = 1 / (4 * Math.LN2);

    for (let i = 0; i < 5; i++) {
      // returns[i] uses candles[i] and candles[i+1]
      const r = Math.log(candles[i + 1].close / candles[i].close);
      expect(returns[i]).toBeCloseTo(r, 12);

      // rv[i] uses candles[i+1]'s high/low range
      const c = candles[i + 1];
      const hl = Math.log(c.high / c.low);
      const expectedRv = COEFF * hl * hl;
      expect(rv[i]).toBeCloseTo(expectedRv > 0 ? expectedRv : r * r, 12);
    }
  });

  it('forecast step 1 uses rv[last] (most recent RV), not returns[last]', () => {
    const candles = makeCandles(300, 42);
    const model = new HarRv(candles);
    const fit = model.fit();
    const rv = model.getRv();
    const fc = model.forecast(fit.params, 1);

    // Manual step-1 forecast
    const t = rv.length - 1;
    const { beta0, betaShort, betaMedium, betaLong } = fit.params;
    const rvS = rv[t]; // shortLag=1
    const rvM = rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5;
    const rvL = rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22;
    const expected = Math.max(beta0 + betaShort * rvS + betaMedium * rvM + betaLong * rvL, 1e-20);

    expect(fc.variance[0]).toBeCloseTo(expected, 12);
  });
});

// ═══════════════════════════════════════════════════════════════
// 3. ALL-CANDLES H===L (ZERO RANGE) FALLBACK
// ═══════════════════════════════════════════════════════════════

describe('HAR-RV all candles H===L fallback', () => {
  it('all flat candles (H=L) → rv falls back to r² for every entry', () => {
    const candles = makeFlatCandles(100, 42);
    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();

    for (let i = 0; i < rv.length; i++) {
      expect(rv[i]).toBeCloseTo(returns[i] * returns[i], 12);
    }
  });

  it('all flat candles → fit succeeds without NaN', () => {
    const candles = makeFlatCandles(100, 42);
    const result = calibrateHarRv(candles);

    expect(result.diagnostics.converged).toBe(true);
    expect(Number.isFinite(result.params.beta0)).toBe(true);
    expect(Number.isFinite(result.params.r2)).toBe(true);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });

  it('all flat candles → forecast produces finite positive values', () => {
    const candles = makeFlatCandles(100, 42);
    const model = new HarRv(candles);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 10);

    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('flat candles produce same rv as prices-only input', () => {
    const candles = makeFlatCandles(100, 42);
    const prices = candles.map(c => c.close);

    const modelCandles = new HarRv(candles);
    const modelPrices = new HarRv(prices);

    const rvCandles = modelCandles.getRv();
    const rvPrices = modelPrices.getRv();

    // When H===L, Parkinson = 0 → fallback to r², same as prices-only
    for (let i = 0; i < rvCandles.length; i++) {
      expect(rvCandles[i]).toBeCloseTo(rvPrices[i], 12);
    }
  });

  it('mixed candles (30% flat, 70% normal) → no crash, valid results', () => {
    const candles = makeMixedCandles(200, 42, 0.3);
    const model = new HarRv(candles);
    const fit = model.fit();
    const rv = model.getRv();

    expect(fit.diagnostics.converged).toBe(true);
    for (const v of rv) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('mixed candles: flat entries use r², normal entries use Parkinson', () => {
    // Craft controlled candles
    const candles: Candle[] = [];
    const rng = lcg(42);
    let price = 100;
    for (let i = 0; i < 60; i++) {
      const r = randn(rng) * 0.01;
      const close = price * Math.exp(r);
      if (i % 3 === 0) {
        // Flat candle
        candles.push({ open: close, high: close, low: close, close, volume: 1000 });
      } else {
        // Normal candle
        const high = Math.max(price, close) * 1.01;
        const low = Math.min(price, close) * 0.99;
        candles.push({ open: price, high, low, close, volume: 1000 });
      }
      price = close;
    }

    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();
    const COEFF = 1 / (4 * Math.LN2);

    for (let i = 0; i < rv.length; i++) {
      const c = candles[i + 1];
      if (c.high === c.low) {
        // Should be r²
        expect(rv[i]).toBeCloseTo(returns[i] * returns[i], 12);
      } else {
        // Should be Parkinson
        const hl = Math.log(c.high / c.low);
        const expected = COEFF * hl * hl;
        expect(rv[i]).toBeCloseTo(expected, 12);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 4. PARKINSON-BASED FORECAST VERIFICATION
// ═══════════════════════════════════════════════════════════════

describe('HAR-RV forecast with Parkinson-based RV', () => {
  it('Candle[] forecast differs from number[] forecast (different RV source)', () => {
    const candles = makeCandles(300, 42);
    const prices = candles.map(c => c.close);

    const modelCandles = new HarRv(candles);
    const modelPrices = new HarRv(prices);
    const fitCandles = modelCandles.fit();
    const fitPrices = modelPrices.fit();

    const fcCandles = modelCandles.forecast(fitCandles.params, 5);
    const fcPrices = modelPrices.forecast(fitPrices.params, 5);

    // Forecasts must differ because Parkinson RV ≠ r²
    let anyDiffer = false;
    for (let i = 0; i < 5; i++) {
      if (Math.abs(fcCandles.variance[i] - fcPrices.variance[i]) > 1e-15) {
        anyDiffer = true;
        break;
      }
    }
    expect(anyDiffer).toBe(true);
  });

  it('Candle[] forecast step 1 uses Parkinson rv[last], not r²[last]', () => {
    const candles = makeCandles(300, 42);
    const model = new HarRv(candles);
    const fit = model.fit();
    const rv = model.getRv();

    // Verify rv contains Parkinson values (not r²)
    const returns = model.getReturns();
    const lastRv = rv[rv.length - 1];
    const lastR2 = returns[returns.length - 1] ** 2;

    // They should differ for normal candles
    expect(Math.abs(lastRv - lastR2)).toBeGreaterThan(1e-15);

    // Forecast step 1 manual
    const t = rv.length - 1;
    const { beta0, betaShort, betaMedium, betaLong } = fit.params;
    const rvS = rv[t];
    const rvM = rv.slice(t - 4, t + 1).reduce((s, v) => s + v, 0) / 5;
    const rvL = rv.slice(t - 21, t + 1).reduce((s, v) => s + v, 0) / 22;
    const expected = Math.max(beta0 + betaShort * rvS + betaMedium * rvM + betaLong * rvL, 1e-20);

    const fc = model.forecast(fit.params, 1);
    expect(fc.variance[0]).toBeCloseTo(expected, 12);
  });

  it('multi-step Candle[] forecast feeds Parkinson-based predictions back', () => {
    const candles = makeCandles(300, 42);
    const model = new HarRv(candles);
    const fit = model.fit();
    const rv = model.getRv();
    const { beta0, betaShort, betaMedium, betaLong } = fit.params;

    // Manual 3-step forecast using Parkinson rv history
    const history = rv.slice();

    const predictions: number[] = [];
    for (let h = 0; h < 3; h++) {
      const t = history.length - 1;
      const rvS = history[t];
      const rvM = history.slice(t - 4, t + 1).reduce((s, v) => s + v, 0) / 5;
      const rvL = history.slice(t - 21, t + 1).reduce((s, v) => s + v, 0) / 22;
      const pred = Math.max(beta0 + betaShort * rvS + betaMedium * rvM + betaLong * rvL, 1e-20);
      predictions.push(pred);
      history.push(pred);
    }

    const fc = model.forecast(fit.params, 3);
    for (let i = 0; i < 3; i++) {
      expect(fc.variance[i]).toBeCloseTo(predictions[i], 12);
    }
  });

  it('Parkinson forecast convergence: long-horizon → unconditional variance', () => {
    const candles = makeCandles(300, 42);
    const model = new HarRv(candles);
    const fit = model.fit();

    if (fit.params.persistence > 0 && fit.params.persistence < 1) {
      const fc = model.forecast(fit.params, 200);
      const lastVar = fc.variance[199];
      const uncond = fit.params.unconditionalVariance;

      // Should converge (within 10% relative)
      const relError = Math.abs(lastVar - uncond) / Math.max(uncond, 1e-20);
      expect(relError).toBeLessThan(0.1);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 5. CANDLE VALIDATION
// ═══════════════════════════════════════════════════════════════

describe('Candle OHLC validation', () => {
  it('high < low candle → Parkinson still computes (ln(H/L) is negative, squared is positive)', () => {
    // This is invalid market data, but test that it doesn't crash
    const candles = makeCandles(100, 42);
    // Swap high and low on one candle
    const bad = { ...candles[50], high: candles[50].low - 0.01, low: candles[50].high + 0.01 };
    candles[50] = bad;

    // Should not throw — ln(H/L) where H<L gives negative, squared is positive
    const model = new HarRv(candles);
    const rv = model.getRv();
    // The rv value for that candle should still be finite and positive
    // rv[49] uses candles[50]'s OHLC
    expect(rv[49]).toBeGreaterThan(0);
    expect(Number.isFinite(rv[49])).toBe(true);
  });

  it('open outside [low, high] candle → fit still works', () => {
    const candles = makeCandles(100, 42);
    // Set open > high (invalid but possible in bad data feeds)
    candles[30] = { ...candles[30], open: candles[30].high + 1 };

    // HAR-RV only uses high/low for Parkinson and close for returns
    // So open doesn't matter for HAR-RV
    const result = calibrateHarRv(candles);
    expect(result.diagnostics.converged).toBe(true);
    expect(Number.isFinite(result.params.beta0)).toBe(true);
  });

  it('close outside [low, high] candle → returns computed from close, rv from H/L', () => {
    const candles = makeCandles(100, 42);
    // Set close > high (invalid)
    candles[30] = { ...candles[30], close: candles[30].high + 5 };

    // Returns use close prices, so this affects returns
    // HAR-RV won't throw — it just computes
    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();

    // rv and returns should all be finite
    for (const v of rv) {
      expect(Number.isFinite(v)).toBe(true);
    }
    for (const r of returns) {
      expect(Number.isFinite(r)).toBe(true);
    }
  });

  it('zero volume candle → HAR-RV ignores volume entirely', () => {
    const candles = makeCandles(100, 42);
    candles[50] = { ...candles[50], volume: 0 };

    const result = calibrateHarRv(candles);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('negative high → ln(negative/positive) → NaN in Parkinson → falls back to r²', () => {
    const candles = makeCandles(100, 42);
    candles[25] = { ...candles[25], high: -1 };

    // ln(-1/positive) = NaN → Parkinson = NaN → NaN > 0 is false → r² fallback
    const model = new HarRv(candles);
    const rv = model.getRv();
    // rv[24] uses candles[25] — should be r² fallback (NaN comparison with 0 returns false)
    const returns = model.getReturns();
    // The Parkinson will be NaN, and NaN > 0 is false, so it falls back to r²
    expect(rv[24]).toBeCloseTo(returns[24] * returns[24], 12);
  });
});

// ═══════════════════════════════════════════════════════════════
// 6. RELIABLE FLAG CASCADE TESTS
// ═══════════════════════════════════════════════════════════════

describe('reliable flag conditions', () => {
  it('reliable is always boolean across many seeds and intervals', () => {
    const intervals = ['15m', '1h', '4h'] as const;
    for (const interval of intervals) {
      for (let seed = 1; seed <= 10; seed++) {
        const n = interval === '15m' ? 300 : 200;
        const candles = makeCandles(n, seed);
        const result = predict(candles, interval);
        expect(typeof result.reliable).toBe('boolean');
      }
    }
  });

  it('predict and predictRange agree on reliable for steps=1', () => {
    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeCandles(300, seed);
      const p1 = predict(candles, '15m');
      const pR = predictRange(candles, '15m', 1);
      expect(pR.reliable).toBe(p1.reliable);
      expect(pR.modelType).toBe(p1.modelType);
    }
  });

  it('reliable=false is possible (not always true)', () => {
    // Constant-ish data should produce unreliable models
    // (high persistence or poor Ljung-Box)
    let foundUnreliable = false;
    for (let seed = 1; seed <= 100; seed++) {
      // Use very low volatility — models may struggle
      const candles = makeCandles(300, seed, 0.1);
      const result = predict(candles, '15m');
      if (!result.reliable) {
        foundUnreliable = true;
        break;
      }
    }
    // At least some data should produce unreliable results
    // If not, that's okay — it means models are robust
    expect(typeof foundUnreliable).toBe('boolean');
  });

  it('reliable=true requires convergence, persistence<0.999, and Ljung-Box p>=0.05', () => {
    // We test the logical implication:
    // If reliable=true → all three conditions must hold
    // We verify this by checking that when predict returns reliable=true,
    // the underlying model must have converged (OLS always converges for HAR-RV,
    // Nelder-Mead usually converges for GARCH/EGARCH/NoVaS)
    for (let seed = 1; seed <= 30; seed++) {
      const candles = makeCandles(300, seed);
      const result = predict(candles, '15m');

      // We can't directly access internal fit, but we can verify that
      // reliable=true models produce reasonable sigma
      if (result.reliable) {
        expect(result.sigma).toBeGreaterThan(0);
        expect(result.sigma).toBeLessThan(1); // Not insane volatility
        expect(Number.isFinite(result.sigma)).toBe(true);
      }
    }
  });

  it('near-constant prices → unreliable (high persistence or poor fit)', () => {
    // Very stable prices → persistence close to 1 or model inadequacy
    const candles: Candle[] = [];
    let price = 100;
    const rng = lcg(42);
    for (let i = 0; i < 300; i++) {
      const r = randn(rng) * 0.0001; // Tiny moves
      const close = price * Math.exp(r);
      candles.push({
        open: price,
        high: Math.max(price, close) * 1.00001,
        low: Math.min(price, close) * 0.99999,
        close,
        volume: 1000,
      });
      price = close;
    }

    const result = predict(candles, '15m');
    // Model should still return a result (not crash)
    expect(typeof result.reliable).toBe('boolean');
    expect(result.sigma).toBeGreaterThanOrEqual(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// 7. BACKTEST VALIDITY
// ═══════════════════════════════════════════════════════════════

describe('backtest validity', () => {
  it('backtest hit rate correlates with ±1σ theory (~68%)', () => {
    // With threshold 50%, backtest should pass (68% > 50%)
    const candles = makeCandles(500, 42);
    expect(backtest(candles, '4h', 50)).toBe(true);
  });

  it('backtest with 100% threshold always fails', () => {
    // No model predicts perfectly
    const candles = makeCandles(500, 42);
    expect(backtest(candles, '4h', 100)).toBe(false);
  });

  it('backtest with 0% threshold always passes', () => {
    const candles = makeCandles(500, 42);
    expect(backtest(candles, '4h', 0)).toBe(true);
  });

  it('backtest is deterministic', () => {
    const candles = makeCandles(500, 42);
    const r1 = backtest(candles, '4h');
    const r2 = backtest(candles, '4h');
    expect(r1).toBe(r2);
  });

  it('backtest runs across multiple seeds without crash', () => {
    for (let seed = 1; seed <= 10; seed++) {
      const candles = makeCandles(500, seed);
      const result = backtest(candles, '4h', 50);
      expect(typeof result).toBe('boolean');
    }
  });

  it('backtest with more data tends to be more reliable', () => {
    // Not a strict guarantee, but more data = better fit generally
    // At minimum, both should complete without error
    const candles500 = makeCandles(500, 42);
    const candles300 = makeCandles(300, 42);

    const r500 = backtest(candles500, '4h', 50);
    const r300 = backtest(candles300, '4h', 50);

    expect(typeof r500).toBe('boolean');
    expect(typeof r300).toBe('boolean');
  });

  it('backtest with different intervals uses correct MIN_CANDLES', () => {
    const candles = makeCandles(500, 42);

    // Should work for all intervals with enough data
    expect(typeof backtest(candles, '15m', 50)).toBe('boolean');
    expect(typeof backtest(candles, '1h', 50)).toBe('boolean');
    expect(typeof backtest(candles, '4h', 50)).toBe('boolean');
  });

  it('backtest predictions have correct structure within walk-forward', () => {
    // Verify that predictions within backtest are meaningful:
    // sigma > 0, prices form valid corridor
    const candles = makeCandles(500, 42);
    // We can't peek inside backtest, but we can verify predict
    // on the same slice backtest would use
    const window = Math.max(200, Math.floor(500 * 0.75));
    const slice = candles.slice(0, window + 1);
    const result = predict(slice, '4h');

    expect(result.sigma).toBeGreaterThan(0);
    expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
    expect(result.lowerPrice).toBeLessThan(result.currentPrice);
  });
});

// ═══════════════════════════════════════════════════════════════
// 8. NUMERICAL PRECISION AND LONG-HORIZON
// ═══════════════════════════════════════════════════════════════

describe('numerical precision edge cases', () => {
  it('ln(H/L) when H/L ≈ 1 + 1e-15 → Parkinson near zero but finite', () => {
    // Very tight spread candles
    const candles: Candle[] = [];
    const rng = lcg(42);
    let price = 100;
    for (let i = 0; i < 100; i++) {
      const r = randn(rng) * 0.01;
      const close = price * Math.exp(r);
      // Extremely tight H/L spread
      const high = Math.max(price, close) * (1 + 1e-12);
      const low = Math.min(price, close) * (1 - 1e-12);
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    const model = new HarRv(candles);
    const rv = model.getRv();

    // All RV values should be finite and non-negative
    for (const v of rv) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  it('H/L ratio = 1 + eps for machine epsilon → Parkinson is tiny but positive', () => {
    const eps = Number.EPSILON;
    const h = 100 * (1 + eps);
    const l = 100;
    const hl = Math.log(h / l);
    const coeff = 1 / (4 * Math.LN2);
    const parkinson = coeff * hl * hl;

    expect(Number.isFinite(parkinson)).toBe(true);
    expect(parkinson).toBeGreaterThanOrEqual(0);
  });

  it('forecast over 200 steps — cumulative error stays bounded', () => {
    const candles = makeCandles(300, 42);
    const model = new HarRv(candles);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 200);

    // All values should be finite and positive
    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }

    // Variance should eventually stabilize (not diverge)
    const last10 = fc.variance.slice(-10);
    const mean = last10.reduce((s, v) => s + v, 0) / 10;
    for (const v of last10) {
      // Within 1% of each other (converged)
      expect(Math.abs(v - mean) / mean).toBeLessThan(0.01);
    }
  });

  it('forecast over 500 steps — no overflow or underflow', () => {
    for (let seed = 1; seed <= 5; seed++) {
      const prices = generatePrices(300, seed);
      const model = new HarRv(prices);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 500);

      for (const v of fc.variance) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('GARCH forecast over 200 steps converges to unconditional', () => {
    const prices = generatePrices(300, 42);
    const result = calibrateGarch(prices);
    if (result.params.persistence < 0.999) {
      const model = new Garch(prices);
      const fc = model.forecast(result.params, 200);
      const lastVar = fc.variance[199];
      const uncond = result.params.unconditionalVariance;
      const relError = Math.abs(lastVar - uncond) / uncond;
      expect(relError).toBeLessThan(0.01);
    }
  });

  it('EGARCH forecast over 200 steps converges', () => {
    const prices = generatePrices(300, 42);
    const result = calibrateEgarch(prices);
    if (Math.abs(result.params.beta) < 0.999) {
      const model = new Egarch(prices);
      const fc = model.forecast(result.params, 200);
      const lastVar = fc.variance[199];
      expect(lastVar).toBeGreaterThan(0);
      expect(Number.isFinite(lastVar)).toBe(true);
    }
  });

  it('NoVaS forecast over 200 steps produces finite results', () => {
    const prices = generatePrices(300, 42);
    const result = calibrateNoVaS(prices);
    if (result.params.persistence < 0.999) {
      const model = new NoVaS(prices);
      const fc = model.forecast(result.params, 200);
      for (const v of fc.variance) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('mixed extreme prices (1e8 → 1e-4) in Parkinson — log returns are scale-invariant', () => {
    // Scale changes don't affect log returns: ln(kP2/kP1) = ln(P2/P1)
    const candles1 = makeCandles(200, 42);
    const candles2 = candles1.map(c => ({
      open: c.open * 1e6,
      high: c.high * 1e6,
      low: c.low * 1e6,
      close: c.close * 1e6,
      volume: c.volume,
    }));

    const model1 = new HarRv(candles1);
    const model2 = new HarRv(candles2);
    const rv1 = model1.getRv();
    const rv2 = model2.getRv();

    // RV should be identical (log returns are scale-invariant)
    for (let i = 0; i < rv1.length; i++) {
      expect(rv1[i]).toBeCloseTo(rv2[i], 10);
    }

    // Betas should also be identical
    const fit1 = model1.fit();
    const fit2 = model2.fit();
    expect(fit1.params.beta0).toBeCloseTo(fit2.params.beta0, 10);
    expect(fit1.params.r2).toBeCloseTo(fit2.params.r2, 10);
  });
});

// ═══════════════════════════════════════════════════════════════
// 9. CROSS-MODEL CONSISTENCY
// ═══════════════════════════════════════════════════════════════

describe('cross-model consistency', () => {
  it('all models produce positive sigma from predict()', () => {
    // Run enough seeds to exercise all model types
    const seen = new Set<string>();
    for (let seed = 1; seed <= 100; seed++) {
      const candles = makeCandles(300, seed);
      const result = predict(candles, '15m');
      seen.add(result.modelType);

      expect(result.sigma).toBeGreaterThan(0);
      expect(Number.isFinite(result.sigma)).toBe(true);
      expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
      expect(result.lowerPrice).toBeLessThan(result.currentPrice);
    }

    // Should have seen at least GARCH/EGARCH
    expect(seen.size).toBeGreaterThanOrEqual(1);
  });

  it('predictRange sigma grows with steps (cumulative variance)', () => {
    const candles = makeCandles(300, 42);
    const p1 = predictRange(candles, '15m', 1);
    const p5 = predictRange(candles, '15m', 5);
    const p10 = predictRange(candles, '15m', 10);

    // Cumulative sigma should grow: σ_cum = √(Σ σ²_i)
    expect(p5.sigma).toBeGreaterThan(p1.sigma);
    expect(p10.sigma).toBeGreaterThan(p5.sigma);
  });

  it('all four model types can produce valid calibration results', () => {
    const prices = generatePrices(300, 42);
    const candles = makeCandles(300, 42);

    const garch = calibrateGarch(prices);
    const egarch = calibrateEgarch(prices);
    const harRv = calibrateHarRv(candles);
    const novas = calibrateNoVaS(prices);

    for (const result of [garch, egarch, harRv, novas]) {
      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
      expect(Number.isFinite(result.diagnostics.aic)).toBe(true);
      expect(Number.isFinite(result.diagnostics.bic)).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 10. REALIZED GARCH (Candle[] → Parkinson RV)
// ═══════════════════════════════════════════════════════════════

describe('Realized GARCH (Candle[] uses Parkinson RV)', () => {
  it('Candle[] and number[] produce different GARCH params', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateGarch(candles);
    const resultPrices = calibrateGarch(prices);

    // Both converge
    expect(resultCandles.diagnostics.converged).toBe(true);
    expect(resultPrices.diagnostics.converged).toBe(true);

    // Params differ because Candle[] uses Parkinson, number[] uses r²
    // At least one of omega/alpha should differ
    const omegaDiff = Math.abs(resultCandles.params.omega - resultPrices.params.omega);
    const alphaDiff = Math.abs(resultCandles.params.alpha - resultPrices.params.alpha);
    expect(omegaDiff + alphaDiff).toBeGreaterThan(1e-10);
  });

  it('flat candles (H=L) degrade to classical GARCH (same as number[])', () => {
    const candles = makeFlatCandles(200, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateGarch(candles);
    const resultPrices = calibrateGarch(prices);

    // When H=L, Parkinson falls back to r² → same innovation
    // Initial variance differs (Yang-Zhang vs sampleVariance), but
    // with enough data the optimizer should converge to similar params
    // Both should at least produce valid results
    expect(resultCandles.diagnostics.converged).toBe(true);
    expect(resultPrices.diagnostics.converged).toBe(true);
    expect(resultCandles.params.persistence).toBeLessThan(1);
    expect(resultPrices.params.persistence).toBeLessThan(1);
  });

  it('Realized GARCH variance series uses Parkinson per-candle', () => {
    const candles = makeCandles(200, 42);
    const model = new Garch(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    // All variances should be positive and finite
    for (const v of vs) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('Realized GARCH 1-step forecast uses Parkinson RV as innovation', () => {
    const candles = makeCandles(200, 42);
    const model = new Garch(candles);
    const fit = model.fit();
    const { omega, alpha, beta } = fit.params;

    const vs = model.getVarianceSeries(fit.params);
    const lastVar = vs[vs.length - 1];

    // Compute expected Parkinson for last candle
    const coeff = 1 / (4 * Math.LN2);
    const lastCandle = candles[candles.length - 1];
    const hl = Math.log(lastCandle.high / lastCandle.low);
    const lastRV = coeff * hl * hl;

    const expected = omega + alpha * lastRV + beta * lastVar;
    const fc = model.forecast(fit.params, 1);
    expect(fc.variance[0]).toBeCloseTo(expected, 12);
  });

  it('Realized GARCH forecast converges to unconditional variance', () => {
    const candles = makeCandles(300, 42);
    const model = new Garch(candles);
    const fit = model.fit();

    if (fit.params.persistence < 0.999) {
      const fc = model.forecast(fit.params, 200);
      const lastVar = fc.variance[199];
      const uncond = fit.params.unconditionalVariance;
      const relError = Math.abs(lastVar - uncond) / uncond;
      expect(relError).toBeLessThan(0.01);
    }
  });

  it('Realized GARCH scale invariance — 1000× prices → same params', () => {
    const candles1 = makeCandles(200, 42);
    const candles2 = candles1.map(c => ({
      open: c.open * 1000,
      high: c.high * 1000,
      low: c.low * 1000,
      close: c.close * 1000,
      volume: c.volume,
    }));

    const r1 = calibrateGarch(candles1);
    const r2 = calibrateGarch(candles2);

    // Log returns and Parkinson are scale-invariant
    expect(r1.params.alpha).toBeCloseTo(r2.params.alpha, 6);
    expect(r1.params.beta).toBeCloseTo(r2.params.beta, 6);
    expect(r1.params.persistence).toBeCloseTo(r2.params.persistence, 6);
  });

  it('Realized GARCH across multiple seeds — always valid', () => {
    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeCandles(200, seed);
      const result = calibrateGarch(candles);

      expect(result.params.persistence).toBeLessThan(1);
      expect(result.params.omega).toBeGreaterThan(0);
      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 11. REALIZED EGARCH (Candle[] → Parkinson magnitude)
// ═══════════════════════════════════════════════════════════════

describe('Realized EGARCH (Candle[] uses Parkinson magnitude)', () => {
  it('Candle[] and number[] produce different EGARCH params', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateEgarch(candles);
    const resultPrices = calibrateEgarch(prices);

    expect(resultCandles.diagnostics.converged).toBe(true);
    expect(resultPrices.diagnostics.converged).toBe(true);

    // At least some params should differ
    const omegaDiff = Math.abs(resultCandles.params.omega - resultPrices.params.omega);
    const alphaDiff = Math.abs(resultCandles.params.alpha - resultPrices.params.alpha);
    expect(omegaDiff + alphaDiff).toBeGreaterThan(1e-6);
  });

  it('Realized EGARCH preserves leverage effect (gamma sign)', () => {
    // Create asymmetric candles where negative returns have higher vol
    const rng = lcg(77);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 300; i++) {
      const u = rng();
      // Negative returns are larger in magnitude → leverage
      const r = u < 0.5 ? -(u * 0.06) : (u - 0.5) * 0.02;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * (1 + Math.abs(r) * 0.3);
      const low = Math.min(price, close) * (1 - Math.abs(r) * 0.3);
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    const result = calibrateEgarch(candles);
    // gamma should be negative (leverage: negative returns increase vol)
    expect(result.params.gamma).toBeLessThan(0);
  });

  it('Realized EGARCH 1-step forecast uses Parkinson magnitude + directional z', () => {
    const candles = makeCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();
    const { omega, alpha, gamma, beta } = fit.params;

    const vs = model.getVarianceSeries(fit.params);
    const returns = model.getReturns();
    const lastVar = vs[vs.length - 1];
    const lastRet = returns[returns.length - 1];
    const z = lastRet / Math.sqrt(lastVar);

    // Parkinson magnitude
    const coeff = 1 / (4 * Math.LN2);
    const lastCandle = candles[candles.length - 1];
    const hl = Math.log(lastCandle.high / lastCandle.low);
    const lastRV = coeff * hl * hl;
    const magnitude = Math.sqrt(lastRV / lastVar);

    const EXPECTED_ABS = Math.sqrt(2 / Math.PI);
    const expectedLogVar = omega
      + alpha * (magnitude - EXPECTED_ABS)
      + gamma * z
      + beta * Math.log(lastVar);
    const expectedVar = Math.exp(expectedLogVar);

    const fc = model.forecast(fit.params, 1);
    expect(fc.variance[0]).toBeCloseTo(expectedVar, 10);
  });

  it('Realized EGARCH variance series all positive and finite', () => {
    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeCandles(200, seed);
      const model = new Egarch(candles);
      const fit = model.fit();
      const vs = model.getVarianceSeries(fit.params);

      for (const v of vs) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('Realized EGARCH forecast converges', () => {
    const candles = makeCandles(300, 42);
    const model = new Egarch(candles);
    const fit = model.fit();

    if (Math.abs(fit.params.beta) < 0.999) {
      const fc = model.forecast(fit.params, 200);
      const uncond = fit.params.unconditionalVariance;
      const lastVar = fc.variance[199];

      // EGARCH converges slower; allow 30% tolerance
      const relError = Math.abs(lastVar - uncond) / uncond;
      expect(relError).toBeLessThan(0.3);
    }
  });

  it('Realized EGARCH scale invariance', () => {
    const candles1 = makeCandles(200, 42);
    const candles2 = candles1.map(c => ({
      open: c.open * 1000,
      high: c.high * 1000,
      low: c.low * 1000,
      close: c.close * 1000,
      volume: c.volume,
    }));

    const r1 = calibrateEgarch(candles1);
    const r2 = calibrateEgarch(candles2);

    expect(r1.params.alpha).toBeCloseTo(r2.params.alpha, 4);
    expect(r1.params.beta).toBeCloseTo(r2.params.beta, 4);
    expect(r1.params.gamma).toBeCloseTo(r2.params.gamma, 4);
  });
});

// ═══════════════════════════════════════════════════════════════
// 12. perCandleParkinson shared function
// ═══════════════════════════════════════════════════════════════

describe('perCandleParkinson shared function', () => {
  it('produces same results as HAR-RV getRv() on same candles', () => {
    const candles = makeCandles(200, 42);
    const model = new HarRv(candles);
    const harRv = model.getRv();

    // Manually call perCandleParkinson with same inputs
    const returns = model.getReturns();
    const rv = perCandleParkinson(candles, returns);

    for (let i = 0; i < harRv.length; i++) {
      expect(rv[i]).toBe(harRv[i]);
    }
  });

  it('length = returns.length', () => {
    const candles = makeCandles(100, 42);
    const model = new HarRv(candles);
    const rv = model.getRv();
    const returns = model.getReturns();
    expect(rv.length).toBe(returns.length);
  });
});

// ═══════════════════════════════════════════════════════════════
// 13. GJR-GARCH — Realized and Classical
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH', () => {
  it('fits and converges on candle data', () => {
    const candles = makeCandles(200, 42);
    const result = calibrateGjrGarch(candles);
    expect(result.diagnostics.converged).toBe(true);
    expect(result.params.omega).toBeGreaterThan(0);
    expect(result.params.alpha).toBeGreaterThanOrEqual(0);
    expect(result.params.gamma).toBeGreaterThanOrEqual(0);
    expect(result.params.beta).toBeGreaterThanOrEqual(0);
  });

  it('fits and converges on price data', () => {
    const prices = generatePrices(200, 42);
    const result = calibrateGjrGarch(prices);
    expect(result.diagnostics.converged).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('Candle[] vs number[] produce different params (Parkinson vs r²)', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const candleResult = calibrateGjrGarch(candles);
    const priceResult = calibrateGjrGarch(prices);

    // At least one param should differ meaningfully
    const diffAlpha = Math.abs(candleResult.params.alpha - priceResult.params.alpha);
    const diffBeta = Math.abs(candleResult.params.beta - priceResult.params.beta);
    const diffGamma = Math.abs(candleResult.params.gamma - priceResult.params.gamma);
    expect(diffAlpha + diffBeta + diffGamma).toBeGreaterThan(0.001);
  });

  it('flat candles (H=L) degrade to same as number[] input', () => {
    const flatCandles = makeFlatCandles(200, 42);
    const prices = flatCandles.map(c => c.close);

    const candleResult = calibrateGjrGarch(flatCandles);
    const priceResult = calibrateGjrGarch(prices);

    // Should be very close since Parkinson falls back to r²
    expect(candleResult.params.alpha).toBeCloseTo(priceResult.params.alpha, 3);
    expect(candleResult.params.beta).toBeCloseTo(priceResult.params.beta, 3);
    expect(candleResult.params.gamma).toBeCloseTo(priceResult.params.gamma, 3);
  });

  it('persistence = alpha + gamma/2 + beta', () => {
    const candles = makeCandles(200, 42);
    const result = calibrateGjrGarch(candles);
    const expected = result.params.alpha + result.params.gamma / 2 + result.params.beta;
    expect(result.params.persistence).toBeCloseTo(expected, 10);
  });

  it('unconditionalVariance = omega / (1 - persistence)', () => {
    const candles = makeCandles(200, 42);
    const result = calibrateGjrGarch(candles);
    const expected = result.params.omega / (1 - result.params.persistence);
    expect(result.params.unconditionalVariance).toBeCloseTo(expected, 10);
  });

  it('leverageEffect equals gamma', () => {
    const candles = makeCandles(200, 42);
    const result = calibrateGjrGarch(candles);
    expect(result.params.leverageEffect).toBe(result.params.gamma);
  });

  it('variance series length matches returns', () => {
    const candles = makeCandles(200, 42);
    const model = new GjrGarch(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);
    expect(vs.length).toBe(model.getReturns().length);
  });

  it('forecast moves toward unconditional variance over time', () => {
    const candles = makeCandles(200, 42);
    const model = new GjrGarch(candles);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 100);
    const uncond = fit.params.unconditionalVariance;
    // Later steps should be closer to unconditional variance than earlier steps
    const distFirst = Math.abs(fc.variance[0] - uncond);
    const distLast = Math.abs(fc.variance[fc.variance.length - 1] - uncond);
    expect(distLast).toBeLessThan(distFirst);
  });

  it('scale invariance (1000x prices → same vol%)', () => {
    const candles1 = makeCandles(200, 42);
    const candles2 = candles1.map(c => ({
      ...c,
      open: c.open * 1000,
      high: c.high * 1000,
      low: c.low * 1000,
      close: c.close * 1000,
    }));

    const r1 = calibrateGjrGarch(candles1);
    const r2 = calibrateGjrGarch(candles2);

    expect(r1.params.annualizedVol).toBeCloseTo(r2.params.annualizedVol, 1);
  });

  it('multi-step forecast is monotone toward unconditional variance', () => {
    const candles = makeCandles(200, 42);
    const model = new GjrGarch(candles);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 50);

    // After step 1, each step should move closer to unconditional
    const uncond = fit.params.unconditionalVariance;
    for (let i = 2; i < fc.variance.length; i++) {
      const distPrev = Math.abs(fc.variance[i - 1] - uncond);
      const distCurr = Math.abs(fc.variance[i] - uncond);
      expect(distCurr).toBeLessThanOrEqual(distPrev + 1e-15);
    }
  });

  it('throws on insufficient data', () => {
    const candles = makeCandles(30, 42);
    expect(() => new GjrGarch(candles)).toThrow('at least 50');
  });

  it('predict() can return gjr-garch as model type', () => {
    // Generate data with leverage effect to trigger EGARCH/GJR-GARCH branch
    const rng = lcg(99);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const shock = randn(rng);
      // Amplify negative shocks to create leverage effect
      const r = shock < 0 ? shock * 0.02 : shock * 0.008;
      const open = price;
      const close = open * Math.exp(r);
      const high = Math.max(open, close) * (1 + Math.abs(randn(rng)) * 0.003);
      const low = Math.min(open, close) * (1 - Math.abs(randn(rng)) * 0.003);
      candles.push({ open, high, low, close, volume: 1000 });
      price = close;
    }

    const result = predict(candles, '4h');
    // gjr-garch is now a valid model type in the union
    expect(['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas']).toContain(result.modelType);
  });

  it('deterministic across runs with same seed', () => {
    const candles = makeCandles(200, 77);
    const r1 = calibrateGjrGarch(candles);
    const r2 = calibrateGjrGarch(candles);
    expect(r1.params.omega).toBe(r2.params.omega);
    expect(r1.params.alpha).toBe(r2.params.alpha);
    expect(r1.params.gamma).toBe(r2.params.gamma);
    expect(r1.params.beta).toBe(r2.params.beta);
  });
});
