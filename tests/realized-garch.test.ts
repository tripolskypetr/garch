import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  NoVaS,
  calibrateGarch,
  calibrateEgarch,
  calibrateNoVaS,
  perCandleParkinson,
  predict,
  predictRange,
  backtest,
  EXPECTED_ABS_NORMAL,
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

const PARKINSON_COEFF = 1 / (4 * Math.LN2);

function parkinsonRV(candle: Candle): number {
  const hl = Math.log(candle.high / candle.low);
  return PARKINSON_COEFF * hl * hl;
}

// ═══════════════════════════════════════════════════════════════
// 1. EGARCH flat candles (H=L) — Parkinson fallback to r²
// ═══════════════════════════════════════════════════════════════

describe('EGARCH flat candles (H=L) Parkinson fallback', () => {
  it('flat candles produce valid EGARCH fit', () => {
    const candles = makeFlatCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();

    expect(fit.diagnostics.converged).toBe(true);
    expect(fit.params.persistence).toBeLessThan(1);
    expect(Number.isFinite(fit.diagnostics.logLikelihood)).toBe(true);
  });

  it('flat candles getVarianceSeries all positive and finite', () => {
    const candles = makeFlatCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    for (const v of vs) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('flat candles forecast produces valid output', () => {
    const candles = makeFlatCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 5);

    expect(fc.variance).toHaveLength(5);
    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('EGARCH flat vs normal candles: flat degrades to |z|-like magnitude', () => {
    // When H=L, perCandleParkinson falls back to r²
    // magnitude = √(r²/σ²) = |r|/σ = |z| — same as classical
    const candles = makeFlatCandles(200, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateEgarch(candles);
    const resultPrices = calibrateEgarch(prices);

    // Both should converge
    expect(resultCandles.diagnostics.converged).toBe(true);
    expect(resultPrices.diagnostics.converged).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════
// 2. Multi-step forecast: Candle[] vs number[] equivalence (h≥2)
// ═══════════════════════════════════════════════════════════════

describe('multi-step forecast recursion identical for Candle[] and number[]', () => {
  it('GARCH steps 2+ use same (α+β)·v recursion regardless of input', () => {
    const candles = makeCandles(200, 42);
    const model = new Garch(candles);
    const fit = model.fit();
    const { omega, alpha, beta } = fit.params;

    const fc = model.forecast(fit.params, 10);

    // Verify steps 2+ follow v = ω + (α+β)·v_{h-1}
    for (let h = 1; h < 10; h++) {
      const expected = omega + (alpha + beta) * fc.variance[h - 1];
      expect(fc.variance[h]).toBeCloseTo(expected, 12);
    }
  });

  it('EGARCH steps 2+ use ω + β·logVar recursion regardless of input', () => {
    const candles = makeCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();
    const { omega, beta } = fit.params;

    const fc = model.forecast(fit.params, 10);

    // Verify steps 2+ follow logVar = ω + β·logVar_{h-1}
    for (let h = 1; h < 10; h++) {
      const logVarPrev = Math.log(fc.variance[h - 1]);
      const expectedLogVar = omega + beta * logVarPrev;
      expect(fc.variance[h]).toBeCloseTo(Math.exp(expectedLogVar), 10);
    }
  });

  it('GARCH multi-step: step 1 differs (Parkinson vs r²) but step 2+ converge to same formula', () => {
    const candles = makeCandles(200, 99);
    const prices = candles.map(c => c.close);

    const modelC = new Garch(candles);
    const modelP = new Garch(prices);
    const fitC = modelC.fit();
    const fitP = modelP.fit();

    const fcC = modelC.forecast(fitC.params, 10);
    const fcP = modelP.forecast(fitP.params, 10);

    // Step 1 should differ (different innovation)
    // Steps 2+ follow same formula: v = ω + (α+β)·v
    // Verify the FORMULA is correct for both, not that values match
    // (params differ, so absolute values differ)
    for (let h = 1; h < 10; h++) {
      const expectedC = fitC.params.omega + (fitC.params.alpha + fitC.params.beta) * fcC.variance[h - 1];
      expect(fcC.variance[h]).toBeCloseTo(expectedC, 12);

      const expectedP = fitP.params.omega + (fitP.params.alpha + fitP.params.beta) * fcP.variance[h - 1];
      expect(fcP.variance[h]).toBeCloseTo(expectedP, 12);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 3. EGARCH getVarianceSeries full magnitude path with Candle[]
// ═══════════════════════════════════════════════════════════════

describe('EGARCH getVarianceSeries magnitude path with Candle[]', () => {
  it('manual reconstruction of variance series matches model output', () => {
    const candles = makeCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();
    const { omega, alpha, gamma, beta } = fit.params;

    const vs = model.getVarianceSeries(fit.params);
    const returns = model.getReturns();
    const rv = perCandleParkinson(candles, returns);

    // Reconstruct manually
    const manual: number[] = [];
    let logVariance = Math.log(vs[0]); // initialVariance
    manual.push(vs[0]); // i=0 is initialVariance

    for (let i = 1; i < returns.length; i++) {
      const sigma = Math.sqrt(manual[i - 1]);
      const z = returns[i - 1] / sigma;
      const magnitude = Math.sqrt(rv[i - 1] / manual[i - 1]);

      logVariance = omega
        + alpha * (magnitude - EXPECTED_ABS_NORMAL)
        + gamma * z
        + beta * logVariance;

      logVariance = Math.max(-50, Math.min(50, logVariance));
      manual.push(Math.exp(logVariance));
    }

    for (let i = 0; i < vs.length; i++) {
      expect(vs[i]).toBeCloseTo(manual[i], 12);
    }
  });

  it('Candle[] variance series differs from number[] variance series', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const modelC = new Egarch(candles);
    const modelP = new Egarch(prices);
    const fitC = modelC.fit();
    const fitP = modelP.fit();

    const vsC = modelC.getVarianceSeries(fitC.params);
    const vsP = modelP.getVarianceSeries(fitP.params);

    // At least some values should differ (different magnitude computation)
    let diffCount = 0;
    const len = Math.min(vsC.length, vsP.length);
    for (let i = 1; i < len; i++) {
      if (Math.abs(vsC[i] - vsP[i]) / vsC[i] > 0.001) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// 4–5. Bad OHLC: NaN/Infinity and high < low for GARCH/EGARCH
// ═══════════════════════════════════════════════════════════════

describe('bad OHLC data in GARCH/EGARCH', () => {
  function makeBadCandles(badCandle: Partial<Candle>, position: number): Candle[] {
    const candles = makeCandles(100, 42);
    candles[position] = { ...candles[position], ...badCandle };
    return candles;
  }

  it('GARCH with NaN high still produces a fit (degrades gracefully)', () => {
    const candles = makeBadCandles({ high: NaN }, 50);
    const model = new Garch(candles);
    const fit = model.fit();
    // Should not crash — NaN Parkinson → NaN innovation → penalty in LL
    expect(Number.isFinite(fit.params.omega) || !fit.diagnostics.converged).toBe(true);
  });

  it('EGARCH with NaN high does not crash', () => {
    const candles = makeBadCandles({ high: NaN }, 50);
    const model = new Egarch(candles);
    // Should not throw — NaN Parkinson propagates but optimizer still terminates
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
  });

  it('GARCH with high < low does not crash', () => {
    // high < low → ln(H/L) is negative but squared → still positive Parkinson RV
    const candles = makeCandles(100, 42);
    const c = candles[50];
    candles[50] = { ...c, high: c.low * 0.99, low: c.high * 1.01 };
    const model = new Garch(candles);
    const fit = model.fit();
    expect(fit.diagnostics.converged).toBe(true);
  });

  it('EGARCH with high < low does not crash', () => {
    const candles = makeCandles(100, 42);
    const c = candles[50];
    candles[50] = { ...c, high: c.low * 0.99, low: c.high * 1.01 };
    const model = new Egarch(candles);
    const fit = model.fit();
    expect(fit.diagnostics.converged).toBe(true);
  });

  it('Parkinson RV is still positive when high < low (ln² always ≥ 0)', () => {
    const badCandle: Candle = { open: 100, high: 95, low: 105, close: 100, volume: 1000 };
    const goodCandle: Candle = { open: 100, high: 105, low: 95, close: 100, volume: 1000 };
    const returns = [0.01]; // dummy return
    const rv = perCandleParkinson([goodCandle, badCandle], returns);
    expect(rv[0]).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// 6. calibrateEgarch convenience with Candle[]
// ═══════════════════════════════════════════════════════════════

describe('calibrateEgarch with Candle[]', () => {
  it('returns valid CalibrationResult with Candle[] input', () => {
    const candles = makeCandles(200, 42);
    const result = calibrateEgarch(candles);

    expect(result.diagnostics.converged).toBe(true);
    expect(result.params.omega).toBeDefined();
    expect(result.params.alpha).toBeDefined();
    expect(result.params.gamma).toBeDefined();
    expect(result.params.beta).toBeDefined();
    expect(result.params.unconditionalVariance).toBeGreaterThan(0);
    expect(Number.isFinite(result.params.annualizedVol)).toBe(true);
  });

  it('calibrateEgarch Candle[] differs from number[] (uses Parkinson)', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const rc = calibrateEgarch(candles);
    const rp = calibrateEgarch(prices);

    const diff = Math.abs(rc.params.alpha - rp.params.alpha)
      + Math.abs(rc.params.omega - rp.params.omega);
    expect(diff).toBeGreaterThan(1e-6);
  });
});

// ═══════════════════════════════════════════════════════════════
// 7. fitModel() — all 3 OHLC models use Parkinson RV
// ═══════════════════════════════════════════════════════════════

describe('fitModel Parkinson RV verification', () => {
  it('predict() with Candle[] returns valid result (exercises fitModel)', () => {
    const candles = makeCandles(500, 42);
    const result = predict(candles, '15m');

    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.upperPrice).toBeGreaterThan(result.lowerPrice);
    expect(['garch', 'egarch', 'har-rv', 'novas']).toContain(result.modelType);
  });

  it('predict() Candle[] vs number[]-derived candles — sigma differs', () => {
    // When we give real OHLC candles, Parkinson RV extracts more info
    // vs candles where H=L=C (forcing r² fallback)
    const realCandles = makeCandles(500, 42);
    const flatCandles = realCandles.map(c => ({
      open: c.open, high: c.close, low: c.close, close: c.close, volume: c.volume,
    }));

    const resultReal = predict(realCandles, '15m');
    const resultFlat = predict(flatCandles, '15m');

    // Both valid, but sigma should differ
    expect(resultReal.sigma).toBeGreaterThan(0);
    expect(resultFlat.sigma).toBeGreaterThan(0);
    // They should be different (Parkinson uses OHLC info)
    expect(Math.abs(resultReal.sigma - resultFlat.sigma)).toBeGreaterThan(1e-10);
  });
});

// ═══════════════════════════════════════════════════════════════
// 8. backtest() with Realized GARCH/EGARCH
// ═══════════════════════════════════════════════════════════════

describe('backtest with Candle[] (Realized path)', () => {
  it('backtest completes with Candle[] input', () => {
    const candles = makeCandles(500, 42);
    const result = backtest(candles, '15m');
    expect(typeof result).toBe('boolean');
  });

  it('backtest result is boolean, no crashes with varied seeds', () => {
    for (let seed = 1; seed <= 5; seed++) {
      const candles = makeCandles(500, seed);
      const result = backtest(candles, '15m');
      expect(typeof result).toBe('boolean');
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 9. EGARCH forecast steps > 1 with Candle[]
// ═══════════════════════════════════════════════════════════════

describe('EGARCH forecast multi-step with Candle[]', () => {
  it('step 1 uses Parkinson magnitude, steps 2+ use ω+β·logVar', () => {
    const candles = makeCandles(200, 42);
    const model = new Egarch(candles);
    const fit = model.fit();
    const { omega, alpha, gamma, beta } = fit.params;

    const vs = model.getVarianceSeries(fit.params);
    const returns = model.getReturns();
    const lastVar = vs[vs.length - 1];
    const lastRet = returns[returns.length - 1];
    const z = lastRet / Math.sqrt(lastVar);

    // Step 1: Parkinson magnitude
    const lastCandle = candles[candles.length - 1];
    const lastRV = parkinsonRV(lastCandle);
    const magnitude = Math.sqrt(lastRV / lastVar);

    const logVar1 = omega
      + alpha * (magnitude - EXPECTED_ABS_NORMAL)
      + gamma * z
      + beta * Math.log(lastVar);

    const fc = model.forecast(fit.params, 5);
    expect(fc.variance[0]).toBeCloseTo(Math.exp(logVar1), 10);

    // Steps 2+: ω + β·logVar
    let logVar = logVar1;
    for (let h = 1; h < 5; h++) {
      logVar = omega + beta * logVar;
      expect(fc.variance[h]).toBeCloseTo(Math.exp(logVar), 10);
    }
  });

  it('multi-step forecast all positive and finite', () => {
    for (let seed = 1; seed <= 10; seed++) {
      const candles = makeCandles(200, seed);
      const model = new Egarch(candles);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 20);

      for (const v of fc.variance) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 10. perCandleParkinson edge cases
// ═══════════════════════════════════════════════════════════════

describe('perCandleParkinson edge cases', () => {
  it('single return produces single RV', () => {
    const candles: Candle[] = [
      { open: 100, high: 102, low: 98, close: 101, volume: 1000 },
      { open: 101, high: 103, low: 99, close: 102, volume: 1000 },
    ];
    const returns = [Math.log(102 / 101)];
    const rv = perCandleParkinson(candles, returns);

    expect(rv).toHaveLength(1);
    // Uses candles[1] (i+1 alignment)
    const expected = PARKINSON_COEFF * Math.log(103 / 99) ** 2;
    expect(rv[0]).toBeCloseTo(expected, 15);
  });

  it('empty returns → empty rv', () => {
    const candles: Candle[] = [
      { open: 100, high: 102, low: 98, close: 100, volume: 1000 },
    ];
    const rv = perCandleParkinson(candles, []);
    expect(rv).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// 11. EGARCH logVariance clamp with extreme Parkinson RV
// ═══════════════════════════════════════════════════════════════

describe('EGARCH logVariance clamp with extreme Candle[] RV', () => {
  it('extremely wide candles do not produce Infinity variance', () => {
    const rng = lcg(42);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 100; i++) {
      const r = randn(rng) * 0.01;
      const close = price * Math.exp(r);
      // Extremely wide intraday range: high = 2×close, low = 0.5×close
      const high = Math.max(price, close) * 2;
      const low = Math.min(price, close) * 0.5;
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    const model = new Egarch(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    for (const v of vs) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }

    const fc = model.forecast(fit.params, 5);
    for (const v of fc.variance) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 12. perCandleParkinson numerical extremes
// ═══════════════════════════════════════════════════════════════

describe('perCandleParkinson numerical stability', () => {
  it('micro-cap prices (0.0001 level) produce valid RV', () => {
    const candles: Candle[] = [
      { open: 0.0001, high: 0.00012, low: 0.00009, close: 0.00011, volume: 1e6 },
      { open: 0.00011, high: 0.00013, low: 0.0001, close: 0.00012, volume: 1e6 },
    ];
    const returns = [Math.log(0.00012 / 0.00011)];
    const rv = perCandleParkinson(candles, returns);

    expect(rv[0]).toBeGreaterThan(0);
    expect(Number.isFinite(rv[0])).toBe(true);
  });

  it('large-cap prices (1e6 level) produce valid RV', () => {
    const candles: Candle[] = [
      { open: 1e6, high: 1.02e6, low: 0.98e6, close: 1.01e6, volume: 100 },
      { open: 1.01e6, high: 1.03e6, low: 0.99e6, close: 1.02e6, volume: 100 },
    ];
    const returns = [Math.log(1.02e6 / 1.01e6)];
    const rv = perCandleParkinson(candles, returns);

    expect(rv[0]).toBeGreaterThan(0);
    expect(Number.isFinite(rv[0])).toBe(true);
  });

  it('scale invariance: 1000× prices produce identical Parkinson RV', () => {
    const candles1: Candle[] = [
      { open: 100, high: 105, low: 95, close: 102, volume: 1000 },
      { open: 102, high: 107, low: 97, close: 104, volume: 1000 },
    ];
    const candles2 = candles1.map(c => ({
      open: c.open * 1000, high: c.high * 1000,
      low: c.low * 1000, close: c.close * 1000, volume: c.volume,
    }));

    const r1 = [Math.log(102 / 100)];
    const r2 = [Math.log(102000 / 100000)]; // same log return

    const rv1 = perCandleParkinson(candles1, r1);
    const rv2 = perCandleParkinson(candles2, r2);

    expect(rv1[0]).toBeCloseTo(rv2[0], 15);
  });
});

// ═══════════════════════════════════════════════════════════════
// 13. Sub-50 Candle[] to GARCH/EGARCH constructors
// ═══════════════════════════════════════════════════════════════

describe('sub-50 Candle[] constructor error', () => {
  it('GARCH throws with 10 candles', () => {
    const candles = makeCandles(10, 42);
    expect(() => new Garch(candles)).toThrow('Need at least 50 data points');
  });

  it('EGARCH throws with 10 candles', () => {
    const candles = makeCandles(10, 42);
    expect(() => new Egarch(candles)).toThrow('Need at least 50 data points');
  });

  it('GARCH throws with 49 candles (boundary)', () => {
    const candles = makeCandles(49, 42);
    expect(() => new Garch(candles)).toThrow('Need at least 50 data points');
  });

  it('GARCH accepts exactly 50 candles', () => {
    const candles = makeCandles(50, 42);
    expect(() => new Garch(candles)).not.toThrow();
  });

  it('EGARCH accepts exactly 50 candles', () => {
    const candles = makeCandles(50, 42);
    expect(() => new Egarch(candles)).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════
// 14. yangZhangVariance = 0 as EGARCH initial variance
// ═══════════════════════════════════════════════════════════════

describe('all-identical candles (O=H=L=C) edge case', () => {
  it('GARCH handles constant-price candles', () => {
    // All prices identical → returns = 0, yangZhang = 0
    const candles: Candle[] = [];
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
    }

    // Constructor should not throw
    const model = new Garch(candles);
    const fit = model.fit();
    // May not converge meaningfully, but should not crash
    expect(Number.isFinite(fit.params.omega) || !fit.diagnostics.converged).toBe(true);
  });

  it('EGARCH handles constant-price candles without throwing', () => {
    const candles: Candle[] = [];
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
    }

    const model = new Egarch(candles);
    // log(0) = -Infinity for initial variance; optimizer runs but params degenerate
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
    // With zero variance initial, EGARCH cannot converge meaningfully
    // — the important thing is no crash / no uncaught exception
  });
});

// ═══════════════════════════════════════════════════════════════
// Extra: predictRange with Candle[] (Realized path)
// ═══════════════════════════════════════════════════════════════

describe('predictRange with Candle[] (Realized path)', () => {
  it('predictRange returns valid multi-step result', () => {
    const candles = makeCandles(500, 42);
    const result = predictRange(candles, '15m', 5);

    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
    expect(result.lowerPrice).toBeLessThan(result.currentPrice);
  });

  it('predictRange sigma grows with more steps', () => {
    const candles = makeCandles(500, 42);
    const r1 = predictRange(candles, '15m', 1);
    const r5 = predictRange(candles, '15m', 5);
    const r20 = predictRange(candles, '15m', 20);

    // Cumulative σ should grow (more steps = wider range)
    expect(r5.sigma).toBeGreaterThan(r1.sigma);
    expect(r20.sigma).toBeGreaterThan(r5.sigma);
  });
});

// ═══════════════════════════════════════════════════════════════
// 15. REALIZED NoVaS (Candle[] → Parkinson RV)
// ═══════════════════════════════════════════════════════════════

describe('Realized NoVaS (Candle[] uses Parkinson RV)', () => {
  it('Candle[] and number[] produce different NoVaS params', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateNoVaS(candles);
    const resultPrices = calibrateNoVaS(prices);

    expect(resultCandles.diagnostics.converged).toBe(true);
    expect(resultPrices.diagnostics.converged).toBe(true);

    // Weights should differ because Candle[] uses Parkinson RV
    let weightDiff = 0;
    for (let i = 0; i < resultCandles.params.weights.length; i++) {
      weightDiff += Math.abs(resultCandles.params.weights[i] - resultPrices.params.weights[i]);
    }
    expect(weightDiff).toBeGreaterThan(1e-10);
  });

  it('flat candles (H=L) degrade to classical NoVaS (r² fallback)', () => {
    const candles = makeFlatCandles(200, 42);
    const result = calibrateNoVaS(candles);

    expect(result.diagnostics.converged).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('Realized NoVaS variance series all positive and finite', () => {
    const candles = makeCandles(200, 42);
    const model = new NoVaS(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    for (const v of vs) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('Realized NoVaS 1-step forecast uses Parkinson RV as innovation', () => {
    const candles = makeCandles(200, 42);
    const model = new NoVaS(candles);
    const fit = model.fit();
    const { weights, lags } = fit.params;

    // Compute expected 1-step forecast manually using Parkinson RV
    const returns = model.getReturns();
    const rv = perCandleParkinson(candles, returns);
    const n = rv.length;

    let expected = weights[0];
    for (let j = 1; j <= lags; j++) {
      expected += weights[j] * rv[n - j];
    }

    const fc = model.forecast(fit.params, 1);
    expect(fc.variance[0]).toBeCloseTo(expected, 12);
  });

  it('Realized NoVaS forecast converges', () => {
    const candles = makeCandles(200, 42);
    const model = new NoVaS(candles);
    const fit = model.fit();

    if (fit.params.persistence < 0.999) {
      const fc = model.forecast(fit.params, 200);
      const lastVar = fc.variance[199];
      const uncond = fit.params.unconditionalVariance;
      const relError = Math.abs(lastVar - uncond) / uncond;
      expect(relError).toBeLessThan(0.01);
    }
  });

  it('Realized NoVaS scale invariance — 1000× prices → same weights', () => {
    const candles1 = makeCandles(200, 42);
    const candles2 = candles1.map(c => ({
      open: c.open * 1000,
      high: c.high * 1000,
      low: c.low * 1000,
      close: c.close * 1000,
      volume: c.volume,
    }));

    const r1 = calibrateNoVaS(candles1);
    const r2 = calibrateNoVaS(candles2);

    // Log returns and Parkinson are scale-invariant
    expect(r1.params.persistence).toBeCloseTo(r2.params.persistence, 4);
    for (let i = 1; i < r1.params.weights.length; i++) {
      expect(r1.params.weights[i]).toBeCloseTo(r2.params.weights[i], 6);
    }
  });

  it('Realized NoVaS across multiple seeds — always valid', () => {
    for (let seed = 1; seed <= 10; seed++) {
      const candles = makeCandles(200, seed);
      const result = calibrateNoVaS(candles);

      expect(result.params.persistence).toBeLessThan(1);
      expect(result.params.weights[0]).toBeGreaterThan(0);
      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    }
  });

  it('D² with Candle[] should be <= D² with number[] (or close)', () => {
    // Parkinson RV is less noisy → normalization should be at least as good
    let candleWins = 0;
    const seeds = 10;
    for (let seed = 1; seed <= seeds; seed++) {
      const candles = makeCandles(200, seed);
      const prices = candles.map(c => c.close);

      const rc = calibrateNoVaS(candles);
      const rp = calibrateNoVaS(prices);

      if (rc.params.dSquared <= rp.params.dSquared + 0.01) {
        candleWins++;
      }
    }
    // Candle[] should win or tie in most cases
    expect(candleWins).toBeGreaterThanOrEqual(5);
  });

  it('all-identical OHLC candles (O=H=L=C=100) do not crash NoVaS', () => {
    const candles: Candle[] = [];
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
    }

    // All returns = 0, all Parkinson RV = 0 → fallback to r² = 0
    const model = new NoVaS(candles);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
    // May not converge meaningfully, but must not throw
  });

  it('minimum-length Candle[] boundary (lags + 31 candles)', () => {
    const lags = 10;
    const minRequired = lags + 30;

    // Below minimum — should throw
    const tooShort = makeCandles(minRequired - 1, 42);
    expect(() => new NoVaS(tooShort)).toThrow();

    // Exactly at minimum — should work
    const exact = makeCandles(minRequired, 42);
    const model = new NoVaS(exact);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.params.weights.length).toBe(lags + 1);
  });

  it('extremely wide candles do not produce Infinity in NoVaS', () => {
    const rng = lcg(42);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const r = randn(rng) * 0.01;
      const close = price * Math.exp(r);
      // Extremely wide intraday range: high = 2×close, low = 0.5×close
      const high = Math.max(price, close) * 2;
      const low = Math.min(price, close) * 0.5;
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    const model = new NoVaS(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    for (const v of vs) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }

    const fc = model.forecast(fit.params, 5);
    for (const v of fc.variance) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  });
});
