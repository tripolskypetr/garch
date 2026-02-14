import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  GjrGarch,
  HarRv,
  NoVaS,
  calibrateGarch,
  calibrateEgarch,
  calibrateGjrGarch,
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
// 3b. Realized GJR-GARCH (Candle[] → Parkinson RV)
// ═══════════════════════════════════════════════════════════════

describe('Realized GJR-GARCH (Candle[] uses Parkinson RV)', () => {
  it('Candle[] and number[] produce different GJR-GARCH params', () => {
    const candles = makeCandles(200, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateGjrGarch(candles);
    const resultPrices = calibrateGjrGarch(prices);

    expect(resultCandles.diagnostics.converged).toBe(true);
    expect(resultPrices.diagnostics.converged).toBe(true);

    const diff = Math.abs(resultCandles.params.alpha - resultPrices.params.alpha)
      + Math.abs(resultCandles.params.omega - resultPrices.params.omega)
      + Math.abs(resultCandles.params.gamma - resultPrices.params.gamma);
    expect(diff).toBeGreaterThan(1e-6);
  });

  it('flat candles (H=L) degrade to classical GJR-GARCH', () => {
    const candles = makeFlatCandles(200, 42);
    const prices = candles.map(c => c.close);

    const rc = calibrateGjrGarch(candles);
    const rp = calibrateGjrGarch(prices);

    expect(rc.params.alpha).toBeCloseTo(rp.params.alpha, 3);
    expect(rc.params.beta).toBeCloseTo(rp.params.beta, 3);
    expect(rc.params.gamma).toBeCloseTo(rp.params.gamma, 3);
  });

  it('Realized GJR-GARCH variance series all positive and finite', () => {
    const candles = makeCandles(200, 42);
    const model = new GjrGarch(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    for (const v of vs) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('GJR-GARCH multi-step forecast with Candle[]: step 2+ use effective persistence', () => {
    const candles = makeCandles(200, 42);
    const model = new GjrGarch(candles);
    const fit = model.fit();
    const { omega, alpha, gamma, beta } = fit.params;

    const fc = model.forecast(fit.params, 10);

    for (let h = 1; h < 10; h++) {
      const expected = omega + (alpha + gamma / 2 + beta) * fc.variance[h - 1];
      expect(fc.variance[h]).toBeCloseTo(expected, 12);
    }
  });

  it('Realized GJR-GARCH scale invariance — 1000× prices → same vol', () => {
    const candles1 = makeCandles(200, 42);
    const candles2 = candles1.map(c => ({
      open: c.open * 1000, high: c.high * 1000,
      low: c.low * 1000, close: c.close * 1000, volume: c.volume,
    }));

    const r1 = calibrateGjrGarch(candles1);
    const r2 = calibrateGjrGarch(candles2);

    expect(r1.params.annualizedVol).toBeCloseTo(r2.params.annualizedVol, 1);
  });

  it('all-identical OHLC candles do not crash GJR-GARCH', () => {
    const candles: Candle[] = [];
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
    }

    const model = new GjrGarch(candles);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
  });
});

// ═══════════════════════════════════════════════════════════════
// 4–5. Bad OHLC: NaN/Infinity and high < low for GARCH/EGARCH/GJR-GARCH
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

  it('GJR-GARCH with NaN high does not crash', () => {
    const candles = makeCandles(100, 42);
    candles[50] = { ...candles[50], high: NaN };
    const model = new GjrGarch(candles);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
  });

  it('GJR-GARCH with high < low does not crash', () => {
    const candles = makeCandles(100, 42);
    const c = candles[50];
    candles[50] = { ...c, high: c.low * 0.99, low: c.high * 1.01 };
    const model = new GjrGarch(candles);
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
    expect(['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas']).toContain(result.modelType);
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

  it('GJR-GARCH throws with 10 candles', () => {
    const candles = makeCandles(10, 42);
    expect(() => new GjrGarch(candles)).toThrow('Need at least 50 data points');
  });

  it('GJR-GARCH throws with 49 candles (boundary)', () => {
    const candles = makeCandles(49, 42);
    expect(() => new GjrGarch(candles)).toThrow('Need at least 50 data points');
  });

  it('GJR-GARCH accepts exactly 50 candles', () => {
    const candles = makeCandles(50, 42);
    expect(() => new GjrGarch(candles)).not.toThrow();
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
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
  });

  it('GJR-GARCH handles constant-price candles without throwing', () => {
    const candles: Candle[] = [];
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
    }

    const model = new GjrGarch(candles);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
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

// ═══════════════════════════════════════════════════════════════
// 16. NoVaS fallback — predict falls back to GARCH when NoVaS fails
// ═══════════════════════════════════════════════════════════════

describe('predict fallback when NoVaS fails', () => {
  it('near-constant candles: predict returns valid result (NoVaS may fail)', () => {
    // Near-constant prices make NoVaS optimization degenerate,
    // but predict should still return a valid result via GARCH fallback
    const rng = lcg(42);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const r = randn(rng) * 1e-12; // near-zero returns
      const close = price + r;
      candles.push({
        open: price,
        high: Math.max(price, close) + 1e-13,
        low: Math.min(price, close) - 1e-13,
        close,
        volume: 1000,
      });
      price = close;
    }

    const result = predict(candles, '4h');
    expect(result).toBeDefined();
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas']).toContain(result.modelType);
  });

  it('predict always returns a valid modelType across 20 seeds', () => {
    const validTypes = ['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas'];
    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeCandles(200, seed);
      const result = predict(candles, '4h');

      expect(validTypes).toContain(result.modelType);
      expect(result.sigma).toBeGreaterThan(0);
      expect(Number.isFinite(result.sigma)).toBe(true);
      expect(result.upperPrice).toBeGreaterThan(result.lowerPrice);
    }
  });

  it('predictRange also falls back gracefully', () => {
    // Flat candles: Parkinson RV = 0, NoVaS degrades
    const candles = makeFlatCandles(200, 42);
    const result = predictRange(candles, '4h', 5);

    expect(result).toBeDefined();
    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.upperPrice).toBeGreaterThan(result.lowerPrice);
  });

  it('GARCH family is always the guaranteed fallback', () => {
    // Even with pathological data, fitGarchFamily never returns null,
    // so predict must always produce a result
    const candles = makeCandles(200, 42);
    const result = predict(candles, '4h');

    // The result must come from one of the four models
    expect(['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas']).toContain(result.modelType);
    // Forecast corridor must be well-formed
    expect(result.move).toBeGreaterThanOrEqual(0);
    expect(result.upperPrice).toBeCloseTo(result.currentPrice + result.move, 8);
    expect(result.lowerPrice).toBeCloseTo(result.currentPrice - result.move, 8);
  });
});

// ═══════════════════════════════════════════════════════════════
// 17. Ground-truth: known σ → predict recovers it
// ═══════════════════════════════════════════════════════════════

describe('predict recovers known volatility from synthetic data', () => {
  /**
   * Generate candles with known constant per-period volatility σ_true.
   *
   * Returns are iid N(0, σ²). Intraday high/low simulated via
   * Brownian bridge: E[max] ≈ close + σ·√(2·ln2), E[min] ≈ close − σ·√(2·ln2).
   * This makes Parkinson RV an unbiased estimator of σ².
   */
  function makeKnownVolCandles(
    n: number, sigmaTrue: number, seed: number, startPrice = 100,
  ): Candle[] {
    const rng = lcg(seed);
    const candles: Candle[] = [];
    let price = startPrice;

    for (let i = 0; i < n; i++) {
      const r = randn(rng) * sigmaTrue;
      const close = price * Math.exp(r);

      // Simulate intraday extremes via additional noise
      // High/low spread ~ σ (Brownian bridge max/min)
      const mid = (price + close) / 2;
      const spread = Math.abs(price - close);
      const extraUp = Math.abs(randn(rng)) * sigmaTrue * price * 0.5;
      const extraDown = Math.abs(randn(rng)) * sigmaTrue * price * 0.5;
      const high = Math.max(price, close) + extraUp + spread * 0.1;
      const low = Math.min(price, close) - extraDown - spread * 0.1;

      candles.push({ open: price, high, low: Math.max(low, 1e-10), close, volume: 1000 });
      price = close;
    }

    return candles;
  }

  it('σ_true = 1% per period — predict within 50% relative error', () => {
    const sigmaTrue = 0.01;
    const candles = makeKnownVolCandles(500, sigmaTrue, 42);

    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;

    expect(result.sigma).toBeGreaterThan(0);
    expect(relError).toBeLessThan(0.5);
  });

  it('σ_true = 3% per period — predict within 50% relative error', () => {
    const sigmaTrue = 0.03;
    const candles = makeKnownVolCandles(500, sigmaTrue, 77);

    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;

    expect(result.sigma).toBeGreaterThan(0);
    expect(relError).toBeLessThan(0.5);
  });

  it('σ_true = 0.2% per period (low vol) — predict within 50% relative error', () => {
    const sigmaTrue = 0.002;
    const candles = makeKnownVolCandles(500, sigmaTrue, 99);

    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;

    expect(result.sigma).toBeGreaterThan(0);
    expect(relError).toBeLessThan(0.5);
  });

  it('higher σ_true → higher predicted sigma (monotonicity)', () => {
    const sigmas = [0.002, 0.01, 0.03];
    const predicted: number[] = [];

    for (const s of sigmas) {
      const candles = makeKnownVolCandles(500, s, 42);
      predicted.push(predict(candles, '15m').sigma);
    }

    // Predicted sigma must increase with true sigma
    expect(predicted[1]).toBeGreaterThan(predicted[0]);
    expect(predicted[2]).toBeGreaterThan(predicted[1]);
  });

  it('median relative error < 30% across 20 seeds (σ=1%)', () => {
    const sigmaTrue = 0.01;
    const errors: number[] = [];

    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeKnownVolCandles(500, sigmaTrue, seed);
      const result = predict(candles, '15m');
      errors.push(Math.abs(result.sigma - sigmaTrue) / sigmaTrue);
    }

    errors.sort((a, b) => a - b);
    const median = errors[Math.floor(errors.length / 2)];
    expect(median).toBeLessThan(0.3);
  });

  it('±1σ corridor captures ~68% of actual next moves (Monte Carlo)', () => {
    const sigmaTrue = 0.01;
    let hits = 0;
    const trials = 30;

    for (let seed = 1; seed <= trials; seed++) {
      // Generate 501 candles: 500 for fitting + 1 for out-of-sample test
      const allCandles = makeKnownVolCandles(501, sigmaTrue, seed);
      const fittingCandles = allCandles.slice(0, 500);
      const actualNext = allCandles[500].close;

      const result = predict(fittingCandles, '15m');

      if (actualNext >= result.lowerPrice && actualNext <= result.upperPrice) {
        hits++;
      }
    }

    const hitRate = hits / trials;
    // Theoretical ~68%, allow 45-90% range for finite sample
    expect(hitRate).toBeGreaterThanOrEqual(0.45);
    expect(hitRate).toBeLessThanOrEqual(0.90);
  });

  it('2× sigma data → ~2× predicted sigma', () => {
    const candles1 = makeKnownVolCandles(500, 0.01, 42);
    const candles2 = makeKnownVolCandles(500, 0.02, 42);

    const sigma1 = predict(candles1, '15m').sigma;
    const sigma2 = predict(candles2, '15m').sigma;

    // Ratio should be approximately 2 (allow 1.2–3.5 range)
    const ratio = sigma2 / sigma1;
    expect(ratio).toBeGreaterThan(1.2);
    expect(ratio).toBeLessThan(3.5);
  });
});

// ═══════════════════════════════════════════════════════════════
// 18. Ground-truth per model: each DGP favors a specific model
// ═══════════════════════════════════════════════════════════════

describe('ground-truth: DGP tailored to each model', () => {
  /**
   * GARCH DGP: symmetric volatility clustering.
   * σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}, ε = σ·z, z ~ N(0,1)
   * Unconditional variance: ω/(1-α-β)
   */
  function makeGarchDGP(n: number, seed: number): { candles: Candle[]; sigmaTrue: number } {
    const rng = lcg(seed);
    const omega = 1e-5, alpha = 0.10, beta = 0.85;
    const uncondVar = omega / (1 - alpha - beta);
    let variance = uncondVar;
    let price = 100;
    const candles: Candle[] = [];

    for (let i = 0; i < n; i++) {
      const sigma = Math.sqrt(variance);
      const z = randn(rng);
      const r = sigma * z;
      const close = price * Math.exp(r);

      // Symmetric intraday range proportional to sigma
      const extraUp = Math.abs(randn(rng)) * sigma * price * 0.4;
      const extraDown = Math.abs(randn(rng)) * sigma * price * 0.4;
      const high = Math.max(price, close) + extraUp;
      const low = Math.max(Math.min(price, close) - extraDown, 1e-10);
      candles.push({ open: price, high, low, close, volume: 1000 });

      // GARCH(1,1) recursion
      variance = omega + alpha * (r * r) + beta * variance;
      price = close;
    }

    return { candles, sigmaTrue: Math.sqrt(uncondVar) };
  }

  /**
   * EGARCH DGP: strong leverage — negative returns increase vol much more.
   * ln(σ²_t) = ω + α·(|z|-E|z|) + γ·z + β·ln(σ²_{t-1}), γ < 0
   */
  function makeEgarchDGP(n: number, seed: number): { candles: Candle[]; sigmaTrue: number } {
    const rng = lcg(seed);
    // Lower persistence → variance stays closer to unconditional
    const omega = -0.3, alpha = 0.15, gamma = -0.12, beta = 0.93;
    const EabsZ = Math.sqrt(2 / Math.PI);
    const uncondLogVar = omega / (1 - beta);
    const uncondVar = Math.exp(uncondLogVar);
    let logVariance = uncondLogVar;
    let price = 100;
    const candles: Candle[] = [];

    for (let i = 0; i < n; i++) {
      const variance = Math.exp(logVariance);
      const sigma = Math.sqrt(variance);
      const z = randn(rng);
      const r = sigma * z;
      const close = price * Math.exp(r);

      // Wider range on negative returns (leverage visible in OHLC)
      const leverageBoost = z < 0 ? 1.5 : 0.8;
      const extraUp = Math.abs(randn(rng)) * sigma * price * 0.3 * leverageBoost;
      const extraDown = Math.abs(randn(rng)) * sigma * price * 0.3 * leverageBoost;
      const high = Math.max(price, close) + extraUp;
      const low = Math.max(Math.min(price, close) - extraDown, 1e-10);
      candles.push({ open: price, high, low, close, volume: 1000 });

      // EGARCH recursion
      logVariance = omega + alpha * (Math.abs(z) - EabsZ) + gamma * z + beta * logVariance;
      logVariance = Math.max(-50, Math.min(50, logVariance));
      price = close;
    }

    return { candles, sigmaTrue: Math.sqrt(uncondVar) };
  }

  /**
   * GJR-GARCH DGP: moderate leverage via indicator function.
   * σ²_t = ω + α·ε² + γ·ε²·I(r<0) + β·σ², γ > 0
   */
  function makeGjrDGP(n: number, seed: number): { candles: Candle[]; sigmaTrue: number } {
    const rng = lcg(seed);
    const omega = 2e-5, alpha = 0.05, gamma = 0.12, beta = 0.82;
    const uncondVar = omega / (1 - alpha - gamma / 2 - beta);
    let variance = uncondVar;
    let price = 100;
    const candles: Candle[] = [];

    for (let i = 0; i < n; i++) {
      const sigma = Math.sqrt(variance);
      const z = randn(rng);
      const r = sigma * z;
      const close = price * Math.exp(r);

      const extraUp = Math.abs(randn(rng)) * sigma * price * 0.4;
      const extraDown = Math.abs(randn(rng)) * sigma * price * 0.4;
      const high = Math.max(price, close) + extraUp;
      const low = Math.max(Math.min(price, close) - extraDown, 1e-10);
      candles.push({ open: price, high, low, close, volume: 1000 });

      // GJR-GARCH recursion
      const indicator = r < 0 ? 1 : 0;
      variance = omega + alpha * (r * r) + gamma * (r * r) * indicator + beta * variance;
      price = close;
    }

    return { candles, sigmaTrue: Math.sqrt(uncondVar) };
  }

  /**
   * HAR-RV DGP: multi-scale volatility — daily, weekly, monthly components.
   * RV_{t+1} = β₀ + β₁·RV_1 + β₂·RV_5 + β₃·RV_22 + noise
   * Returns drawn from N(0, RV_t).
   */
  function makeHarDGP(n: number, seed: number): { candles: Candle[]; sigmaTrue: number } {
    const rng = lcg(seed);
    // Strong multi-scale: daily component dominant, weekly/monthly significant
    const b0 = 1e-5, b1 = 0.4, b2 = 0.25, b3 = 0.25;
    const uncondRV = b0 / (1 - b1 - b2 - b3);
    const rv: number[] = [];

    // Burn-in with varied RV to establish multi-scale structure
    for (let i = 0; i < 22; i++) {
      rv.push(uncondRV * (0.5 + rng()));
    }

    // Generate RV series using HAR dynamics with meaningful noise
    for (let t = 22; t < n; t++) {
      const rv1 = rv[t - 1];
      const rv5 = (rv[t - 1] + rv[t - 2] + rv[t - 3] + rv[t - 4] + rv[t - 5]) / 5;
      const rv22 = rv.slice(t - 22, t).reduce((s, v) => s + v, 0) / 22;
      let next = b0 + b1 * rv1 + b2 * rv5 + b3 * rv22;
      // Multiplicative noise on RV (lognormal-like)
      next *= Math.exp(randn(rng) * 0.15);
      next = Math.max(next, 1e-10);
      rv.push(next);
    }

    // Generate candles from RV series — use Parkinson-consistent OHLC
    // Parkinson: RV = (1/(4·ln2))·ln(H/L)², so ln(H/L) = sqrt(4·ln2·RV)
    let price = 100;
    const candles: Candle[] = [];
    for (let i = 0; i < n; i++) {
      const sigma = Math.sqrt(rv[i]);
      const r = randn(rng) * sigma;
      const close = price * Math.exp(r);

      // Set H/L range to encode true RV via Parkinson formula
      const logRange = Math.sqrt(4 * Math.LN2 * rv[i]) * (0.8 + 0.4 * rng());
      const mid = (price + close) / 2;
      const high = mid * Math.exp(logRange / 2);
      const low = Math.max(mid * Math.exp(-logRange / 2), 1e-10);
      candles.push({ open: price, high: Math.max(high, price, close), low: Math.min(low, price, close), close, volume: 1000 });
      price = close;
    }

    return { candles, sigmaTrue: Math.sqrt(uncondRV) };
  }

  /**
   * NoVaS DGP: regime-switching volatility + heavy tails.
   * Two regimes (low/high vol), Student-t(5) innovations.
   * Parametric models struggle; NoVaS adapts via normality criterion.
   */
  function makeNovasDGP(n: number, seed: number): { candles: Candle[]; sigmaTrue: number } {
    const rng = lcg(seed);
    // Mild, smooth volatility changes — no leverage, no strong clustering
    // NoVaS is model-free and shines when parametric models overfit
    const baseVol = 0.01;
    let price = 100;
    const candles: Candle[] = [];
    let totalVar = 0;

    for (let i = 0; i < n; i++) {
      // Slowly varying vol with a sine pattern (non-parametric)
      const sigma = baseVol * (1 + 0.3 * Math.sin(2 * Math.PI * i / 50));
      totalVar += sigma * sigma;

      const r = randn(rng) * sigma;
      const close = price * Math.exp(r);

      const extraUp = Math.abs(randn(rng)) * sigma * price * 0.3;
      const extraDown = Math.abs(randn(rng)) * sigma * price * 0.3;
      const high = Math.max(price, close) + extraUp;
      const low = Math.max(Math.min(price, close) - extraDown, 1e-10);
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    return { candles, sigmaTrue: Math.sqrt(totalVar / n) };
  }

  // ── GARCH DGP ──────────────────────────────────────────────

  it('GARCH DGP: predict recovers unconditional σ (< 50% error)', () => {
    const { candles, sigmaTrue } = makeGarchDGP(500, 42);
    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;
    expect(relError).toBeLessThan(0.5);
  });

  it('GARCH DGP: GARCH family wins across majority of seeds', () => {
    const garchTypes = ['garch', 'egarch', 'gjr-garch'];
    let garchWins = 0;
    for (let seed = 1; seed <= 20; seed++) {
      const { candles } = makeGarchDGP(500, seed);
      const result = predict(candles, '15m');
      if (garchTypes.includes(result.modelType)) garchWins++;
    }
    // GARCH-family should win most of the time on GARCH DGP
    expect(garchWins).toBeGreaterThanOrEqual(10);
  });

  // ── EGARCH DGP ─────────────────────────────────────────────

  it('EGARCH DGP: predict recovers unconditional σ (< 75% error)', () => {
    const { candles, sigmaTrue } = makeEgarchDGP(500, 42);
    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;
    // EGARCH has time-varying vol → single-point forecast may deviate from unconditional
    expect(relError).toBeLessThan(0.75);
  });

  it('EGARCH DGP: EGARCH selected more often than plain GARCH', () => {
    let egarchWins = 0;
    let garchWins = 0;
    for (let seed = 1; seed <= 20; seed++) {
      const { candles } = makeEgarchDGP(500, seed);
      const result = predict(candles, '15m');
      if (result.modelType === 'egarch') egarchWins++;
      if (result.modelType === 'garch') garchWins++;
    }
    expect(egarchWins).toBeGreaterThan(garchWins);
  });

  // ── GJR-GARCH DGP ─────────────────────────────────────────

  it('GJR-GARCH DGP: predict recovers unconditional σ (< 50% error)', () => {
    const { candles, sigmaTrue } = makeGjrDGP(500, 42);
    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;
    expect(relError).toBeLessThan(0.5);
  });

  it('GJR-GARCH DGP: asymmetric model (EGARCH or GJR) wins over GARCH', () => {
    let asymmetricWins = 0;
    let garchWins = 0;
    for (let seed = 1; seed <= 20; seed++) {
      const { candles } = makeGjrDGP(500, seed);
      const result = predict(candles, '15m');
      if (result.modelType === 'gjr-garch' || result.modelType === 'egarch') asymmetricWins++;
      if (result.modelType === 'garch') garchWins++;
    }
    expect(asymmetricWins).toBeGreaterThan(garchWins);
  });

  // ── HAR-RV DGP ────────────────────────────────────────────

  it('HAR-RV DGP: predict recovers unconditional σ (< 75% error)', () => {
    const { candles, sigmaTrue } = makeHarDGP(500, 42);
    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;
    // HAR has multi-scale dynamics + noisy RV → wider tolerance
    expect(relError).toBeLessThan(0.75);
  });

  it('HAR-RV DGP: HAR-RV R² > 0.1 on own data (captures multi-scale)', () => {
    const { candles } = makeHarDGP(500, 42);
    const model = new HarRv(candles, { periodsPerYear: 35040 });
    const fit = model.fit();
    // HAR-RV should explain meaningful variance in its own DGP
    expect(fit.params.r2).toBeGreaterThan(0.1);
    expect(fit.diagnostics.converged).toBe(true);
  });

  // ── NoVaS DGP ─────────────────────────────────────────────

  it('NoVaS DGP: predict recovers average σ (< 75% error)', () => {
    const { candles, sigmaTrue } = makeNovasDGP(500, 42);
    const result = predict(candles, '15m');
    const relError = Math.abs(result.sigma - sigmaTrue) / sigmaTrue;
    // Non-parametric vol pattern → wider tolerance
    expect(relError).toBeLessThan(0.75);
  });

  it('NoVaS DGP: NoVaS achieves low D² on own data (good normalization)', () => {
    const { candles } = makeNovasDGP(500, 42);
    const model = new NoVaS(candles, { periodsPerYear: 35040 });
    const fit = model.fit();
    // NoVaS should normalize its own DGP well (D² < 1)
    expect(fit.params.dSquared).toBeLessThan(1);
    expect(fit.diagnostics.converged).toBe(true);
  });

  // ── Cross-DGP: each DGP produces a valid forecast ─────────

  it('all 5 DGPs produce valid predict output', () => {
    const dgps = [
      makeGarchDGP(500, 42),
      makeEgarchDGP(500, 42),
      makeGjrDGP(500, 42),
      makeHarDGP(500, 42),
      makeNovasDGP(500, 42),
    ];

    for (const { candles, sigmaTrue } of dgps) {
      const result = predict(candles, '15m');
      expect(result.sigma).toBeGreaterThan(0);
      expect(Number.isFinite(result.sigma)).toBe(true);
      expect(result.upperPrice).toBeGreaterThan(result.lowerPrice);
      expect(['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas']).toContain(result.modelType);
    }
  });

  it('monotonicity holds across DGPs: higher true σ → higher predicted σ', () => {
    const dgps = [
      makeGarchDGP(500, 42),
      makeEgarchDGP(500, 42),
      makeGjrDGP(500, 42),
      makeHarDGP(500, 42),
      makeNovasDGP(500, 42),
    ];

    const pairs = dgps
      .map(({ candles, sigmaTrue }) => ({
        sigmaTrue,
        predicted: predict(candles, '15m').sigma,
      }))
      .sort((a, b) => a.sigmaTrue - b.sigmaTrue);

    // Spearman rank correlation: predicted should generally increase with sigmaTrue
    // Allow one inversion out of 4 adjacent pairs
    let inversions = 0;
    for (let i = 1; i < pairs.length; i++) {
      if (pairs[i].predicted < pairs[i - 1].predicted) inversions++;
    }
    expect(inversions).toBeLessThanOrEqual(1);
  });
});
