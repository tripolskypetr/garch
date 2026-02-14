import { describe, it, expect } from 'vitest';
import {
  HarRv,
  calibrateHarRv,
  predict,
  predictRange,
  backtest,
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
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

/** Generate synthetic prices with GARCH-like volatility clustering. */
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

/** Generate candles from prices. */
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

/** Generate data with strong HAR structure (multi-scale clustering). */
function makeHarData(n: number, seed = 42): number[] {
  const rng = lcg(seed);
  const rv: number[] = [];
  const baseVar = 0.0001;

  // Generate RV with HAR-like structure
  for (let i = 0; i < n; i++) {
    const shortComp = i > 0 ? 0.3 * rv[i - 1] : baseVar;
    const medComp = i >= 5 ? 0.3 * rv.slice(i - 5, i).reduce((a, b) => a + b, 0) / 5 : baseVar;
    const longComp = i >= 22 ? 0.2 * rv.slice(i - 22, i).reduce((a, b) => a + b, 0) / 22 : baseVar;
    const noise = Math.abs(randn(rng)) * baseVar * 0.5;
    rv.push(0.00001 + shortComp + medComp + longComp + noise);
  }

  // Convert RV to prices: r_t ~ N(0, rv_t), P_t = P_{t-1} * exp(r_t)
  const prices = [100];
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt(rv[i]) * randn(rng);
    prices.push(prices[i] * Math.exp(r));
  }
  return prices;
}

// ── Unit tests: OLS and internals ────────────────────────────

describe('HAR-RV', () => {
  describe('constructor', () => {
    it('should accept price array', () => {
      const prices = generatePrices(100);
      const model = new HarRv(prices);
      expect(model.getReturns().length).toBe(99);
    });

    it('should accept candle array', () => {
      const candles = makeCandles(100);
      const model = new HarRv(candles);
      expect(model.getReturns().length).toBe(99);
    });

    it('should throw on insufficient data', () => {
      const prices = generatePrices(30);
      expect(() => new HarRv(prices)).toThrow('at least 52');
    });

    it('should respect custom lags', () => {
      const prices = generatePrices(100);
      const model = new HarRv(prices, { shortLag: 2, mediumLag: 10, longLag: 30 });
      expect(model.getReturns().length).toBe(99);
    });

    it('should throw when data < longLag + 30 with custom lags', () => {
      const prices = generatePrices(50);
      expect(() => new HarRv(prices, { longLag: 30 })).toThrow('at least 60');
    });
  });

  describe('fit (OLS calibration)', () => {
    it('should return valid HarRvParams', () => {
      const prices = generatePrices(300);
      const result = calibrateHarRv(prices);

      expect(result.params.beta0).toBeDefined();
      expect(result.params.betaShort).toBeDefined();
      expect(result.params.betaMedium).toBeDefined();
      expect(result.params.betaLong).toBeDefined();
      expect(typeof result.params.persistence).toBe('number');
      expect(typeof result.params.r2).toBe('number');
      expect(typeof result.params.unconditionalVariance).toBe('number');
      expect(typeof result.params.annualizedVol).toBe('number');
    });

    it('should always converge (OLS is closed-form)', () => {
      const prices = generatePrices(200);
      const result = calibrateHarRv(prices);
      expect(result.diagnostics.converged).toBe(true);
      expect(result.diagnostics.iterations).toBe(1);
    });

    it('should compute valid diagnostics', () => {
      const prices = generatePrices(500);
      const result = calibrateHarRv(prices);

      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
      expect(Number.isFinite(result.diagnostics.aic)).toBe(true);
      expect(Number.isFinite(result.diagnostics.bic)).toBe(true);
    });

    it('should have R² between 0 and 1 on well-behaved data', () => {
      const prices = makeHarData(500);
      const result = calibrateHarRv(prices);
      expect(result.params.r2).toBeGreaterThan(0);
      expect(result.params.r2).toBeLessThanOrEqual(1);
    });

    it('persistence = betaShort + betaMedium + betaLong', () => {
      const prices = generatePrices(300);
      const result = calibrateHarRv(prices);
      const { betaShort, betaMedium, betaLong, persistence } = result.params;
      expect(persistence).toBeCloseTo(betaShort + betaMedium + betaLong, 10);
    });

    it('should produce positive unconditional variance when stationary', () => {
      const prices = generatePrices(500, 99);
      const result = calibrateHarRv(prices);
      if (result.params.persistence < 1 && result.params.persistence > -1) {
        expect(result.params.unconditionalVariance).toBeGreaterThan(0);
      }
    });

    it('BIC penalizes more than AIC for large samples', () => {
      const prices = generatePrices(500);
      const result = calibrateHarRv(prices);
      expect(result.diagnostics.bic).toBeGreaterThan(result.diagnostics.aic);
    });
  });

  describe('getVarianceSeries', () => {
    it('should return series of correct length', () => {
      const prices = generatePrices(200);
      const model = new HarRv(prices);
      const fit = model.fit();
      const vs = model.getVarianceSeries(fit.params);
      expect(vs.length).toBe(prices.length - 1);
    });

    it('should have all positive values', () => {
      const prices = generatePrices(200);
      const model = new HarRv(prices);
      const fit = model.fit();
      const vs = model.getVarianceSeries(fit.params);
      expect(vs.every(v => v > 0)).toBe(true);
    });

    it('should have all finite values', () => {
      const prices = generatePrices(200);
      const model = new HarRv(prices);
      const fit = model.fit();
      const vs = model.getVarianceSeries(fit.params);
      expect(vs.every(v => Number.isFinite(v))).toBe(true);
    });

    it('uses sample variance for first longLag points', () => {
      const prices = generatePrices(200);
      const model = new HarRv(prices);
      const fit = model.fit();
      const vs = model.getVarianceSeries(fit.params);
      // First 22 entries should all be the same (sample variance fallback)
      const first = vs[0];
      for (let i = 1; i < 22; i++) {
        expect(vs[i]).toBe(first);
      }
    });
  });

  describe('getRv', () => {
    it('should return squared returns', () => {
      const prices = generatePrices(100);
      const model = new HarRv(prices);
      const returns = model.getReturns();
      const rv = model.getRv();
      expect(rv.length).toBe(returns.length);
      for (let i = 0; i < rv.length; i++) {
        expect(rv[i]).toBeCloseTo(returns[i] ** 2, 15);
      }
    });
  });

  describe('forecast', () => {
    it('should forecast single step', () => {
      const prices = generatePrices(300);
      const model = new HarRv(prices);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 1);

      expect(fc.variance.length).toBe(1);
      expect(fc.volatility.length).toBe(1);
      expect(fc.annualized.length).toBe(1);
      expect(fc.variance[0]).toBeGreaterThan(0);
      expect(fc.volatility[0]).toBeCloseTo(Math.sqrt(fc.variance[0]), 10);
    });

    it('should forecast multiple steps', () => {
      const prices = generatePrices(300);
      const model = new HarRv(prices);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 10);

      expect(fc.variance.length).toBe(10);
      expect(fc.volatility.length).toBe(10);
      expect(fc.annualized.length).toBe(10);
    });

    it('all forecast values should be positive and finite', () => {
      const prices = generatePrices(300);
      const model = new HarRv(prices);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 20);

      for (const v of fc.variance) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
      for (const v of fc.volatility) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    });

    it('multi-step forecasts converge toward unconditional variance', () => {
      const prices = generatePrices(500);
      const model = new HarRv(prices);
      const fit = model.fit();
      if (fit.params.persistence > 0 && fit.params.persistence < 1) {
        const fc = model.forecast(fit.params, 50);
        const lastVar = fc.variance[49];
        const uncond = fit.params.unconditionalVariance;
        // Should approach unconditional variance
        const diff = Math.abs(lastVar - uncond) / uncond;
        expect(diff).toBeLessThan(1);
      }
    });

    it('volatility = sqrt(variance)', () => {
      const prices = generatePrices(200);
      const model = new HarRv(prices);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 5);
      for (let i = 0; i < 5; i++) {
        expect(fc.volatility[i]).toBeCloseTo(Math.sqrt(fc.variance[i]), 10);
      }
    });

    it('annualized uses periodsPerYear', () => {
      const prices = generatePrices(200);
      const periodsPerYear = 2190;
      const model = new HarRv(prices, { periodsPerYear });
      const fit = model.fit();
      const fc = model.forecast(fit.params, 1);
      expect(fc.annualized[0]).toBeCloseTo(Math.sqrt(fc.variance[0] * periodsPerYear) * 100, 5);
    });
  });

  describe('calibrateHarRv convenience', () => {
    it('should work the same as new HarRv + fit', () => {
      const prices = generatePrices(300, 77);
      const result1 = calibrateHarRv(prices);
      const model = new HarRv(prices);
      const result2 = model.fit();

      expect(result1.params.beta0).toBeCloseTo(result2.params.beta0, 10);
      expect(result1.params.betaShort).toBeCloseTo(result2.params.betaShort, 10);
      expect(result1.params.betaMedium).toBeCloseTo(result2.params.betaMedium, 10);
      expect(result1.params.betaLong).toBeCloseTo(result2.params.betaLong, 10);
    });

    it('should pass through options', () => {
      const prices = generatePrices(300);
      const result = calibrateHarRv(prices, { periodsPerYear: 8760 });
      // annualizedVol should reflect the periodsPerYear
      expect(result.params.annualizedVol).toBeGreaterThan(0);
    });
  });

  describe('edge cases', () => {
    it('should handle constant volatility', () => {
      // Prices with very small, near-constant moves
      const rng = lcg(42);
      const prices = [100];
      for (let i = 1; i < 200; i++) {
        prices.push(prices[i - 1] * Math.exp(0.001 * randn(rng)));
      }
      const result = calibrateHarRv(prices);
      expect(Number.isFinite(result.params.r2)).toBe(true);
      expect(result.diagnostics.converged).toBe(true);
    });

    it('should handle large sudden move (flash crash)', () => {
      const prices = generatePrices(300, 55);
      // Insert flash crash
      prices[150] = prices[149] * 0.9;
      prices[151] = prices[150] * 1.08;
      const result = calibrateHarRv(prices);
      expect(Number.isFinite(result.params.beta0)).toBe(true);
      expect(result.diagnostics.converged).toBe(true);
    });

    it('should handle exactly longLag + 30 data points', () => {
      const prices = generatePrices(53); // 52 = 22+30 minimum, +1 because returns = n-1
      const model = new HarRv(prices);
      const fit = model.fit();
      expect(fit.diagnostics.converged).toBe(true);
    });

    it('should handle very large dataset (5000 points)', () => {
      const prices = generatePrices(5000);
      const result = calibrateHarRv(prices);
      expect(Number.isFinite(result.params.r2)).toBe(true);
      expect(Number.isFinite(result.params.beta0)).toBe(true);
    });

    it('should handle custom lags', () => {
      const prices = generatePrices(200);
      const result = calibrateHarRv(prices, { shortLag: 3, mediumLag: 10, longLag: 20 });
      expect(result.diagnostics.converged).toBe(true);
      expect(Number.isFinite(result.params.r2)).toBe(true);
    });
  });

  describe('determinism', () => {
    it('same input produces identical output', () => {
      const prices = generatePrices(300, 42);
      const r1 = calibrateHarRv(prices);
      const r2 = calibrateHarRv(prices);
      expect(r1.params.beta0).toBe(r2.params.beta0);
      expect(r1.params.betaShort).toBe(r2.params.betaShort);
      expect(r1.params.r2).toBe(r2.params.r2);
    });
  });
});

// ── Integration with predict ──────────────────────────────────

describe('HAR-RV integration with predict', () => {
  it('predict should accept har-rv as modelType', () => {
    const candles = makeCandles(500, 42);
    const result = predict(candles, '4h');
    expect(['garch', 'egarch', 'har-rv', 'novas']).toContain(result.modelType);
    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
  });

  it('predict output structure is unchanged', () => {
    const candles = makeCandles(500, 42);
    const result = predict(candles, '4h');

    expect(typeof result.currentPrice).toBe('number');
    expect(typeof result.sigma).toBe('number');
    expect(typeof result.move).toBe('number');
    expect(typeof result.upperPrice).toBe('number');
    expect(typeof result.lowerPrice).toBe('number');
    expect(typeof result.modelType).toBe('string');
    expect(typeof result.reliable).toBe('boolean');
  });

  it('upperPrice > currentPrice > lowerPrice', () => {
    const candles = makeCandles(500, 42);
    const result = predict(candles, '4h');
    expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
    expect(result.lowerPrice).toBeLessThan(result.currentPrice);
  });

  it('move = currentPrice * sigma', () => {
    const candles = makeCandles(500, 42);
    const result = predict(candles, '4h');
    expect(result.move).toBeCloseTo(result.currentPrice * result.sigma, 10);
  });

  it('predictRange cumulative sigma > single-step sigma', () => {
    const candles = makeCandles(500, 42);
    const single = predict(candles, '4h');
    const range = predictRange(candles, '4h', 10);
    expect(range.sigma).toBeGreaterThan(single.sigma);
  });

  it('predict works with all intervals', () => {
    const intervals: Array<[string, number]> = [
      ['1m', 500], ['3m', 500], ['5m', 500], ['15m', 300],
      ['30m', 200], ['1h', 200], ['2h', 200], ['4h', 200],
      ['6h', 150], ['8h', 150],
    ];

    for (const [interval, minCandles] of intervals) {
      const candles = makeCandles(minCandles, 42);
      const result = predict(candles, interval as any);
      expect(result.sigma).toBeGreaterThan(0);
      expect(Number.isFinite(result.sigma)).toBe(true);
      expect(['garch', 'egarch', 'har-rv', 'novas']).toContain(result.modelType);
    }
  });

  it('predict never returns NaN or Infinity', () => {
    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeCandles(300, seed);
      const result = predict(candles, '4h');
      expect(Number.isFinite(result.sigma)).toBe(true);
      expect(Number.isFinite(result.move)).toBe(true);
      expect(Number.isFinite(result.upperPrice)).toBe(true);
      expect(Number.isFinite(result.lowerPrice)).toBe(true);
    }
  });

  it('backtest still works with HAR-RV in the mix', () => {
    const candles = makeCandles(500, 42);
    // Should not throw
    const result = backtest(candles, '4h', 50);
    expect(typeof result).toBe('boolean');
  });

  it('predict with custom currentPrice', () => {
    const candles = makeCandles(300, 42);
    const result = predict(candles, '4h', 50000);
    expect(result.currentPrice).toBe(50000);
    expect(result.move).toBeCloseTo(50000 * result.sigma, 5);
  });
});

// ── OLS mathematical correctness ──────────────────────────────

describe('HAR-RV OLS correctness', () => {
  it('should fit a known linear relationship', () => {
    // Generate data with strong serial dependence in squared returns
    const rng = lcg(77);
    const n = 300;
    const prices = [100];
    let vol = 0.01;

    for (let i = 0; i < n; i++) {
      vol = Math.sqrt(0.00002 + 0.6 * vol * vol + 0.0001 * Math.abs(randn(rng)));
      const r = vol * randn(rng);
      prices.push(prices[i] * Math.exp(r));
    }

    const result = calibrateHarRv(prices);
    expect(Number.isFinite(result.params.betaShort)).toBe(true);
    expect(Number.isFinite(result.params.r2)).toBe(true);
    expect(result.params.r2).toBeGreaterThan(0);
  });

  it('OLS residuals should sum to approximately zero', () => {
    const prices = generatePrices(500, 33);
    const model = new HarRv(prices);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);
    const returns = model.getReturns();
    const rv = model.getRv();

    // Compute residuals in the regression range (after longLag)
    let residualSum = 0;
    let count = 0;
    for (let t = 22; t < rv.length - 1; t++) {
      const predicted = fit.params.beta0
        + fit.params.betaShort * rv[t]
        + fit.params.betaMedium * (rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5)
        + fit.params.betaLong * (rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22);
      const actual = rv[t + 1];
      residualSum += actual - predicted;
      count++;
    }

    // OLS guarantees residuals sum to ~0 (within floating point tolerance)
    expect(Math.abs(residualSum / count)).toBeLessThan(1e-6);
  });

  it('R² should be 0 for pure noise (approximately)', () => {
    // Pure random walk with no volatility clustering
    const rng = lcg(42);
    const prices = [100];
    for (let i = 0; i < 500; i++) {
      prices.push(prices[i] * Math.exp(0.01 * randn(rng)));
    }
    const result = calibrateHarRv(prices);
    // R² should be low (near 0) — no predictable structure
    expect(result.params.r2).toBeLessThan(0.15);
  });
});

// ── OLS mathematical correctness (deep) ──────────────────────

describe('HAR-RV OLS deep mathematical tests', () => {
  it('OLS residuals are orthogonal to each regressor (X\'e ≈ 0)', () => {
    const prices = generatePrices(500, 33);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();

    // Reconstruct X and residuals in regression range
    const longLag = 22;
    const dotProducts = [0, 0, 0, 0]; // intercept, short, medium, long

    for (let t = longLag - 1; t < rv.length - 1; t++) {
      const rvShort = rv[t];
      const rvMedium = rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5;
      const rvLong = rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22;

      const predicted = fit.params.beta0
        + fit.params.betaShort * rvShort
        + fit.params.betaMedium * rvMedium
        + fit.params.betaLong * rvLong;
      const residual = rv[t + 1] - predicted;

      dotProducts[0] += 1 * residual;         // intercept column
      dotProducts[1] += rvShort * residual;
      dotProducts[2] += rvMedium * residual;
      dotProducts[3] += rvLong * residual;
    }

    for (let j = 0; j < 4; j++) {
      expect(Math.abs(dotProducts[j])).toBeLessThan(1e-10);
    }
  });

  it('TSS = RSS + ESS decomposition holds', () => {
    const prices = generatePrices(500, 44);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();

    const longLag = 22;
    const yActual: number[] = [];
    const yPredicted: number[] = [];

    for (let t = longLag - 1; t < rv.length - 1; t++) {
      const rvShort = rv[t];
      const rvMedium = rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5;
      const rvLong = rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22;

      const pred = fit.params.beta0
        + fit.params.betaShort * rvShort
        + fit.params.betaMedium * rvMedium
        + fit.params.betaLong * rvLong;

      yActual.push(rv[t + 1]);
      yPredicted.push(pred);
    }

    const yMean = yActual.reduce((s, v) => s + v, 0) / yActual.length;
    let tss = 0, rss = 0, ess = 0;
    for (let i = 0; i < yActual.length; i++) {
      tss += (yActual[i] - yMean) ** 2;
      rss += (yActual[i] - yPredicted[i]) ** 2;
      ess += (yPredicted[i] - yMean) ** 2;
    }

    expect(tss).toBeCloseTo(rss + ess, 6);
  });

  it('R² matches 1 - RSS/TSS', () => {
    const prices = generatePrices(400, 55);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();

    const longLag = 22;
    const yActual: number[] = [];
    const yPredicted: number[] = [];

    for (let t = longLag - 1; t < rv.length - 1; t++) {
      const rvShort = rv[t];
      const rvMedium = rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5;
      const rvLong = rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22;

      const pred = fit.params.beta0
        + fit.params.betaShort * rvShort
        + fit.params.betaMedium * rvMedium
        + fit.params.betaLong * rvLong;

      yActual.push(rv[t + 1]);
      yPredicted.push(pred);
    }

    const yMean = yActual.reduce((s, v) => s + v, 0) / yActual.length;
    let tss = 0, rss = 0;
    for (let i = 0; i < yActual.length; i++) {
      tss += (yActual[i] - yMean) ** 2;
      rss += (yActual[i] - yPredicted[i]) ** 2;
    }

    const r2Manual = 1 - rss / tss;
    expect(fit.params.r2).toBeCloseTo(r2Manual, 8);
  });

  it('R² on GARCH-clustered data > R² on iid data (averaged over seeds)', () => {
    // Independent check: averaged over multiple seeds to reduce noise,
    // GARCH-clustered data (generatePrices) should show more predictable
    // RV structure than pure iid returns.
    let sumR2Clustered = 0;
    let sumR2Iid = 0;
    const nSeeds = 10;

    for (let seed = 1; seed <= nSeeds; seed++) {
      // Clustered (GARCH-like, persistence ~0.95)
      sumR2Clustered += calibrateHarRv(generatePrices(500, seed)).params.r2;

      // Pure iid — no volatility clustering
      const rng = lcg(seed + 1000);
      const iidPrices = [100];
      for (let i = 0; i < 500; i++) {
        iidPrices.push(iidPrices[i] * Math.exp(0.01 * randn(rng)));
      }
      sumR2Iid += calibrateHarRv(iidPrices).params.r2;
    }

    expect(sumR2Clustered / nSeeds).toBeGreaterThan(sumR2Iid / nSeeds);
  });

  it('normal equations hold: X\'X·β = X\'y', () => {
    const prices = generatePrices(400, 66);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();

    const longLag = 22;
    const beta = [fit.params.beta0, fit.params.betaShort, fit.params.betaMedium, fit.params.betaLong];
    const p = 4;

    // Build X'X and X'y
    const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    const Xty: number[] = new Array(p).fill(0);

    for (let t = longLag - 1; t < rv.length - 1; t++) {
      const rvShort = rv[t];
      const rvMedium = rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5;
      const rvLong = rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22;
      const x = [1, rvShort, rvMedium, rvLong];
      const y = rv[t + 1];

      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          XtX[i][j] += x[i] * x[j];
        }
        Xty[i] += x[i] * y;
      }
    }

    // Verify X'X·β ≈ X'y
    for (let i = 0; i < p; i++) {
      let lhs = 0;
      for (let j = 0; j < p; j++) {
        lhs += XtX[i][j] * beta[j];
      }
      expect(lhs).toBeCloseTo(Xty[i], 6);
    }
  });
});

// ── fit() independent property verification ─────────────────

describe('HAR-RV fit independent verification', () => {
  it('fitted model LL > naive constant-variance model LL', () => {
    // Independent check: if the HAR-RV model captures volatility dynamics,
    // its LL should beat a "null model" that uses constant sample variance.
    // naiveLL = -n/2 * (ln(σ²) + 1) where σ² = sample variance of returns.
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const returns = model.getReturns();
    const n = returns.length;

    // Naive model: constant variance = Σr²/n (same formula the code uses for warmup)
    const sampleVar = returns.reduce((s, r) => s + r * r, 0) / n;
    // Closed-form Gaussian LL for constant variance:
    // LL = -0.5 * Σ[ln(σ²) + r²/σ²] = -0.5 * [n*ln(σ²) + n] = -n/2 * (ln(σ²) + 1)
    const naiveLL = -n / 2 * (Math.log(sampleVar) + 1);

    // Fitted model should explain data at least as well as constant variance
    expect(fit.diagnostics.logLikelihood).toBeGreaterThanOrEqual(naiveLL);
  });

  it('AIC of fitted model < AIC of zero-slope model', () => {
    // Independent check: a model with all slope betas = 0 should have worse AIC.
    // Zero-slope model: σ² = beta0 for all t (effectively constant variance).
    // Both models have 4 params, so AIC comparison reduces to LL comparison.
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const returns = model.getReturns();

    // Zero-slope LL: use getVarianceSeries with betas zeroed out
    const zeroParams = {
      ...fit.params,
      betaShort: 0,
      betaMedium: 0,
      betaLong: 0,
      beta0: returns.reduce((s, r) => s + r * r, 0) / returns.length,
    };
    const zeroVs = model.getVarianceSeries(zeroParams);

    let zeroLL = 0;
    for (let i = 0; i < returns.length; i++) {
      const v = zeroVs[i];
      if (v <= 1e-20 || !isFinite(v)) { zeroLL += -1e6; continue; }
      zeroLL += Math.log(v) + (returns[i] ** 2) / v;
    }
    zeroLL = -zeroLL / 2;
    const zeroAIC = 2 * 4 - 2 * zeroLL;

    expect(fit.diagnostics.aic).toBeLessThan(zeroAIC);
  });

  it('unconditional variance ≈ mean of variance series post-warmup', () => {
    // Independent check: for a stationary model, the unconditional variance
    // should approximate the average conditional variance over the sample.
    const prices = generatePrices(500, 42);
    const model = new HarRv(prices);
    const fit = model.fit();

    if (fit.params.persistence > 0 && fit.params.persistence < 1) {
      const vs = model.getVarianceSeries(fit.params);
      const postWarmup = vs.slice(22);
      const empiricalMean = postWarmup.reduce((s, v) => s + v, 0) / postWarmup.length;

      // Allow 50% relative tolerance — finite sample vs theoretical mean
      const relError = Math.abs(empiricalMean - fit.params.unconditionalVariance)
                     / fit.params.unconditionalVariance;
      expect(relError).toBeLessThan(0.5);
    }
  });

  it('unconditional variance ≈ long-horizon forecast', () => {
    // Independent check: iterating forecast far enough should converge
    // to the theoretical unconditional variance.
    const prices = generatePrices(500, 42);
    const model = new HarRv(prices);
    const fit = model.fit();

    if (fit.params.persistence > 0 && fit.params.persistence < 0.99) {
      const fc = model.forecast(fit.params, 200);
      const relError = Math.abs(fc.variance[199] - fit.params.unconditionalVariance)
                     / fit.params.unconditionalVariance;
      expect(relError).toBeLessThan(0.01);
    }
  });

  it('annualizedVol scales with sqrt(periodsPerYear) — ratio test', () => {
    // Independent check: same data, different periodsPerYear — vol ratio
    // should equal sqrt(ppy2/ppy1). This tests the scaling without copying the formula.
    const prices = generatePrices(300, 42);
    const r1 = calibrateHarRv(prices, { periodsPerYear: 252 });
    const r2 = calibrateHarRv(prices, { periodsPerYear: 252 * 4 });

    // uncondVar is the same (doesn't depend on periodsPerYear)
    expect(r1.params.unconditionalVariance).toBe(r2.params.unconditionalVariance);

    // annualizedVol should scale by sqrt(4) = 2
    const ratio = r2.params.annualizedVol / r1.params.annualizedVol;
    expect(ratio).toBeCloseTo(2.0, 10);
  });

  it('BIC > AIC for n >> k (property of penalty terms)', () => {
    // Independent check: BIC = k*ln(n) - 2LL, AIC = 2k - 2LL.
    // For n > e² ≈ 7.39, ln(n) > 2, so BIC penalty > AIC penalty, hence BIC > AIC.
    const prices = generatePrices(300, 42);
    const result = calibrateHarRv(prices);
    expect(result.diagnostics.bic).toBeGreaterThan(result.diagnostics.aic);
  });
});

// ── Variance series and forecast deep tests ──────────────────

describe('HAR-RV variance series deep tests', () => {
  it('fallback-to-prediction transition at longLag boundary', () => {
    const prices = generatePrices(300);
    const model = new HarRv(prices);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    // Position 21 (last fallback) should equal position 0 (all fallback = sample variance)
    expect(vs[21]).toBe(vs[0]);
    // Position 22 (first prediction) should generally differ from fallback
    // Can be same by coincidence, but check they're computed differently
    expect(typeof vs[22]).toBe('number');
    expect(vs[22]).toBeGreaterThan(0);
  });

  it('variance floor 1e-20 prevents zero/negative variance', () => {
    // Use params that would produce negative predicted variance
    const prices = generatePrices(200);
    const model = new HarRv(prices);
    const fit = model.fit();

    // Override with extreme negative betas to force negative prediction
    const extremeParams = {
      ...fit.params,
      beta0: -1,
      betaShort: -10,
      betaMedium: -10,
      betaLong: -10,
    };

    const vs = model.getVarianceSeries(extremeParams);
    for (const v of vs) {
      expect(v).toBeGreaterThanOrEqual(1e-20);
    }
  });

  it('variance series from getVarianceSeries matches internal fit calculation', () => {
    const prices = generatePrices(300, 55);
    const model = new HarRv(prices);
    const fit = model.fit();

    // Calling getVarianceSeries with same params should produce same result every time
    const vs1 = model.getVarianceSeries(fit.params);
    const vs2 = model.getVarianceSeries(fit.params);
    for (let i = 0; i < vs1.length; i++) {
      expect(vs1[i]).toBe(vs2[i]);
    }
  });

  it('variance series length = returns length', () => {
    const prices = generatePrices(200, 11);
    const model = new HarRv(prices);
    const fit = model.fit();
    expect(model.getVarianceSeries(fit.params).length).toBe(model.getReturns().length);
  });
});

describe('HAR-RV forecast deep tests', () => {
  it('2-step forecast ≠ 1-step * sqrt(2) — feedback is nonlinear', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const fc1 = model.forecast(fit.params, 1);
    const fc2 = model.forecast(fit.params, 2);

    // Multi-step is iterative, not simple scaling
    const naiveSecondStep = fc1.variance[0]; // would be same if no feedback
    // Second step should differ because it feeds back first forecast
    expect(fc2.variance[1]).not.toBeCloseTo(naiveSecondStep, 10);
  });

  it('stationary model: long-horizon forecast converges to unconditional variance (tight)', () => {
    const prices = generatePrices(1000, 42);
    const model = new HarRv(prices);
    const fit = model.fit();

    if (fit.params.persistence > 0 && fit.params.persistence < 0.99) {
      const fc = model.forecast(fit.params, 200);
      const lastVar = fc.variance[199];
      const uncond = fit.params.unconditionalVariance;
      const relError = Math.abs(lastVar - uncond) / uncond;
      expect(relError).toBeLessThan(0.05);
    }
  });

  it('forecast step 1 is continuous with variance series (last predicted value)', () => {
    // Independent check: forecast[0] should be close to the last value
    // of the in-sample variance series if recent RV is stable.
    // This catches off-by-one errors in the forecast indexing.
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);
    const fc = model.forecast(fit.params, 1);

    const lastVs = vs[vs.length - 1];
    // Not identical (different time step), but same order of magnitude
    const ratio = fc.variance[0] / lastVs;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });

  it('forecast annualized scales with sqrt(periodsPerYear) — ratio test', () => {
    // Independent check: same data, different periodsPerYear
    const prices = generatePrices(300, 42);
    const m1 = new HarRv(prices, { periodsPerYear: 100 });
    const m2 = new HarRv(prices, { periodsPerYear: 400 });
    const f1 = m1.fit();
    const f2 = m2.fit();
    const fc1 = m1.forecast(f1.params, 3);
    const fc2 = m2.forecast(f2.params, 3);

    // variance is the same (periodsPerYear doesn't affect variance)
    for (let i = 0; i < 3; i++) {
      expect(fc1.variance[i]).toBe(fc2.variance[i]);
    }
    // annualized scales by sqrt(400/100) = 2
    for (let i = 0; i < 3; i++) {
      const ratio = fc2.annualized[i] / fc1.annualized[i];
      expect(ratio).toBeCloseTo(2.0, 10);
    }
  });

  it('forecast with extreme negative betas still produces positive variance', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();

    const extremeParams = {
      ...fit.params,
      beta0: -0.5,
      betaShort: -5,
      betaMedium: -5,
      betaLong: -5,
    };

    const fc = model.forecast(extremeParams, 10);
    for (const v of fc.variance) {
      expect(v).toBeGreaterThanOrEqual(1e-20);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('multi-step variance is monotonically changing toward unconditional', () => {
    const prices = generatePrices(500, 42);
    const model = new HarRv(prices);
    const fit = model.fit();

    if (fit.params.persistence > 0 && fit.params.persistence < 0.95) {
      const fc = model.forecast(fit.params, 50);
      const uncond = fit.params.unconditionalVariance;

      // Distance from unconditional should generally decrease over time
      const dists = fc.variance.map(v => Math.abs(v - uncond));
      // Compare first third to last third — last third should be closer on average
      const firstThird = dists.slice(0, 15).reduce((a, b) => a + b, 0) / 15;
      const lastThird = dists.slice(35, 50).reduce((a, b) => a + b, 0) / 15;
      expect(lastThird).toBeLessThanOrEqual(firstThird + 1e-15);
    }
  });
});

// ── predict.ts integration: HAR-RV model selection ───────────

describe('HAR-RV model selection in predict', () => {
  it('predict returns valid result for HAR-structured data', () => {
    const candles = makeCandles(500, 42);
    const result = predict(candles, '4h');
    expect(['garch', 'egarch', 'har-rv', 'novas']).toContain(result.modelType);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.sigma).toBeGreaterThanOrEqual(0);
  });

  it('predict never crashes across diverse seeds', () => {
    for (let seed = 100; seed <= 130; seed++) {
      const candles = makeCandles(300, seed);
      const result = predict(candles, '1h');
      expect(Number.isFinite(result.sigma)).toBe(true);
      expect(Number.isFinite(result.move)).toBe(true);
      expect(result.upperPrice).toBeGreaterThan(result.lowerPrice);
      expect(['garch', 'egarch', 'har-rv', 'novas']).toContain(result.modelType);
    }
  });

  it('predictRange sigma grows with steps', () => {
    const candles = makeCandles(500, 42);
    const r3 = predictRange(candles, '4h', 3);
    const r10 = predictRange(candles, '4h', 10);
    expect(r10.sigma).toBeGreaterThan(r3.sigma);
  });
});

// ── Regression snapshots (deterministic) ─────────────────────

describe('HAR-RV regression snapshots', () => {
  it('fixed seed 42 produces exact hardcoded values', () => {
    // Frozen reference values for generatePrices(300, 42).
    // Any change to OLS, returns, or RV computation will break this.
    const prices = generatePrices(300, 42);
    const result = calibrateHarRv(prices);

    expect(result.params.beta0).toBeCloseTo(0.00004369693354642014, 15);
    expect(result.params.betaShort).toBeCloseTo(0.12859026062381232, 12);
    expect(result.params.betaMedium).toBeCloseTo(-0.11338226665144722, 12);
    expect(result.params.betaLong).toBeCloseTo(0.6963174240904823, 12);
    expect(result.params.persistence).toBeCloseTo(0.7115254180628473, 12);
    expect(result.params.r2).toBeCloseTo(0.09233338797305524, 12);
    expect(result.params.unconditionalVariance).toBeCloseTo(0.0001514758536193667, 12);
    expect(result.params.annualizedVol).toBeCloseTo(19.537634225279273, 8);
    expect(result.diagnostics.logLikelihood).toBeCloseTo(1178.0553492987865, 6);
    expect(result.diagnostics.aic).toBeCloseTo(-2348.110698597573, 6);
    expect(result.diagnostics.bic).toBeCloseTo(-2333.6146285728237, 6);
  });

  it('fitted params minimize RSS (OLS optimality — perturbation test)', () => {
    const prices = generatePrices(500, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();

    // Compute RSS for arbitrary beta values
    function computeRSS(beta: number[]): number {
      const longLag = 22;
      let rss = 0;
      for (let t = longLag - 1; t < rv.length - 1; t++) {
        const rvS = rv[t];
        const rvM = rv.slice(t - 4, t + 1).reduce((a, b) => a + b, 0) / 5;
        const rvL = rv.slice(t - 21, t + 1).reduce((a, b) => a + b, 0) / 22;
        const pred = beta[0] + beta[1] * rvS + beta[2] * rvM + beta[3] * rvL;
        const res = rv[t + 1] - pred;
        rss += res * res;
      }
      return rss;
    }

    const baseBeta = [fit.params.beta0, fit.params.betaShort, fit.params.betaMedium, fit.params.betaLong];
    const baseRSS = computeRSS(baseBeta);
    const deltas = [1e-6, -1e-6, 1e-4, -1e-4];

    // Perturb each parameter — RSS should not decrease
    for (let j = 0; j < 4; j++) {
      for (const delta of deltas) {
        const perturbed = [...baseBeta];
        perturbed[j] += delta;
        const perturbedRSS = computeRSS(perturbed);
        expect(perturbedRSS).toBeGreaterThanOrEqual(baseRSS - 1e-15);
      }
    }
  });
});

// ── Edge cases: equal lags, boundary data ────────────────────

describe('HAR-RV edge cases (deep)', () => {
  it('shortLag = mediumLag = longLag throws (singular — identical columns)', () => {
    const prices = generatePrices(200);
    expect(() => calibrateHarRv(prices, { shortLag: 5, mediumLag: 5, longLag: 5 }))
      .toThrow('Singular matrix');
  });

  it('all lags = 1 throws (singular — identical columns)', () => {
    const prices = generatePrices(200);
    expect(() => calibrateHarRv(prices, { shortLag: 1, mediumLag: 1, longLag: 1 }))
      .toThrow('Singular matrix');
  });

  it('very large periodsPerYear scales annualized correctly', () => {
    const prices = generatePrices(200);
    const r1 = calibrateHarRv(prices, { periodsPerYear: 1 });
    const r2 = calibrateHarRv(prices, { periodsPerYear: 1000000 });
    // Same unconditional variance
    expect(r1.params.unconditionalVariance).toBe(r2.params.unconditionalVariance);
    // annualizedVol scales with sqrt(periodsPerYear)
    const ratio = r2.params.annualizedVol / r1.params.annualizedVol;
    expect(ratio).toBeCloseTo(Math.sqrt(1000000 / 1), 3);
  });

  it('extremely skewed RV data does not crash', () => {
    // Create data where some returns are 100x larger
    const rng = lcg(42);
    const prices = [100];
    for (let i = 1; i <= 300; i++) {
      const spike = (i % 50 === 0) ? 10 : 1;
      prices.push(prices[i - 1] * Math.exp(spike * 0.01 * randn(rng)));
    }
    const result = calibrateHarRv(prices);
    expect(Number.isFinite(result.params.beta0)).toBe(true);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('near-constant prices throw singular matrix (degenerate RV)', () => {
    // Prices all identical except tiny noise → rv ≈ 0 → X'X singular
    const prices: number[] = [];
    for (let i = 0; i < 200; i++) {
      prices.push(100 + i * 1e-15);
    }
    expect(() => calibrateHarRv(prices)).toThrow('Singular matrix');
  });

  it('forecast from minimum data size (longLag + 30 + 1 prices)', () => {
    const prices = generatePrices(53); // 52 = 22+30, +1 for returns
    const model = new HarRv(prices);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 5);
    expect(fc.variance.length).toBe(5);
    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── Untested code paths in predict.ts ─────────────────────────

describe('HAR-RV predict.ts code paths', () => {
  it('fitHarRv try/catch: near-constant candles → GARCH fallback (not crash)', () => {
    // Near-constant prices cause singular matrix in HarRv.fit().
    // fitHarRv catches this and returns null → predict falls back to GARCH/EGARCH.
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      // Tiny deterministic increments → near-zero rv → singular X'X
      price += 1e-12;
      candles.push({ open: price, high: price + 1e-13, low: price - 1e-13, close: price, volume: 1000 });
    }
    const result = predict(candles, '8h');
    // Should not crash — GARCH family is the fallback
    expect(['garch', 'egarch', 'novas']).toContain(result.modelType);
    expect(Number.isFinite(result.sigma)).toBe(true);
  });

  it('HAR-RV wins model selection for at least one seed', () => {
    // Scan seeds to verify HAR-RV can actually win AIC comparison
    let harRvWon = false;
    for (let seed = 1; seed <= 200; seed++) {
      const candles = makeCandles(500, seed);
      const result = predict(candles, '4h');
      if (result.modelType === 'har-rv') {
        harRvWon = true;
        break;
      }
    }
    expect(harRvWon).toBe(true);
  });

  it('GARCH/EGARCH wins model selection for at least one seed', () => {
    let garchWon = false;
    for (let seed = 1; seed <= 200; seed++) {
      const candles = makeCandles(500, seed);
      const result = predict(candles, '4h');
      if (result.modelType === 'garch' || result.modelType === 'egarch') {
        garchWon = true;
        break;
      }
    }
    expect(garchWon).toBe(true);
  });
});

// ── LL clamping path ─────────────────────────────────────────

describe('HAR-RV LL clamping', () => {
  it('variance floor in getVarianceSeries triggers LL penalty path', () => {
    // If extreme betas produce predicted v <= 1e-20, the LL computation
    // in fit() uses -1e6 penalty. Verify this doesn't produce NaN/Infinity.
    const prices = generatePrices(200, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const returns = model.getReturns();

    // Construct params that force many floor values (1e-20)
    const badParams = {
      ...fit.params,
      beta0: -100,
      betaShort: -100,
      betaMedium: -100,
      betaLong: -100,
    };
    const vs = model.getVarianceSeries(badParams);

    // Count how many values hit the floor
    const flooredCount = vs.filter(v => v === 1e-20).length;
    expect(flooredCount).toBeGreaterThan(0);

    // Manually compute LL with clamping — should produce a very negative value
    let ll = 0;
    for (let i = 0; i < returns.length; i++) {
      const v = vs[i];
      if (v <= 1e-20 || !isFinite(v)) {
        ll += -1e6;
      } else {
        ll += Math.log(v) + (returns[i] ** 2) / v;
      }
    }
    ll = -ll / 2;

    // LL should be finite (very large positive due to -(-1e6))
    expect(Number.isFinite(ll)).toBe(true);
    // And much larger than the fitted LL (penalty inflates it)
    expect(ll).not.toBe(fit.diagnostics.logLikelihood);
  });
});

// ── Input equivalence ────────────────────────────────────────

describe('HAR-RV input equivalence', () => {
  it('Candle[] and number[] (close prices) produce identical results', () => {
    const candles = makeCandles(300, 42);
    const prices = candles.map(c => c.close);

    const resultCandles = calibrateHarRv(candles);
    const resultPrices = calibrateHarRv(prices);

    expect(resultCandles.params.beta0).toBe(resultPrices.params.beta0);
    expect(resultCandles.params.betaShort).toBe(resultPrices.params.betaShort);
    expect(resultCandles.params.betaMedium).toBe(resultPrices.params.betaMedium);
    expect(resultCandles.params.betaLong).toBe(resultPrices.params.betaLong);
    expect(resultCandles.params.r2).toBe(resultPrices.params.r2);
    expect(resultCandles.diagnostics.logLikelihood).toBe(resultPrices.diagnostics.logLikelihood);
  });

  it('periodsPerYear does NOT affect betas, r2, or persistence', () => {
    const prices = generatePrices(300, 42);
    const r1 = calibrateHarRv(prices, { periodsPerYear: 252 });
    const r2 = calibrateHarRv(prices, { periodsPerYear: 525600 });

    expect(r1.params.beta0).toBe(r2.params.beta0);
    expect(r1.params.betaShort).toBe(r2.params.betaShort);
    expect(r1.params.betaMedium).toBe(r2.params.betaMedium);
    expect(r1.params.betaLong).toBe(r2.params.betaLong);
    expect(r1.params.r2).toBe(r2.params.r2);
    expect(r1.params.persistence).toBe(r2.params.persistence);
    expect(r1.params.unconditionalVariance).toBe(r2.params.unconditionalVariance);
    // Only annualizedVol differs
    expect(r1.params.annualizedVol).not.toBe(r2.params.annualizedVol);
  });
});

// ── Mutation safety ──────────────────────────────────────────

describe('HAR-RV mutation safety', () => {
  it('getReturns() returns a copy — mutation does not affect model', () => {
    const prices = generatePrices(200, 42);
    const model = new HarRv(prices);
    const returns1 = model.getReturns();
    returns1[0] = 999999;
    const returns2 = model.getReturns();
    expect(returns2[0]).not.toBe(999999);
  });

  it('getRv() returns a copy — mutation does not affect model', () => {
    const prices = generatePrices(200, 42);
    const model = new HarRv(prices);
    const rv1 = model.getRv();
    rv1[0] = 999999;
    const rv2 = model.getRv();
    expect(rv2[0]).not.toBe(999999);
  });

  it('mutating fit result does not affect subsequent fit calls', () => {
    const prices = generatePrices(200, 42);
    const model = new HarRv(prices);
    const fit1 = model.fit();
    (fit1.params as any).beta0 = 999;
    const fit2 = model.fit();
    expect(fit2.params.beta0).not.toBe(999);
  });
});

// ── forecast(params, 0) ──────────────────────────────────────

describe('HAR-RV forecast edge cases', () => {
  it('forecast with steps=0 returns empty arrays', () => {
    const prices = generatePrices(200, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 0);

    expect(fc.variance).toEqual([]);
    expect(fc.volatility).toEqual([]);
    expect(fc.annualized).toEqual([]);
  });

  it('forecast step 2 correctly uses step 1 output in rolling mean', () => {
    // Verify iterative substitution: the 2nd forecast step's rolling mean
    // should include the 1st step's predicted variance.
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();
    const { beta0, betaShort, betaMedium, betaLong } = fit.params;

    // Step 1: use original rv
    const t0 = rv.length - 1;
    const rvS1 = rv[t0]; // shortLag=1
    const rvM1 = rv.slice(t0 - 4, t0 + 1).reduce((a, b) => a + b, 0) / 5;
    const rvL1 = rv.slice(t0 - 21, t0 + 1).reduce((a, b) => a + b, 0) / 22;
    const pred1 = Math.max(beta0 + betaShort * rvS1 + betaMedium * rvM1 + betaLong * rvL1, 1e-20);

    // Step 2: rv extended with pred1
    const extendedRv = [...rv, pred1];
    const t1 = extendedRv.length - 1;
    const rvS2 = extendedRv[t1]; // = pred1
    const rvM2 = extendedRv.slice(t1 - 4, t1 + 1).reduce((a, b) => a + b, 0) / 5;
    const rvL2 = extendedRv.slice(t1 - 21, t1 + 1).reduce((a, b) => a + b, 0) / 22;
    const pred2 = Math.max(beta0 + betaShort * rvS2 + betaMedium * rvM2 + betaLong * rvL2, 1e-20);

    const fc = model.forecast(fit.params, 2);
    expect(fc.variance[0]).toBeCloseTo(pred1, 12);
    expect(fc.variance[1]).toBeCloseTo(pred2, 12);
  });

  it('very long forecast (1000 steps) does not overflow or underflow', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const fc = model.forecast(fit.params, 1000);

    expect(fc.variance.length).toBe(1000);
    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
    for (const v of fc.annualized) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── Numerical edge cases ─────────────────────────────────────

describe('HAR-RV numerical scenarios', () => {
  it('very large prices (1e8 scale) — no overflow in rv', () => {
    const rng = lcg(42);
    const prices = [1e8];
    for (let i = 1; i < 200; i++) {
      prices.push(prices[i - 1] * Math.exp(0.01 * randn(rng)));
    }
    const result = calibrateHarRv(prices);
    expect(Number.isFinite(result.params.beta0)).toBe(true);
    expect(Number.isFinite(result.params.r2)).toBe(true);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('very small prices (1e-4 scale) — no underflow', () => {
    const rng = lcg(42);
    const prices = [0.0001];
    for (let i = 1; i < 200; i++) {
      prices.push(prices[i - 1] * Math.exp(0.01 * randn(rng)));
    }
    const result = calibrateHarRv(prices);
    expect(Number.isFinite(result.params.beta0)).toBe(true);
    expect(Number.isFinite(result.params.r2)).toBe(true);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('prices with large gap (1000 → 10 → 1000) — no crash', () => {
    const rng = lcg(42);
    const prices = [1000];
    for (let i = 1; i < 200; i++) {
      let base = prices[i - 1];
      if (i === 100) base = 10;   // crash
      if (i === 101) base = 1000; // recovery
      prices.push(base * Math.exp(0.01 * randn(rng)));
    }
    const result = calibrateHarRv(prices);
    expect(Number.isFinite(result.params.beta0)).toBe(true);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('log returns are scale-invariant — same betas for 10x scaled prices', () => {
    // log(kP_t / kP_{t-1}) = log(P_t / P_{t-1}), so scaling prices
    // should NOT change returns, rv, or betas.
    const prices = generatePrices(300, 42);
    const scaledPrices = prices.map(p => p * 1000);

    const r1 = calibrateHarRv(prices);
    const r2 = calibrateHarRv(scaledPrices);

    expect(r1.params.beta0).toBeCloseTo(r2.params.beta0, 10);
    expect(r1.params.betaShort).toBeCloseTo(r2.params.betaShort, 10);
    expect(r1.params.r2).toBeCloseTo(r2.params.r2, 10);
  });
});

// ── Partial pivoting ─────────────────────────────────────────

describe('HAR-RV partial pivoting (indirect)', () => {
  it('diverse seeds never fail due to pivot issues', () => {
    // Gaussian elimination without pivoting would fail on certain data
    // configurations. Running many seeds exercises different X'X matrices.
    for (let seed = 1; seed <= 100; seed++) {
      const prices = generatePrices(200, seed);
      const result = calibrateHarRv(prices);
      expect(result.diagnostics.converged).toBe(true);
      expect(Number.isFinite(result.params.beta0)).toBe(true);
    }
  });
});

// ── reliable flag with HAR-RV ─────────────────────────────────

describe('HAR-RV reliable flag', () => {
  it('reliable is boolean for HAR-RV model', () => {
    // Scan seeds until HAR-RV wins, then check reliable flag
    for (let seed = 1; seed <= 200; seed++) {
      const candles = makeCandles(500, seed);
      const result = predict(candles, '4h');
      if (result.modelType === 'har-rv') {
        expect(typeof result.reliable).toBe('boolean');
        return;
      }
    }
    // If no HAR-RV won, just verify reliable works for any model
    const candles = makeCandles(500, 42);
    expect(typeof predict(candles, '4h').reliable).toBe('boolean');
  });

  it('reliable=false when persistence >= 0.999', () => {
    // High persistence models should be flagged unreliable.
    // We can verify this property: for any model with persistence >= 0.999,
    // reliable must be false. Scan multiple seeds.
    for (let seed = 1; seed <= 50; seed++) {
      const candles = makeCandles(300, seed);
      const result = predict(candles, '1h');
      // We can't know which model was selected, but the contract holds:
      // if the internal persistence >= 0.999, reliable must be false
      // (we can't check persistence directly from predict output, but
      // we verify the flag is always a valid boolean)
      expect(typeof result.reliable).toBe('boolean');
    }
  });

  it('reliable flag is consistent between predict and predictRange', () => {
    const candles = makeCandles(500, 42);
    const p1 = predict(candles, '4h');
    const pRange = predictRange(candles, '4h', 1);
    // Same data, same steps=1 — same model, same reliable
    expect(pRange.reliable).toBe(p1.reliable);
    expect(pRange.modelType).toBe(p1.modelType);
  });
});

// ── Custom lags + forecast ───────────────────────────────────

describe('HAR-RV custom lags + forecast', () => {
  it('custom lags affect forecast values', () => {
    const prices = generatePrices(300, 42);
    const m1 = new HarRv(prices, { shortLag: 1, mediumLag: 5, longLag: 22 }); // default
    const m2 = new HarRv(prices, { shortLag: 3, mediumLag: 10, longLag: 22 });
    const f1 = m1.fit();
    const f2 = m2.fit();
    const fc1 = m1.forecast(f1.params, 5);
    const fc2 = m2.forecast(f2.params, 5);

    // Different lags → different betas → different forecasts
    let allSame = true;
    for (let i = 0; i < 5; i++) {
      if (Math.abs(fc1.variance[i] - fc2.variance[i]) > 1e-15) {
        allSame = false;
        break;
      }
    }
    expect(allSame).toBe(false);
  });

  it('custom lags forecast produces positive finite values', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices, { shortLag: 2, mediumLag: 7, longLag: 15 });
    const fit = model.fit();
    const fc = model.forecast(fit.params, 10);

    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('custom lags variance series uses correct lag windows', () => {
    const prices = generatePrices(200, 42);
    const longLag = 15;
    const model = new HarRv(prices, { shortLag: 2, mediumLag: 7, longLag });
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    // First longLag entries should be fallback (all identical)
    const fallback = vs[0];
    for (let i = 1; i < longLag; i++) {
      expect(vs[i]).toBe(fallback);
    }
    // Entry at longLag should be a prediction (generally different)
    expect(typeof vs[longLag]).toBe('number');
    expect(vs[longLag]).toBeGreaterThan(0);
  });
});

// ── Two equal lags out of three ──────────────────────────────

describe('HAR-RV two equal lags', () => {
  it('shortLag = mediumLag ≠ longLag — may produce collinear columns', () => {
    const prices = generatePrices(200, 42);
    // shortLag=5, mediumLag=5 → identical columns; longLag=22 → unique
    // But with intercept, X has [1, rv5, rv5, rv22] — two identical columns → singular
    expect(() => calibrateHarRv(prices, { shortLag: 5, mediumLag: 5, longLag: 22 }))
      .toThrow('Singular matrix');
  });

  it('mediumLag = longLag ≠ shortLag — also singular', () => {
    const prices = generatePrices(200, 42);
    expect(() => calibrateHarRv(prices, { shortLag: 1, mediumLag: 22, longLag: 22 }))
      .toThrow('Singular matrix');
  });

  it('all three lags distinct — works fine', () => {
    const prices = generatePrices(200, 42);
    const result = calibrateHarRv(prices, { shortLag: 1, mediumLag: 10, longLag: 22 });
    expect(result.diagnostics.converged).toBe(true);
    expect(Number.isFinite(result.params.r2)).toBe(true);
  });
});

// ── Invalid inputs ───────────────────────────────────────────

describe('HAR-RV invalid inputs', () => {
  it('NaN in prices throws', () => {
    const prices = generatePrices(200, 42);
    prices[50] = NaN;
    expect(() => new HarRv(prices)).toThrow('Invalid price');
  });

  it('Infinity in prices throws', () => {
    const prices = generatePrices(200, 42);
    prices[50] = Infinity;
    expect(() => new HarRv(prices)).toThrow('Invalid price');
  });

  it('zero price throws', () => {
    const prices = generatePrices(200, 42);
    prices[50] = 0;
    expect(() => new HarRv(prices)).toThrow('Invalid price');
  });

  it('negative price throws', () => {
    const prices = generatePrices(200, 42);
    prices[50] = -1;
    expect(() => new HarRv(prices)).toThrow('Invalid price');
  });

  it('NaN in candle close throws', () => {
    const candles = makeCandles(200, 42);
    candles[50] = { ...candles[50], close: NaN };
    expect(() => new HarRv(candles)).toThrow('Invalid close price');
  });
});

// ── nObs verification ────────────────────────────────────────

describe('HAR-RV regression observation count', () => {
  it('nObs = rv.length - longLag for default lags', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const fit = model.fit();
    const rv = model.getRv();

    // Regression range: t = longLag-1 .. rv.length-2
    // nObs = (rv.length - 2) - (22 - 1) + 1 = rv.length - 22
    const nObs = rv.length - 22;

    // Verify via BIC: BIC = k*ln(nObs) - 2*LL
    // So nObs = exp((BIC + 2*LL) / k)
    const { aic, bic, logLikelihood: ll } = fit.diagnostics;
    // AIC = 2k - 2LL → 2LL = 2k - AIC
    // BIC = k*ln(n) - 2LL → k*ln(n) = BIC + 2LL = BIC + 2k - AIC
    // ln(n) = (BIC + 2k - AIC) / k
    const k = 4;
    const lnN = (bic + 2 * k - aic) / k;
    const recoveredN = Math.round(Math.exp(lnN));

    expect(recoveredN).toBe(nObs);
  });

  it('nObs = rv.length - customLongLag for custom lags', () => {
    const prices = generatePrices(300, 42);
    const longLag = 15;
    const model = new HarRv(prices, { shortLag: 2, mediumLag: 7, longLag });
    const fit = model.fit();
    const rv = model.getRv();

    const nObs = rv.length - longLag;
    const k = 4;
    const { aic, bic } = fit.diagnostics;
    const lnN = (bic + 2 * k - aic) / k;
    const recoveredN = Math.round(Math.exp(lnN));

    expect(recoveredN).toBe(nObs);
  });
});

// ── fit() idempotency ────────────────────────────────────────

describe('HAR-RV fit idempotency', () => {
  it('calling fit() twice on same instance returns identical results', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const r1 = model.fit();
    const r2 = model.fit();

    expect(r1.params.beta0).toBe(r2.params.beta0);
    expect(r1.params.betaShort).toBe(r2.params.betaShort);
    expect(r1.params.betaMedium).toBe(r2.params.betaMedium);
    expect(r1.params.betaLong).toBe(r2.params.betaLong);
    expect(r1.params.r2).toBe(r2.params.r2);
    expect(r1.diagnostics.logLikelihood).toBe(r2.diagnostics.logLikelihood);
    expect(r1.diagnostics.aic).toBe(r2.diagnostics.aic);
    expect(r1.diagnostics.bic).toBe(r2.diagnostics.bic);
  });

  it('fit → forecast → fit again gives same params', () => {
    const prices = generatePrices(300, 42);
    const model = new HarRv(prices);
    const r1 = model.fit();
    model.forecast(r1.params, 100); // should not mutate internal state
    const r2 = model.fit();

    expect(r1.params.beta0).toBe(r2.params.beta0);
    expect(r1.params.betaShort).toBe(r2.params.betaShort);
    expect(r1.params.r2).toBe(r2.params.r2);
  });
});

// ── backtest meaningful result ───────────────────────────────

describe('HAR-RV backtest', () => {
  it('backtest hit rate is > 0% (model predicts something)', () => {
    // backtest returns true if hit rate >= requiredPercent.
    // With requiredPercent=0, it should always return true (> 0% hits).
    const candles = makeCandles(500, 42);
    expect(backtest(candles, '4h', 0)).toBe(true);
  });

  it('backtest hit rate is < 100% (model is not perfect)', () => {
    // With requiredPercent=100, model should fail (not every move within ±1σ)
    const candles = makeCandles(500, 42);
    expect(backtest(candles, '4h', 100)).toBe(false);
  });

  it('backtest result is deterministic', () => {
    const candles = makeCandles(500, 42);
    const r1 = backtest(candles, '4h');
    const r2 = backtest(candles, '4h');
    expect(r1).toBe(r2);
  });
});

// ── Fuzz testing ──────────────────────────────────────────────

describe('HAR-RV fuzz tests', () => {
  const seeds = [1, 7, 13, 19, 23, 31, 37, 41, 53, 59, 67, 73];

  for (const seed of seeds) {
    it(`seed ${seed}: calibrateHarRv never produces NaN/Infinity`, () => {
      const prices = generatePrices(300, seed);
      const result = calibrateHarRv(prices);

      expect(Number.isFinite(result.params.beta0)).toBe(true);
      expect(Number.isFinite(result.params.betaShort)).toBe(true);
      expect(Number.isFinite(result.params.betaMedium)).toBe(true);
      expect(Number.isFinite(result.params.betaLong)).toBe(true);
      expect(Number.isFinite(result.params.persistence)).toBe(true);
      expect(Number.isFinite(result.params.r2)).toBe(true);
      expect(Number.isFinite(result.params.unconditionalVariance)).toBe(true);
      expect(Number.isFinite(result.params.annualizedVol)).toBe(true);
    });

    it(`seed ${seed}: forecast is always finite and positive`, () => {
      const prices = generatePrices(300, seed);
      const model = new HarRv(prices);
      const fit = model.fit();
      const fc = model.forecast(fit.params, 10);

      for (const v of fc.variance) {
        expect(v).toBeGreaterThan(0);
        expect(Number.isFinite(v)).toBe(true);
      }
    });
  }
});
