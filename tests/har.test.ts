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
    expect(['garch', 'egarch', 'har-rv']).toContain(result.modelType);
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
      expect(['garch', 'egarch', 'har-rv']).toContain(result.modelType);
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
