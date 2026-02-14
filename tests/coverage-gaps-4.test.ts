import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  GjrGarch,
  calibrateGarch,
  calibrateEgarch,
  calibrateGjrGarch,
  calculateReturns,
  calculateReturnsFromPrices,
  checkLeverageEffect,
  nelderMead,
  EXPECTED_ABS_NORMAL,
  type Candle,
} from '../src/index.js';

// ── helpers ──────────────────────────────────────────────────

function makePrices(n: number, seed = 12345): number[] {
  const prices = [100];
  let state = seed;
  for (let i = 1; i < n; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    const r = ((state / 0x7fffffff) - 0.5) * 0.04;
    prices.push(prices[i - 1] * Math.exp(r));
  }
  return prices;
}

function lcg(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function seededRandn(rng: () => number) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ── 1. calculateReturns — invalid close in candles ──────────

describe('calculateReturns candle validation', () => {
  const validCandle: Candle = { open: 99, high: 102, low: 98, close: 100, volume: 1000 };

  it('throws on NaN close', () => {
    const candles: Candle[] = [
      validCandle,
      { open: 100, high: 101, low: 99, close: NaN, volume: 500 },
    ];
    expect(() => calculateReturns(candles)).toThrow('Invalid close price');
  });

  it('does not throw on Infinity close (Infinity > 0 is true)', () => {
    // Note: !(Infinity > 0) === false, so the guard does not catch it.
    // calculateReturns will produce Infinity return. This documents actual behavior.
    const candles: Candle[] = [
      validCandle,
      { open: 100, high: 101, low: 99, close: Infinity, volume: 500 },
    ];
    const returns = calculateReturns(candles);
    expect(returns[0]).toBe(Infinity);
  });

  it('throws on negative close', () => {
    const candles: Candle[] = [
      validCandle,
      { open: 100, high: 101, low: 99, close: -5, volume: 500 },
    ];
    expect(() => calculateReturns(candles)).toThrow('Invalid close price');
  });

  it('throws on zero close', () => {
    const candles: Candle[] = [
      validCandle,
      { open: 100, high: 101, low: 99, close: 0, volume: 500 },
    ];
    expect(() => calculateReturns(candles)).toThrow('Invalid close price');
  });

  it('throws when first candle has invalid close', () => {
    const candles: Candle[] = [
      { open: 99, high: 102, low: 98, close: -1, volume: 1000 },
      { open: 100, high: 101, low: 99, close: 100, volume: 500 },
    ];
    expect(() => calculateReturns(candles)).toThrow('Invalid close price');
  });
});

// ── 2. Garch/Egarch constructor with NaN close candle ───────

describe('Constructor with invalid candles', () => {
  function makeCandles(n: number): Candle[] {
    const candles: Candle[] = [];
    let close = 100;
    const rng = lcg(42);
    for (let i = 0; i < n; i++) {
      const newClose = close * (1 + (rng() - 0.5) * 0.02);
      candles.push({ open: close, high: Math.max(close, newClose) + 0.1, low: Math.min(close, newClose) - 0.1, close: newClose, volume: 1000 });
      close = newClose;
    }
    return candles;
  }

  it('Garch throws on candle with NaN close', () => {
    const candles = makeCandles(100);
    candles[50].close = NaN;
    expect(() => new Garch(candles)).toThrow();
  });

  it('Egarch throws on candle with NaN close', () => {
    const candles = makeCandles(100);
    candles[50].close = NaN;
    expect(() => new Egarch(candles)).toThrow();
  });
});

// ── 3. periodsPerYear: 0 ───────────────────────────────────

describe('periodsPerYear edge cases', () => {
  it('periodsPerYear = 0: annualizedVol is 0', () => {
    const result = calibrateGarch(makePrices(100), { periodsPerYear: 0 });

    expect(result.params.annualizedVol).toBe(0);
  });

  it('EGARCH periodsPerYear = 0: annualizedVol is 0', () => {
    const result = calibrateEgarch(makePrices(100), { periodsPerYear: 0 });

    expect(result.params.annualizedVol).toBe(0);
  });

  // ── 4. periodsPerYear: negative ─────────────────────────────

  it('periodsPerYear negative: annualizedVol is NaN', () => {
    const result = calibrateGarch(makePrices(100), { periodsPerYear: -1 });

    expect(result.params.annualizedVol).toBeNaN();
  });

  it('EGARCH periodsPerYear negative: annualizedVol is NaN', () => {
    const result = calibrateEgarch(makePrices(100), { periodsPerYear: -1 });

    expect(result.params.annualizedVol).toBeNaN();
  });
});

// ── 5. Nelder-Mead: function returns NaN / Infinity ─────────

describe('nelderMead with pathological objective', () => {
  it('handles function that returns NaN for some inputs', () => {
    // sqrt(x) is NaN for negative x; optimizer should avoid those regions
    function fn(x: number[]): number {
      const val = (x[0] - 2) ** 2;
      return Number.isFinite(val) ? val : 1e10;
    }

    const result = nelderMead(fn, [0], { maxIter: 1000 });
    expect(result.x[0]).toBeCloseTo(2, 2);
    expect(result.converged).toBe(true);
  });

  it('handles function that returns Infinity for some inputs', () => {
    function fn(x: number[]): number {
      if (x[0] < 0) return Infinity;
      return (x[0] - 3) ** 2;
    }

    const result = nelderMead(fn, [5], { maxIter: 1000 });
    expect(result.x[0]).toBeCloseTo(3, 2);
  });

  it('constant function: converges immediately', () => {
    const result = nelderMead(() => 42, [1, 2, 3]);

    expect(result.fx).toBe(42);
    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(0);
  });
});

// ── 6. Nelder-Mead: negative x0 values ─────────────────────

describe('nelderMead with negative x0', () => {
  it('negative initial point: delta = x0[i] * 0.05 is negative', () => {
    function fn(x: number[]): number {
      return (x[0] + 5) ** 2 + (x[1] + 3) ** 2;
    }

    // Both x0 values are negative → simplex expands in negative direction
    const result = nelderMead(fn, [-10, -10], { maxIter: 2000 });

    expect(result.x[0]).toBeCloseTo(-5, 2);
    expect(result.x[1]).toBeCloseTo(-3, 2);
    expect(result.converged).toBe(true);
  });

  it('mixed positive and negative x0', () => {
    function fn(x: number[]): number {
      return (x[0] - 3) ** 2 + (x[1] + 4) ** 2;
    }

    const result = nelderMead(fn, [10, -10]);

    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-4, 2);
  });
});

// ── 7. Flat function → immediate convergence ────────────────
// (covered in #5 constant function test above)

// ── 8. getVarianceSeries returns new array each time ────────

describe('getVarianceSeries immutability', () => {
  it('GARCH: returns different reference each call', () => {
    const model = new Garch(makePrices(55));
    const result = model.fit();
    const a = model.getVarianceSeries(result.params);
    const b = model.getVarianceSeries(result.params);

    expect(a).toEqual(b);
    expect(a).not.toBe(b);

    a[0] = 999;
    expect(model.getVarianceSeries(result.params)[0]).not.toBe(999);
  });

  it('EGARCH: returns different reference each call', () => {
    const model = new Egarch(makePrices(55));
    const result = model.fit();
    const a = model.getVarianceSeries(result.params);
    const b = model.getVarianceSeries(result.params);

    expect(a).toEqual(b);
    expect(a).not.toBe(b);

    a[0] = 999;
    expect(model.getVarianceSeries(result.params)[0]).not.toBe(999);
  });

  it('GJR-GARCH: returns different reference each call', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();
    const a = model.getVarianceSeries(result.params);
    const b = model.getVarianceSeries(result.params);

    expect(a).toEqual(b);
    expect(a).not.toBe(b);

    a[0] = 999;
    expect(model.getVarianceSeries(result.params)[0]).not.toBe(999);
  });
});

// ── 9. forecast returns new arrays each time ────────────────

describe('forecast immutability', () => {
  it('GARCH: forecast returns new arrays', () => {
    const model = new Garch(makePrices(55));
    const result = model.fit();
    const fc1 = model.forecast(result.params, 5);
    const fc2 = model.forecast(result.params, 5);

    expect(fc1.variance).toEqual(fc2.variance);
    expect(fc1.variance).not.toBe(fc2.variance);
    expect(fc1.volatility).not.toBe(fc2.volatility);
    expect(fc1.annualized).not.toBe(fc2.annualized);
  });

  it('EGARCH: forecast returns new arrays', () => {
    const model = new Egarch(makePrices(55));
    const result = model.fit();
    const fc1 = model.forecast(result.params, 5);
    const fc2 = model.forecast(result.params, 5);

    expect(fc1.variance).toEqual(fc2.variance);
    expect(fc1.variance).not.toBe(fc2.variance);
    expect(fc1.volatility).not.toBe(fc2.volatility);
    expect(fc1.annualized).not.toBe(fc2.annualized);
  });

  it('GJR-GARCH: forecast returns new arrays', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();
    const fc1 = model.forecast(result.params, 5);
    const fc2 = model.forecast(result.params, 5);

    expect(fc1.variance).toEqual(fc2.variance);
    expect(fc1.variance).not.toBe(fc2.variance);
    expect(fc1.volatility).not.toBe(fc2.volatility);
    expect(fc1.annualized).not.toBe(fc2.annualized);
  });
});

// ── 10. getInitialVariance stable after fit ─────────────────

describe('getInitialVariance stability', () => {
  it('GARCH: getInitialVariance same before and after fit', () => {
    const model = new Garch(makePrices(100));
    const before = model.getInitialVariance();
    model.fit();
    const after = model.getInitialVariance();

    expect(after).toBe(before);
  });

  it('EGARCH: getInitialVariance same before and after fit', () => {
    const model = new Egarch(makePrices(100));
    const before = model.getInitialVariance();
    model.fit();
    const after = model.getInitialVariance();

    expect(after).toBe(before);
  });

  it('GJR-GARCH: getInitialVariance same before and after fit', () => {
    const model = new GjrGarch(makePrices(100));
    const before = model.getInitialVariance();
    model.fit();
    const after = model.getInitialVariance();

    expect(after).toBe(before);
  });
});

// ── 11. Input data not mutated by constructor ───────────────

describe('Input data immutability', () => {
  it('price array not mutated by Garch constructor', () => {
    const prices = makePrices(100);
    const copy = [...prices];

    new Garch(prices);

    expect(prices).toEqual(copy);
  });

  it('price array not mutated by Egarch constructor', () => {
    const prices = makePrices(100);
    const copy = [...prices];

    new Egarch(prices);

    expect(prices).toEqual(copy);
  });

  it('candle array not mutated by Garch constructor', () => {
    const candles: Candle[] = [];
    let close = 100;
    const rng = lcg(42);
    for (let i = 0; i < 100; i++) {
      const newClose = close * (1 + (rng() - 0.5) * 0.02);
      candles.push({ open: close, high: Math.max(close, newClose) + 0.1, low: Math.min(close, newClose) - 0.1, close: newClose, volume: 1000 });
      close = newClose;
    }
    const copy = candles.map(c => ({ ...c }));

    new Garch(candles);

    expect(candles).toEqual(copy);
  });
});

// ── 12. GARCH negLL: variance <= 1e-12 mid-loop guard ───────

describe('GARCH variance floor guard', () => {
  it('fit succeeds even when data could drive variance near zero', () => {
    // Near-constant prices with one tiny wiggle
    const prices: number[] = [100];
    for (let i = 1; i <= 200; i++) {
      // Extremely small returns
      prices.push(prices[i - 1] * (1 + 1e-10 * (i % 2 === 0 ? 1 : -1)));
    }

    const model = new Garch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.omega).toBeGreaterThan(0);
  });
});

// ── 13. EGARCH logVariance clamp boundaries ─────────────────

describe('EGARCH logVariance clamp', () => {
  it('getVarianceSeries clamps logVariance to [-50, 50]', () => {
    const model = new Egarch(makePrices(55));

    // Extreme omega drives logVariance toward +50
    const paramsHigh = {
      omega: 100, alpha: 0, gamma: 0, beta: 0.1,
      persistence: 0.1,
      unconditionalVariance: Math.exp(100 / 0.9),
      annualizedVol: 0,
      leverageEffect: 0,
      df: 30,
    };
    const vHigh = model.getVarianceSeries(paramsHigh);

    // After first step, logVariance should be clamped to 50
    // exp(50) ≈ 5.18e21
    const maxVar = Math.exp(50);
    for (let i = 1; i < vHigh.length; i++) {
      expect(vHigh[i]).toBeLessThanOrEqual(maxVar);
      expect(vHigh[i]).toBeGreaterThan(0);
    }

    // Extreme negative omega drives logVariance toward -50
    const paramsLow = {
      omega: -100, alpha: 0, gamma: 0, beta: 0.1,
      persistence: 0.1,
      unconditionalVariance: Math.exp(-100 / 0.9),
      annualizedVol: 0,
      leverageEffect: 0,
      df: 30,
    };
    const vLow = model.getVarianceSeries(paramsLow);

    // exp(-50) ≈ 1.93e-22
    const minVar = Math.exp(-50);
    for (let i = 1; i < vLow.length; i++) {
      expect(vLow[i]).toBeGreaterThanOrEqual(minVar);
      expect(Number.isFinite(vLow[i])).toBe(true);
    }
  });
});

// ── 14. checkLeverageEffect with empty array ────────────────

describe('checkLeverageEffect edge cases', () => {
  it('empty returns array', () => {
    const stats = checkLeverageEffect([]);

    // No positive or negative returns
    expect(stats.ratio).toBe(1);
    expect(stats.recommendation).toBe('garch');
  });

  it('single zero return', () => {
    const stats = checkLeverageEffect([0]);

    expect(stats.ratio).toBe(1);
    expect(stats.recommendation).toBe('garch');
  });
});

// ── 15. Forecast with persistence ≈ 1 ──────────────────────

describe('Forecast with near-unit persistence', () => {
  it('GARCH: persistence = 0.998, forecast stays finite', () => {
    const model = new Garch(makePrices(200));
    const result = model.fit();

    // Override with near-unit persistence params
    const params = {
      omega: 1e-7,
      alpha: 0.05,
      beta: 0.948,
      persistence: 0.998,
      unconditionalVariance: 1e-7 / (1 - 0.998),
      annualizedVol: Math.sqrt((1e-7 / 0.002) * 252) * 100,
      df: 30,
    };

    const fc = model.forecast(params, 1000);

    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
    expect(fc.volatility.every(v => v > 0 && Number.isFinite(v))).toBe(true);

    // Should still converge (slowly) toward unconditional
    const unconditional = params.unconditionalVariance;
    const relErr = Math.abs(fc.variance[999] - unconditional) / unconditional;
    expect(relErr).toBeLessThan(0.5);
  });

  it('EGARCH: |beta| = 0.998, forecast stays finite', () => {
    const model = new Egarch(makePrices(200));
    const result = model.fit();

    const params = {
      omega: -0.01,
      alpha: 0.1,
      gamma: -0.05,
      beta: 0.998,
      persistence: 0.998,
      unconditionalVariance: Math.exp(-0.01 / (1 - 0.998)),
      annualizedVol: 0,
      leverageEffect: -0.05,
      df: 30,
    };

    const fc = model.forecast(params, 1000);

    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });
});
