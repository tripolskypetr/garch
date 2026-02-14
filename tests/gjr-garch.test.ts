import { describe, it, expect } from 'vitest';
import {
  GjrGarch,
  calibrateGjrGarch,
  calibrateGarch,
  calibrateEgarch,
  calculateReturnsFromPrices,
  sampleVariance,
  yangZhangVariance,
  perCandleParkinson,
  predict,
  predictRange,
  backtest,
  type Candle,
} from '../src/index.js';

// ── helpers ──────────────────────────────────────────────────

function lcg(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function seededRandn(rng: () => number) {
  const u1 = rng() || 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

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

function makeCandles(n: number, seed = 42): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const r = seededRandn(rng) * 0.01;
    const close = price * Math.exp(r);
    const high = Math.max(price, close) * (1 + Math.abs(seededRandn(rng)) * 0.002);
    const low = Math.min(price, close) * (1 - Math.abs(seededRandn(rng)) * 0.002);
    candles.push({ open: price, high, low, close, volume: 1000 + rng() * 500 });
    price = close;
  }
  return candles;
}

function makeFlatCandles(n: number, seed = 42): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const r = seededRandn(rng) * 0.01;
    const close = price * Math.exp(r);
    candles.push({ open: close, high: close, low: close, close, volume: 1000 });
    price = close;
  }
  return candles;
}

/** Generate data with GJR-GARCH DGP (data generating process) */
function generateGjrGarchData(
  n: number, omega: number, alpha: number, gamma: number, beta: number, seed = 42,
): number[] {
  const rng = lcg(seed);
  let v = omega / (1 - alpha - gamma / 2 - beta);
  const prices = [100];
  for (let i = 0; i < n; i++) {
    const eps = Math.sqrt(v) * seededRandn(rng);
    prices.push(prices[prices.length - 1] * Math.exp(eps));
    const indicator = eps < 0 ? 1 : 0;
    v = omega + alpha * eps ** 2 + gamma * eps ** 2 * indicator + beta * v;
  }
  return prices;
}

function gjrParams(omega: number, alpha: number, gamma: number, beta: number) {
  const persistence = alpha + gamma / 2 + beta;
  return {
    omega, alpha, gamma, beta, persistence,
    unconditionalVariance: omega / (1 - persistence),
    annualizedVol: Math.sqrt((omega / (1 - persistence)) * 252) * 100,
    leverageEffect: gamma,
  };
}

function computeLL(returns: number[], variance: number[]): number {
  let sum = 0;
  for (let i = 0; i < returns.length; i++) {
    sum += Math.log(variance[i]) + returns[i] ** 2 / variance[i];
  }
  return -0.5 * sum;
}

// ═══════════════════════════════════════════════════════════════
// 1. calibrateGjrGarch — basic estimation
// ═══════════════════════════════════════════════════════════════

describe('calibrateGjrGarch', () => {
  it('should estimate parameters close to true values on DGP data', () => {
    const prices = generateGjrGarchData(1000, 0.00001, 0.05, 0.1, 0.85);
    const result = calibrateGjrGarch(prices);

    expect(result.params.alpha).toBeGreaterThan(0);
    expect(result.params.alpha).toBeLessThan(0.3);
    expect(result.params.gamma).toBeGreaterThanOrEqual(0);
    expect(result.params.beta).toBeGreaterThan(0.5);
    expect(result.params.beta).toBeLessThan(0.99);
    expect(result.params.persistence).toBeLessThan(1);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('should handle price array input', () => {
    const prices = makePrices(200);
    const result = calibrateGjrGarch(prices);

    expect(result.params.omega).toBeGreaterThan(0);
    expect(result.params.alpha).toBeGreaterThanOrEqual(0);
    expect(result.params.gamma).toBeGreaterThanOrEqual(0);
    expect(result.params.beta).toBeGreaterThanOrEqual(0);
  });

  it('should handle candle input', () => {
    const candles = makeCandles(200);
    const result = calibrateGjrGarch(candles);

    expect(result.params.omega).toBeGreaterThan(0);
    expect(result.params.persistence).toBeLessThan(1);
    expect(result.diagnostics.converged).toBe(true);
  });

  it('should throw on insufficient data', () => {
    const prices = [100, 101, 102];
    expect(() => calibrateGjrGarch(prices)).toThrow('at least 50');
  });
});

// ═══════════════════════════════════════════════════════════════
// 2. GjrGarch class
// ═══════════════════════════════════════════════════════════════

describe('GjrGarch class', () => {
  it('should compute variance series', () => {
    const prices = makePrices(200);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const variance = model.getVarianceSeries(result.params);

    expect(variance.length).toBe(prices.length - 1);
    expect(variance.every(v => v > 0)).toBe(true);
  });

  it('should forecast variance', () => {
    const prices = makePrices(200);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const forecast = model.forecast(result.params, 10);

    expect(forecast.variance.length).toBe(10);
    expect(forecast.volatility.length).toBe(10);
    expect(forecast.annualized.length).toBe(10);

    const lastForecast = forecast.variance[9];
    const unconditional = result.params.unconditionalVariance;
    expect(Math.abs(lastForecast - unconditional) / unconditional).toBeLessThan(0.5);
  });

  it('should return correct returns', () => {
    const prices = makePrices(55);
    const model = new GjrGarch(prices);
    const returns = model.getReturns();

    expect(returns.length).toBe(prices.length - 1);
    expect(returns[0]).toBeCloseTo(Math.log(prices[1] / prices[0]));
  });

  it('Candle[] end-to-end: construct → fit → forecast', () => {
    const candles = makeCandles(200);
    const model = new GjrGarch(candles);
    const result = model.fit();
    const fc = model.forecast(result.params, 10);

    expect(result.params.persistence).toBeLessThan(1);
    expect(result.diagnostics.converged).toBe(true);
    expect(fc.variance).toHaveLength(10);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════
// 3. Variance recursion verification
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH variance recursion', () => {
  it('variance[0] equals initialVariance (sampleVariance for number[])', () => {
    const prices = makePrices(100);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const vs = model.getVarianceSeries(result.params);

    expect(vs[0]).toBe(model.getInitialVariance());
  });

  it('variance[0] equals initialVariance (yangZhang for Candle[])', () => {
    const candles = makeCandles(100);
    const model = new GjrGarch(candles);
    const result = model.fit();
    const vs = model.getVarianceSeries(result.params);

    expect(vs[0]).toBe(model.getInitialVariance());
  });

  it('number[] recursion: σ²ₜ = ω + α·r²ₜ₋₁ + γ·r²ₜ₋₁·I(rₜ₋₁<0) + β·σ²ₜ₋₁', () => {
    const prices = makePrices(100);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const { omega, alpha, gamma, beta } = result.params;
    const vs = model.getVarianceSeries(result.params);
    const returns = model.getReturns();

    for (let i = 1; i < returns.length; i++) {
      const innovation = returns[i - 1] ** 2;
      const indicator = returns[i - 1] < 0 ? 1 : 0;
      const expected = omega + alpha * innovation + gamma * innovation * indicator + beta * vs[i - 1];
      expect(vs[i]).toBeCloseTo(expected, 12);
    }
  });

  it('Candle[] recursion: σ²ₜ = ω + α·RV + γ·RV·I(r<0) + β·σ²ₜ₋₁', () => {
    const candles = makeCandles(200);
    const model = new GjrGarch(candles);
    const result = model.fit();
    const { omega, alpha, gamma, beta } = result.params;
    const vs = model.getVarianceSeries(result.params);
    const returns = model.getReturns();
    const rv = perCandleParkinson(candles, returns);

    for (let i = 1; i < returns.length; i++) {
      const innovation = rv[i - 1];
      const indicator = returns[i - 1] < 0 ? 1 : 0;
      const expected = omega + alpha * innovation + gamma * innovation * indicator + beta * vs[i - 1];
      expect(vs[i]).toBeCloseTo(expected, 12);
    }
  });

  it('indicator I(r<0) activates only on negative returns', () => {
    const prices = makePrices(100);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const { omega, alpha, gamma, beta } = result.params;
    const vs = model.getVarianceSeries(result.params);
    const returns = model.getReturns();

    for (let i = 1; i < returns.length; i++) {
      const r2 = returns[i - 1] ** 2;
      if (returns[i - 1] >= 0) {
        // No leverage term
        const expected = omega + alpha * r2 + beta * vs[i - 1];
        expect(vs[i]).toBeCloseTo(expected, 12);
      } else {
        // With leverage term
        const expected = omega + (alpha + gamma) * r2 + beta * vs[i - 1];
        expect(vs[i]).toBeCloseTo(expected, 12);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 4. Forecast formula verification
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH forecast formula', () => {
  it('one-step forecast uses actual last return and indicator', () => {
    const prices = makePrices(100);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const { omega, alpha, gamma, beta } = result.params;

    const vs = model.getVarianceSeries(result.params);
    const returns = model.getReturns();
    const lastVar = vs[vs.length - 1];
    const lastR2 = returns[returns.length - 1] ** 2;
    const indicator = returns[returns.length - 1] < 0 ? 1 : 0;

    const expected = omega + alpha * lastR2 + gamma * lastR2 * indicator + beta * lastVar;
    const fc = model.forecast(result.params, 1);

    expect(fc.variance[0]).toBeCloseTo(expected, 12);
  });

  it('multi-step forecast: v = ω + (α + γ/2 + β)·v (E[I(r<0)] = 0.5)', () => {
    const prices = makePrices(200);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const { omega, alpha, gamma, beta } = result.params;

    const fc = model.forecast(result.params, 10);

    for (let h = 1; h < 10; h++) {
      const expected = omega + (alpha + gamma / 2 + beta) * fc.variance[h - 1];
      expect(fc.variance[h]).toBeCloseTo(expected, 12);
    }
  });

  it('Candle[] one-step forecast uses Parkinson RV', () => {
    const candles = makeCandles(200);
    const model = new GjrGarch(candles);
    const result = model.fit();
    const { omega, alpha, gamma, beta } = result.params;

    const vs = model.getVarianceSeries(result.params);
    const returns = model.getReturns();
    const rv = perCandleParkinson(candles, returns);
    const lastVar = vs[vs.length - 1];
    const lastRV = rv[rv.length - 1];
    const indicator = returns[returns.length - 1] < 0 ? 1 : 0;

    const expected = omega + alpha * lastRV + gamma * lastRV * indicator + beta * lastVar;
    const fc = model.forecast(result.params, 1);

    expect(fc.variance[0]).toBeCloseTo(expected, 12);
  });

  it('forecast(params, 1) returns single-element arrays', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();
    const fc = model.forecast(result.params, 1);

    expect(fc.variance).toHaveLength(1);
    expect(fc.volatility).toHaveLength(1);
    expect(fc.annualized).toHaveLength(1);
  });

  it('long horizon → ω/(1−α−γ/2−β)', () => {
    const model = new GjrGarch(makePrices(200));
    const result = model.fit();
    const unconditional = result.params.unconditionalVariance;

    const fc = model.forecast(result.params, 500);
    const relErr = Math.abs(fc.variance[499] - unconditional) / unconditional;

    expect(relErr).toBeLessThan(0.001);
  });

  it('forecast is monotonic toward unconditional', () => {
    const model = new GjrGarch(makePrices(200));
    const result = model.fit();
    const unconditional = result.params.unconditionalVariance;

    const fc = model.forecast(result.params, 50);

    const above = fc.variance[0] >= unconditional;
    for (let h = 1; h < 50; h++) {
      if (above) {
        expect(fc.variance[h]).toBeLessThanOrEqual(fc.variance[h - 1] + 1e-20);
      } else {
        expect(fc.variance[h]).toBeGreaterThanOrEqual(fc.variance[h - 1] - 1e-20);
      }
    }
  });

  it('forecast(params, 0) returns one step', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();
    const fc0 = model.forecast(result.params, 0);
    const fc1 = model.forecast(result.params, 1);

    expect(fc0.variance).toHaveLength(1);
    expect(fc0.variance[0]).toBe(fc1.variance[0]);
  });
});

// ═══════════════════════════════════════════════════════════════
// 5. Constraint barriers
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH constraint barriers', () => {
  it('fit never returns omega <= 0', () => {
    for (const seed of [1, 42, 123, 999]) {
      const result = calibrateGjrGarch(makePrices(100, seed));
      expect(result.params.omega).toBeGreaterThan(0);
    }
  });

  it('fit never returns alpha < 0', () => {
    const result = calibrateGjrGarch(makePrices(200));
    expect(result.params.alpha).toBeGreaterThanOrEqual(0);
  });

  it('fit never returns gamma < 0', () => {
    const result = calibrateGjrGarch(makePrices(200));
    expect(result.params.gamma).toBeGreaterThanOrEqual(0);
  });

  it('fit never returns beta < 0', () => {
    const result = calibrateGjrGarch(makePrices(200));
    expect(result.params.beta).toBeGreaterThanOrEqual(0);
  });

  it('fit never returns alpha + gamma/2 + beta >= 1', () => {
    for (const seed of [1, 42, 123, 999, 7777]) {
      const result = calibrateGjrGarch(makePrices(100, seed));
      expect(result.params.persistence).toBeLessThan(1);
    }
  });

  it('getVarianceSeries always positive even with extreme returns', () => {
    const rng = lcg(42);
    const prices = [100];
    for (let i = 0; i < 200; i++) {
      prices.push(prices[i] * Math.exp((rng() - 0.5) * 0.2));
    }

    const model = new GjrGarch(prices);
    const result = model.fit();
    const vs = model.getVarianceSeries(result.params);

    for (const v of vs) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 6. Computed fields
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH computed fields', () => {
  it('persistence = alpha + gamma/2 + beta', () => {
    const result = calibrateGjrGarch(makePrices(200));
    const expected = result.params.alpha + result.params.gamma / 2 + result.params.beta;
    expect(result.params.persistence).toBeCloseTo(expected, 14);
  });

  it('unconditionalVariance = omega / (1 - persistence)', () => {
    const result = calibrateGjrGarch(makePrices(200));
    const expected = result.params.omega / (1 - result.params.persistence);
    expect(result.params.unconditionalVariance).toBeCloseTo(expected, 14);
  });

  it('annualizedVol = sqrt(unconditionalVariance * periodsPerYear) * 100', () => {
    const result = calibrateGjrGarch(makePrices(200));
    const expected = Math.sqrt(result.params.unconditionalVariance * 252) * 100;
    expect(result.params.annualizedVol).toBeCloseTo(expected, 10);
  });

  it('annualizedVol uses custom periodsPerYear', () => {
    const prices = makePrices(200);
    const r365 = calibrateGjrGarch(prices, { periodsPerYear: 365 });
    const expected = Math.sqrt(r365.params.unconditionalVariance * 365) * 100;
    expect(r365.params.annualizedVol).toBeCloseTo(expected, 10);
  });

  it('leverageEffect equals gamma', () => {
    const result = calibrateGjrGarch(makePrices(200));
    expect(result.params.leverageEffect).toBe(result.params.gamma);
  });
});

// ═══════════════════════════════════════════════════════════════
// 7. AIC / BIC
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH AIC/BIC', () => {
  it('numParams = 4: AIC = 2·4 − 2·LL', () => {
    const prices = makePrices(200);
    const result = calibrateGjrGarch(prices);
    const expected = 2 * 4 - 2 * result.diagnostics.logLikelihood;
    expect(result.diagnostics.aic).toBeCloseTo(expected, 8);
  });

  it('numParams = 4: BIC = 4·ln(n) − 2·LL', () => {
    const prices = makePrices(200);
    const model = new GjrGarch(prices);
    const n = model.getReturns().length;
    const result = model.fit();
    const expected = 4 * Math.log(n) - 2 * result.diagnostics.logLikelihood;
    expect(result.diagnostics.bic).toBeCloseTo(expected, 8);
  });

  it('BIC > AIC for large samples (n > e²)', () => {
    const result = calibrateGjrGarch(makePrices(500));
    expect(result.diagnostics.bic).toBeGreaterThan(result.diagnostics.aic);
  });
});

// ═══════════════════════════════════════════════════════════════
// 8. Estimation properties
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH estimation properties', () => {
  it('fitted params are local LL maximum: perturbations decrease LL', () => {
    const prices = makePrices(300);
    const model = new GjrGarch(prices);
    const result = model.fit();
    const returns = model.getReturns();
    const baseVariance = model.getVarianceSeries(result.params);
    const baseLL = computeLL(returns, baseVariance);

    const delta = 1e-4;
    const fields = ['omega', 'alpha', 'gamma', 'beta'] as const;

    for (const field of fields) {
      for (const sign of [1, -1]) {
        const perturbed = {
          ...result.params,
          [field]: result.params[field] + sign * delta,
        };
        if (perturbed.omega <= 0 || perturbed.alpha < 0 || perturbed.gamma < 0 || perturbed.beta < 0) continue;
        if (perturbed.alpha + perturbed.gamma / 2 + perturbed.beta >= 1) continue;

        const pVariance = model.getVarianceSeries(perturbed);
        const pLL = computeLL(returns, pVariance);

        expect(pLL).toBeLessThanOrEqual(baseLL + 1e-4);
      }
    }
  });

  it('double fit on same data gives identical params', () => {
    const prices = makePrices(200);
    const model = new GjrGarch(prices);
    const r1 = model.fit();
    const r2 = model.fit();

    expect(r1.params.omega).toBe(r2.params.omega);
    expect(r1.params.alpha).toBe(r2.params.alpha);
    expect(r1.params.gamma).toBe(r2.params.gamma);
    expect(r1.params.beta).toBe(r2.params.beta);
    expect(r1.diagnostics.logLikelihood).toBe(r2.diagnostics.logLikelihood);
  });
});

// ═══════════════════════════════════════════════════════════════
// 9. Numerical stability
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH numerical stability', () => {
  it('near-constant prices: fits without crashing', () => {
    const prices: number[] = [100];
    let state = 42;
    for (let i = 1; i <= 54; i++) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      prices.push(prices[i - 1] * (1 + ((state / 0x7fffffff) - 0.5) * 1e-6));
    }

    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.params.omega)).toBe(true);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });

  it('exactly constant prices: does not throw or hang', () => {
    const model = new GjrGarch(Array(55).fill(100));
    expect(() => model.fit()).not.toThrow();
  });

  it('extreme outlier: single 50% drop', () => {
    const prices = makePrices(200);
    prices[100] *= 0.5;

    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('very volatile data (±50% swings)', () => {
    const rng = lcg(42);
    const prices = [100];
    for (let i = 0; i < 200; i++) {
      prices.push(prices[i] * Math.exp((rng() - 0.5) * 1.0));
    }

    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.params.omega)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('very small prices (~0.0001)', () => {
    const rng = lcg(42);
    const prices = [0.0001];
    for (let i = 0; i < 100; i++) {
      prices.push(prices[i] * Math.exp((rng() - 0.5) * 0.04));
    }
    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.params.omega)).toBe(true);
  });

  it('very large prices (~1e8)', () => {
    const rng = lcg(42);
    const prices = [1e8];
    for (let i = 0; i < 100; i++) {
      prices.push(prices[i] * Math.exp((rng() - 0.5) * 0.04));
    }
    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.params.omega)).toBe(true);
  });

  it('trending data: monotonically rising prices', () => {
    const prices = [100];
    for (let i = 0; i < 100; i++) {
      prices.push(prices[i] * 1.003);
    }
    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.params.omega)).toBe(true);
  });

  it('alternating up/down pattern', () => {
    const prices = [100];
    for (let i = 0; i < 100; i++) {
      prices.push(prices[i] * (i % 2 === 0 ? 1.02 : 0.98));
    }
    const result = new GjrGarch(prices).fit();
    expect(Number.isFinite(result.params.omega)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('10K prices: fits without error', () => {
    const result = new GjrGarch(makePrices(10001)).fit();
    expect(result.diagnostics.converged).toBe(true);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════
// 10. Input validation
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH input validation', () => {
  it('50 prices (boundary): does not throw', () => {
    expect(() => new GjrGarch(makePrices(50))).not.toThrow();
  });

  it('49 prices: throws', () => {
    expect(() => new GjrGarch(makePrices(49))).toThrow('at least 50');
  });

  it('10 candles: throws', () => {
    expect(() => new GjrGarch(makeCandles(10))).toThrow('at least 50');
  });

  it('50 candles: does not throw', () => {
    expect(() => new GjrGarch(makeCandles(50))).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════
// 11. Edge cases
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH edge cases', () => {
  it('forecast with negative steps returns 1 element', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();
    const fc = model.forecast(result.params, -1);
    expect(fc.variance).toHaveLength(1);
  });

  it('forecast 10000 steps: all finite, converges', () => {
    const model = new GjrGarch(makePrices(200));
    const result = model.fit();
    const fc = model.forecast(result.params, 10000);

    expect(fc.variance).toHaveLength(10000);
    for (const v of fc.variance) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }

    const relErr = Math.abs(fc.variance[9999] - result.params.unconditionalVariance)
      / result.params.unconditionalVariance;
    expect(relErr).toBeLessThan(1e-6);
  });
});

// ═══════════════════════════════════════════════════════════════
// 12. Degenerate parameters
// ═══════════════════════════════════════════════════════════════

describe('Degenerate GJR-GARCH params', () => {
  it('gamma = 0: degrades to GARCH(1,1)', () => {
    const model = new GjrGarch(makePrices(100));
    const params = gjrParams(0.00001, 0.1, 0, 0.85);
    const vs = model.getVarianceSeries(params);
    const returns = model.getReturns();

    for (let i = 1; i < returns.length; i++) {
      const expected = 0.00001 + 0.1 * returns[i - 1] ** 2 + 0.85 * vs[i - 1];
      expect(vs[i]).toBeCloseTo(expected, 12);
    }
  });

  it('alpha = 0: only leverage term responds to shocks', () => {
    const model = new GjrGarch(makePrices(100));
    const params = gjrParams(0.00001, 0, 0.2, 0.85);
    const vs = model.getVarianceSeries(params);
    const returns = model.getReturns();

    for (let i = 1; i < returns.length; i++) {
      const indicator = returns[i - 1] < 0 ? 1 : 0;
      const expected = 0.00001 + 0.2 * returns[i - 1] ** 2 * indicator + 0.85 * vs[i - 1];
      expect(vs[i]).toBeCloseTo(expected, 12);
    }
  });

  it('beta = 0: no memory, variance depends only on previous shock', () => {
    const model = new GjrGarch(makePrices(100));
    const returns = model.getReturns();
    const params = gjrParams(0.0001, 0.3, 0.1, 0);
    const vs = model.getVarianceSeries(params);

    for (let i = 1; i < returns.length; i++) {
      const r2 = returns[i - 1] ** 2;
      const indicator = returns[i - 1] < 0 ? 1 : 0;
      const fromScratch = 0.0001 + 0.3 * r2 + 0.1 * r2 * indicator;
      expect(vs[i]).toBeCloseTo(fromScratch, 12);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 13. Realized GJR-GARCH path (Candle[] vs number[])
// ═══════════════════════════════════════════════════════════════

describe('Realized GJR-GARCH (Candle path)', () => {
  it('Candle[] uses yangZhangVariance for initial variance', () => {
    const candles = makeCandles(100);
    const model = new GjrGarch(candles);
    const expected = yangZhangVariance(candles);
    expect(model.getInitialVariance()).toBe(expected);
  });

  it('number[] uses sampleVariance for initial variance', () => {
    const prices = makePrices(100);
    const model = new GjrGarch(prices);
    const returns = calculateReturnsFromPrices(prices);
    const expected = sampleVariance(returns);
    expect(model.getInitialVariance()).toBe(expected);
  });

  it('Candle[] and number[] produce different params (Parkinson vs r²)', () => {
    const candles = makeCandles(200);
    const prices = candles.map(c => c.close);

    const rc = calibrateGjrGarch(candles);
    const rp = calibrateGjrGarch(prices);

    const diff = Math.abs(rc.params.alpha - rp.params.alpha)
      + Math.abs(rc.params.omega - rp.params.omega)
      + Math.abs(rc.params.gamma - rp.params.gamma);
    expect(diff).toBeGreaterThan(1e-6);
  });

  it('flat candles (H=L) degrade to same as number[]', () => {
    const candles = makeFlatCandles(200, 42);
    const prices = candles.map(c => c.close);

    const rc = calibrateGjrGarch(candles);
    const rp = calibrateGjrGarch(prices);

    expect(rc.params.alpha).toBeCloseTo(rp.params.alpha, 3);
    expect(rc.params.beta).toBeCloseTo(rp.params.beta, 3);
    expect(rc.params.gamma).toBeCloseTo(rp.params.gamma, 3);
  });

  it('variance series differs for Candle[] vs number[]', () => {
    const candles = makeCandles(200);
    const prices = candles.map(c => c.close);

    const modelC = new GjrGarch(candles);
    const modelP = new GjrGarch(prices);
    const fitC = modelC.fit();
    const fitP = modelP.fit();

    const vsC = modelC.getVarianceSeries(fitC.params);
    const vsP = modelP.getVarianceSeries(fitP.params);

    let diffCount = 0;
    const len = Math.min(vsC.length, vsP.length);
    for (let i = 1; i < len; i++) {
      if (Math.abs(vsC[i] - vsP[i]) / vsC[i] > 0.001) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(0);
  });

  it('multi-step forecast: step 1 uses Parkinson RV, steps 2+ use effective persistence', () => {
    const candles = makeCandles(200, 99);
    const model = new GjrGarch(candles);
    const fit = model.fit();
    const { omega, alpha, gamma, beta } = fit.params;

    const fc = model.forecast(fit.params, 10);

    for (let h = 1; h < 10; h++) {
      const expected = omega + (alpha + gamma / 2 + beta) * fc.variance[h - 1];
      expect(fc.variance[h]).toBeCloseTo(expected, 12);
    }
  });

  it('bad OHLC: NaN high does not crash', () => {
    const candles = makeCandles(100, 42);
    candles[50] = { ...candles[50], high: NaN };
    const model = new GjrGarch(candles);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
  });

  it('bad OHLC: high < low does not crash', () => {
    const candles = makeCandles(100, 42);
    const c = candles[50];
    candles[50] = { ...c, high: c.low * 0.99, low: c.high * 1.01 };
    const model = new GjrGarch(candles);
    const fit = model.fit();
    expect(fit.diagnostics.converged).toBe(true);
  });

  it('all-identical candles (O=H=L=C) do not crash', () => {
    const candles: Candle[] = [];
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
    }

    const model = new GjrGarch(candles);
    const fit = model.fit();
    expect(fit.params).toBeDefined();
    expect(fit.diagnostics).toBeDefined();
  });

  it('extremely wide candles do not produce Infinity variance', () => {
    const rng = lcg(42);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 100; i++) {
      const r = seededRandn(rng) * 0.01;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * 2;
      const low = Math.min(price, close) * 0.5;
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }

    const model = new GjrGarch(candles);
    const fit = model.fit();
    const vs = model.getVarianceSeries(fit.params);

    for (const v of vs) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 14. Options
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH options', () => {
  it('periodsPerYear scales annualized vol as √(periodsPerYear)', () => {
    const prices = makePrices(200);
    const r252 = new GjrGarch(prices, { periodsPerYear: 252 }).fit();
    const r365 = new GjrGarch(prices, { periodsPerYear: 365 }).fit();

    // Core params identical
    expect(r365.params.omega).toBe(r252.params.omega);
    expect(r365.params.alpha).toBe(r252.params.alpha);
    expect(r365.params.gamma).toBe(r252.params.gamma);
    expect(r365.params.beta).toBe(r252.params.beta);

    // Annualized vol ratio = √(365/252)
    expect(r365.params.annualizedVol / r252.params.annualizedVol)
      .toBeCloseTo(Math.sqrt(365 / 252), 10);
  });

  it('low maxIter prevents convergence', () => {
    const result = new GjrGarch(makePrices(200)).fit({ maxIter: 5 });
    expect(result.diagnostics.converged).toBe(false);
    expect(result.diagnostics.iterations).toBeLessThanOrEqual(5);
  });

  it('tighter tol requires more iterations', () => {
    const prices = makePrices(200);
    const loose = new GjrGarch(prices).fit({ tol: 1e-4 });
    const tight = new GjrGarch(prices).fit({ tol: 1e-12 });

    expect(tight.diagnostics.iterations).toBeGreaterThanOrEqual(
      loose.diagnostics.iterations,
    );
  });

  it('calibrateGjrGarch forwards periodsPerYear', () => {
    const prices = makePrices(200);
    const direct = new GjrGarch(prices, { periodsPerYear: 365 }).fit();
    const convenience = calibrateGjrGarch(prices, { periodsPerYear: 365 });

    expect(convenience.params.omega).toBe(direct.params.omega);
    expect(convenience.params.annualizedVol).toBe(direct.params.annualizedVol);
  });

  it('calibrateGjrGarch forwards maxIter', () => {
    const prices = makePrices(200);
    const result = calibrateGjrGarch(prices, { maxIter: 5 });
    expect(result.diagnostics.converged).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════
// 15. Immutability
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH immutability', () => {
  it('getReturns() returns a copy', () => {
    const model = new GjrGarch(makePrices(55));
    const a = model.getReturns();
    const b = model.getReturns();

    expect(a).toEqual(b);
    expect(a).not.toBe(b);

    a[0] = 999;
    expect(model.getReturns()[0]).not.toBe(999);
  });

  it('getVarianceSeries returns different reference each call', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();

    const a = model.getVarianceSeries(result.params);
    const b = model.getVarianceSeries(result.params);

    expect(a).toEqual(b);
    expect(a).not.toBe(b);
  });

  it('forecast returns new arrays each call', () => {
    const model = new GjrGarch(makePrices(55));
    const result = model.fit();

    const f1 = model.forecast(result.params, 3);
    const f2 = model.forecast(result.params, 3);

    expect(f1.variance).toEqual(f2.variance);
    expect(f1.variance).not.toBe(f2.variance);
  });

  it('getInitialVariance same before and after fit', () => {
    const model = new GjrGarch(makePrices(55));
    const before = model.getInitialVariance();
    model.fit();
    expect(model.getInitialVariance()).toBe(before);
  });
});

// ═══════════════════════════════════════════════════════════════
// 16. Instance isolation
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH instance isolation', () => {
  it('two instances do not share state', () => {
    const model1 = new GjrGarch(makePrices(100, 111));
    const model2 = new GjrGarch(makePrices(100, 222));

    const r1 = model1.fit();
    const r2 = model2.fit();

    // Different data → different params
    expect(r1.params.omega).not.toBe(r2.params.omega);

    // First model unchanged
    const r1again = model1.fit();
    expect(r1again.params.omega).toBe(r1.params.omega);
  });
});

// ═══════════════════════════════════════════════════════════════
// 17. Cross-model consistency
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH cross-model consistency', () => {
  it('GJR-GARCH and GARCH: similar unconditional variance on same data', () => {
    const prices = makePrices(500);
    const gjrUV = calibrateGjrGarch(prices).params.unconditionalVariance;
    const gUV = calibrateGarch(prices).params.unconditionalVariance;

    const ratio = gjrUV / gUV;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });

  it('forecast annualized vol converges to params.annualizedVol', () => {
    const model = new GjrGarch(makePrices(200));
    const result = model.fit();
    const fc = model.forecast(result.params, 500);

    const relErr = Math.abs(fc.annualized[499] - result.params.annualizedVol)
      / result.params.annualizedVol;

    expect(relErr).toBeLessThan(0.01);
  });
});

// ═══════════════════════════════════════════════════════════════
// 18. Regression snapshot
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH regression snapshot', () => {
  it('params on makePrices(100) are deterministic', () => {
    const prices = makePrices(100);
    const r1 = calibrateGjrGarch(prices);
    const r2 = calibrateGjrGarch(prices);

    expect(r1.params.omega).toBe(r2.params.omega);
    expect(r1.params.alpha).toBe(r2.params.alpha);
    expect(r1.params.gamma).toBe(r2.params.gamma);
    expect(r1.params.beta).toBe(r2.params.beta);
    expect(r1.diagnostics.logLikelihood).toBe(r2.diagnostics.logLikelihood);
  });
});

// ═══════════════════════════════════════════════════════════════
// 19. Property-based (fuzz)
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH property-based invariants', () => {
  const seeds = [1, 42, 123, 999, 7777];

  it('persistence < 1, ω > 0, LL finite for diverse seeds', () => {
    for (const seed of seeds) {
      const result = calibrateGjrGarch(makePrices(100, seed));

      expect(result.params.persistence).toBeLessThan(1);
      expect(result.params.omega).toBeGreaterThan(0);
      expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    }
  });

  it('all conditional variances > 0 for diverse seeds', () => {
    for (const seed of seeds) {
      const prices = makePrices(100, seed);
      const model = new GjrGarch(prices);
      const result = model.fit();
      const variance = model.getVarianceSeries(result.params);

      for (const v of variance) {
        expect(v).toBeGreaterThan(0);
      }
    }
  });

  it('forecast converges for diverse seeds', () => {
    for (const seed of seeds) {
      const prices = makePrices(100, seed);
      const model = new GjrGarch(prices);
      const result = model.fit();
      const fc = model.forecast(result.params, 100);
      const unconditional = result.params.unconditionalVariance;

      const relErr = Math.abs(fc.variance[99] - unconditional) / unconditional;
      expect(relErr).toBeLessThan(0.1);
    }
  });
});

// ═══════════════════════════════════════════════════════════════
// 20. Scale invariance
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH scale invariance', () => {
  it('1000× prices → identical params (log returns are scale-invariant)', () => {
    const prices1 = makePrices(200);
    const prices2 = prices1.map(p => p * 1000);

    const r1 = calibrateGjrGarch(prices1);
    const r2 = calibrateGjrGarch(prices2);

    expect(r1.params.omega).toBeCloseTo(r2.params.omega, 10);
    expect(r1.params.alpha).toBeCloseTo(r2.params.alpha, 10);
    expect(r1.params.gamma).toBeCloseTo(r2.params.gamma, 10);
    expect(r1.params.beta).toBeCloseTo(r2.params.beta, 10);
  });

  it('Candle[] 1000× → same annualized vol', () => {
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
});

// ═══════════════════════════════════════════════════════════════
// 21. Integration — predict pipeline
// ═══════════════════════════════════════════════════════════════

describe('GJR-GARCH integration with predict pipeline', () => {
  it('predict() can select gjr-garch model type', () => {
    // Try several seeds — gjr-garch should be selectable
    const validTypes = ['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas'];
    for (let seed = 1; seed <= 20; seed++) {
      const candles = makeCandles(200, seed);
      const result = predict(candles, '4h');
      expect(validTypes).toContain(result.modelType);
    }
  });

  it('predictRange works when gjr-garch is a candidate', () => {
    const candles = makeCandles(500, 42);
    const result = predictRange(candles, '15m', 5);

    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
    expect(result.lowerPrice).toBeLessThan(result.currentPrice);
  });

  it('backtest completes when gjr-garch is a candidate', () => {
    const candles = makeCandles(500, 42);
    const result = backtest(candles, '15m');
    expect(typeof result).toBe('boolean');
  });

  it('forecast annualized scaling: annualized = sqrt(variance * periodsPerYear) * 100', () => {
    const candles = makeCandles(200);
    const model = new GjrGarch(candles, { periodsPerYear: 252 });
    const fit = model.fit();
    const fc = model.forecast(fit.params, 5);

    for (let i = 0; i < fc.variance.length; i++) {
      expect(fc.annualized[i]).toBeCloseTo(Math.sqrt(fc.variance[i] * 252) * 100, 10);
      expect(fc.volatility[i]).toBeCloseTo(Math.sqrt(fc.variance[i]), 10);
    }
  });
});
