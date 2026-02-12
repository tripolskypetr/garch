import { describe, it, expect } from 'vitest';
import {
  Garch,
  calibrateGarch,
  calculateReturnsFromPrices,
  sampleVariance,
} from '../src/index.js';

// Generate synthetic GARCH(1,1) data for testing
function generateGarchData(
  n: number,
  omega: number,
  alpha: number,
  beta: number,
  seed: number = 42
): number[] {
  // Simple seeded random (not cryptographic, just for reproducibility)
  let state = seed;
  function random(): number {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  }

  // Box-Muller for normal distribution
  function randn(): number {
    const u1 = random();
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  const returns: number[] = [];
  let variance = omega / (1 - alpha - beta); // Start at unconditional variance

  for (let i = 0; i < n; i++) {
    const epsilon = Math.sqrt(variance) * randn();
    returns.push(epsilon);
    variance = omega + alpha * epsilon ** 2 + beta * variance;
  }

  // Convert returns to prices
  const prices: number[] = [100];
  for (const r of returns) {
    prices.push(prices[prices.length - 1] * Math.exp(r));
  }

  return prices;
}

describe('GARCH', () => {
  describe('calibrateGarch', () => {
    it('should estimate parameters close to true values', () => {
      const trueOmega = 0.00001;
      const trueAlpha = 0.1;
      const trueBeta = 0.85;

      const prices = generateGarchData(1000, trueOmega, trueAlpha, trueBeta);
      const result = calibrateGarch(prices);

      // Parameters should be in reasonable range
      expect(result.params.alpha).toBeGreaterThan(0.01);
      expect(result.params.alpha).toBeLessThan(0.3);
      expect(result.params.beta).toBeGreaterThan(0.6);
      expect(result.params.beta).toBeLessThan(0.99);
      expect(result.params.persistence).toBeLessThan(1);

      // Should converge
      expect(result.diagnostics.converged).toBe(true);
    });

    it('should handle price array input', () => {
      const prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105];
      // Extend with more data
      for (let i = 0; i < 100; i++) {
        prices.push(100 + Math.sin(i / 10) * 5 + (Math.random() - 0.5) * 2);
      }

      const result = calibrateGarch(prices);

      expect(result.params.omega).toBeGreaterThan(0);
      expect(result.params.alpha).toBeGreaterThanOrEqual(0);
      expect(result.params.beta).toBeGreaterThanOrEqual(0);
    });

    it('should handle candle input', () => {
      const candles = Array.from({ length: 100 }, (_, i) => ({
        open: 100 + i * 0.1,
        high: 101 + i * 0.1,
        low: 99 + i * 0.1,
        close: 100 + i * 0.1 + (Math.random() - 0.5),
        volume: 1000,
      }));

      const result = calibrateGarch(candles);

      expect(result.params.omega).toBeGreaterThan(0);
      expect(result.params.persistence).toBeLessThan(1);
    });

    it('should throw on insufficient data', () => {
      const prices = [100, 101, 102];

      expect(() => calibrateGarch(prices)).toThrow('at least 50');
    });
  });

  describe('Garch class', () => {
    it('should compute variance series', () => {
      const prices = generateGarchData(200, 0.00001, 0.1, 0.85);
      const model = new Garch(prices);
      const result = model.fit();
      const variance = model.getVarianceSeries(result.params);

      expect(variance.length).toBe(prices.length - 1);
      expect(variance.every(v => v > 0)).toBe(true);
    });

    it('should forecast variance', () => {
      const prices = generateGarchData(200, 0.00001, 0.1, 0.85);
      const model = new Garch(prices);
      const result = model.fit();
      const forecast = model.forecast(result.params, 10);

      expect(forecast.variance.length).toBe(10);
      expect(forecast.volatility.length).toBe(10);
      expect(forecast.annualized.length).toBe(10);

      // Forecasts should converge toward unconditional variance
      const lastForecast = forecast.variance[9];
      const unconditional = result.params.unconditionalVariance;
      expect(Math.abs(lastForecast - unconditional) / unconditional).toBeLessThan(0.5);
    });

    it('should return correct returns', () => {
      const prices = [100, 110, 105, 115, 108];
      for (let i = 0; i < 50; i++) {
        prices.push(100 + Math.random() * 20);
      }

      const model = new Garch(prices);
      const returns = model.getReturns();

      expect(returns.length).toBe(prices.length - 1);
      expect(returns[0]).toBeCloseTo(Math.log(110 / 100));
    });
  });

  describe('diagnostics', () => {
    it('should compute AIC and BIC', () => {
      const prices = generateGarchData(500, 0.00001, 0.1, 0.85);
      const result = calibrateGarch(prices);

      expect(result.diagnostics.aic).toBeDefined();
      expect(result.diagnostics.bic).toBeDefined();
      expect(result.diagnostics.logLikelihood).toBeDefined();

      // BIC should penalize more than AIC for large samples
      expect(result.diagnostics.bic).toBeGreaterThan(result.diagnostics.aic);
    });
  });
});

describe('utils', () => {
  describe('calculateReturnsFromPrices', () => {
    it('should compute log returns', () => {
      const prices = [100, 110, 99];
      const returns = calculateReturnsFromPrices(prices);

      expect(returns.length).toBe(2);
      expect(returns[0]).toBeCloseTo(Math.log(1.1));
      expect(returns[1]).toBeCloseTo(Math.log(99 / 110));
    });
  });

  describe('sampleVariance', () => {
    it('should compute variance with zero mean', () => {
      const returns = [0.01, -0.01, 0.02, -0.02];
      const variance = sampleVariance(returns);

      const expected = (0.01 ** 2 + 0.01 ** 2 + 0.02 ** 2 + 0.02 ** 2) / 4;
      expect(variance).toBeCloseTo(expected);
    });
  });
});
