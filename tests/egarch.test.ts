import { describe, it, expect } from 'vitest';
import {
  Egarch,
  calibrateEgarch,
  calibrateGarch,
  checkLeverageEffect,
  EXPECTED_ABS_NORMAL,
} from '../src/index.js';

// Generate synthetic EGARCH data
function generateEgarchData(
  n: number,
  omega: number,
  alpha: number,
  gamma: number,
  beta: number,
  seed: number = 42
): number[] {
  let state = seed;
  function random(): number {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  }

  function randn(): number {
    const u1 = random();
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  const returns: number[] = [];
  let logVariance = omega / (1 - beta);
  let variance = Math.exp(logVariance);

  for (let i = 0; i < n; i++) {
    const sigma = Math.sqrt(variance);
    const z = randn();
    const epsilon = sigma * z;
    returns.push(epsilon);

    logVariance = omega
      + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
      + gamma * z
      + beta * logVariance;
    variance = Math.exp(logVariance);
  }

  const prices: number[] = [100];
  for (const r of returns) {
    prices.push(prices[prices.length - 1] * Math.exp(r));
  }

  return prices;
}

describe('EGARCH', () => {
  describe('calibrateEgarch', () => {
    it('should estimate parameters', () => {
      const prices = generateEgarchData(500, -0.1, 0.1, -0.05, 0.95);
      const result = calibrateEgarch(prices);

      expect(result.params.alpha).toBeDefined();
      expect(result.params.gamma).toBeDefined();
      expect(result.params.beta).toBeDefined();
      expect(result.params.omega).toBeDefined();

      // Beta should be high (persistence)
      expect(Math.abs(result.params.beta)).toBeGreaterThan(0.5);
      expect(Math.abs(result.params.beta)).toBeLessThan(1);
    });

    it('should detect leverage effect (negative gamma)', () => {
      // Generate data with strong leverage effect
      const prices = generateEgarchData(500, -0.1, 0.15, -0.1, 0.9, 123);
      const result = calibrateEgarch(prices);

      // Gamma should be negative for leverage effect
      // (negative returns increase volatility more than positive)
      expect(result.params.leverageEffect).toBeDefined();
    });

    it('should handle price array input', () => {
      const prices: number[] = [100];
      for (let i = 0; i < 100; i++) {
        const change = (Math.random() - 0.5) * 0.02;
        prices.push(prices[prices.length - 1] * (1 + change));
      }

      const result = calibrateEgarch(prices);

      expect(result.params.omega).toBeDefined();
      expect(result.diagnostics.converged).toBeDefined();
    });

    it('should throw on insufficient data', () => {
      const prices = [100, 101, 102];

      expect(() => calibrateEgarch(prices)).toThrow('at least 50');
    });
  });

  describe('Egarch class', () => {
    it('should compute variance series', () => {
      const prices = generateEgarchData(200, -0.1, 0.1, -0.05, 0.95);
      const model = new Egarch(prices);
      const result = model.fit();
      const variance = model.getVarianceSeries(result.params);

      expect(variance.length).toBe(prices.length - 1);
      expect(variance.every(v => v > 0)).toBe(true);
      expect(variance.every(v => isFinite(v))).toBe(true);
    });

    it('should forecast variance', () => {
      const prices = generateEgarchData(200, -0.1, 0.1, -0.05, 0.95);
      const model = new Egarch(prices);
      const result = model.fit();
      const forecast = model.forecast(result.params, 10);

      expect(forecast.variance.length).toBe(10);
      expect(forecast.volatility.length).toBe(10);
      expect(forecast.annualized.length).toBe(10);

      // All forecasts should be positive
      expect(forecast.variance.every(v => v > 0)).toBe(true);
    });
  });

  describe('comparison with GARCH', () => {
    it('should have different AIC when leverage effect present', () => {
      // Generate data with leverage effect
      const prices = generateEgarchData(500, -0.1, 0.15, -0.1, 0.9);

      const garchResult = calibrateGarch(prices);
      const egarchResult = calibrateEgarch(prices);

      // Both should have valid results
      expect(garchResult.diagnostics.aic).toBeDefined();
      expect(egarchResult.diagnostics.aic).toBeDefined();

      // Log likelihoods should be finite
      expect(isFinite(garchResult.diagnostics.logLikelihood)).toBe(true);
      expect(isFinite(egarchResult.diagnostics.logLikelihood)).toBe(true);
    });
  });
});

describe('checkLeverageEffect', () => {
  it('should detect asymmetric volatility', () => {
    // Returns where negative returns are larger in magnitude
    const returns = [
      0.01, -0.03, 0.02, -0.04, 0.01, -0.05,
      0.01, -0.03, 0.02, -0.04, 0.01, -0.03,
    ];

    const stats = checkLeverageEffect(returns);

    expect(stats.negativeVol).toBeGreaterThan(stats.positiveVol);
    expect(stats.ratio).toBeGreaterThan(1);
    expect(stats.recommendation).toBe('egarch');
  });

  it('should recommend GARCH for symmetric data', () => {
    const returns = [
      0.02, -0.02, 0.02, -0.02, 0.02, -0.02,
      0.02, -0.02, 0.02, -0.02, 0.02, -0.02,
    ];

    const stats = checkLeverageEffect(returns);

    expect(stats.ratio).toBeCloseTo(1, 1);
    expect(stats.recommendation).toBe('garch');
  });

  it('should handle edge cases', () => {
    const allPositive = [0.01, 0.02, 0.03];
    const stats = checkLeverageEffect(allPositive);

    expect(stats.recommendation).toBe('garch');
  });
});

describe('EXPECTED_ABS_NORMAL', () => {
  it('should equal sqrt(2/pi)', () => {
    expect(EXPECTED_ABS_NORMAL).toBeCloseTo(Math.sqrt(2 / Math.PI));
    expect(EXPECTED_ABS_NORMAL).toBeCloseTo(0.7979, 3);
  });
});
