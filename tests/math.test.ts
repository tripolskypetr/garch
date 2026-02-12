import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  calibrateGarch,
  calibrateEgarch,
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  sampleVarianceWithMean,
  EXPECTED_ABS_NORMAL,
} from '../src/index.js';
import { calculateAIC, calculateBIC } from '../src/utils.js';

// ── helpers ──────────────────────────────────────────────────

/** Deterministic prices via LCG */
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

function generateGarchData(
  n: number, omega: number, alpha: number, beta: number, seed = 42,
): number[] {
  let state = seed;
  const random = () => { state = (state * 1103515245 + 12345) & 0x7fffffff; return state / 0x7fffffff; };
  const randn = () => { const u1 = random(), u2 = random(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); };
  const returns: number[] = [];
  let variance = omega / (1 - alpha - beta);
  for (let i = 0; i < n; i++) {
    const eps = Math.sqrt(variance) * randn();
    returns.push(eps);
    variance = omega + alpha * eps ** 2 + beta * variance;
  }
  const prices = [100];
  for (const r of returns) prices.push(prices[prices.length - 1] * Math.exp(r));
  return prices;
}

function generateEgarchData(
  n: number, omega: number, alpha: number, gamma: number, beta: number, seed = 42,
): number[] {
  let state = seed;
  const random = () => { state = (state * 1103515245 + 12345) & 0x7fffffff; return state / 0x7fffffff; };
  const randn = () => { const u1 = random(), u2 = random(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); };
  const returns: number[] = [];
  let logVar = omega / (1 - beta);
  let variance = Math.exp(logVar);
  for (let i = 0; i < n; i++) {
    const z = randn();
    returns.push(Math.sqrt(variance) * z);
    logVar = omega + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL) + gamma * z + beta * logVar;
    variance = Math.exp(logVar);
  }
  const prices = [100];
  for (const r of returns) prices.push(prices[prices.length - 1] * Math.exp(r));
  return prices;
}

function garchParams(omega: number, alpha: number, beta: number) {
  const persistence = alpha + beta;
  return {
    omega, alpha, beta, persistence,
    unconditionalVariance: omega / (1 - persistence),
    annualizedVol: Math.sqrt((omega / (1 - persistence)) * 252) * 100,
  };
}

function egarchParams(omega: number, alpha: number, gamma: number, beta: number) {
  return {
    omega, alpha, gamma, beta,
    persistence: beta,
    unconditionalVariance: Math.exp(omega / (1 - beta)),
    annualizedVol: Math.sqrt(Math.exp(omega / (1 - beta)) * 252) * 100,
    leverageEffect: gamma,
  };
}

// ── 1. GARCH variance recursion ─────────────────────────────

describe('GARCH variance recursion', () => {
  it('σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁ step-by-step', () => {
    const prices = makePrices(55);
    const model = new Garch(prices);
    const returns = model.getReturns();
    const initVar = model.getInitialVariance();
    const params = garchParams(0.00001, 0.1, 0.85);

    const series = model.getVarianceSeries(params);

    expect(series[0]).toBe(initVar);
    for (let i = 1; i < returns.length; i++) {
      const expected = params.omega
        + params.alpha * returns[i - 1] ** 2
        + params.beta * series[i - 1];
      expect(series[i]).toBeCloseTo(expected, 10);
    }
  });

  it('all variances are strictly positive', () => {
    const prices = makePrices(200);
    const model = new Garch(prices);
    const result = model.fit();
    const series = model.getVarianceSeries(result.params);
    for (const v of series) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── 2. GARCH log-likelihood ─────────────────────────────────

describe('GARCH log-likelihood', () => {
  it('logLikelihood = −0.5·Σ[ln(σ²ₜ) + εₜ²/σ²ₜ]', () => {
    const prices = makePrices(200);
    const model = new Garch(prices);
    const result = model.fit();
    const returns = model.getReturns();
    const variance = model.getVarianceSeries(result.params);

    let sum = 0;
    for (let i = 0; i < returns.length; i++) {
      sum += Math.log(variance[i]) + returns[i] ** 2 / variance[i];
    }
    const expectedLL = -0.5 * sum;

    expect(result.diagnostics.logLikelihood).toBeCloseTo(expectedLL, 4);
  });
});

// ── 3. GARCH forecast ───────────────────────────────────────

describe('GARCH forecast formulas', () => {
  const prices = makePrices(200);
  const model = new Garch(prices);
  const result = model.fit();
  const { omega, alpha, beta } = result.params;
  const varianceSeries = model.getVarianceSeries(result.params);
  const returns = model.getReturns();
  const lastVar = varianceSeries[varianceSeries.length - 1];
  const lastRet = returns[returns.length - 1];
  const fc = model.forecast(result.params, 50);

  it('one-step: σ²ₜ₊₁ = ω + α·εₜ² + β·σ²ₜ', () => {
    const expected = omega + alpha * lastRet ** 2 + beta * lastVar;
    expect(fc.variance[0]).toBeCloseTo(expected, 10);
  });

  it('multi-step: σ²ₜ₊ₕ = ω + (α+β)·σ²ₜ₊ₕ₋₁', () => {
    let v = fc.variance[0];
    for (let h = 1; h < 50; h++) {
      v = omega + (alpha + beta) * v;
      expect(fc.variance[h]).toBeCloseTo(v, 10);
    }
  });

  it('volatility = √σ², annualized = √(σ²·252)·100', () => {
    for (let h = 0; h < fc.variance.length; h++) {
      expect(fc.volatility[h]).toBeCloseTo(Math.sqrt(fc.variance[h]), 10);
      expect(fc.annualized[h]).toBeCloseTo(
        Math.sqrt(fc.variance[h] * 252) * 100, 8,
      );
    }
  });

  it('converges to unconditional variance ω/(1−α−β)', () => {
    const unconditional = omega / (1 - alpha - beta);
    const last = fc.variance[49];
    const relErr = Math.abs(last - unconditional) / unconditional;
    expect(relErr).toBeLessThan(0.1);
  });
});

// ── 4. EGARCH log-variance recursion ────────────────────────

describe('EGARCH log-variance recursion', () => {
  it('ln(σ²ₜ) = ω + α·(|zₜ₋₁|−E[|z|]) + γ·zₜ₋₁ + β·ln(σ²ₜ₋₁)', () => {
    const prices = makePrices(55);
    const model = new Egarch(prices);
    const returns = model.getReturns();
    const initVar = model.getInitialVariance();
    const params = egarchParams(-0.1, 0.1, -0.05, 0.95);

    const series = model.getVarianceSeries(params);

    expect(series[0]).toBe(initVar);

    let logVar = Math.log(initVar);
    for (let i = 1; i < returns.length; i++) {
      const sigma = Math.sqrt(series[i - 1]);
      const z = returns[i - 1] / sigma;
      logVar = params.omega
        + params.alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
        + params.gamma * z
        + params.beta * logVar;
      logVar = Math.max(-50, Math.min(50, logVar));

      expect(series[i]).toBeCloseTo(Math.exp(logVar), 10);
    }
  });

  it('all variances are positive and finite', () => {
    const prices = makePrices(200);
    const model = new Egarch(prices);
    const result = model.fit();
    const series = model.getVarianceSeries(result.params);
    for (const v of series) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── 5. EGARCH log-likelihood ────────────────────────────────

describe('EGARCH log-likelihood', () => {
  it('logLikelihood = −0.5·Σ[ln(σ²ₜ) + εₜ²/σ²ₜ]', () => {
    const prices = makePrices(200);
    const model = new Egarch(prices);
    const result = model.fit();
    const returns = model.getReturns();
    const variance = model.getVarianceSeries(result.params);

    let sum = 0;
    for (let i = 0; i < returns.length; i++) {
      sum += Math.log(variance[i]) + returns[i] ** 2 / variance[i];
    }
    const expectedLL = -0.5 * sum;

    expect(result.diagnostics.logLikelihood).toBeCloseTo(expectedLL, 4);
  });
});

// ── 6. EGARCH forecast ──────────────────────────────────────

describe('EGARCH forecast formulas', () => {
  const prices = makePrices(200);
  const model = new Egarch(prices);
  const result = model.fit();
  const { omega, alpha, gamma, beta } = result.params;
  const varianceSeries = model.getVarianceSeries(result.params);
  const returns = model.getReturns();
  const lastVar = varianceSeries[varianceSeries.length - 1];
  const lastRet = returns[returns.length - 1];
  const fc = model.forecast(result.params, 50);

  it('one-step: uses actual last standardized residual', () => {
    const sigma = Math.sqrt(lastVar);
    const z = lastRet / sigma;
    const logVar = omega
      + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
      + gamma * z
      + beta * Math.log(lastVar);

    expect(fc.variance[0]).toBeCloseTo(Math.exp(logVar), 10);
  });

  it('multi-step: ln(σ²ₕ) = ω + β·ln(σ²ₕ₋₁)  (E[z]=0, E[|z|]=√(2/π) cancel)', () => {
    let logVar = Math.log(fc.variance[0]);
    for (let h = 1; h < 50; h++) {
      logVar = omega + beta * logVar;
      expect(fc.variance[h]).toBeCloseTo(Math.exp(logVar), 10);
    }
  });

  it('volatility = √σ², annualized = √(σ²·252)·100', () => {
    for (let h = 0; h < fc.variance.length; h++) {
      expect(fc.volatility[h]).toBeCloseTo(Math.sqrt(fc.variance[h]), 10);
      expect(fc.annualized[h]).toBeCloseTo(
        Math.sqrt(fc.variance[h] * 252) * 100, 8,
      );
    }
  });
});

// ── 7. EGARCH leverage detection ────────────────────────────

describe('EGARCH leverage detection', () => {
  it('γ < 0 when data generated with negative γ', () => {
    const prices = generateEgarchData(1000, -0.1, 0.15, -0.1, 0.9, 123);
    const result = calibrateEgarch(prices);

    expect(result.params.gamma).toBeLessThan(0);
    expect(result.params.leverageEffect).toBeLessThan(0);
  });
});

// ── 8. GARCH parameter recovery (tighter bounds) ────────────

describe('GARCH parameter recovery', () => {
  it('α and β within 50 % relative error (n = 2000)', () => {
    const trueAlpha = 0.1;
    const trueBeta = 0.85;
    const prices = generateGarchData(2000, 0.00001, trueAlpha, trueBeta);
    const result = calibrateGarch(prices);

    expect(Math.abs(result.params.alpha - trueAlpha) / trueAlpha).toBeLessThan(0.5);
    expect(Math.abs(result.params.beta - trueBeta) / trueBeta).toBeLessThan(0.5);
    expect(result.params.persistence).toBeLessThan(1);
    expect(result.diagnostics.converged).toBe(true);
  });
});

// ── 9. sampleVarianceWithMean ───────────────────────────────

describe('sampleVarianceWithMean', () => {
  it('Σ(rᵢ − μ)² / (n−1) hand-calculated', () => {
    const returns = [0.01, -0.01, 0.02, -0.02, 0.03];
    const mean = (0.01 + (-0.01) + 0.02 + (-0.02) + 0.03) / 5;
    const sumSq =
      (0.01 - mean) ** 2 +
      (-0.01 - mean) ** 2 +
      (0.02 - mean) ** 2 +
      (-0.02 - mean) ** 2 +
      (0.03 - mean) ** 2;
    const expected = sumSq / 4; // Bessel's correction: n-1

    expect(sampleVarianceWithMean(returns)).toBeCloseTo(expected, 14);
  });

  it('differs from sampleVariance when mean ≠ 0', () => {
    const returns = [0.05, 0.06, 0.04, 0.07, 0.03];

    expect(sampleVariance(returns)).toBeGreaterThan(sampleVarianceWithMean(returns));
  });
});

// ── 10. calculateReturns from candles ───────────────────────

describe('calculateReturns (candles)', () => {
  it('computes ln(closeₜ / closeₜ₋₁)', () => {
    const candles = [
      { open: 99, high: 102, low: 98, close: 100, volume: 1000 },
      { open: 100, high: 112, low: 99, close: 110, volume: 1200 },
      { open: 109, high: 111, low: 96, close: 99, volume: 800 },
    ];
    const returns = calculateReturns(candles);

    expect(returns).toHaveLength(2);
    expect(returns[0]).toBeCloseTo(Math.log(110 / 100), 14);
    expect(returns[1]).toBeCloseTo(Math.log(99 / 110), 14);
  });

  it('throws on non-positive close', () => {
    const candles = [
      { open: 99, high: 102, low: 98, close: 100, volume: 1000 },
      { open: 100, high: 101, low: 0, close: 0, volume: 500 },
    ];

    expect(() => calculateReturns(candles)).toThrow();
  });
});

// ── 11. AIC / BIC formulas ──────────────────────────────────

describe('AIC and BIC', () => {
  it('AIC = 2k − 2·LL', () => {
    expect(calculateAIC(-100, 3)).toBeCloseTo(2 * 3 - 2 * (-100), 10);
    expect(calculateAIC(50, 2)).toBeCloseTo(2 * 2 - 2 * 50, 10);
    expect(calculateAIC(0, 1)).toBeCloseTo(2, 10);
  });

  it('BIC = k·ln(n) − 2·LL', () => {
    expect(calculateBIC(-100, 3, 100)).toBeCloseTo(
      3 * Math.log(100) - 2 * (-100), 10,
    );
    expect(calculateBIC(50, 2, 500)).toBeCloseTo(
      2 * Math.log(500) - 2 * 50, 10,
    );
  });

  it('BIC − AIC = k·(ln(n) − 2)  for same LL', () => {
    const ll = -100, k = 3, n = 200;
    const diff = calculateBIC(ll, k, n) - calculateAIC(ll, k);

    expect(diff).toBeCloseTo(k * (Math.log(n) - 2), 10);
  });

  it('model diagnostics: AIC and BIC consistent with logLikelihood', () => {
    const prices = makePrices(200);
    const result = calibrateGarch(prices);
    const { logLikelihood, aic, bic } = result.diagnostics;
    const n = prices.length - 1; // number of returns
    const k = 3; // omega, alpha, beta

    expect(aic).toBeCloseTo(2 * k - 2 * logLikelihood, 6);
    expect(bic).toBeCloseTo(k * Math.log(n) - 2 * logLikelihood, 6);
  });
});
