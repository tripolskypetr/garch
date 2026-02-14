import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  GjrGarch,
  calibrateGarch,
  calibrateEgarch,
  calibrateGjrGarch,
  sampleVariance,
  EXPECTED_ABS_NORMAL,
  logGamma,
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

function seededRandn(stateFn: () => number) {
  const u1 = stateFn(), u2 = stateFn();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function lcg(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function generateGarchData(
  n: number, omega: number, alpha: number, beta: number, seed = 42,
): number[] {
  const rng = lcg(seed);
  let v = omega / (1 - alpha - beta);
  const prices = [100];
  for (let i = 0; i < n; i++) {
    const eps = Math.sqrt(v) * seededRandn(rng);
    prices.push(prices[prices.length - 1] * Math.exp(eps));
    v = omega + alpha * eps ** 2 + beta * v;
  }
  return prices;
}

function generateEgarchData(
  n: number, omega: number, alpha: number, gamma: number, beta: number, seed = 42,
): number[] {
  const rng = lcg(seed);
  let logVar = omega / (1 - beta);
  let v = Math.exp(logVar);
  const prices = [100];
  for (let i = 0; i < n; i++) {
    const z = seededRandn(rng);
    prices.push(prices[prices.length - 1] * Math.exp(Math.sqrt(v) * z));
    logVar = omega + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL) + gamma * z + beta * logVar;
    v = Math.exp(logVar);
  }
  return prices;
}

function computeNegLL(returns: number[], variance: number[], df: number): number {
  const n = returns.length;
  const halfDfPlus1 = (df + 1) / 2;
  const dfMinus2 = df - 2;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += 0.5 * Math.log(variance[i]) + halfDfPlus1 * Math.log(1 + returns[i] ** 2 / (dfMinus2 * variance[i]));
  }
  const constant = n * (logGamma(df / 2) - logGamma(halfDfPlus1) + 0.5 * Math.log(Math.PI * dfMinus2));
  return sum + constant;
}

// ── Estimation properties ───────────────────────────────────

describe('Estimation properties', () => {
  it('fitted params are local LL maximum: perturbations decrease LL', () => {
    const prices = makePrices(300);
    const model = new Garch(prices);
    const result = model.fit();
    const returns = model.getReturns();
    const df = result.params.df;
    const baseVariance = model.getVarianceSeries(result.params);
    const baseNegLL = computeNegLL(returns, baseVariance, df);

    const delta = 1e-4;
    const fields = ['omega', 'alpha', 'beta'] as const;

    for (const field of fields) {
      for (const sign of [1, -1]) {
        const perturbed = {
          ...result.params,
          [field]: result.params[field] + sign * delta,
        };
        if (perturbed.omega <= 0 || perturbed.alpha < 0 || perturbed.beta < 0) continue;
        if (perturbed.alpha + perturbed.beta >= 1) continue;

        const pVariance = model.getVarianceSeries(perturbed);
        const pNegLL = computeNegLL(returns, pVariance, df);

        expect(pNegLL).toBeGreaterThanOrEqual(baseNegLL - 1e-4);
      }
    }
  });

  it('double fit on same data gives identical params', () => {
    const prices = makePrices(200);
    const model = new Garch(prices);
    const r1 = model.fit();
    const r2 = model.fit();

    expect(r1.params.omega).toBe(r2.params.omega);
    expect(r1.params.alpha).toBe(r2.params.alpha);
    expect(r1.params.beta).toBe(r2.params.beta);
    expect(r1.diagnostics.logLikelihood).toBe(r2.diagnostics.logLikelihood);
  });

  it('unconditional variance ≈ sample variance for large n', () => {
    const trueOmega = 0.00001;
    const trueAlpha = 0.1;
    const trueBeta = 0.85;
    const trueUnconditional = trueOmega / (1 - trueAlpha - trueBeta);

    const prices = generateGarchData(5000, trueOmega, trueAlpha, trueBeta);
    const returns = new Garch(prices).getReturns();
    const sampleVar = sampleVariance(returns);

    const relErr = Math.abs(sampleVar - trueUnconditional) / trueUnconditional;
    expect(relErr).toBeLessThan(0.5);
  });
});

// ── Model selection ─────────────────────────────────────────

describe('Model selection', () => {
  it('symmetric data: AIC prefers GARCH over EGARCH', () => {
    const prices = generateGarchData(1000, 0.00001, 0.1, 0.85, 999);
    const garchAIC = calibrateGarch(prices).diagnostics.aic;
    const egarchAIC = calibrateEgarch(prices).diagnostics.aic;

    expect(garchAIC).toBeLessThan(egarchAIC);
  });

  it('leverage data: AIC prefers EGARCH over GARCH', () => {
    const prices = generateEgarchData(1000, -0.1, 0.15, -0.15, 0.9, 777);
    const garchAIC = calibrateGarch(prices).diagnostics.aic;
    const egarchAIC = calibrateEgarch(prices).diagnostics.aic;

    expect(egarchAIC).toBeLessThan(garchAIC);
  });

  it('EGARCH on symmetric data: |γ| is small', () => {
    const prices = generateGarchData(1000, 0.00001, 0.1, 0.85, 555);
    const result = calibrateEgarch(prices);

    expect(Math.abs(result.params.gamma)).toBeLessThan(0.1);
  });

  it('GJR-GARCH on symmetric data: γ is small', () => {
    const prices = generateGarchData(1000, 0.00001, 0.1, 0.85, 555);
    const result = calibrateGjrGarch(prices);

    expect(result.params.gamma).toBeLessThan(0.15);
  });
});

// ── Forecast properties ─────────────────────────────────────

describe('Forecast properties', () => {
  it('forecast(params, 1) returns single-element arrays', () => {
    const model = new Garch(makePrices(55));
    const result = model.fit();
    const fc = model.forecast(result.params, 1);

    expect(fc.variance).toHaveLength(1);
    expect(fc.volatility).toHaveLength(1);
    expect(fc.annualized).toHaveLength(1);
  });

  it('GARCH long horizon → ω/(1−α−β)', () => {
    const model = new Garch(makePrices(200));
    const result = model.fit();
    const { omega, alpha, beta } = result.params;
    const unconditional = omega / (1 - alpha - beta);
    const persistence = alpha + beta;

    // Use enough steps for convergence given the persistence level
    const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);
    const relErr = Math.abs(fc.variance[steps - 1] - unconditional) / unconditional;

    expect(relErr).toBeLessThan(0.01);
  });

  it('EGARCH long horizon → exp(ω/(1−β))', () => {
    const prices = generateEgarchData(500, -0.1, 0.1, -0.05, 0.9, 42);
    const model = new Egarch(prices);
    const result = model.fit();
    const { omega, beta } = result.params;
    const unconditional = Math.exp(omega / (1 - beta));
    const persistence = Math.abs(beta);

    const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);
    const relErr = Math.abs(fc.variance[steps - 1] - unconditional) / unconditional;

    expect(relErr).toBeLessThan(0.01);
  });

  it('GJR-GARCH long horizon → ω/(1−α−γ/2−β)', () => {
    const model = new GjrGarch(makePrices(200));
    const result = model.fit();
    const { alpha, gamma, beta } = result.params;
    const unconditional = result.params.unconditionalVariance;
    const persistence = alpha + gamma / 2 + beta;

    // Use enough steps for convergence given the persistence level
    const steps = Math.max(500, Math.ceil(Math.log(0.01) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);
    const relErr = Math.abs(fc.variance[steps - 1] - unconditional) / unconditional;

    expect(relErr).toBeLessThan(0.01);
  });

  it('GARCH forecast is monotonic toward unconditional', () => {
    const model = new Garch(makePrices(200));
    const result = model.fit();
    const { omega, alpha, beta } = result.params;
    const unconditional = omega / (1 - alpha - beta);

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
});

// ── Options ─────────────────────────────────────────────────

describe('Options', () => {
  it('periodsPerYear scales annualized vol as √(periodsPerYear)', () => {
    const prices = makePrices(200);
    const r252 = new Garch(prices, { periodsPerYear: 252 }).fit();
    const r365 = new Garch(prices, { periodsPerYear: 365 }).fit();

    // Core params identical (periodsPerYear doesn't affect optimization)
    expect(r365.params.omega).toBe(r252.params.omega);
    expect(r365.params.alpha).toBe(r252.params.alpha);
    expect(r365.params.beta).toBe(r252.params.beta);

    // Annualized vol ratio = √(365/252)
    expect(r365.params.annualizedVol / r252.params.annualizedVol)
      .toBeCloseTo(Math.sqrt(365 / 252), 10);
  });

  it('low maxIter prevents convergence', () => {
    const result = new Garch(makePrices(200)).fit({ maxIter: 5 });

    expect(result.diagnostics.converged).toBe(false);
    expect(result.diagnostics.iterations).toBeLessThanOrEqual(5);
  });

  it('tighter tol requires more iterations', () => {
    const prices = makePrices(200);
    const loose = new Garch(prices).fit({ tol: 1e-4 });
    const tight = new Garch(prices).fit({ tol: 1e-12 });

    expect(tight.diagnostics.iterations).toBeGreaterThanOrEqual(
      loose.diagnostics.iterations,
    );
  });
});
