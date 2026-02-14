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
  sampleVariance,
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

// ── 1. GARCH negLogLikelihood penalty paths ─────────────────

describe('GARCH constraint barriers', () => {
  it('fit never returns omega <= 0', () => {
    const seeds = [1, 42, 100, 999, 5555];
    for (const seed of seeds) {
      const result = calibrateGarch(makePrices(100, seed));
      expect(result.params.omega).toBeGreaterThan(0);
    }
  });

  it('fit never returns alpha < 0', () => {
    const result = calibrateGarch(makePrices(200));
    expect(result.params.alpha).toBeGreaterThanOrEqual(0);
  });

  it('fit never returns beta < 0', () => {
    const result = calibrateGarch(makePrices(200));
    expect(result.params.beta).toBeGreaterThanOrEqual(0);
  });

  it('fit never returns alpha + beta >= 1', () => {
    const seeds = [1, 42, 100, 999, 5555];
    for (const seed of seeds) {
      const result = calibrateGarch(makePrices(200, seed));
      expect(result.params.alpha + result.params.beta).toBeLessThan(1);
    }
  });

  it('getVarianceSeries always positive even with extreme returns', () => {
    const prices = makePrices(200);
    // Inject extreme moves
    prices[50] *= 2;
    prices[100] *= 0.3;
    const model = new Garch(prices);
    const result = model.fit();
    const variance = model.getVarianceSeries(result.params);

    for (const v of variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── 2. EGARCH negLogLikelihood — !isFinite guard ────────────

describe('EGARCH negLogLikelihood guards', () => {
  it('fit returns finite logLikelihood even with extreme input', () => {
    const prices = makePrices(200);
    prices[50] *= 3;
    prices[51] *= 0.2;
    const model = new Egarch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(Math.abs(result.params.beta)).toBeLessThan(1);
  });

  it('fit never returns |beta| >= 1', () => {
    const seeds = [1, 42, 100, 999, 5555];
    for (const seed of seeds) {
      const result = calibrateEgarch(makePrices(100, seed));
      expect(Math.abs(result.params.beta)).toBeLessThan(1);
    }
  });
});

// ── 3. Nelder-Mead branch: expansion rejected (fe >= fr) ────

describe('Nelder-Mead expansion rejection branch', () => {
  it('converges on function where expansion overshoots', () => {
    // Booth function: f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
    // minimum at (1, 3)
    // Has a narrow valley that causes expansion to overshoot
    function booth(x: number[]): number {
      return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2;
    }

    const result = nelderMead(booth, [-5, -5], { maxIter: 2000 });

    expect(result.x[0]).toBeCloseTo(1, 1);
    expect(result.x[1]).toBeCloseTo(3, 1);
    expect(result.converged).toBe(true);
  });
});

// ── 4. Instance isolation ───────────────────────────────────

describe('Instance isolation', () => {
  it('two Garch instances do not share state', () => {
    const prices1 = makePrices(100, 111);
    const prices2 = makePrices(100, 222);

    const model1 = new Garch(prices1);
    const model2 = new Garch(prices2);

    const r1 = model1.fit();
    const r2 = model2.fit();

    // Different data → different params
    expect(r1.params.omega).not.toBe(r2.params.omega);

    // Fitting model2 didn't affect model1
    const r1again = model1.fit();
    expect(r1again.params.omega).toBe(r1.params.omega);
    expect(r1again.params.alpha).toBe(r1.params.alpha);
    expect(r1again.params.beta).toBe(r1.params.beta);
  });

  it('two Egarch instances do not share state', () => {
    const prices1 = makePrices(100, 111);
    const prices2 = makePrices(100, 222);

    const model1 = new Egarch(prices1);
    const model2 = new Egarch(prices2);

    const r1 = model1.fit();
    const r2 = model2.fit();

    expect(r1.params.omega).not.toBe(r2.params.omega);

    const r1again = model1.fit();
    expect(r1again.params.omega).toBe(r1.params.omega);
  });

  it('two GjrGarch instances do not share state', () => {
    const prices1 = makePrices(100, 111);
    const prices2 = makePrices(100, 222);

    const model1 = new GjrGarch(prices1);
    const model2 = new GjrGarch(prices2);

    const r1 = model1.fit();
    const r2 = model2.fit();

    expect(r1.params.omega).not.toBe(r2.params.omega);

    const r1again = model1.fit();
    expect(r1again.params.omega).toBe(r1.params.omega);
  });
});

// ── 5 & 6. forecast annualized scales with periodsPerYear ───

describe('Forecast annualized scales with periodsPerYear', () => {
  it('GARCH forecast annualized uses periodsPerYear', () => {
    const prices = makePrices(200);
    const model252 = new Garch(prices, { periodsPerYear: 252 });
    const model365 = new Garch(prices, { periodsPerYear: 365 });
    const r252 = model252.fit();
    const r365 = model365.fit();

    const fc252 = model252.forecast(r252.params, 5);
    const fc365 = model365.forecast(r365.params, 5);

    // Core variance identical (periodsPerYear doesn't affect it)
    for (let i = 0; i < 5; i++) {
      expect(fc252.variance[i]).toBeCloseTo(fc365.variance[i], 14);
    }

    // Annualized ratio = √(365/252)
    const ratio = Math.sqrt(365 / 252);
    for (let i = 0; i < 5; i++) {
      expect(fc365.annualized[i] / fc252.annualized[i]).toBeCloseTo(ratio, 10);
    }
  });

  it('EGARCH forecast annualized uses periodsPerYear', () => {
    const prices = makePrices(200);
    const model252 = new Egarch(prices, { periodsPerYear: 252 });
    const model365 = new Egarch(prices, { periodsPerYear: 365 });
    const r252 = model252.fit();
    const r365 = model365.fit();

    const fc252 = model252.forecast(r252.params, 5);
    const fc365 = model365.forecast(r365.params, 5);

    for (let i = 0; i < 5; i++) {
      expect(fc252.variance[i]).toBeCloseTo(fc365.variance[i], 14);
    }

    const ratio = Math.sqrt(365 / 252);
    for (let i = 0; i < 5; i++) {
      expect(fc365.annualized[i] / fc252.annualized[i]).toBeCloseTo(ratio, 10);
    }
  });
});

// ── 7. Large data smoke test ────────────────────────────────

describe('Large data (n > 5000)', () => {
  it('GARCH fits 10K prices without error', () => {
    const prices = makePrices(10001, 77);
    const model = new Garch(prices);
    const result = model.fit();

    expect(result.diagnostics.converged).toBe(true);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('EGARCH fits 10K prices without error', () => {
    const prices = makePrices(10001, 77);
    const model = new Egarch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(Math.abs(result.params.beta)).toBeLessThan(1);
  });

  it('GJR-GARCH fits 10K prices without error', () => {
    const prices = makePrices(10001, 77);
    const model = new GjrGarch(prices);
    const result = model.fit();

    expect(result.diagnostics.converged).toBe(true);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });
});

// ── 8. getVarianceSeries length for EGARCH ──────────────────

describe('EGARCH getVarianceSeries length', () => {
  it('length equals returns.length (= prices.length - 1)', () => {
    const prices = makePrices(100);
    const model = new Egarch(prices);
    const result = model.fit();
    const variance = model.getVarianceSeries(result.params);
    const returns = model.getReturns();

    expect(variance.length).toBe(returns.length);
    expect(variance.length).toBe(prices.length - 1);
  });
});

// ── 9. GARCH unconditionalVariance & annualizedVol formulas ─

describe('GARCH computed fields', () => {
  it('unconditionalVariance = omega / (1 - alpha - beta)', () => {
    const result = calibrateGarch(makePrices(200));
    const { omega, alpha, beta, unconditionalVariance } = result.params;

    expect(unconditionalVariance).toBeCloseTo(omega / (1 - alpha - beta), 14);
  });

  it('annualizedVol = sqrt(unconditionalVariance * 252) * 100', () => {
    const result = calibrateGarch(makePrices(200));
    const { unconditionalVariance, annualizedVol } = result.params;

    expect(annualizedVol).toBeCloseTo(Math.sqrt(unconditionalVariance * 252) * 100, 10);
  });

  it('annualizedVol uses custom periodsPerYear', () => {
    const result = calibrateGarch(makePrices(200), { periodsPerYear: 365 });
    const { unconditionalVariance, annualizedVol } = result.params;

    expect(annualizedVol).toBeCloseTo(Math.sqrt(unconditionalVariance * 365) * 100, 10);
  });
});

// ── 10. EGARCH unconditionalVariance formula ────────────────

describe('EGARCH computed fields', () => {
  it('unconditionalVariance = exp(omega / (1 - beta))', () => {
    const result = calibrateEgarch(makePrices(200));
    const { omega, beta, unconditionalVariance } = result.params;

    expect(unconditionalVariance).toBeCloseTo(Math.exp(omega / (1 - beta)), 10);
  });

  it('annualizedVol = sqrt(unconditionalVariance * 252) * 100', () => {
    const result = calibrateEgarch(makePrices(200));
    const { unconditionalVariance, annualizedVol } = result.params;

    expect(annualizedVol).toBeCloseTo(Math.sqrt(unconditionalVariance * 252) * 100, 10);
  });

  it('annualizedVol uses custom periodsPerYear', () => {
    const result = calibrateEgarch(makePrices(200), { periodsPerYear: 365 });
    const { unconditionalVariance, annualizedVol } = result.params;

    expect(annualizedVol).toBeCloseTo(Math.sqrt(unconditionalVariance * 365) * 100, 10);
  });
});

// ── 11. EGARCH persistence === beta ─────────────────────────

describe('EGARCH persistence field', () => {
  it('persistence equals beta', () => {
    const seeds = [1, 42, 123, 999];
    for (const seed of seeds) {
      const result = calibrateEgarch(makePrices(100, seed));
      expect(result.params.persistence).toBe(result.params.beta);
    }
  });
});

// ── 12. EGARCH leverageEffect === gamma ─────────────────────

describe('EGARCH leverageEffect field', () => {
  it('leverageEffect equals gamma', () => {
    const seeds = [1, 42, 123, 999];
    for (const seed of seeds) {
      const result = calibrateEgarch(makePrices(100, seed));
      expect(result.params.leverageEffect).toBe(result.params.gamma);
    }
  });
});

// ── 13. EGARCH AIC < BIC for large samples ──────────────────

describe('EGARCH AIC vs BIC', () => {
  it('BIC > AIC for large samples (n > e²)', () => {
    const result = calibrateEgarch(makePrices(500));

    expect(result.diagnostics.bic).toBeGreaterThan(result.diagnostics.aic);
  });
});

// ── 14. Very volatile data ──────────────────────────────────

describe('Very volatile data', () => {
  it('GARCH handles ±50% swings', () => {
    const rng = lcg(42);
    const prices = [100];
    for (let i = 1; i <= 200; i++) {
      const r = (rng() - 0.5) * 1.0; // ±50%
      prices.push(prices[i - 1] * Math.exp(r));
    }

    const model = new Garch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
    expect(result.params.omega).toBeGreaterThan(0);
  });

  it('EGARCH handles ±50% swings', () => {
    const rng = lcg(42);
    const prices = [100];
    for (let i = 1; i <= 200; i++) {
      const r = (rng() - 0.5) * 1.0;
      prices.push(prices[i - 1] * Math.exp(r));
    }

    const model = new Egarch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(Math.abs(result.params.beta)).toBeLessThan(1);
  });
});

// ── 15. Trending data ───────────────────────────────────────

describe('Trending data', () => {
  it('GARCH handles monotonically rising prices', () => {
    const prices: number[] = [];
    for (let i = 0; i <= 200; i++) {
      prices.push(100 * Math.exp(0.001 * i + Math.sin(i) * 0.001));
    }

    const model = new Garch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.omega).toBeGreaterThan(0);
  });

  it('GARCH handles monotonically falling prices', () => {
    const prices: number[] = [];
    for (let i = 0; i <= 200; i++) {
      prices.push(100 * Math.exp(-0.001 * i + Math.sin(i) * 0.001));
    }

    const model = new Garch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });

  it('EGARCH handles monotonically rising prices', () => {
    const prices: number[] = [];
    for (let i = 0; i <= 200; i++) {
      prices.push(100 * Math.exp(0.001 * i + Math.sin(i) * 0.001));
    }

    const model = new Egarch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });
});

// ── 16. Candles with zero return (close == open) ────────────

describe('Candles with zero return', () => {
  it('some periods with close == previous close', () => {
    const candles: Candle[] = [];
    let close = 100;
    const rng = lcg(42);
    for (let i = 0; i < 100; i++) {
      // Every 5th candle has zero return
      const newClose = (i % 5 === 0) ? close : close * (1 + (rng() - 0.5) * 0.02);
      candles.push({
        open: close,
        high: Math.max(close, newClose) + 0.1,
        low: Math.min(close, newClose) - 0.1,
        close: newClose,
        volume: 1000,
      });
      close = newClose;
    }

    const returns = calculateReturns(candles);
    // Should have some exact zeros
    expect(returns.some(r => r === 0)).toBe(true);

    const model = new Garch(candles);
    const result = model.fit();
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });
});

// ── 17. Minimum edge: exactly 51 points end-to-end ──────────

describe('Minimum data (51 prices)', () => {
  it('GARCH: fit → getVarianceSeries → forecast chain', () => {
    const prices = makePrices(51);
    const model = new Garch(prices);
    const result = model.fit();
    const variance = model.getVarianceSeries(result.params);
    const fc = model.forecast(result.params, 5);

    expect(result.params.omega).toBeGreaterThan(0);
    expect(variance.length).toBe(50);
    expect(fc.variance.length).toBe(5);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });

  it('EGARCH: fit → getVarianceSeries → forecast chain', () => {
    const prices = makePrices(51);
    const model = new Egarch(prices);
    const result = model.fit();
    const variance = model.getVarianceSeries(result.params);
    const fc = model.forecast(result.params, 5);

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(variance.length).toBe(50);
    expect(fc.variance.length).toBe(5);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });
});

// ── 19. forecast() default steps = 1 ───────────────────────

describe('forecast default steps', () => {
  it('GARCH forecast() with no steps arg returns 1 element', () => {
    const model = new Garch(makePrices(100));
    const result = model.fit();
    const fc = model.forecast(result.params);

    expect(fc.variance).toHaveLength(1);
    expect(fc.volatility).toHaveLength(1);
    expect(fc.annualized).toHaveLength(1);
  });

  it('GARCH forecast() default matches forecast(params, 1)', () => {
    const model = new Garch(makePrices(100));
    const result = model.fit();
    const fcDefault = model.forecast(result.params);
    const fc1 = model.forecast(result.params, 1);

    expect(fcDefault.variance[0]).toBe(fc1.variance[0]);
    expect(fcDefault.volatility[0]).toBe(fc1.volatility[0]);
    expect(fcDefault.annualized[0]).toBe(fc1.annualized[0]);
  });

  it('EGARCH forecast() with no steps arg returns 1 element', () => {
    const model = new Egarch(makePrices(100));
    const result = model.fit();
    const fc = model.forecast(result.params);

    expect(fc.variance).toHaveLength(1);
    expect(fc.volatility).toHaveLength(1);
    expect(fc.annualized).toHaveLength(1);
  });

  it('EGARCH forecast() default matches forecast(params, 1)', () => {
    const model = new Egarch(makePrices(100));
    const result = model.fit();
    const fcDefault = model.forecast(result.params);
    const fc1 = model.forecast(result.params, 1);

    expect(fcDefault.variance[0]).toBe(fc1.variance[0]);
    expect(fcDefault.volatility[0]).toBe(fc1.volatility[0]);
    expect(fcDefault.annualized[0]).toBe(fc1.annualized[0]);
  });
});
