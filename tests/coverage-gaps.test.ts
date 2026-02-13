import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  calibrateGarch,
  calibrateEgarch,
  checkLeverageEffect,
  calculateReturnsFromPrices,
  nelderMead,
  EXPECTED_ABS_NORMAL,
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

// ── 1. checkLeverageEffect: allNegative ─────────────────────

describe('checkLeverageEffect edge cases', () => {
  it('all negative returns: ratio = 1, recommendation = garch', () => {
    const allNegative = [-0.01, -0.02, -0.03];
    const stats = checkLeverageEffect(allNegative);

    expect(stats.positiveVol).toBe(0);
    expect(stats.ratio).toBe(1);
    expect(stats.recommendation).toBe('garch');
  });
});

// ── 2. calculateReturnsFromPrices: validation errors ────────

describe('calculateReturnsFromPrices validation', () => {
  it('throws on zero price', () => {
    expect(() => calculateReturnsFromPrices([100, 0, 50])).toThrow();
  });

  it('throws on negative price', () => {
    expect(() => calculateReturnsFromPrices([100, -1, 50])).toThrow();
  });

  it('throws on NaN price', () => {
    expect(() => calculateReturnsFromPrices([100, NaN, 50])).toThrow();
  });

  it('throws on Infinity price', () => {
    expect(() => calculateReturnsFromPrices([100, Infinity, 50])).toThrow();
  });

  it('throws on -Infinity price', () => {
    expect(() => calculateReturnsFromPrices([100, -Infinity, 50])).toThrow();
  });
});

// ── 3. Nelder-Mead: custom alpha/gamma/rho/sigma ───────────

describe('nelderMead custom coefficients', () => {
  function quadratic(x: number[]): number {
    return (x[0] - 3) ** 2 + (x[1] + 2) ** 2;
  }

  it('custom alpha (reflection coefficient)', () => {
    const result = nelderMead(quadratic, [0, 0], { alpha: 1.5 });

    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-2, 2);
    expect(result.converged).toBe(true);
  });

  it('custom gamma (expansion coefficient)', () => {
    const result = nelderMead(quadratic, [0, 0], { gamma: 3 });

    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-2, 2);
    expect(result.converged).toBe(true);
  });

  it('custom rho (contraction coefficient)', () => {
    const result = nelderMead(quadratic, [0, 0], { rho: 0.25 });

    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-2, 2);
    expect(result.converged).toBe(true);
  });

  it('custom sigma (shrink coefficient)', () => {
    const result = nelderMead(quadratic, [0, 0], { sigma: 0.25 });

    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-2, 2);
    expect(result.converged).toBe(true);
  });

  it('all custom coefficients together', () => {
    const result = nelderMead(quadratic, [0, 0], {
      alpha: 1.2,
      gamma: 2.5,
      rho: 0.4,
      sigma: 0.6,
    });

    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-2, 2);
    expect(result.converged).toBe(true);
  });
});

// ── 4. EGARCH periodsPerYear ────────────────────────────────

describe('EGARCH periodsPerYear', () => {
  it('periodsPerYear scales annualized vol as √(periodsPerYear)', () => {
    const prices = makePrices(200);
    const r252 = new Egarch(prices, { periodsPerYear: 252 }).fit();
    const r365 = new Egarch(prices, { periodsPerYear: 365 }).fit();

    // Core params identical (periodsPerYear doesn't affect optimization)
    expect(r365.params.omega).toBe(r252.params.omega);
    expect(r365.params.alpha).toBe(r252.params.alpha);
    expect(r365.params.gamma).toBe(r252.params.gamma);
    expect(r365.params.beta).toBe(r252.params.beta);

    // Annualized vol ratio = √(365/252)
    expect(r365.params.annualizedVol / r252.params.annualizedVol)
      .toBeCloseTo(Math.sqrt(365 / 252), 10);
  });
});

// ── 5. EGARCH maxIter / tol options ─────────────────────────

describe('EGARCH fit options', () => {
  it('low maxIter prevents convergence', () => {
    const result = new Egarch(makePrices(200)).fit({ maxIter: 5 });

    expect(result.diagnostics.converged).toBe(false);
    expect(result.diagnostics.iterations).toBeLessThanOrEqual(5);
  });

  it('tighter tol requires more iterations', () => {
    const prices = makePrices(200);
    const loose = new Egarch(prices).fit({ tol: 1e-4 });
    const tight = new Egarch(prices).fit({ tol: 1e-12 });

    expect(tight.diagnostics.iterations).toBeGreaterThanOrEqual(
      loose.diagnostics.iterations,
    );
  });
});

// ── 6. calibrateGarch/calibrateEgarch options forwarding ────

describe('Convenience function options forwarding', () => {
  it('calibrateGarch forwards periodsPerYear', () => {
    const prices = makePrices(200);
    const direct = new Garch(prices, { periodsPerYear: 365 }).fit();
    const convenience = calibrateGarch(prices, { periodsPerYear: 365 });

    expect(convenience.params.omega).toBe(direct.params.omega);
    expect(convenience.params.alpha).toBe(direct.params.alpha);
    expect(convenience.params.beta).toBe(direct.params.beta);
    expect(convenience.params.annualizedVol).toBe(direct.params.annualizedVol);
  });

  it('calibrateGarch forwards maxIter', () => {
    const prices = makePrices(200);
    const result = calibrateGarch(prices, { maxIter: 5 });

    expect(result.diagnostics.converged).toBe(false);
    expect(result.diagnostics.iterations).toBeLessThanOrEqual(5);
  });

  it('calibrateEgarch forwards periodsPerYear', () => {
    const prices = makePrices(200);
    const direct = new Egarch(prices, { periodsPerYear: 365 }).fit();
    const convenience = calibrateEgarch(prices, { periodsPerYear: 365 });

    expect(convenience.params.omega).toBe(direct.params.omega);
    expect(convenience.params.alpha).toBe(direct.params.alpha);
    expect(convenience.params.gamma).toBe(direct.params.gamma);
    expect(convenience.params.beta).toBe(direct.params.beta);
    expect(convenience.params.annualizedVol).toBe(direct.params.annualizedVol);
  });

  it('calibrateEgarch forwards maxIter', () => {
    const prices = makePrices(200);
    const result = calibrateEgarch(prices, { maxIter: 5 });

    expect(result.diagnostics.converged).toBe(false);
    expect(result.diagnostics.iterations).toBeLessThanOrEqual(5);
  });
});

// ── 7. EGARCH forecast monotonicity ─────────────────────────

describe('EGARCH forecast convergence', () => {
  it('forecast log-variance converges monotonically toward unconditional', () => {
    const prices = generateEgarchData(500, -0.1, 0.1, -0.05, 0.9, 42);
    const model = new Egarch(prices);
    const result = model.fit();
    const { omega, beta } = result.params;
    const unconditionalLogVar = omega / (1 - beta);

    const fc = model.forecast(result.params, 50);
    const logVars = fc.variance.map(v => Math.log(v));

    const above = logVars[0] >= unconditionalLogVar;
    for (let h = 1; h < 50; h++) {
      if (above) {
        expect(logVars[h]).toBeLessThanOrEqual(logVars[h - 1] + 1e-10);
      } else {
        expect(logVars[h]).toBeGreaterThanOrEqual(logVars[h - 1] - 1e-10);
      }
    }
  });
});

// ── 8. EGARCH negative beta ─────────────────────────────────

describe('EGARCH negative beta', () => {
  it('getVarianceSeries produces valid output with negative beta', () => {
    const model = new Egarch(makePrices(55));
    const params = {
      omega: -0.5, alpha: 0.1, gamma: -0.05, beta: -0.5,
      persistence: -0.5,
      unconditionalVariance: Math.exp(-0.5 / (1 - (-0.5))),
      annualizedVol: 0,
      leverageEffect: -0.05,
    };
    const variance = model.getVarianceSeries(params);

    for (const v of variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('forecast produces valid output with negative beta', () => {
    const model = new Egarch(makePrices(55));
    const result = model.fit();

    // Force negative beta for this test
    const params = {
      ...result.params,
      beta: -0.5,
      persistence: -0.5,
    };

    const fc = model.forecast(params, 10);

    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ── 9. Nelder-Mead x0[i] === 0 initial delta ───────────────

describe('nelderMead initial simplex with zeros', () => {
  it('x0 = [0] uses delta = 0.00025 (not 0)', () => {
    // If delta were 0, simplex would be degenerate and couldn't optimize
    function parabola(x: number[]): number {
      return (x[0] - 1) ** 2;
    }

    const result = nelderMead(parabola, [0]);

    expect(result.x[0]).toBeCloseTo(1, 2);
    expect(result.converged).toBe(true);
  });

  it('x0 = [0, 0, 0] all zeros: still converges', () => {
    function sphere(x: number[]): number {
      return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2;
    }

    const result = nelderMead(sphere, [0, 0, 0], { maxIter: 5000 });

    expect(result.x[0]).toBeCloseTo(1, 1);
    expect(result.x[1]).toBeCloseTo(2, 1);
    expect(result.x[2]).toBeCloseTo(3, 1);
  });

  it('mixed zeros and non-zeros: different deltas used', () => {
    function target(x: number[]): number {
      return (x[0] - 5) ** 2 + (x[1] - 3) ** 2;
    }

    // x0[0] = 0 → delta = 0.00025; x0[1] = 10 → delta = 0.5
    const result = nelderMead(target, [0, 10]);

    expect(result.x[0]).toBeCloseTo(5, 2);
    expect(result.x[1]).toBeCloseTo(3, 2);
    expect(result.converged).toBe(true);
  });
});
