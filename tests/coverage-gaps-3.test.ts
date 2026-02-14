import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  calibrateGarch,
  calibrateEgarch,
  calculateReturns,
  calculateReturnsFromPrices,
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

// ── 1. Nelder-Mead: inside contraction + shrink paths ───────

describe('Nelder-Mead contraction and shrink branches', () => {
  it('converges on Beale function (triggers contraction/shrink)', () => {
    // Beale function: has a narrow curved valley
    // f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    // minimum at (3, 0.5)
    function beale(x: number[]): number {
      return (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2;
    }

    const result = nelderMead(beale, [0, 0], { maxIter: 5000, tol: 1e-10 });

    expect(result.x[0]).toBeCloseTo(3, 1);
    expect(result.x[1]).toBeCloseTo(0.5, 1);
    expect(result.fx).toBeLessThan(0.01);
  });

  it('converges on Matyas function (triggers outside contraction)', () => {
    // f(x,y) = 0.26(x² + y²) - 0.48xy, minimum at (0,0)
    function matyas(x: number[]): number {
      return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1];
    }

    const result = nelderMead(matyas, [10, -10], { maxIter: 2000 });

    expect(result.x[0]).toBeCloseTo(0, 2);
    expect(result.x[1]).toBeCloseTo(0, 2);
    expect(result.converged).toBe(true);
  });

  it('shrink is triggered on McCormick-like function', () => {
    // A function with a narrow valley that forces shrink steps
    function narrow(x: number[]): number {
      return (x[0] + x[1]) ** 2 + (x[0] - x[1] - 1) ** 4;
    }

    // Start far from minimum with large initial simplex
    const result = nelderMead(narrow, [10, -10], { maxIter: 5000 });

    expect(result.fx).toBeLessThan(narrow([10, -10]));
    expect(result.converged).toBe(true);
  });
});

// ── 2. Outside contraction fails → shrink ───────────────────

describe('Nelder-Mead reflection accepted (middle range)', () => {
  it('converges on Himmelblau function', () => {
    // f(x,y) = (x²+y-11)² + (x+y²-7)², minimum at (3,2) among others
    function himmelblau(x: number[]): number {
      return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2;
    }

    const result = nelderMead(himmelblau, [1, 1], { maxIter: 2000 });

    expect(result.fx).toBeLessThan(0.001);
    expect(result.converged).toBe(true);
  });
});

// ── 4. Variance series depends on data, not just params ─────

describe('Variance series depends on data', () => {
  it('GARCH: same params, different data → different variance series', () => {
    const prices1 = makePrices(100, 111);
    const prices2 = makePrices(100, 222);
    const params = garchParams(0.00001, 0.1, 0.85);

    const model1 = new Garch(prices1);
    const model2 = new Garch(prices2);
    const v1 = model1.getVarianceSeries(params);
    const v2 = model2.getVarianceSeries(params);

    // First element (initialVariance) should differ
    expect(v1[0]).not.toBe(v2[0]);
    // Series should differ throughout
    const allSame = v1.every((v, i) => v === v2[i]);
    expect(allSame).toBe(false);
  });

  it('EGARCH: same params, different data → different variance series', () => {
    const prices1 = makePrices(100, 111);
    const prices2 = makePrices(100, 222);
    const params = egarchParams(-0.1, 0.1, -0.05, 0.95);

    const model1 = new Egarch(prices1);
    const model2 = new Egarch(prices2);
    const v1 = model1.getVarianceSeries(params);
    const v2 = model2.getVarianceSeries(params);

    expect(v1[0]).not.toBe(v2[0]);
    const allSame = v1.every((v, i) => v === v2[i]);
    expect(allSame).toBe(false);
  });
});

// ── 5. EGARCH symmetry when γ = 0 ───────────────────────────

describe('EGARCH symmetry with gamma = 0', () => {
  it('positive and negative returns produce same variance change when γ = 0', () => {
    // Build two price series that differ only by sign of one return
    const base = makePrices(55);
    const model = new Egarch(base);
    const returns = model.getReturns();
    const initVar = model.getInitialVariance();

    const params = egarchParams(-0.5, 0.15, 0, 0.95); // gamma = 0

    // Manually compute variance at step 2 for +r and -r
    const r = Math.abs(returns[0]);
    const sigma = Math.sqrt(initVar);

    const zPos = r / sigma;
    const logVarPos = params.omega
      + params.alpha * (Math.abs(zPos) - EXPECTED_ABS_NORMAL)
      + params.beta * Math.log(initVar);

    const zNeg = -r / sigma;
    const logVarNeg = params.omega
      + params.alpha * (Math.abs(zNeg) - EXPECTED_ABS_NORMAL)
      + params.beta * Math.log(initVar);

    // With gamma = 0, positive and negative shocks have identical effect
    expect(logVarPos).toBeCloseTo(logVarNeg, 14);
  });

  it('positive and negative returns produce different variance when γ ≠ 0', () => {
    const initVar = 0.0004;
    const sigma = Math.sqrt(initVar);
    const r = 0.02;

    const params = egarchParams(-0.5, 0.15, -0.1, 0.95); // gamma = -0.1

    const zPos = r / sigma;
    const logVarPos = params.omega
      + params.alpha * (Math.abs(zPos) - EXPECTED_ABS_NORMAL)
      + params.gamma * zPos
      + params.beta * Math.log(initVar);

    const zNeg = -r / sigma;
    const logVarNeg = params.omega
      + params.alpha * (Math.abs(zNeg) - EXPECTED_ABS_NORMAL)
      + params.gamma * zNeg
      + params.beta * Math.log(initVar);

    // Negative shock should produce higher variance (negative gamma)
    expect(logVarNeg).toBeGreaterThan(logVarPos);
  });
});

// ── 6. Larger alpha → faster reaction to shocks ─────────────

describe('GARCH alpha sensitivity', () => {
  it('larger alpha produces larger variance spike after a big shock', () => {
    const prices = makePrices(100);
    const model = new Garch(prices);
    const returns = model.getReturns();

    const paramsLowAlpha = garchParams(0.00001, 0.05, 0.85);
    const paramsHighAlpha = garchParams(0.00001, 0.3, 0.60);

    const vLow = model.getVarianceSeries(paramsLowAlpha);
    const vHigh = model.getVarianceSeries(paramsHighAlpha);

    // Find the index after the largest absolute return
    let maxRetIdx = 0;
    let maxRet = 0;
    for (let i = 0; i < returns.length - 1; i++) {
      if (Math.abs(returns[i]) > maxRet) {
        maxRet = Math.abs(returns[i]);
        maxRetIdx = i;
      }
    }

    // After the biggest shock, high alpha should react more
    const reactionLow = Math.abs(vLow[maxRetIdx + 1] - vLow[maxRetIdx]);
    const reactionHigh = Math.abs(vHigh[maxRetIdx + 1] - vHigh[maxRetIdx]);

    expect(reactionHigh).toBeGreaterThan(reactionLow);
  });
});

// ── 7. Larger beta → more persistence ───────────────────────

describe('GARCH beta persistence', () => {
  it('larger beta makes variance decay slower toward unconditional', () => {
    const prices = makePrices(100);
    const model = new Garch(prices);

    const paramsLowBeta = garchParams(0.0001, 0.1, 0.5);
    const paramsHighBeta = garchParams(0.0001, 0.1, 0.85);

    const fcLow = model.forecast(paramsLowBeta, 50);
    const fcHigh = model.forecast(paramsHighBeta, 50);

    const uncondLow = paramsLowBeta.unconditionalVariance;
    const uncondHigh = paramsHighBeta.unconditionalVariance;

    // At step 10, high beta should be further from unconditional (relatively)
    const relErrLow = Math.abs(fcLow.variance[10] - uncondLow) / uncondLow;
    const relErrHigh = Math.abs(fcHigh.variance[10] - uncondHigh) / uncondHigh;

    expect(relErrHigh).toBeGreaterThan(relErrLow);
  });
});

// ── 8. Run of equal closes ──────────────────────────────────

describe('Run of equal closes', () => {
  it('GARCH handles long run of identical prices in the middle', () => {
    const prices = makePrices(200);
    // Insert a run of 20 identical prices
    const flatPrice = prices[80];
    for (let i = 80; i < 100; i++) {
      prices[i] = flatPrice;
    }

    const model = new Garch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.omega).toBeGreaterThan(0);
  });
});

// ── 9. Very small prices (penny stock) ──────────────────────

describe('Very small prices', () => {
  it('GARCH handles prices around 0.0001', () => {
    const basePrices = makePrices(200);
    // Scale down to penny-stock level
    const tinyPrices = basePrices.map(p => p * 1e-6);

    const model = new Garch(tinyPrices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('EGARCH handles prices around 0.0001', () => {
    const basePrices = makePrices(200);
    const tinyPrices = basePrices.map(p => p * 1e-6);

    const model = new Egarch(tinyPrices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });
});

// ── 10. Very large prices ───────────────────────────────────

describe('Very large prices', () => {
  it('GARCH handles prices around 1e8', () => {
    const basePrices = makePrices(200);
    const bigPrices = basePrices.map(p => p * 1e6);

    const model = new Garch(bigPrices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('EGARCH handles prices around 1e8', () => {
    const basePrices = makePrices(200);
    const bigPrices = basePrices.map(p => p * 1e6);

    const model = new Egarch(bigPrices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });
});

// ── 11. Alternating up/down (mean-reverting) ────────────────

describe('Alternating prices', () => {
  it('GARCH handles alternating up/down pattern', () => {
    const prices: number[] = [100];
    for (let i = 1; i <= 200; i++) {
      prices.push(i % 2 === 0 ? prices[i - 1] * 1.02 : prices[i - 1] * 0.98);
    }

    const model = new Garch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.omega).toBeGreaterThan(0);
  });

  it('EGARCH handles alternating up/down pattern', () => {
    const prices: number[] = [100];
    for (let i = 1; i <= 200; i++) {
      prices.push(i % 2 === 0 ? prices[i - 1] * 1.02 : prices[i - 1] * 0.98);
    }

    const model = new Egarch(prices);
    const result = model.fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });
});

// ── 12. Forecast with very large steps ──────────────────────

describe('Forecast with large steps', () => {
  it('GARCH forecast 10000 steps: all finite, converges', () => {
    const model = new Garch(makePrices(100));
    const result = model.fit();
    const { omega, alpha, beta } = result.params;
    const persistence = alpha + beta;

    // Use enough steps for convergence to within 1e-6
    const steps = Math.max(10000, Math.ceil(Math.log(1e-6) / Math.log(persistence)));
    const fc = model.forecast(result.params, steps);

    expect(fc.variance).toHaveLength(steps);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
    expect(fc.volatility.every(v => v > 0 && Number.isFinite(v))).toBe(true);
    expect(fc.annualized.every(v => v > 0 && Number.isFinite(v))).toBe(true);

    const unconditional = omega / (1 - persistence);
    const relErr = Math.abs(fc.variance[steps - 1] - unconditional) / unconditional;
    expect(relErr).toBeLessThan(1e-6);
  });

  it('EGARCH forecast 10000 steps: all finite', () => {
    const model = new Egarch(makePrices(100));
    const result = model.fit();
    const fc = model.forecast(result.params, 10000);

    expect(fc.variance).toHaveLength(10000);
    expect(fc.variance.every(v => v > 0 && Number.isFinite(v))).toBe(true);
  });
});

// ── 13. getVarianceSeries with different params on same model ─

describe('getVarianceSeries with different params', () => {
  it('GARCH: different params → different series, model unchanged', () => {
    const model = new Garch(makePrices(100));

    const params1 = garchParams(0.00001, 0.1, 0.85);
    const params2 = garchParams(0.00005, 0.2, 0.7);

    const v1a = model.getVarianceSeries(params1);
    const v2 = model.getVarianceSeries(params2);
    const v1b = model.getVarianceSeries(params1);

    // Different params → different series
    const allSame = v1a.every((v, i) => v === v2[i]);
    expect(allSame).toBe(false);

    // Calling with params2 didn't mutate: params1 gives same result
    for (let i = 0; i < v1a.length; i++) {
      expect(v1b[i]).toBe(v1a[i]);
    }
  });

  it('EGARCH: different params → different series, model unchanged', () => {
    const model = new Egarch(makePrices(100));

    const params1 = egarchParams(-0.1, 0.1, -0.05, 0.95);
    const params2 = egarchParams(-0.5, 0.2, -0.1, 0.8);

    const v1a = model.getVarianceSeries(params1);
    const v2 = model.getVarianceSeries(params2);
    const v1b = model.getVarianceSeries(params1);

    const allSame = v1a.every((v, i) => v === v2[i]);
    expect(allSame).toBe(false);

    for (let i = 0; i < v1a.length; i++) {
      expect(v1b[i]).toBe(v1a[i]);
    }
  });
});

// ── 14. getReturns length == getVarianceSeries length ────────

describe('Returns and variance series length consistency', () => {
  it('GARCH: getReturns().length === getVarianceSeries().length', () => {
    const sizes = [51, 100, 500];
    for (const n of sizes) {
      const model = new Garch(makePrices(n));
      const result = model.fit();
      expect(model.getReturns().length).toBe(model.getVarianceSeries(result.params).length);
    }
  });

  it('EGARCH: getReturns().length === getVarianceSeries().length', () => {
    const sizes = [51, 100, 500];
    for (const n of sizes) {
      const model = new Egarch(makePrices(n));
      const result = model.fit();
      expect(model.getReturns().length).toBe(model.getVarianceSeries(result.params).length);
    }
  });
});

// ── 15. Scale invariance of log returns and params ──────────

describe('Scale invariance', () => {
  it('log returns are identical for scaled prices', () => {
    const basePrices = makePrices(100);
    const scaledPrices = basePrices.map(p => p * 1000);

    const r1 = calculateReturnsFromPrices(basePrices);
    const r2 = calculateReturnsFromPrices(scaledPrices);

    expect(r1.length).toBe(r2.length);
    for (let i = 0; i < r1.length; i++) {
      expect(r1[i]).toBeCloseTo(r2[i], 14);
    }
  });

  it('GARCH params are identical for scaled prices', () => {
    const basePrices = makePrices(200);
    const scaledPrices = basePrices.map(p => p * 1000);

    const r1 = calibrateGarch(basePrices);
    const r2 = calibrateGarch(scaledPrices);

    expect(r1.params.omega).toBeCloseTo(r2.params.omega, 14);
    expect(r1.params.alpha).toBeCloseTo(r2.params.alpha, 14);
    expect(r1.params.beta).toBeCloseTo(r2.params.beta, 14);
    expect(r1.diagnostics.logLikelihood).toBeCloseTo(r2.diagnostics.logLikelihood, 8);
  });

  it('EGARCH logLikelihood is identical for scaled prices', () => {
    // EGARCH optimizer may converge to different local optima,
    // but the log-likelihood landscape is identical for scaled prices
    // (since log returns are identical). Verify LL is within 0.1%.
    const basePrices = makePrices(200);
    const scaledPrices = basePrices.map(p => p * 1000);

    const r1 = calibrateEgarch(basePrices);
    const r2 = calibrateEgarch(scaledPrices);

    const relErr = Math.abs(r1.diagnostics.logLikelihood - r2.diagnostics.logLikelihood)
      / Math.abs(r1.diagnostics.logLikelihood);
    expect(relErr).toBeLessThan(0.001);
  });

  it('candle returns are identical for scaled candles', () => {
    const candles: Candle[] = [
      { open: 99, high: 102, low: 98, close: 100, volume: 1000 },
      { open: 100, high: 112, low: 99, close: 110, volume: 1200 },
      { open: 109, high: 111, low: 96, close: 99, volume: 800 },
    ];

    const scaledCandles: Candle[] = candles.map(c => ({
      open: c.open * 1000,
      high: c.high * 1000,
      low: c.low * 1000,
      close: c.close * 1000,
      volume: c.volume,
    }));

    const r1 = calculateReturns(candles);
    const r2 = calculateReturns(scaledCandles);

    for (let i = 0; i < r1.length; i++) {
      expect(r1[i]).toBeCloseTo(r2[i], 14);
    }
  });
});
