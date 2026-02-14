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
  type Candle,
} from '../src/index.js';
import { calculateAIC, calculateBIC } from '../src/utils.js';

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

// ── 1. calculateReturnsFromPrices: first element invalid ────

describe('calculateReturnsFromPrices first element validation', () => {
  it('throws when first price is negative', () => {
    expect(() => calculateReturnsFromPrices([-1, 100, 50])).toThrow('Invalid price');
  });

  it('throws when first price is zero', () => {
    expect(() => calculateReturnsFromPrices([0, 100, 50])).toThrow('Invalid price');
  });

  it('throws when first price is NaN', () => {
    expect(() => calculateReturnsFromPrices([NaN, 100, 50])).toThrow('Invalid price');
  });

  it('throws when first price is Infinity', () => {
    expect(() => calculateReturnsFromPrices([Infinity, 100, 50])).toThrow('Invalid price');
  });
});

// ── 2. calculateReturns with -Infinity close ────────────────

describe('calculateReturns -Infinity close', () => {
  it('throws on -Infinity close', () => {
    const candles: Candle[] = [
      { open: 99, high: 102, low: 98, close: 100, volume: 1000 },
      { open: 100, high: 101, low: 99, close: -Infinity, volume: 500 },
    ];
    expect(() => calculateReturns(candles)).toThrow('Invalid close price');
  });
});

// ── 3. forecast with negative steps ─────────────────────────

describe('forecast with negative steps', () => {
  it('GARCH forecast(params, -1) returns 1 element (loop body skipped)', () => {
    const model = new Garch(makePrices(55));
    const result = model.fit();
    const fc = model.forecast(result.params, -1);

    // One-step push always happens, multi-step loop: h=1 < -1 is false
    expect(fc.variance).toHaveLength(1);
    expect(fc.variance[0]).toBeGreaterThan(0);
  });

  it('EGARCH forecast(params, -1) returns 1 element', () => {
    const model = new Egarch(makePrices(55));
    const result = model.fit();
    const fc = model.forecast(result.params, -1);

    expect(fc.variance).toHaveLength(1);
    expect(fc.variance[0]).toBeGreaterThan(0);
  });
});

// ── 4. checkLeverageEffect: ratio exactly 1.2 (boundary) ───

describe('checkLeverageEffect boundary', () => {
  it('ratio = 1.2 exactly → recommendation is garch (not egarch)', () => {
    // We need negativeVol / positiveVol = 1.2
    // negativeVol = sqrt(sum(r²)/n) for negative r
    // positiveVol = sqrt(sum(r²)/n) for positive r
    // If all negative returns have magnitude a, and all positive have magnitude b,
    // then ratio = a/b. We want a/b = 1.2, e.g. a = 0.012, b = 0.01
    const returns = [
      0.01, -0.012, 0.01, -0.012, 0.01, -0.012,
      0.01, -0.012, 0.01, -0.012, 0.01, -0.012,
    ];

    const stats = checkLeverageEffect(returns);

    // negativeVol = 0.012, positiveVol = 0.01, ratio = 1.2
    expect(stats.ratio).toBeCloseTo(1.2, 10);
    // ratio > 1.2 is false when ratio === 1.2, so recommendation is 'garch'
    expect(stats.recommendation).toBe('garch');
  });

  it('ratio = 1.2001 → recommendation is egarch', () => {
    // Slightly above 1.2
    const returns = [
      0.01, -0.012001, 0.01, -0.012001, 0.01, -0.012001,
      0.01, -0.012001, 0.01, -0.012001, 0.01, -0.012001,
    ];

    const stats = checkLeverageEffect(returns);

    expect(stats.ratio).toBeGreaterThan(1.2);
    expect(stats.recommendation).toBe('egarch');
  });
});

// ── 5. checkLeverageEffect: verify vol formulas numerically ─

describe('checkLeverageEffect vol formulas', () => {
  it('negativeVol and positiveVol match hand calculation', () => {
    const returns = [0.01, -0.03, 0.02, -0.04, 0.015, -0.025];

    const stats = checkLeverageEffect(returns);

    const negative = [-0.03, -0.04, -0.025];
    const positive = [0.01, 0.02, 0.015];

    const expectedNegVol = Math.sqrt(
      (0.03 ** 2 + 0.04 ** 2 + 0.025 ** 2) / 3
    );
    const expectedPosVol = Math.sqrt(
      (0.01 ** 2 + 0.02 ** 2 + 0.015 ** 2) / 3
    );

    expect(stats.negativeVol).toBeCloseTo(expectedNegVol, 14);
    expect(stats.positiveVol).toBeCloseTo(expectedPosVol, 14);
    expect(stats.ratio).toBeCloseTo(expectedNegVol / expectedPosVol, 14);
  });
});

// ── 6. nelderMead: x0 is not mutated ────────────────────────

describe('nelderMead input immutability', () => {
  it('x0 array is not mutated', () => {
    const x0 = [5, -3];
    const copy = [...x0];

    nelderMead((x) => x[0] ** 2 + x[1] ** 2, x0);

    expect(x0).toEqual(copy);
  });
});

// ── 7. GARCH uses numParams = 3 for AIC/BIC ────────────────

describe('GARCH numParams = 3', () => {
  it('AIC = 2·3 − 2·LL', () => {
    const result = calibrateGarch(makePrices(200));
    const { logLikelihood, aic } = result.diagnostics;

    expect(aic).toBeCloseTo(2 * 3 - 2 * logLikelihood, 10);
  });

  it('BIC = 3·ln(n) − 2·LL where n = number of returns', () => {
    const prices = makePrices(200);
    const result = calibrateGarch(prices);
    const n = prices.length - 1; // number of returns
    const { logLikelihood, bic } = result.diagnostics;

    expect(bic).toBeCloseTo(3 * Math.log(n) - 2 * logLikelihood, 10);
  });
});

// ── 8. EGARCH uses numParams = 4 for AIC/BIC ───────────────

describe('EGARCH numParams = 4', () => {
  it('AIC = 2·4 − 2·LL', () => {
    const result = calibrateEgarch(makePrices(200));
    const { logLikelihood, aic } = result.diagnostics;

    expect(aic).toBeCloseTo(2 * 4 - 2 * logLikelihood, 10);
  });

  it('BIC = 4·ln(n) − 2·LL where n = number of returns', () => {
    const prices = makePrices(200);
    const result = calibrateEgarch(prices);
    const n = prices.length - 1;
    const { logLikelihood, bic } = result.diagnostics;

    expect(bic).toBeCloseTo(4 * Math.log(n) - 2 * logLikelihood, 10);
  });
});

// ── 9. GJR-GARCH uses numParams = 4 for AIC/BIC ────────────

describe('GJR-GARCH numParams = 4', () => {
  it('AIC = 2·4 − 2·LL', () => {
    const result = calibrateGjrGarch(makePrices(200));
    const { logLikelihood, aic } = result.diagnostics;

    expect(aic).toBeCloseTo(2 * 4 - 2 * logLikelihood, 10);
  });

  it('BIC = 4·ln(n) − 2·LL where n = number of returns', () => {
    const prices = makePrices(200);
    const result = calibrateGjrGarch(prices);
    const n = prices.length - 1;
    const { logLikelihood, bic } = result.diagnostics;

    expect(bic).toBeCloseTo(4 * Math.log(n) - 2 * logLikelihood, 10);
  });
});
