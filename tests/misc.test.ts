import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  GjrGarch,
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  sampleVarianceWithMean,
  nelderMead,
  type Candle,
} from '../src/index.js';

// ── Empty / zero-length inputs ──────────────────────────────

describe('Empty inputs', () => {
  it('calculateReturnsFromPrices([]) → empty array', () => {
    expect(calculateReturnsFromPrices([])).toEqual([]);
  });

  it('calculateReturnsFromPrices([100]) → empty array', () => {
    expect(calculateReturnsFromPrices([100])).toEqual([]);
  });

  it('calculateReturns([]) → empty array', () => {
    expect(calculateReturns([])).toEqual([]);
  });

  it('calculateReturns([single candle]) → empty array', () => {
    const candle: Candle = { open: 99, high: 101, low: 98, close: 100, volume: 500 };
    expect(calculateReturns([candle])).toEqual([]);
  });

  it('sampleVariance([]) → NaN (0/0)', () => {
    expect(sampleVariance([])).toBeNaN();
  });

  it('sampleVarianceWithMean([]) → -0 (0 / (n-1 = -1))', () => {
    expect(sampleVarianceWithMean([])).toBe(-0);
  });

  it('sampleVarianceWithMean([x]) → NaN (0/(n-1=0))', () => {
    // Bessel's correction: divide by n-1 = 0
    expect(sampleVarianceWithMean([0.05])).toBeNaN();
  });
});

// ── Nelder-Mead zero dimension ──────────────────────────────

describe('nelderMead edge cases', () => {
  it('empty x0: returns immediately', () => {
    const result = nelderMead(() => 42, []);

    expect(result.x).toEqual([]);
    expect(result.fx).toBe(42);
    expect(result.converged).toBe(true);
  });
});

// ── Immutability ────────────────────────────────────────────

describe('Immutability', () => {
  function makePrices(n: number): number[] {
    const prices = [100];
    let state = 12345;
    for (let i = 1; i < n; i++) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      prices.push(prices[i - 1] * Math.exp(((state / 0x7fffffff) - 0.5) * 0.04));
    }
    return prices;
  }

  it('Garch.getReturns() returns a copy', () => {
    const model = new Garch(makePrices(55));
    const a = model.getReturns();
    const b = model.getReturns();

    expect(a).toEqual(b);
    expect(a).not.toBe(b); // different reference

    a[0] = 999;
    expect(model.getReturns()[0]).not.toBe(999);
  });

  it('Egarch.getReturns() returns a copy', () => {
    const model = new Egarch(makePrices(55));
    const a = model.getReturns();
    const b = model.getReturns();

    expect(a).toEqual(b);
    expect(a).not.toBe(b);

    a[0] = 999;
    expect(model.getReturns()[0]).not.toBe(999);
  });

  it('GjrGarch.getReturns() returns a copy', () => {
    const model = new GjrGarch(makePrices(55));
    const a = model.getReturns();
    const b = model.getReturns();

    expect(a).toEqual(b);
    expect(a).not.toBe(b);

    a[0] = 999;
    expect(model.getReturns()[0]).not.toBe(999);
  });
});

// ── Candle timestamp field ──────────────────────────────────

describe('Candle timestamp', () => {
  const base: Candle[] = [
    { open: 99, high: 102, low: 98, close: 100, volume: 1000 },
    { open: 100, high: 112, low: 99, close: 110, volume: 1200 },
    { open: 109, high: 111, low: 96, close: 99, volume: 800 },
  ];

  it('works without timestamp', () => {
    const returns = calculateReturns(base);
    expect(returns).toHaveLength(2);
  });

  it('works with timestamp (ignored)', () => {
    const withTs: Candle[] = base.map((c, i) => ({ ...c, timestamp: 1700000000 + i * 86400 }));
    const returns = calculateReturns(withTs);

    expect(returns).toHaveLength(2);
    expect(returns).toEqual(calculateReturns(base));
  });
});
