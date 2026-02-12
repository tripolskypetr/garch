import { describe, it, expect } from 'vitest';
import { Garch, Egarch } from '../src/index.js';

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

// ── Numerical stability ─────────────────────────────────────

describe('Numerical stability', () => {
  it('near-constant prices: fits without crashing', () => {
    const prices: number[] = [100];
    let state = 42;
    for (let i = 1; i <= 54; i++) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      prices.push(prices[i - 1] * (1 + ((state / 0x7fffffff) - 0.5) * 1e-6));
    }

    const result = new Garch(prices).fit();

    expect(Number.isFinite(result.params.omega)).toBe(true);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });

  it('exactly constant prices: does not throw or hang', () => {
    const model = new Garch(Array(55).fill(100));
    expect(() => model.fit()).not.toThrow();
  });

  it('extreme outlier: single 50 % drop', () => {
    const prices = makePrices(200);
    prices[100] *= 0.5;

    const result = new Garch(prices).fit();

    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
    expect(result.params.persistence).toBeLessThan(1);
  });

  it('high persistence (α+β ≈ 0.998): stays stationary', () => {
    let state = 77;
    const random = () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
    const randn = () => {
      const u1 = random(), u2 = random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    const omega = 1e-6, alpha = 0.05, beta = 0.948;
    let v = omega / (1 - alpha - beta);
    const prices = [100];
    for (let i = 0; i < 500; i++) {
      const eps = Math.sqrt(v) * randn();
      prices.push(prices[prices.length - 1] * Math.exp(eps));
      v = omega + alpha * eps ** 2 + beta * v;
    }

    const result = new Garch(prices).fit();

    expect(result.params.persistence).toBeLessThan(1);
    expect(result.params.persistence).toBeGreaterThan(0);
    expect(Number.isFinite(result.diagnostics.logLikelihood)).toBe(true);
  });

  it('EGARCH: extreme params trigger clamp, variance stays finite', () => {
    const model = new Egarch(makePrices(55));

    const series = model.getVarianceSeries({
      omega: -50, alpha: 5, gamma: 3, beta: 0.99,
      persistence: 0.99,
      unconditionalVariance: 0,
      annualizedVol: 0,
      leverageEffect: 3,
    });

    for (const v of series) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  });
});

// ── Input validation ────────────────────────────────────────

describe('Input validation', () => {
  it('50 prices (boundary): does not throw', () => {
    expect(() => new Garch(makePrices(50))).not.toThrow();
    expect(() => new Egarch(makePrices(50))).not.toThrow();
  });

  it('49 prices: throws', () => {
    expect(() => new Garch(makePrices(49))).toThrow('at least 50');
    expect(() => new Egarch(makePrices(49))).toThrow('at least 50');
  });

  it('negative price: throws', () => {
    const prices = makePrices(55);
    prices[25] = -1;
    expect(() => new Garch(prices)).toThrow();
  });

  it('NaN in prices: throws', () => {
    const prices = makePrices(55);
    prices[10] = NaN;
    expect(() => new Garch(prices)).toThrow();
  });

  it('Infinity in prices: throws', () => {
    const prices = makePrices(55);
    prices[10] = Infinity;
    expect(() => new Garch(prices)).toThrow();
  });
});
