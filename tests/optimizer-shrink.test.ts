import { describe, it, expect } from 'vitest';
import { nelderMead } from '../src/index.js';

// ── Nelder-Mead shrink branch coverage ──────────────────────
//
// The shrink() function (optimizer.ts:122-135) is called when
// contraction fails to improve the worst vertex. We need:
//   1. Outside contraction → shrink  (lines 94-97)
//   2. Inside contraction → shrink   (lines 107-109)
//
// Strategy: use non-smooth / rugged functions where the contraction
// point lands in a worse region, forcing a full simplex shrink.

describe('Nelder-Mead shrink paths', () => {
  it('triggers shrink via outside contraction failure (ridged landscape)', () => {
    // High-frequency ridges make contraction land on bumps.
    const fn = (x: number[]): number => {
      const base = x[0] * x[0] + x[1] * x[1];
      const ridges = 2 * Math.sin(30 * x[0]) * Math.sin(30 * x[1]);
      return base + ridges;
    };

    const result = nelderMead(fn, [5, 5], {
      maxIter: 5000,
      tol: 1e-12,
    });

    // Improved from starting point; exact convergence not expected
    expect(result.fx).toBeLessThan(fn([5, 5]));
    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('triggers shrink via inside contraction failure (L-infinity norm)', () => {
    // max(|x|, |y|) is non-smooth: flat faces + sharp edges
    // reliably defeat contraction near the kink at the origin.
    const fn = (x: number[]): number => {
      return Math.max(Math.abs(x[0]), Math.abs(x[1]));
    };

    const result = nelderMead(fn, [10, 7], {
      maxIter: 5000,
      tol: 1e-10,
    });

    expect(result.fx).toBeLessThan(10);
    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('triggers shrink on discontinuous step function', () => {
    // floor-based: contraction lands on same step → no improvement → shrink
    const fn = (x: number[]): number => {
      return Math.floor(Math.abs(x[0])) + Math.floor(Math.abs(x[1]));
    };

    const result = nelderMead(fn, [5.5, 3.5], {
      maxIter: 5000,
      tol: 1e-12,
    });

    // May not converge well, but should run without error and improve
    expect(result.fx).toBeLessThanOrEqual(fn([5.5, 3.5]));
    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('triggers shrink on Schwefel-like function with deep local minima', () => {
    const fn = (x: number[]): number => {
      const a = x[0] - 1;
      const b = x[1] - 1;
      return a * a + b * b + 5 * (Math.sin(10 * a) ** 2 + Math.sin(10 * b) ** 2);
    };

    const result = nelderMead(fn, [8, -6], {
      maxIter: 10000,
      tol: 1e-14,
    });

    expect(Number.isFinite(result.fx)).toBe(true);
    expect(result.fx).toBeLessThan(fn([8, -6]));
  });

  it('triggers shrink with aggressive contraction (rho=0.9)', () => {
    // Ackley function + aggressive rho → contraction overshoots into bumps
    const fn = (x: number[]): number => {
      const sum1 = x[0] * x[0] + x[1] * x[1];
      const sum2 = Math.cos(2 * Math.PI * x[0]) + Math.cos(2 * Math.PI * x[1]);
      return -20 * Math.exp(-0.2 * Math.sqrt(sum1 / 2)) - Math.exp(sum2 / 2) + 20 + Math.E;
    };

    const result = nelderMead(fn, [4, 4], {
      maxIter: 10000,
      tol: 1e-14,
      rho: 0.9,
    });

    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('1D shrink path: staircase function', () => {
    const fn = (x: number[]): number => {
      return Math.ceil(Math.abs(x[0]));
    };

    const result = nelderMead(fn, [7], {
      maxIter: 5000,
      tol: 1e-12,
    });

    expect(result.fx).toBeLessThanOrEqual(fn([7]));
    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('high-dimensional shrink with L-infinity norm (4D)', () => {
    const fn = (x: number[]): number => {
      return Math.max(...x.map(Math.abs));
    };

    const result = nelderMead(fn, [3, -4, 5, -2], {
      maxIter: 10000,
      tol: 1e-10,
    });

    expect(result.fx).toBeLessThan(fn([3, -4, 5, -2]));
    expect(Number.isFinite(result.fx)).toBe(true);
  });
});
