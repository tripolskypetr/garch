import { describe, it, expect } from 'vitest';
import { nelderMead } from '../src/index.js';

describe('nelderMead', () => {
  it('should minimize Rosenbrock function', () => {
    // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Minimum at (1, 1)
    function rosenbrock(x: number[]): number {
      return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2;
    }

    const result = nelderMead(rosenbrock, [0, 0], { maxIter: 2000, tol: 1e-8 });

    expect(result.x[0]).toBeCloseTo(1, 2);
    expect(result.x[1]).toBeCloseTo(1, 2);
    expect(result.fx).toBeCloseTo(0, 4);
  });

  it('should minimize quadratic function', () => {
    // f(x) = (x-3)² + (y+2)²
    // Minimum at (3, -2)
    function quadratic(x: number[]): number {
      return (x[0] - 3) ** 2 + (x[1] + 2) ** 2;
    }

    const result = nelderMead(quadratic, [0, 0]);

    expect(result.x[0]).toBeCloseTo(3, 3);
    expect(result.x[1]).toBeCloseTo(-2, 3);
    expect(result.fx).toBeCloseTo(0, 4);
    expect(result.converged).toBe(true);
  });

  it('should handle 1D optimization', () => {
    function parabola(x: number[]): number {
      return (x[0] - 5) ** 2;
    }

    const result = nelderMead(parabola, [0]);

    expect(result.x[0]).toBeCloseTo(5, 2);
    expect(result.converged).toBe(true);
  });

  it('should handle 3D optimization', () => {
    function sphere(x: number[]): number {
      return x[0] ** 2 + x[1] ** 2 + x[2] ** 2;
    }

    const result = nelderMead(sphere, [1, 2, 3]);

    expect(result.x[0]).toBeCloseTo(0, 3);
    expect(result.x[1]).toBeCloseTo(0, 3);
    expect(result.x[2]).toBeCloseTo(0, 3);
  });

  it('should respect maxIter', () => {
    function rosenbrock(x: number[]): number {
      return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2;
    }

    const result = nelderMead(rosenbrock, [0, 0], { maxIter: 10 });

    expect(result.iterations).toBeLessThanOrEqual(10);
    expect(result.converged).toBe(false);
  });

  it('should work with custom tolerance', () => {
    function quadratic(x: number[]): number {
      return x[0] ** 2;
    }

    const looseTol = nelderMead(quadratic, [10], { tol: 1 });
    const tightTol = nelderMead(quadratic, [10], { tol: 1e-12 });

    expect(tightTol.iterations).toBeGreaterThanOrEqual(looseTol.iterations);
  });
});
