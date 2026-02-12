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

  it('should converge quickly when starting near optimum', () => {
    function quadratic(x: number[]): number {
      return (x[0] - 3) ** 2 + (x[1] + 2) ** 2;
    }

    const result = nelderMead(quadratic, [3.001, -2.001], { tol: 1e-6 });

    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(50);
    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(-2, 2);
  });

  it('should find local minimum of non-convex function', () => {
    // Rastrigin: global min at origin, many local minima
    function rastrigin(x: number[]): number {
      return 20
        + (x[0] ** 2 - 10 * Math.cos(2 * Math.PI * x[0]))
        + (x[1] ** 2 - 10 * Math.cos(2 * Math.PI * x[1]));
    }

    const x0 = [3.5, 4.5];
    const result = nelderMead(rastrigin, x0, { maxIter: 5000 });

    expect(result.fx).toBeLessThan(rastrigin(x0));
    expect(result.converged).toBe(true);
  });

  it('should handle 10D optimization', () => {
    function sphere(x: number[]): number {
      return x.reduce((s, v) => s + v * v, 0);
    }

    const x0 = Array.from({ length: 10 }, (_, i) => (i + 1) * 0.1);
    const result = nelderMead(sphere, x0, { maxIter: 50000, tol: 1e-6 });

    expect(result.fx).toBeLessThan(sphere(x0) * 0.01);
  });
});
