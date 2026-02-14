import type { OptimizerResult } from './types.js';

export function nelderMead(
  fn: (x: number[]) => number,
  x0: number[],
  options: {
    maxIter?: number;
    tol?: number;
    alpha?: number;
    gamma?: number;
    rho?: number;
    sigma?: number;
  } = {}
): OptimizerResult {
  const {
    maxIter = 1000,
    tol = 1e-8,
    alpha = 1,    // reflection
    gamma = 2,    // expansion
    rho = 0.5,    // contraction
    sigma = 0.5,  // shrink
  } = options;

  const n = x0.length;

  // Initialize simplex
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const point = x0.slice();
    const delta = point[i] === 0 ? 0.00025 : point[i] * 0.20;
    point[i] += delta;
    simplex.push(point);
  }

  let values = simplex.map(fn);
  let iterations = 0;
  let converged = false;

  for (iterations = 0; iterations < maxIter; iterations++) {
    // Sort simplex by function values
    const indices = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
    const sortedSimplex = indices.map(i => simplex[i]);
    const sortedValues = indices.map(i => values[i]);

    for (let i = 0; i <= n; i++) {
      simplex[i] = sortedSimplex[i];
      values[i] = sortedValues[i];
    }

    // Check convergence
    const range = values[n] - values[0];
    if (range < tol) {
      converged = true;
      break;
    }

    // Centroid of all points except worst
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        centroid[j] += simplex[i][j] / n;
      }
    }

    // Reflection
    const reflected = centroid.map((c, j) => c + alpha * (c - simplex[n][j]));
    const fr = fn(reflected);

    if (fr < values[0]) {
      // Expansion
      const expanded = centroid.map((c, j) => c + gamma * (reflected[j] - c));
      const fe = fn(expanded);

      if (fe < fr) {
        simplex[n] = expanded;
        values[n] = fe;
      } else {
        simplex[n] = reflected;
        values[n] = fr;
      }
    } else if (fr < values[n - 1]) {
      simplex[n] = reflected;
      values[n] = fr;
    } else {
      // Contraction
      if (fr < values[n]) {
        // Outside contraction
        const contracted = centroid.map((c, j) => c + rho * (reflected[j] - c));
        const fc = fn(contracted);

        if (fc <= fr) {
          simplex[n] = contracted;
          values[n] = fc;
        } else {
          // Shrink
          shrink(simplex, values, sigma, fn, n);
        }
      } else {
        // Inside contraction
        const contracted = centroid.map((c, j) => c + rho * (simplex[n][j] - c));
        const fc = fn(contracted);

        if (fc < values[n]) {
          simplex[n] = contracted;
          values[n] = fc;
        } else {
          // Shrink
          shrink(simplex, values, sigma, fn, n);
        }
      }
    }
  }

  return {
    x: simplex[0],
    fx: values[0],
    iterations,
    converged,
  };
}

function shrink(
  simplex: number[][],
  values: number[],
  sigma: number,
  fn: (x: number[]) => number,
  n: number
): void {
  for (let i = 1; i <= n; i++) {
    for (let j = 0; j < n; j++) {
      simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
    }
    values[i] = fn(simplex[i]);
  }
}

/**
 * Multi-start Nelder-Mead: runs NM from multiple deterministic starting
 * points and returns the best result. Escapes local minima by exploring
 * different basins of attraction.
 *
 * Perturbation uses golden-ratio quasi-random sequence for uniform
 * coverage of the search space without clustering.
 */
const PHI = (1 + Math.sqrt(5)) / 2; // golden ratio

export function nelderMeadMultiStart(
  fn: (x: number[]) => number,
  x0: number[],
  options: {
    maxIter?: number;
    tol?: number;
    restarts?: number;
  } = {}
): OptimizerResult {
  const { maxIter = 1000, tol = 1e-8, restarts = 3 } = options;
  const n = x0.length;

  // Run from original starting point
  let best = nelderMead(fn, x0, { maxIter, tol });

  // Run from perturbed starting points
  for (let k = 1; k <= restarts; k++) {
    const perturbed = new Array(n);
    for (let i = 0; i < n; i++) {
      // Quasi-random perturbation: golden-ratio sequence mapped to [-0.5, +0.5]
      const frac = (k * (i + 1) * PHI) % 1;
      const scale = frac - 0.5; // range [-0.5, +0.5]
      perturbed[i] = x0[i] === 0
        ? 0.001 * scale
        : x0[i] * (1 + scale);
    }

    const result = nelderMead(fn, perturbed, { maxIter, tol });
    if (result.fx < best.fx) {
      best = result;
    }
  }

  return best;
}
