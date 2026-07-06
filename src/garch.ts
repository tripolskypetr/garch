import type { Candle, GarchParams, CalibrationResult, VolatilityForecast } from './types.js';
import { nelderMeadMultiStart } from './optimizer.js';
import {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  yangZhangVariance,
  perCandleParkinson,
  calculateAIC,
  calculateBIC,
  logGamma,
  validateCandles,
} from './utils.js';

export interface GarchOptions {
  periodsPerYear?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * GARCH(1,1) model
 *
 * σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
 *
 * where:
 * - ω (omega) > 0: constant term
 * - α (alpha) ≥ 0: ARCH parameter (reaction to shocks)
 * - β (beta) ≥ 0: GARCH parameter (persistence)
 * - α + β < 1: stationarity condition
 */
export class Garch {
  private returns: number[];
  private rv: number[] | null;
  private periodsPerYear: number;
  private initialVariance: number;

  constructor(data: Candle[] | number[], options: GarchOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;

    if (data.length < 50) {
      throw new Error('Need at least 50 data points for GARCH estimation');
    }

    // Determine if input is candles or prices
    // Variance floor keeps degenerate (constant-price) data from producing
    // log(0)/division-by-zero downstream instead of a graceful bad fit.
    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
      this.initialVariance = Math.max(sampleVariance(this.returns), 1e-300);
      this.rv = null;
    } else {
      const candles = data as Candle[];
      validateCandles(candles);
      this.returns = calculateReturns(candles);
      this.initialVariance = Math.max(yangZhangVariance(candles), 1e-300);
      // Parkinson (1980) per-candle RV: ~5× more efficient than r²
      this.rv = perCandleParkinson(candles, this.returns);
    }
  }

  /**
   * Calibrate GARCH(1,1) parameters using Maximum Likelihood Estimation
   */
  fit(
    options: { maxIter?: number; tol?: number; forgetting?: number; warmStart?: GarchParams } = {},
  ): CalibrationResult<GarchParams> {
    const { maxIter = 1000, tol = 1e-8, forgetting = 1 } = options;
    const n = this.returns.length;

    // Exponential forgetting: observation t contributes with weight
    // λ^(n−1−t) (newest = 1), so the fit adapts to regime shifts instead of
    // weighting a year-old candle like yesterday's. λ = 1 disables.
    const weights = new Array(n).fill(1);
    if (forgetting < 1) {
      for (let t = 0; t < n; t++) weights[t] = Math.pow(forgetting, n - 1 - t);
    }
    const wTotal = weights.reduce((a, b) => a + b, 0);

    // Calibrate in normalized space: returns are scaled to unit initial
    // variance, so the likelihood floors, penalty constants, and optimizer
    // tolerances are scale-free — a stablecoin pair and a high-vol altcoin
    // follow the same optimizer path. Parameters are mapped back to the
    // data scale after optimization.
    const s2 = 1 / this.initialVariance;
    const s = Math.sqrt(s2);
    const returns = this.returns.map(r => r * s);
    const rv = this.rv ? this.rv.map(v => v * s2) : null;
    const initVar = 1;
    const varFloor = 1e-12;

    // Student-t negative log-likelihood function
    function negLogLikelihood(params: number[]): number {
      const [omega, alpha, beta, df] = params;

      // Constraints
      if (omega <= varFloor) return 1e10;
      if (alpha < 0 || beta < 0) return 1e10;
      if (alpha + beta >= 0.9999) return 1e10;
      if (df <= 2.01 || df > 100) return 1e10;

      const halfDfPlus1 = (df + 1) / 2;
      const dfMinus2 = df - 2;
      const constant = wTotal * (logGamma(halfDfPlus1) - logGamma(df / 2) - 0.5 * Math.log(Math.PI * dfMinus2));

      let variance = initVar;
      let ll = 0;

      for (let i = 0; i < n; i++) {
        if (i > 0) {
          const innovation = rv ? rv[i - 1] : returns[i - 1] ** 2;
          variance = omega + alpha * innovation + beta * variance;
        }

        if (variance <= varFloor) return 1e10;

        // Student-t log-likelihood
        ll += weights[i] * (-0.5 * Math.log(variance) - halfDfPlus1 * Math.log(1 + (returns[i] ** 2) / (dfMinus2 * variance)));
      }

      return -(ll + constant);
    }

    // Initial guesses: variance targeting — ω₀ implied by the sample
    // variance and the persistence seed, so the optimizer starts at the
    // observed volatility level for any asset/interval.
    const alpha0 = 0.1;
    const beta0 = 0.85;
    const omega0 = initVar * (1 - alpha0 - beta0);
    const df0 = 5;

    // Warm start (previous window's optimum) replaces the cold seed: the
    // multi-start perturbations then explore its neighborhood, and rolling
    // refits converge in a fraction of the cold multi-start cost. The
    // constraint set is window-independent, so a previous optimum is
    // always feasible.
    const wp = options.warmStart;
    const warmValid = !!(wp && isFinite(wp.omega) && wp.omega > 0
      && isFinite(wp.alpha) && isFinite(wp.beta) && isFinite(wp.df));
    const x0 = warmValid
      ? [wp!.omega * s2, wp!.alpha, wp!.beta, wp!.df]
      : [omega0, alpha0, beta0, df0];

    const result = nelderMeadMultiStart(negLogLikelihood, x0, {
      maxIter,
      tol,
      restarts: warmValid ? 1 : 3,
    });

    // Map back to the data scale: ω scales with variance, α/β/df are scale-free
    const [omegaScaled, alpha, beta, df] = result.x;
    const omega = omegaScaled / s2;
    const persistence = alpha + beta;
    const unconditionalVariance = omega / (1 - persistence);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    // Jacobian of the rescaling: LL in data units = LL(scaled) + n·ln s
    const logLikelihood = -result.fx + n * Math.log(s);
    const numParams = 4;

    return {
      params: {
        omega,
        alpha,
        beta,
        persistence,
        unconditionalVariance,
        annualizedVol,
        df,
      },
      diagnostics: {
        logLikelihood,
        aic: calculateAIC(logLikelihood, numParams),
        bic: calculateBIC(logLikelihood, numParams, n),
        iterations: result.iterations,
        converged: result.converged,
      },
    };
  }

  /**
   * Calculate conditional variance series given parameters
   */
  getVarianceSeries(params: GarchParams): number[] {
    const { omega, alpha, beta } = params;
    const variance: number[] = [];

    for (let i = 0; i < this.returns.length; i++) {
      if (i === 0) {
        variance.push(this.initialVariance);
      } else {
        const innovation = this.rv ? this.rv[i - 1] : this.returns[i - 1] ** 2;
        const v = omega + alpha * innovation + beta * variance[i - 1];
        variance.push(v);
      }
    }

    return variance;
  }

  /**
   * Forecast variance forward
   */
  forecast(params: GarchParams, steps: number = 1): VolatilityForecast {
    const { omega, alpha, beta } = params;
    const variance: number[] = [];

    // Get last variance
    const varianceSeries = this.getVarianceSeries(params);
    const lastVariance = varianceSeries[varianceSeries.length - 1];
    const lastInnovation = this.rv
      ? this.rv[this.rv.length - 1]
      : this.returns[this.returns.length - 1] ** 2;

    // One-step ahead
    let v = omega + alpha * lastInnovation + beta * lastVariance;
    variance.push(v);

    // Multi-step ahead (converges to unconditional variance)
    for (let h = 1; h < steps; h++) {
      v = omega + (alpha + beta) * v;
      variance.push(v);
    }

    return {
      variance,
      volatility: variance.map(v => Math.sqrt(v)),
      annualized: variance.map(v => Math.sqrt(v * this.periodsPerYear) * 100),
    };
  }

  /**
   * Get the return series
   */
  getReturns(): number[] {
    return [...this.returns];
  }

  /**
   * Get initial variance estimate
   */
  getInitialVariance(): number {
    return this.initialVariance;
  }
}

/**
 * Convenience function to calibrate GARCH(1,1) from candles
 */
export function calibrateGarch(
  data: Candle[] | number[],
  options: GarchOptions = {}
): CalibrationResult<GarchParams> {
  const model = new Garch(data, options);
  return model.fit(options);
}
