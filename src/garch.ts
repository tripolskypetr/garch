import type { Candle, GarchParams, CalibrationResult, VolatilityForecast } from './types.js';
import { nelderMead } from './optimizer.js';
import {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  yangZhangVariance,
  perCandleParkinson,
  calculateAIC,
  calculateBIC,
  logGamma,
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
    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
      this.initialVariance = sampleVariance(this.returns);
      this.rv = null;
    } else {
      const candles = data as Candle[];
      this.returns = calculateReturns(candles);
      this.initialVariance = yangZhangVariance(candles);
      // Parkinson (1980) per-candle RV: ~5× more efficient than r²
      this.rv = perCandleParkinson(candles, this.returns);
    }
  }

  /**
   * Calibrate GARCH(1,1) parameters using Maximum Likelihood Estimation
   */
  fit(options: { maxIter?: number; tol?: number } = {}): CalibrationResult<GarchParams> {
    const { maxIter = 1000, tol = 1e-8 } = options;
    const returns = this.returns;
    const n = returns.length;
    const initVar = this.initialVariance;

    const rv = this.rv;

    // Student-t negative log-likelihood function
    function negLogLikelihood(params: number[]): number {
      const [omega, alpha, beta, df] = params;

      // Constraints
      if (omega <= 1e-12) return 1e10;
      if (alpha < 0 || beta < 0) return 1e10;
      if (alpha + beta >= 0.9999) return 1e10;
      if (df <= 2.01 || df > 100) return 1e10;

      const halfDfPlus1 = (df + 1) / 2;
      const dfMinus2 = df - 2;
      const constant = n * (logGamma(halfDfPlus1) - logGamma(df / 2) - 0.5 * Math.log(Math.PI * dfMinus2));

      let variance = initVar;
      let ll = 0;

      for (let i = 0; i < n; i++) {
        if (i > 0) {
          const innovation = rv ? rv[i - 1] : returns[i - 1] ** 2;
          variance = omega + alpha * innovation + beta * variance;
        }

        if (variance <= 1e-12) return 1e10;

        // Student-t log-likelihood
        ll += -0.5 * Math.log(variance) - halfDfPlus1 * Math.log(1 + (returns[i] ** 2) / (dfMinus2 * variance));
      }

      return -(ll + constant);
    }

    // Initial guesses
    const omega0 = initVar * 0.05;
    const alpha0 = 0.1;
    const beta0 = 0.85;
    const df0 = 5;

    const result = nelderMead(negLogLikelihood, [omega0, alpha0, beta0, df0], { maxIter, tol });

    const [omega, alpha, beta, df] = result.x;
    const persistence = alpha + beta;
    const unconditionalVariance = omega / (1 - persistence);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    const logLikelihood = -result.fx;
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
