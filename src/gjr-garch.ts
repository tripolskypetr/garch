import type { Candle, GjrGarchParams, CalibrationResult, VolatilityForecast } from './types.js';
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

export interface GjrGarchOptions {
  periodsPerYear?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * GJR-GARCH(1,1) model (Glosten, Jagannathan & Runkle, 1993)
 *
 * σ²ₜ = ω + α·ε²ₜ₋₁ + γ·ε²ₜ₋₁·I(rₜ₋₁<0) + β·σ²ₜ₋₁
 *
 * where:
 * - ω (omega) > 0: constant term
 * - α (alpha) ≥ 0: symmetric shock response
 * - γ (gamma) ≥ 0: asymmetric leverage coefficient
 * - β (beta) ≥ 0: persistence
 * - I(r<0) = 1 when return is negative, 0 otherwise
 * - Stationarity: α + γ/2 + β < 1
 *
 * With Candle[] input, ε² is replaced by Parkinson per-candle RV.
 * Leverage direction still comes from close-to-close return sign.
 */
export class GjrGarch {
  private returns: number[];
  private rv: number[] | null;
  private periodsPerYear: number;
  private initialVariance: number;

  constructor(data: Candle[] | number[], options: GjrGarchOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;

    if (data.length < 50) {
      throw new Error('Need at least 50 data points for GJR-GARCH estimation');
    }

    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
      this.initialVariance = sampleVariance(this.returns);
      this.rv = null;
    } else {
      const candles = data as Candle[];
      this.returns = calculateReturns(candles);
      this.initialVariance = yangZhangVariance(candles);
      this.rv = perCandleParkinson(candles, this.returns);
    }
  }

  /**
   * Calibrate GJR-GARCH(1,1) parameters using Maximum Likelihood Estimation
   */
  fit(options: { maxIter?: number; tol?: number } = {}): CalibrationResult<GjrGarchParams> {
    const { maxIter = 1000, tol = 1e-8 } = options;
    const returns = this.returns;
    const n = returns.length;
    const initVar = this.initialVariance;

    const rv = this.rv;

    function negLogLikelihood(params: number[]): number {
      const [omega, alpha, gamma, beta, df] = params;

      if (omega <= 1e-12) return 1e10;
      if (alpha < 0 || gamma < 0 || beta < 0) return 1e10;
      if (alpha + gamma / 2 + beta >= 0.9999) return 1e10;
      if (df <= 2.01 || df > 100) return 1e10;

      const halfDfPlus1 = (df + 1) / 2;
      const dfMinus2 = df - 2;
      const constant = n * (logGamma(halfDfPlus1) - logGamma(df / 2) - 0.5 * Math.log(Math.PI * dfMinus2));

      let variance = initVar;
      let ll = 0;

      for (let i = 0; i < n; i++) {
        if (i > 0) {
          const innovation = rv ? rv[i - 1] : returns[i - 1] ** 2;
          const indicator = returns[i - 1] < 0 ? 1 : 0;
          variance = omega + alpha * innovation + gamma * innovation * indicator + beta * variance;
        }

        if (variance <= 1e-12) return 1e10;

        // Student-t log-likelihood
        ll += -0.5 * Math.log(variance) - halfDfPlus1 * Math.log(1 + (returns[i] ** 2) / (dfMinus2 * variance));
      }

      return -(ll + constant);
    }

    const omega0 = initVar * 0.05;
    const alpha0 = 0.05;
    const gamma0 = 0.1;
    const beta0 = 0.85;
    const df0 = 5;

    const result = nelderMead(negLogLikelihood, [omega0, alpha0, gamma0, beta0, df0], { maxIter, tol });

    const [omega, alpha, gamma, beta, df] = result.x;
    const persistence = alpha + gamma / 2 + beta;
    const unconditionalVariance = omega / (1 - persistence);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    const logLikelihood = -result.fx;
    const numParams = 5;

    return {
      params: {
        omega,
        alpha,
        gamma,
        beta,
        persistence,
        unconditionalVariance,
        annualizedVol,
        leverageEffect: gamma,
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
  getVarianceSeries(params: GjrGarchParams): number[] {
    const { omega, alpha, gamma, beta } = params;
    const variance: number[] = [];

    for (let i = 0; i < this.returns.length; i++) {
      if (i === 0) {
        variance.push(this.initialVariance);
      } else {
        const innovation = this.rv ? this.rv[i - 1] : this.returns[i - 1] ** 2;
        const indicator = this.returns[i - 1] < 0 ? 1 : 0;
        const v = omega + alpha * innovation + gamma * innovation * indicator + beta * variance[i - 1];
        variance.push(v);
      }
    }

    return variance;
  }

  /**
   * Forecast variance forward
   */
  forecast(params: GjrGarchParams, steps: number = 1): VolatilityForecast {
    const { omega, alpha, gamma, beta } = params;
    const variance: number[] = [];

    const varianceSeries = this.getVarianceSeries(params);
    const lastVariance = varianceSeries[varianceSeries.length - 1];
    const lastInnovation = this.rv
      ? this.rv[this.rv.length - 1]
      : this.returns[this.returns.length - 1] ** 2;
    const lastIndicator = this.returns[this.returns.length - 1] < 0 ? 1 : 0;

    // One-step ahead using actual last return
    let v = omega + alpha * lastInnovation + gamma * lastInnovation * lastIndicator + beta * lastVariance;
    variance.push(v);

    // Multi-step: E[I(r<0)] = 0.5, so effective persistence = α + γ/2 + β
    for (let h = 1; h < steps; h++) {
      v = omega + (alpha + gamma / 2 + beta) * v;
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
 * Convenience function to calibrate GJR-GARCH(1,1) from candles
 */
export function calibrateGjrGarch(
  data: Candle[] | number[],
  options: GjrGarchOptions = {}
): CalibrationResult<GjrGarchParams> {
  const model = new GjrGarch(data, options);
  return model.fit(options);
}
