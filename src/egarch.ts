import type { Candle, EgarchParams, CalibrationResult, VolatilityForecast } from './types.js';
import { nelderMead } from './optimizer.js';
import {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  calculateAIC,
  calculateBIC,
  EXPECTED_ABS_NORMAL,
} from './utils.js';

export interface EgarchOptions {
  periodsPerYear?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * EGARCH(1,1) model (Nelson, 1991)
 *
 * ln(σ²ₜ) = ω + α·(|zₜ₋₁| - E[|z|]) + γ·zₜ₋₁ + β·ln(σ²ₜ₋₁)
 *
 * where:
 * - zₜ = εₜ/σₜ (standardized residual)
 * - ω (omega): constant term
 * - α (alpha): magnitude effect
 * - γ (gamma): leverage effect (typically negative)
 * - β (beta): persistence
 * - E[|z|] = √(2/π) for standard normal
 */
export class Egarch {
  private returns: number[];
  private periodsPerYear: number;
  private initialVariance: number;

  constructor(data: Candle[] | number[], options: EgarchOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;

    if (data.length < 50) {
      throw new Error('Need at least 50 data points for EGARCH estimation');
    }

    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
    } else {
      this.returns = calculateReturns(data as Candle[]);
    }

    this.initialVariance = sampleVariance(this.returns);
  }

  /**
   * Calibrate EGARCH(1,1) parameters using Maximum Likelihood Estimation
   */
  fit(options: { maxIter?: number; tol?: number } = {}): CalibrationResult<EgarchParams> {
    const { maxIter = 1000, tol = 1e-8 } = options;
    const returns = this.returns;
    const n = returns.length;
    const initLogVar = Math.log(this.initialVariance);

    function negLogLikelihood(params: number[]): number {
      const [omega, alpha, gamma, beta] = params;

      // EGARCH allows negative gamma, but beta should ensure stationarity
      if (Math.abs(beta) >= 0.9999) return 1e10;

      let logVariance = initLogVar;
      let variance = Math.exp(logVariance);
      let ll = 0;

      for (let i = 0; i < n; i++) {
        if (i > 0) {
          const sigma = Math.sqrt(variance);
          const z = returns[i - 1] / sigma;

          logVariance = omega
            + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
            + gamma * z
            + beta * logVariance;

          // Prevent extreme values
          logVariance = Math.max(-50, Math.min(50, logVariance));
          variance = Math.exp(logVariance);
        }

        if (variance <= 1e-12 || !isFinite(variance)) return 1e10;

        ll += Math.log(variance) + (returns[i] ** 2) / variance;
      }

      return ll / 2;
    }

    // Initial guesses
    // omega approximates log of unconditional variance when other params are small
    const omega0 = initLogVar * 0.1;
    const alpha0 = 0.1;
    const gamma0 = -0.05; // Negative for typical leverage effect
    const beta0 = 0.95;

    const result = nelderMead(
      negLogLikelihood,
      [omega0, alpha0, gamma0, beta0],
      { maxIter, tol }
    );

    const [omega, alpha, gamma, beta] = result.x;

    // For EGARCH, unconditional variance: E[ln(σ²)] = ω/(1-β)
    // So E[σ²] ≈ exp(ω/(1-β)) when α and γ effects average out
    const unconditionalLogVar = omega / (1 - beta);
    const unconditionalVariance = Math.exp(unconditionalLogVar);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    const logLikelihood = -result.fx * 2;
    const numParams = 4;

    return {
      params: {
        omega,
        alpha,
        gamma,
        beta,
        persistence: beta, // In EGARCH, persistence is primarily driven by beta
        unconditionalVariance,
        annualizedVol,
        leverageEffect: gamma,
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
  getVarianceSeries(params: EgarchParams): number[] {
    const { omega, alpha, gamma, beta } = params;
    const variance: number[] = [];
    let logVariance = Math.log(this.initialVariance);

    for (let i = 0; i < this.returns.length; i++) {
      if (i === 0) {
        variance.push(this.initialVariance);
      } else {
        const sigma = Math.sqrt(variance[i - 1]);
        const z = this.returns[i - 1] / sigma;

        logVariance = omega
          + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
          + gamma * z
          + beta * logVariance;

        logVariance = Math.max(-50, Math.min(50, logVariance));
        variance.push(Math.exp(logVariance));
      }
    }

    return variance;
  }

  /**
   * Forecast variance forward
   *
   * Note: EGARCH forecasts are more complex because they depend on
   * the path of shocks. This provides an approximation assuming
   * expected values of future shocks.
   */
  forecast(params: EgarchParams, steps: number = 1): VolatilityForecast {
    const { omega, alpha, gamma, beta } = params;
    const variance: number[] = [];

    const varianceSeries = this.getVarianceSeries(params);
    const lastVariance = varianceSeries[varianceSeries.length - 1];
    const lastReturn = this.returns[this.returns.length - 1];

    // One-step ahead using actual last return
    const sigma = Math.sqrt(lastVariance);
    const z = lastReturn / sigma;
    let logVariance = omega
      + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
      + gamma * z
      + beta * Math.log(lastVariance);

    variance.push(Math.exp(logVariance));

    // Multi-step: assume E[z] = 0, E[|z|] = √(2/π)
    // So the α and γ terms contribute 0 on average
    for (let h = 1; h < steps; h++) {
      logVariance = omega + beta * logVariance;
      variance.push(Math.exp(logVariance));
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
 * Convenience function to calibrate EGARCH(1,1) from candles
 */
export function calibrateEgarch(
  data: Candle[] | number[],
  options: EgarchOptions = {}
): CalibrationResult<EgarchParams> {
  const model = new Egarch(data, options);
  return model.fit(options);
}
