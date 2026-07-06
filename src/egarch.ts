import type { Candle, EgarchParams, CalibrationResult, VolatilityForecast } from './types.js';
import { nelderMeadMultiStart } from './optimizer.js';
import {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  yangZhangVariance,
  perCandleParkinson,
  calculateAIC,
  calculateBIC,
  EXPECTED_ABS_NORMAL,
  logGamma,
  expectedAbsStudentT,
  validateCandles,
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
 * - E[|z|] = expectedAbsStudentT(df) for Student-t(df)
 */
export class Egarch {
  private returns: number[];
  private rv: number[] | null;
  private periodsPerYear: number;
  private initialVariance: number;

  constructor(data: Candle[] | number[], options: EgarchOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;

    if (data.length < 50) {
      throw new Error('Need at least 50 data points for EGARCH estimation');
    }

    // Variance floor keeps degenerate (constant-price) data from producing
    // ω = ln(0) = -Infinity instead of a graceful bad fit.
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
   * Calibrate EGARCH(1,1) parameters using Maximum Likelihood Estimation
   */
  fit(
    options: { maxIter?: number; tol?: number; forgetting?: number; warmStart?: EgarchParams } = {},
  ): CalibrationResult<EgarchParams> {
    const { maxIter = 1000, tol = 1e-8, forgetting = 1 } = options;
    const n = this.returns.length;
    const initLogVarOrig = Math.log(this.initialVariance);

    // Exponential forgetting: observation t contributes with weight
    // λ^(n−1−t) (newest = 1); λ = 1 disables.
    const weights = new Array(n).fill(1);
    if (forgetting < 1) {
      for (let t = 0; t < n; t++) weights[t] = Math.pow(forgetting, n - 1 - t);
    }
    const wTotal = weights.reduce((a, b) => a + b, 0);

    // Calibrate in normalized space: returns are scaled to unit initial
    // variance, so the likelihood floors, the ±50 log-variance clamp, and
    // optimizer tolerances are scale-free — a stablecoin pair and a
    // high-vol altcoin follow the same optimizer path. ω is mapped back
    // to the data scale after optimization.
    const s2 = 1 / this.initialVariance;
    const s = Math.sqrt(s2);
    const returns = this.returns.map(r => r * s);
    const rv = this.rv ? this.rv.map(v => v * s2) : null;
    const initLogVar = 0; // ln of the scaled initial variance
    const varFloor = 1e-12;

    function negLogLikelihood(params: number[]): number {
      const [omega, alpha, gamma, beta, df] = params;

      // EGARCH allows negative gamma, but beta should ensure stationarity
      if (Math.abs(beta) >= 0.9999) return 1e10;
      if (df <= 2.01 || df > 100) return 1e10;

      // ω/(1−β) is the implied unconditional ln(σ²). When β rides toward ±1
      // on weakly-identified data the likelihood surface is flat along this
      // ridge and the implied long-run variance drifts orders of magnitude
      // away from the sample variance measured on the same data. Hard wall
      // at 4 orders of magnitude, plus a weak Gaussian prior (sd = 1 in
      // log-variance) that resolves the flat direction toward the sample
      // level while leaving well-identified fits untouched.
      const impliedLogVar = omega / (1 - beta);
      if (!isFinite(impliedLogVar) || Math.abs(impliedLogVar - initLogVar) > Math.log(1e4)) return 1e10;
      const priorDev = impliedLogVar - initLogVar;
      const prior = 0.5 * priorDev * priorDev;

      const eAbsZ = expectedAbsStudentT(df);
      const halfDfPlus1 = (df + 1) / 2;
      const dfMinus2 = df - 2;
      const constant = wTotal * (logGamma(halfDfPlus1) - logGamma(df / 2) - 0.5 * Math.log(Math.PI * dfMinus2));

      let logVariance = initLogVar;
      let variance = Math.exp(logVariance);
      let ll = 0;

      for (let i = 0; i < n; i++) {
        if (i > 0) {
          const sigma = Math.sqrt(variance);
          const z = returns[i - 1] / sigma; // directional — kept for leverage

          // Magnitude: √(RV/σ²) for candles, |z| for prices
          const magnitude = rv
            ? Math.sqrt(rv[i - 1] / variance)
            : Math.abs(z);

          logVariance = omega
            + alpha * (magnitude - eAbsZ)
            + gamma * z
            + beta * logVariance;

          // Prevent extreme values
          logVariance = Math.max(-50, Math.min(50, logVariance));
          variance = Math.exp(logVariance);
        }

        if (variance <= varFloor || !isFinite(variance)) return 1e10;

        // Student-t log-likelihood
        ll += weights[i] * (-0.5 * Math.log(variance) - halfDfPlus1 * Math.log(1 + (returns[i] ** 2) / (dfMinus2 * variance)));
      }

      return -(ll + constant) + prior;
    }

    // Initial guesses: variance targeting in log space —
    // E[ln σ²] = ω/(1−β), so ω₀ = ln(σ̂²)·(1−β₀) starts the optimizer at
    // the observed volatility level (ω₀ = 0.1·ln σ̂² implied a level of
    // 2·ln σ̂² with β₀ = 0.95 — orders of magnitude off).
    const beta0 = 0.95;
    const omega0 = initLogVar * (1 - beta0);
    const alpha0 = 0.1;
    const gamma0 = -0.05; // Negative for typical leverage effect
    const df0 = 5;

    // Warm start (previous window's optimum) replaces the cold seed with a
    // reduced restart budget. The hard wall on the implied unconditional
    // level moves with the sample variance between windows, so an
    // out-of-wall warm seed falls back to the cold start.
    const wp = options.warmStart;
    let warmX0: number[] | null = null;
    if (wp && isFinite(wp.omega) && isFinite(wp.beta) && Math.abs(wp.beta) < 1) {
      const cand = [wp.omega - (1 - wp.beta) * initLogVarOrig, wp.alpha, wp.gamma, wp.beta, wp.df];
      if (Math.abs(cand[0] / (1 - wp.beta)) <= Math.log(1e4)) warmX0 = cand;
    }

    const result = nelderMeadMultiStart(
      negLogLikelihood,
      warmX0 ?? [omega0, alpha0, gamma0, beta0, df0],
      { maxIter, tol, restarts: warmX0 ? 1 : 4 }
    );

    // Map back to the data scale: ln σ²_orig = ln σ²_scaled + ln σ̂²_orig,
    // so ω_orig = ω_scaled + (1−β)·ln σ̂²_orig; α/γ/β/df are scale-free
    const [omegaScaled, alpha, gamma, beta, df] = result.x;
    const omega = omegaScaled + (1 - beta) * initLogVarOrig;

    // For EGARCH, unconditional variance: E[ln(σ²)] = ω/(1-β)
    // So E[σ²] ≈ exp(ω/(1-β)) when α and γ effects average out
    const unconditionalLogVar = omega / (1 - beta);
    const unconditionalVariance = Math.exp(unconditionalLogVar);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    // Report the pure likelihood: strip the shrinkage prior evaluated at
    // the optimum (the prior deviation is scale-invariant), then add the
    // Jacobian of the rescaling: LL in data units = LL(scaled) + n·ln s
    const priorAtOptimum = 0.5 * (unconditionalLogVar - initLogVarOrig) ** 2;
    const logLikelihood = -(result.fx - priorAtOptimum) + n * Math.log(s);
    const numParams = 5;

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
  getVarianceSeries(params: EgarchParams): number[] {
    const { omega, alpha, gamma, beta, df } = params;
    const eAbsZ = df > 2 ? expectedAbsStudentT(df) : EXPECTED_ABS_NORMAL;
    const variance: number[] = [];
    let logVariance = Math.log(this.initialVariance);

    for (let i = 0; i < this.returns.length; i++) {
      if (i === 0) {
        variance.push(this.initialVariance);
      } else {
        const sigma = Math.sqrt(variance[i - 1]);
        const z = this.returns[i - 1] / sigma;
        const magnitude = this.rv
          ? Math.sqrt(this.rv[i - 1] / variance[i - 1])
          : Math.abs(z);

        logVariance = omega
          + alpha * (magnitude - eAbsZ)
          + gamma * z
          + beta * logVariance;

        logVariance = Math.max(-50, Math.min(50, logVariance));
        variance.push(Math.exp(logVariance));
      }
    }

    return variance;
  }

  /**
   * Mean drift of the magnitude term under the fitted dynamics.
   *
   * With RV magnitude, E[√(RV/σ²)] ≠ E[|z|]: the in-sample recursion
   * carries a mean offset α·m̄ per step that ω absorbed during fitting.
   * A multi-step forecast that drops the α term entirely would therefore
   * converge to a level systematically below the fitted dynamics.
   * Returns m̄ = mean(magnitude − E|z|) over the sample (0 for prices-only
   * input, where magnitude = |z| and the offset is sampling noise).
   */
  magnitudeDrift(params: EgarchParams): number {
    if (!this.rv) return 0;
    const { df } = params;
    const eAbsZ = df > 2 ? expectedAbsStudentT(df) : EXPECTED_ABS_NORMAL;
    const series = this.getVarianceSeries(params);
    let sum = 0;
    let count = 0;
    for (let i = 1; i < this.returns.length; i++) {
      const m = Math.sqrt(this.rv[i - 1] / series[i - 1]);
      if (!isFinite(m)) continue;
      sum += m - eAbsZ;
      count++;
    }
    return count > 0 ? sum / count : 0;
  }

  /**
   * Forecast variance forward
   *
   * Note: EGARCH forecasts are more complex because they depend on
   * the path of shocks. This provides an approximation assuming
   * expected values of future shocks.
   */
  forecast(params: EgarchParams, steps: number = 1): VolatilityForecast {
    const { omega, alpha, gamma, beta, df } = params;
    const eAbsZ = df > 2 ? expectedAbsStudentT(df) : EXPECTED_ABS_NORMAL;
    const variance: number[] = [];

    const varianceSeries = this.getVarianceSeries(params);
    const lastVariance = varianceSeries[varianceSeries.length - 1];
    const lastReturn = this.returns[this.returns.length - 1];

    // One-step ahead using actual last return
    const sigma = Math.sqrt(lastVariance);
    const z = lastReturn / sigma;
    const magnitude = this.rv
      ? Math.sqrt(this.rv[this.rv.length - 1] / lastVariance)
      : Math.abs(z);
    let logVariance = omega
      + alpha * (magnitude - eAbsZ)
      + gamma * z
      + beta * Math.log(lastVariance);

    logVariance = Math.max(-50, Math.min(50, logVariance));
    variance.push(Math.exp(logVariance));

    // Multi-step: assume E[z] = 0. With RV magnitude the α term has a
    // nonzero mean α·m̄ under the fitted dynamics — keep it as drift so
    // the forecast converges to the same level the fit implies.
    const drift = alpha * this.magnitudeDrift(params);
    for (let h = 1; h < steps; h++) {
      logVariance = omega + drift + beta * logVariance;
      logVariance = Math.max(-50, Math.min(50, logVariance));
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
