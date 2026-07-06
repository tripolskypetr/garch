import type { Candle, RealizedGarchParams, CalibrationResult, VolatilityForecast } from './types.js';
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

export interface RealizedGarchOptions {
  periodsPerYear?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * Realized GARCH(1,1) (Hansen, Huang & Shek, 2012), log-linear, φ = 1.
 *
 *   r_t = σ_t·z_t,                     z_t ~ standardized t(df)
 *   ln σ²_t = ω + β·ln σ²_{t−1} + γ·ln RV_{t−1}
 *   ln RV_t = ξ + ln σ²_t + τ₁·z_t + τ₂·(z²_t − 1) + u_t,   u_t ~ N(0, σ²_u)
 *
 * Unlike the RV-in-place-of-ε² hybrids, the measurement equation estimates
 * the bias (ξ) and noise (σ_u) of the realized measure inside the joint
 * likelihood: RV information is weighted by how trustworthy it actually is,
 * and leverage enters through τ₁. Stationarity: β + γ < 1 (with φ = 1).
 */
export class RealizedGarch {
  private returns: number[];
  private lnRv: number[];
  private periodsPerYear: number;
  private initialVariance: number;

  constructor(data: Candle[] | number[], options: RealizedGarchOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;

    if (data.length < 50) {
      throw new Error('Need at least 50 data points for Realized GARCH estimation');
    }

    let rv: number[];
    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
      this.initialVariance = Math.max(sampleVariance(this.returns), 1e-300);
      // Prices only — squared returns as the (noisy) realized measure
      rv = this.returns.map(r => r * r);
    } else {
      const candles = data as Candle[];
      validateCandles(candles);
      this.returns = calculateReturns(candles);
      this.initialVariance = Math.max(yangZhangVariance(candles), 1e-300);
      rv = perCandleParkinson(candles, this.returns);
    }

    // ln RV with a floor at half the smallest positive observation: the
    // measurement equation lives in logs and a literal zero (flat candle)
    // must not inject −Infinity.
    let minPos = Infinity;
    for (const v of rv) {
      if (v > 0 && v < minPos) minPos = v;
    }
    if (!isFinite(minPos)) {
      throw new Error('Realized GARCH needs at least one positive realized-variance observation');
    }
    const floor = minPos * 0.5;
    this.lnRv = rv.map(v => Math.log(Math.max(v, floor)));
  }

  /**
   * Calibrate by joint MLE over returns and the realized measure.
   */
  fit(
    options: { maxIter?: number; tol?: number; forgetting?: number; warmStart?: RealizedGarchParams } = {},
  ): CalibrationResult<RealizedGarchParams> {
    const { maxIter = 1000, tol = 1e-8, forgetting = 1 } = options;
    const n = this.returns.length;
    const initLogVarOrig = Math.log(this.initialVariance);

    // Normalized space: returns to unit initial variance, ln RV shifted
    // accordingly — ξ, τ, σ_u, df are scale-free; ω is mapped back below.
    const s2 = 1 / this.initialVariance;
    const s = Math.sqrt(s2);
    const returns = this.returns.map(r => r * s);
    const lnRv = this.lnRv.map(v => v + Math.log(s2));

    // Exponential forgetting: w_t = λ^(n−1−t), newest observation weight 1
    const weights = new Array(n).fill(1);
    if (forgetting < 1) {
      for (let t = 0; t < n; t++) weights[t] = Math.pow(forgetting, n - 1 - t);
    }
    const wTotal = weights.reduce((a, b) => a + b, 0);

    const LOG_2PI = Math.log(2 * Math.PI);

    function negLogLikelihood(params: number[]): number {
      const [omega, beta, gamma, xi, tau1, tau2, lnSigmaU, df] = params;

      if (beta < 0 || gamma < 0) return 1e10;
      if (beta + gamma >= 0.9999) return 1e10;
      if (df <= 2.01 || df > 100) return 1e10;
      if (Math.abs(lnSigmaU) > 5) return 1e10;

      // Same flat-ridge treatment as EGARCH: hard wall on the implied
      // unconditional level plus a weak Gaussian prior toward the sample
      // variance (scaled space: ln σ̂² = 0)
      const impliedLogVar = (omega + gamma * xi) / (1 - beta - gamma);
      if (!isFinite(impliedLogVar) || Math.abs(impliedLogVar) > Math.log(1e4)) return 1e10;
      const prior = 0.5 * impliedLogVar * impliedLogVar;

      const sigmaU2 = Math.exp(2 * lnSigmaU);
      const halfDfPlus1 = (df + 1) / 2;
      const dfMinus2 = df - 2;
      const constant = wTotal * (
        logGamma(halfDfPlus1) - logGamma(df / 2) - 0.5 * Math.log(Math.PI * dfMinus2)
        - 0.5 * (LOG_2PI + 2 * lnSigmaU)
      );

      let logVariance = 0; // scaled initial variance = 1
      let ll = 0;

      for (let i = 0; i < n; i++) {
        if (i > 0) {
          logVariance = omega + beta * logVariance + gamma * lnRv[i - 1];
          logVariance = Math.max(-50, Math.min(50, logVariance));
        }
        const variance = Math.exp(logVariance);
        const z = returns[i] / Math.sqrt(variance);

        // Return likelihood (Student-t) + measurement likelihood (Gaussian)
        const u = lnRv[i] - xi - logVariance - tau1 * z - tau2 * (z * z - 1);
        ll += weights[i] * (
          -0.5 * logVariance - halfDfPlus1 * Math.log(1 + (z * z) / dfMinus2)
          - 0.5 * (u * u) / sigmaU2
        );
        if (!isFinite(ll)) return 1e10;
      }

      return -(ll + constant) + prior;
    }

    // Initial guesses: variance targeting in scaled log space (level 0)
    const beta0 = 0.55;
    const gamma0 = 0.4;
    const xi0 = lnRv.reduce((a, b) => a + b, 0) / n; // E[ln RV] − E[ln σ²] with E[ln σ²]=0
    const omega0 = -gamma0 * xi0;
    const x0 = [omega0, beta0, gamma0, xi0, -0.05, 0.1, Math.log(0.5), 5];

    // Warm start (previous window's optimum) replaces the cold seed with a
    // reduced restart budget; the level wall moves with the sample, so an
    // out-of-wall warm seed falls back to the cold start.
    let warmX0: number[] | null = null;
    const wp = options.warmStart;
    if (wp && isFinite(wp.omega) && isFinite(wp.beta) && isFinite(wp.gamma) && wp.beta + wp.gamma < 1) {
      const cand = [
        wp.omega - (1 - wp.beta - wp.gamma) * initLogVarOrig,
        wp.beta,
        wp.gamma,
        wp.xi,
        wp.tau1,
        wp.tau2,
        Math.log(Math.max(wp.sigmaU, 1e-4)),
        wp.df,
      ];
      const implied = (cand[0] + wp.gamma * wp.xi) / (1 - wp.beta - wp.gamma);
      if (Math.abs(implied) <= Math.log(1e4)) warmX0 = cand;
    }

    // Cold restarts kept low: the variance-targeted seed plus the level
    // prior already resolve the ω/β/γ ridge, and the adaptive multi-start
    // extends the budget on its own when restarts keep improving.
    const result = nelderMeadMultiStart(negLogLikelihood, warmX0 ?? x0, {
      maxIter,
      tol,
      restarts: warmX0 ? 1 : 2,
    });

    // Map back: ln σ²_orig = ln σ²_scaled + ln σ̂²_orig, ξ is invariant, so
    // ω_orig = ω_scaled + (1 − β − γ)·ln σ̂²_orig
    const [omegaScaled, beta, gamma, xi, tau1, tau2, lnSigmaU, df] = result.x;
    const omega = omegaScaled + (1 - beta - gamma) * initLogVarOrig;
    const sigmaU = Math.exp(lnSigmaU);

    const persistence = beta + gamma;
    const unconditionalLogVar = (omega + gamma * xi) / (1 - persistence);
    const unconditionalVariance = Math.exp(unconditionalLogVar);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    // Strip the shrinkage prior (deviation is scale-invariant) and add the
    // returns Jacobian; the ln RV measurement density is shift-invariant.
    const priorAtOptimum = 0.5 * (unconditionalLogVar - initLogVarOrig) ** 2;
    const logLikelihood = -(result.fx - priorAtOptimum) + n * Math.log(s);
    const numParams = 8;

    return {
      params: {
        omega,
        beta,
        gamma,
        xi,
        tau1,
        tau2,
        sigmaU,
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
   * Conditional variance series (data scale). σ²_t is driven by RV_{t−1}
   * through the log recursion — no look-ahead.
   */
  getVarianceSeries(params: RealizedGarchParams): number[] {
    const { omega, beta, gamma } = params;
    const variance: number[] = [];
    let logVariance = Math.log(this.initialVariance);

    for (let i = 0; i < this.returns.length; i++) {
      if (i > 0) {
        logVariance = omega + beta * logVariance + gamma * this.lnRv[i - 1];
        logVariance = Math.max(-720, Math.min(50, logVariance));
      }
      variance.push(Math.exp(logVariance));
    }

    return variance;
  }

  /**
   * Forecast variance forward. One step uses the actual last RV; further
   * steps substitute E[ln RV_t] = ξ + ln σ²_t, giving the reduced recursion
   * ln σ²_{t+h} = (ω + γξ) + (β + γ)·ln σ²_{t+h−1}.
   */
  forecast(params: RealizedGarchParams, steps: number = 1): VolatilityForecast {
    const { omega, beta, gamma, xi } = params;
    const series = this.getVarianceSeries(params);
    const variance: number[] = [];

    let logVariance = omega
      + beta * Math.log(series[series.length - 1])
      + gamma * this.lnRv[this.lnRv.length - 1];
    logVariance = Math.max(-720, Math.min(50, logVariance));
    variance.push(Math.exp(logVariance));

    for (let h = 1; h < steps; h++) {
      logVariance = omega + gamma * xi + (beta + gamma) * logVariance;
      logVariance = Math.max(-720, Math.min(50, logVariance));
      variance.push(Math.exp(logVariance));
    }

    return {
      variance,
      volatility: variance.map(v => Math.sqrt(v)),
      annualized: variance.map(v => Math.sqrt(v * this.periodsPerYear) * 100),
    };
  }

  /**
   * Get the return series.
   */
  getReturns(): number[] {
    return [...this.returns];
  }

  /**
   * Get initial variance estimate.
   */
  getInitialVariance(): number {
    return this.initialVariance;
  }
}

/**
 * Convenience function to calibrate Realized GARCH from candles or prices.
 */
export function calibrateRealizedGarch(
  data: Candle[] | number[],
  options: RealizedGarchOptions = {},
): CalibrationResult<RealizedGarchParams> {
  const model = new RealizedGarch(data, options);
  return model.fit(options);
}
