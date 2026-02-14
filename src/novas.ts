import type { Candle, NoVaSParams, CalibrationResult, VolatilityForecast } from './types.js';
import { nelderMead } from './optimizer.js';
import {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  perCandleParkinson,
  calculateAIC,
  calculateBIC,
  studentTNegLL,
  profileStudentTDf,
} from './utils.js';

export interface NoVaSOptions {
  periodsPerYear?: number;
  lags?: number;
  maxIter?: number;
  tol?: number;
}

const DEFAULT_LAGS = 10;

/**
 * NoVaS (Normalizing and Variance-Stabilizing) model (Politis, 2003)
 *
 * Model-free volatility prediction using the ARCH frame:
 *
 *   σ²_t = a_0 + a_1·X²_{t-1} + a_2·X²_{t-2} + ... + a_p·X²_{t-p}
 *   W_t  = X_t / σ_t
 *
 * Parameters a_0, ..., a_p are chosen to minimize the non-normality
 * of the transformed series {W_t}:
 *
 *   D² = S² + (K - 3)²
 *
 * where S = skewness and K = kurtosis of {W_t}.
 *
 * Unlike GARCH (MLE), NoVaS is model-free — no distributional assumptions.
 * Constraints: a_0 > 0, a_j >= 0, sum(a_1..a_p) < 1 (stationarity).
 */
export class NoVaS {
  private returns: number[];
  private rv: number[] | null;
  private periodsPerYear: number;
  private lags: number;

  constructor(data: Candle[] | number[], options: NoVaSOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;
    this.lags = options.lags ?? DEFAULT_LAGS;

    const minRequired = this.lags + 30;

    if (data.length < minRequired) {
      throw new Error(`Need at least ${minRequired} data points for NoVaS estimation`);
    }

    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
      this.rv = null;
    } else {
      const candles = data as Candle[];
      this.returns = calculateReturns(candles);
      // Parkinson (1980) per-candle RV: ~5× more efficient than r²
      this.rv = perCandleParkinson(candles, this.returns);
    }
  }

  /**
   * Calibrate NoVaS weights by minimizing D² = S² + (K-3)²
   * of the transformed series W_t = X_t / σ_t.
   */
  fit(options: { maxIter?: number; tol?: number } = {}): CalibrationResult<NoVaSParams> {
    const { maxIter = 2000, tol = 1e-8 } = options;
    const returns = this.returns;
    const n = returns.length;
    const p = this.lags;
    const initVar = sampleVariance(returns);

    // Innovation: Parkinson RV for candles, r² for prices
    const r2 = this.rv ?? returns.map(r => r * r);

    /**
     * Compute D² for a given weight vector.
     * D² = S² + (K - 3)² where S, K are skewness and kurtosis of W_t.
     */
    function objectiveD2(rawWeights: number[]): number {
      // Enforce constraints: a_j >= 0 via abs, a_0 > epsilon
      const weights = rawWeights.map(w => Math.abs(w));
      if (weights[0] < 1e-15) return 1e10;

      // Stationarity: sum(a_1..a_p) < 1
      let lagSum = 0;
      for (let j = 1; j <= p; j++) lagSum += weights[j];
      if (lagSum >= 0.9999) return 1e10;

      // Compute transformed series W_t = r_t / sqrt(sigma^2_t)
      let sumW = 0;
      let sumW2 = 0;
      let sumW3 = 0;
      let sumW4 = 0;
      let count = 0;

      for (let t = p; t < n; t++) {
        let variance = weights[0];
        for (let j = 1; j <= p; j++) {
          variance += weights[j] * r2[t - j];
        }
        if (variance <= 1e-15) return 1e10;

        const w = returns[t] / Math.sqrt(variance);
        if (!isFinite(w)) return 1e10;

        sumW += w;
        sumW2 += w * w;
        sumW3 += w * w * w;
        sumW4 += w * w * w * w;
        count++;
      }

      if (count < 10) return 1e10;

      const mean = sumW / count;
      const m2 = sumW2 / count - mean * mean;
      if (m2 <= 1e-15) return 1e10;

      const m3 = sumW3 / count - 3 * mean * sumW2 / count + 2 * mean * mean * mean;
      const m4 = sumW4 / count - 4 * mean * sumW3 / count
        + 6 * mean * mean * sumW2 / count - 3 * mean * mean * mean * mean;

      const skewness = m3 / (m2 * Math.sqrt(m2));
      const kurtosis = m4 / (m2 * m2);

      if (!isFinite(skewness) || !isFinite(kurtosis)) return 1e10;

      return skewness * skewness + (kurtosis - 3) * (kurtosis - 3);
    }

    // Initial guess: intercept in variance units, lag weights dimensionless
    const lambda = 0.7;
    const x0: number[] = [initVar * 0.1];
    for (let j = 1; j <= p; j++) {
      x0.push(0.9 * (1 - lambda) * Math.pow(lambda, j - 1));
    }

    const d2Result = nelderMead(objectiveD2, x0, { maxIter, tol });
    const d2Weights = d2Result.x.map(w => Math.abs(w));
    const dSquared = d2Result.fx;

    // Stage 2: MLE refinement — optimize Student-t LL directly
    // Use generic initial values (not D² weights) for better LL starting point
    const self = this;
    function negLogLikelihood(params: number[]): number {
      const w = params.slice(0, p + 1).map(v => Math.abs(v));
      const df = params[p + 1];
      if (w[0] < 1e-15) return 1e10;
      if (df <= 2.01 || df > 100) return 1e10;

      let lagSum = 0;
      for (let j = 1; j <= p; j++) lagSum += w[j];
      if (lagSum >= 0.9999) return 1e10;

      const varSeries = self.getVarianceSeriesInternal(w);
      return studentTNegLL(returns, varSeries, df);
    }

    const mleX0: number[] = [initVar * 0.1];
    for (let j = 1; j <= p; j++) {
      mleX0.push(0.9 * (1 - lambda) * Math.pow(lambda, j - 1));
    }
    mleX0.push(5); // df

    const mleResult = nelderMead(
      negLogLikelihood,
      mleX0,
      { maxIter, tol },
    );

    const weights = mleResult.x.slice(0, p + 1).map(w => Math.abs(w));
    const df = Math.max(2.01, Math.min(100, mleResult.x[p + 1]));

    let persistence = 0;
    for (let j = 1; j <= p; j++) persistence += weights[j];

    const unconditionalVariance = persistence < 1 && persistence > -1
      ? Math.max(weights[0] / (1 - persistence), 1e-20)
      : sampleVariance(returns);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    const ll = -mleResult.fx;
    const numParams = p + 2; // weights + df
    const nObs = n - p; // usable observations

    return {
      params: {
        weights,
        lags: p,
        persistence,
        unconditionalVariance,
        annualizedVol,
        dSquared,
        df,
      },
      diagnostics: {
        logLikelihood: ll,
        aic: calculateAIC(ll, numParams),
        bic: calculateBIC(ll, numParams, nObs),
        iterations: d2Result.iterations + mleResult.iterations,
        converged: d2Result.converged, // D² is primary objective; MLE refines for AIC
      },
    };
  }

  /**
   * Internal: compute variance series from weight vector.
   */
  private getVarianceSeriesInternal(weights: number[]): number[] {
    const { returns, lags } = this;
    const n = returns.length;
    const r2 = this.rv ?? returns.map(r => r * r);
    const fallback = sampleVariance(returns);
    const series: number[] = [];

    for (let t = 0; t < n; t++) {
      if (t < lags) {
        series.push(fallback);
      } else {
        let variance = weights[0];
        for (let j = 1; j <= lags; j++) {
          variance += weights[j] * r2[t - j];
        }
        series.push(Math.max(variance, 1e-20));
      }
    }

    return series;
  }

  /**
   * Calculate conditional variance series given parameters.
   */
  getVarianceSeries(params: NoVaSParams): number[] {
    return this.getVarianceSeriesInternal(params.weights);
  }

  /**
   * Forecast variance forward.
   *
   * One-step: σ²_{t+1} = a_0 + Σ a_j · X²_{t+1-j}
   * Multi-step: replace future X² with σ² (E[X²] = σ²).
   */
  forecast(params: NoVaSParams, steps: number = 1): VolatilityForecast {
    const { weights, lags } = params;
    const r2 = this.rv ?? this.returns.map(r => r * r);

    // Working buffer: past innovation values + forecasted variances
    const history = r2.slice();
    const variance: number[] = [];

    for (let h = 0; h < steps; h++) {
      const t = history.length;
      let v = weights[0];
      for (let j = 1; j <= lags; j++) {
        v += weights[j] * history[t - j];
      }
      v = Math.max(v, 1e-20);
      variance.push(v);
      history.push(v); // future E[X²] = σ²
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
}

/**
 * Convenience function to calibrate NoVaS from candles or prices.
 */
export function calibrateNoVaS(
  data: Candle[] | number[],
  options: NoVaSOptions = {},
): CalibrationResult<NoVaSParams> {
  const model = new NoVaS(data, options);
  return model.fit(options);
}
