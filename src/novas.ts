import type { Candle, NoVaSParams, CalibrationResult, VolatilityForecast } from './types.js';
import { nelderMead, nelderMeadMultiStart } from './optimizer.js';
import {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  perCandleParkinson,
  calculateAIC,
  calculateBIC,
  studentTNegLL,
  profileStudentTDf,
  validateCandles,
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
 * Two-stage calibration:
 *
 * Stage 1 — D² minimization (model-free normality):
 *   σ²_t = a_0 + a_1·X²_{t-1} + a_2·X²_{t-2} + ... + a_p·X²_{t-p}
 *   W_t  = X_t / σ_t
 *   Minimize D² = S² + (K - 3)² where S, K are skewness and kurtosis of {W_t}.
 *
 * Stage 2 — OLS rescaling (forecast-optimal):
 *   RV_{t+1} = β₀ + β₁·σ²_t(D²)
 *   The D²-discovered σ²_t acts as a data-driven smoother over RV lags.
 *   OLS rescales it to minimize forecast error (RSS on RV).
 *   Only 2 parameters → robust on small samples with noisy per-candle RV.
 *
 * D² discovers lag structure (model-free). OLS rescales for prediction accuracy.
 * Both weight sets are stored in params — no identity loss.
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
      validateCandles(candles);
      this.returns = calculateReturns(candles);
      // Parkinson (1980) per-candle RV: ~5× more efficient than r²
      this.rv = perCandleParkinson(candles, this.returns);
    }
  }

  /**
   * Calibrate NoVaS weights via two-stage procedure:
   * Stage 1: D² minimization (normality of W_t)
   * Stage 2: OLS rescaling of D²-variance (forecast-optimal)
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

    // Screening: the multi-start perturbation scales x0 multiplicatively and
    // preserves its exponential-decay shape, so far-lag weight structures are
    // unreachable from x0 alone (measured: 5.6× worse D² on two-spike ARCH
    // ground truth). D² costs one pass to evaluate, so scan a deterministic
    // low-discrepancy cloud of sparse weight shapes and hand the best few to
    // Nelder-Mead as explicit extra starts.
    const screened: Array<{ d2: number; w: number[] }> = [];
    {
      // Kronecker sequence: one irrational stride per dimension
      const strides: number[] = [];
      for (let i = 0, prime = 2; i <= p + 1; prime++) {
        let isPrime = true;
        for (let q = 2; q * q <= prime; q++) if (prime % q === 0) { isPrime = false; break; }
        if (!isPrime) continue;
        strides.push(Math.sqrt(prime) % 1);
        i++;
      }
      for (let s = 1; s <= 384; s++) {
        // a0 log-uniform in [0.01, 1]·initVar
        const a0 = initVar * Math.pow(10, ((s * strides[0]) % 1) * 2 - 2);
        const raw: number[] = [];
        let rawSum = 0;
        for (let j = 1; j <= p; j++) {
          // ~half the lags exactly zero — spikes and gaps are in the span
          const u = (s * strides[j]) % 1;
          const v = Math.max(0, u * 2 - 1);
          raw.push(v);
          rawSum += v;
        }
        if (rawSum <= 0) continue;
        const target = 0.1 + 0.85 * ((s * strides[p + 1]) % 1);
        const w = [a0, ...raw.map(v => (v * target) / rawSum)];
        screened.push({ d2: objectiveD2(w), w });
      }
      screened.sort((a, b) => a.d2 - b.d2);
    }
    const extraStarts = screened.slice(0, 3).map(c => c.w);

    // Stage-2 OLS (RV_t ~ β₀ + β₁·σ²_t) for a given weight vector.
    // σ²_t is built from r2[t-1..t-p], so it already IS the one-step-ahead
    // prediction of rv[t] — pairing it with r2[t+1] (as before) calibrated
    // β one bar off from how getForecastVarianceSeries and forecast use it.
    // Used both to pick among D² candidates and for the final rescaling.
    const stage2 = (w: number[]): { beta0: number; beta1: number; rss: number; r2: number } | null => {
      const dv = this.getVarianceSeriesInternal(w);
      let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0, count = 0;
      for (let t = p; t < n; t++) {
        const x = dv[t];
        const y = r2[t];
        sumX += x;
        sumY += y;
        sumXX += x * x;
        sumXY += x * y;
        count++;
      }
      const denom = count * sumXX - sumX * sumX;
      if (Math.abs(denom) < 1e-30) return null;
      const beta1 = (count * sumXY - sumX * sumY) / denom;
      const beta0 = (sumY - beta1 * sumX) / count;
      const yMean = sumY / count;
      let rss = 0, tss = 0;
      for (let t = p; t < n; t++) {
        const yHat = beta0 + beta1 * dv[t];
        rss += (r2[t] - yHat) ** 2;
        tss += (r2[t] - yMean) ** 2;
      }
      return { beta0, beta1, rss, r2: tss > 0 ? 1 - rss / tss : 0 };
    };

    // D² is underdetermined: two moment conditions (skewness, kurtosis)
    // constrain p+1 weights, so its minimum is a manifold and near-zero D²
    // differences are sampling noise — sd of (K−3)² at n ≈ 500 is ~0.05.
    // Run NM from the exp-decay seed and from each screened start, then
    // among candidates within that noise floor of the best D² pick the best
    // RV forecaster: normality identifies the set, prediction picks the point.
    const D2_NOISE = 0.05;
    const candidateRuns = [
      nelderMeadMultiStart(objectiveD2, x0, { maxIter, tol, restarts: 6 }),
      ...extraStarts.map(s => nelderMead(objectiveD2, s, { maxIter, tol })),
    ];
    const bestD2 = Math.min(...candidateRuns.map(r => r.fx));
    let result = candidateRuns[0];
    let bestRss = Infinity;
    for (const run of candidateRuns) {
      if (run.fx > bestD2 + D2_NOISE) continue;
      const s2 = stage2(run.x.map(Math.abs));
      const rss = s2 ? s2.rss : Infinity;
      if (rss < bestRss || (rss === bestRss && run.fx < result.fx)) {
        bestRss = rss;
        result = run;
      }
    }

    // Extract final weights (abs for constraint enforcement)
    const weights = result.x.map(w => Math.abs(w));

    let persistence = 0;
    for (let j = 1; j <= p; j++) persistence += weights[j];

    const unconditionalVariance = persistence < 1 && persistence > -1
      ? Math.max(weights[0] / (1 - persistence), 1e-20)
      : sampleVariance(returns);
    const annualizedVol = Math.sqrt(unconditionalVariance * this.periodsPerYear) * 100;

    // ── Stage 2: OLS rescaling of D²-variance ──────────────
    // RV_{t+1} = β₀ + β₁·σ²_t(D²)
    // D² weights discover lag structure; OLS rescales for forecast accuracy.
    // Only 2 parameters → robust on small samples with noisy per-candle RV.
    const d2Variance = this.getVarianceSeriesInternal(weights);
    const s2Final = stage2(weights);
    // OLS failed (degenerate variance series) — fall back to identity rescaling
    const forecastWeights = s2Final ? [s2Final.beta0, s2Final.beta1] : [0, 1];
    const olsR2 = s2Final ? s2Final.r2 : 0;

    // Student-t log-likelihood for AIC comparison with GARCH/EGARCH/HAR-RV
    const df = profileStudentTDf(returns, d2Variance);
    const ll = -studentTNegLL(returns, d2Variance, df);

    const numParams = p + 2; // weights + df
    const nObs = n - p; // usable observations for D²

    return {
      params: {
        weights,
        forecastWeights,
        lags: p,
        persistence,
        unconditionalVariance,
        annualizedVol,
        dSquared: result.fx,
        r2: olsR2,
        df,
      },
      diagnostics: {
        logLikelihood: ll,
        aic: calculateAIC(ll, numParams),
        bic: calculateBIC(ll, numParams, nObs),
        iterations: result.iterations,
        converged: result.converged,
      },
    };
  }

  /**
   * Internal: compute variance series from D² weight vector.
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
   * Calculate conditional variance series using D² weights (normalization identity).
   */
  getVarianceSeries(params: NoVaSParams): number[] {
    return this.getVarianceSeriesInternal(params.weights);
  }

  /**
   * Calculate forecast variance series using OLS-rescaled D² variance.
   * forecast_σ²_t = β₀ + β₁·σ²_t(D²)
   * Used for QLIKE model comparison — measures forecast quality.
   */
  getForecastVarianceSeries(params: NoVaSParams): number[] {
    const d2Series = this.getVarianceSeriesInternal(params.weights);
    const [beta0, beta1] = params.forecastWeights;
    return d2Series.map(v => Math.max(beta0 + beta1 * v, 1e-20));
  }

  /**
   * Forecast variance forward using OLS-rescaled D² weights.
   *
   * Step 1: compute D²-based σ²_{t+h} using D² weights
   * Step 2: rescale via β₀ + β₁·σ²_{t+h}
   */
  forecast(params: NoVaSParams, steps: number = 1): VolatilityForecast {
    const { weights, forecastWeights, lags } = params;
    const [beta0, beta1] = forecastWeights;
    const r2 = this.rv ?? this.returns.map(r => r * r);

    // Working buffer: past innovation values + forecasted variances
    const history = r2.slice();
    const variance: number[] = [];

    for (let h = 0; h < steps; h++) {
      const t = history.length;
      // D²-based variance at this step
      let d2v = weights[0];
      for (let j = 1; j <= lags; j++) {
        d2v += weights[j] * history[t - j];
      }
      d2v = Math.max(d2v, 1e-20);

      // OLS-rescaled forecast
      const v = Math.max(beta0 + beta1 * d2v, 1e-20);
      variance.push(v);
      history.push(v); // future E[RV] = σ²
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
