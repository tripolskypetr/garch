import type { Candle, HarRvParams, CalibrationResult, VolatilityForecast } from './types.js';
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

export interface HarRvOptions {
  periodsPerYear?: number;
  shortLag?: number;
  mediumLag?: number;
  longLag?: number;
}

const DEFAULT_SHORT = 1;
const DEFAULT_MEDIUM = 5;
const DEFAULT_LONG = 22;

/**
 * Solve linear system Ax = b via Gaussian elimination with partial pivoting.
 * A is n×n, b is n-vector. Returns x.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    if (Math.abs(M[col][col]) < 1e-15) {
      throw new Error('Singular matrix in HAR-RV OLS');
    }

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= M[i][j] * x[j];
    }
    x[i] /= M[i][i];
  }

  return x;
}

/**
 * OLS regression: y = Xβ + ε
 * Returns coefficients, residuals, R², RSS, TSS.
 */
function ols(X: number[][], y: number[]): {
  beta: number[];
  residuals: number[];
  rss: number;
  tss: number;
  r2: number;
} {
  const n = X.length;
  const p = X[0].length;

  // X'X
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < n; k++) {
        XtX[i][j] += X[k][i] * X[k][j];
      }
    }
  }

  // X'y
  const Xty: number[] = new Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    for (let k = 0; k < n; k++) {
      Xty[i] += X[k][i] * y[k];
    }
  }

  const beta = solveLinearSystem(XtX, Xty);

  const yMean = y.reduce((s, v) => s + v, 0) / n;
  let rss = 0;
  let tss = 0;
  const residuals: number[] = [];
  for (let i = 0; i < n; i++) {
    let yHat = 0;
    for (let j = 0; j < p; j++) {
      yHat += X[i][j] * beta[j];
    }
    const res = y[i] - yHat;
    residuals.push(res);
    rss += res * res;
    tss += (y[i] - yMean) ** 2;
  }

  const r2 = tss > 0 ? 1 - rss / tss : 0;

  return { beta, residuals, rss, tss, r2 };
}

/**
 * Compute rolling mean of rv[t-lag+1 .. t] (inclusive).
 */
function rollingMean(rv: number[], t: number, lag: number): number {
  let sum = 0;
  for (let j = 0; j < lag; j++) {
    sum += rv[t - j];
  }
  return sum / lag;
}

/**
 * HAR-RV model (Corsi, 2009)
 *
 * RV_{t+1} = β₀ + β₁·RV_short + β₂·RV_medium + β₃·RV_long + ε
 *
 * where:
 * - RV_short  = mean(rv[t-s+1..t])  (default s=1)
 * - RV_medium = mean(rv[t-m+1..t])  (default m=5)
 * - RV_long   = mean(rv[t-l+1..t])  (default l=22)
 * - rv[t] = Parkinson(candle_t) for OHLC data, r[t]² for prices-only
 *
 * Parkinson (1980): RV = (1/(4·ln2))·(ln(H/L))², ~5x more efficient than r².
 *
 * Uses OLS for estimation — closed-form, always converges.
 */
export class HarRv {
  private returns: number[];
  private rv: number[];
  private periodsPerYear: number;
  private shortLag: number;
  private mediumLag: number;
  private longLag: number;

  constructor(data: Candle[] | number[], options: HarRvOptions = {}) {
    this.periodsPerYear = options.periodsPerYear ?? 252;
    this.shortLag = options.shortLag ?? DEFAULT_SHORT;
    this.mediumLag = options.mediumLag ?? DEFAULT_MEDIUM;
    this.longLag = options.longLag ?? DEFAULT_LONG;

    const minRequired = this.longLag + 30;

    if (data.length < minRequired) {
      throw new Error(`Need at least ${minRequired} data points for HAR-RV estimation`);
    }

    if (typeof data[0] === 'number') {
      this.returns = calculateReturnsFromPrices(data as number[]);
      // Prices only — no OHLC, fall back to squared returns
      this.rv = this.returns.map(r => r * r);
    } else {
      const candles = data as Candle[];
      this.returns = calculateReturns(candles);
      // Parkinson (1980) per-candle RV: (1/(4·ln2))·(ln(H/L))²
      this.rv = perCandleParkinson(candles, this.returns);
    }
  }

  /**
   * Calibrate HAR-RV via OLS + Student-t MLE refinement.
   *
   * Stage 1: OLS on RV_{t+1} = β₀ + β₁·RV_short + β₂·RV_medium + β₃·RV_long
   * Stage 2: Nelder-Mead refines betas + df to maximize Student-t log-likelihood
   *
   * This ensures HAR-RV competes fairly with GARCH family on AIC.
   */
  fit(options: { maxIter?: number; tol?: number } = {}): CalibrationResult<HarRvParams> {
    const { maxIter = 1000, tol = 1e-8 } = options;
    const { rv, shortLag, mediumLag, longLag } = this;
    const n = rv.length;

    // Build regression data
    // Usable range: t = longLag-1 .. n-2 (need longLag history, and rv[t+1] as target)
    const startIdx = longLag - 1;
    const endIdx = n - 2;
    const nObs = endIdx - startIdx + 1;

    const X: number[][] = [];
    const y: number[] = [];

    for (let t = startIdx; t <= endIdx; t++) {
      const rvShort = rollingMean(rv, t, shortLag);
      const rvMedium = rollingMean(rv, t, mediumLag);
      const rvLong = rollingMean(rv, t, longLag);
      X.push([1, rvShort, rvMedium, rvLong]);
      y.push(rv[t + 1]);
    }

    // Stage 1: OLS for initial estimates
    const olsResult = ols(X, y);

    // Stage 2: MLE refinement — optimize Student-t LL directly
    // Use generic initial values (not OLS betas) to avoid flat LL surface
    const returns = this.returns;
    const initVar = sampleVariance(returns);

    const self = this;
    function negLogLikelihood(params: number[]): number {
      const beta = params.slice(0, 4);
      const df = params[4];
      if (df <= 2.01 || df > 100) return 1e10;
      const varSeries = self.getVarianceSeriesInternal(beta);
      return studentTNegLL(returns, varSeries, df);
    }

    const mleResult = nelderMead(
      negLogLikelihood,
      [initVar * 0.05, 0.1, 0.3, 0.5, 5],
      { maxIter, tol },
    );

    const [beta0, betaShort, betaMedium, betaLong, df] = mleResult.x;

    const persistence = betaShort + betaMedium + betaLong;
    const unconditionalVariance = persistence < 1 && persistence > -1
      ? Math.max(beta0 / (1 - persistence), 1e-20)
      : sampleVariance(this.returns);
    const annualizedVol = Math.sqrt(Math.abs(unconditionalVariance) * this.periodsPerYear) * 100;

    const ll = -mleResult.fx;
    const numParams = 5; // beta0, betaShort, betaMedium, betaLong, df

    return {
      params: {
        beta0,
        betaShort,
        betaMedium,
        betaLong,
        persistence,
        unconditionalVariance,
        annualizedVol,
        r2: olsResult.r2,
        df,
      },
      diagnostics: {
        logLikelihood: ll,
        aic: calculateAIC(ll, numParams),
        bic: calculateBIC(ll, numParams, nObs),
        iterations: mleResult.iterations + 1,
        converged: mleResult.converged,
      },
    };
  }

  /**
   * Internal: compute variance series from beta vector.
   */
  private getVarianceSeriesInternal(beta: number[]): number[] {
    const { rv, shortLag, mediumLag, longLag } = this;
    const n = rv.length;
    const fallback = sampleVariance(this.returns);
    const series: number[] = [];

    for (let i = 0; i < n; i++) {
      if (i < longLag) {
        // Not enough history — use sample variance
        series.push(fallback);
      } else {
        // HAR prediction for rv[i] based on rv[..i-1]
        const t = i - 1;
        const rvS = rollingMean(rv, t, shortLag);
        const rvM = rollingMean(rv, t, mediumLag);
        const rvL = rollingMean(rv, t, longLag);
        const predicted = beta[0] + beta[1] * rvS + beta[2] * rvM + beta[3] * rvL;
        series.push(Math.max(predicted, 1e-20));
      }
    }

    return series;
  }

  /**
   * Calculate conditional variance series given parameters.
   */
  getVarianceSeries(params: HarRvParams): number[] {
    const beta = [params.beta0, params.betaShort, params.betaMedium, params.betaLong];
    return this.getVarianceSeriesInternal(beta);
  }

  /**
   * Forecast variance forward.
   *
   * Uses iterative substitution: each forecast step feeds back
   * into the rolling RV components for subsequent steps.
   */
  forecast(params: HarRvParams, steps: number = 1): VolatilityForecast {
    const { rv, shortLag, mediumLag, longLag } = this;
    const { beta0, betaShort, betaMedium, betaLong } = params;

    // Working copy of recent rv values + forecasts appended
    const history = rv.slice();
    const variance: number[] = [];

    for (let h = 0; h < steps; h++) {
      const t = history.length - 1;
      const rvS = rollingMean(history, t, shortLag);
      const rvM = rollingMean(history, t, mediumLag);
      const rvL = rollingMean(history, t, longLag);
      const predicted = beta0 + betaShort * rvS + betaMedium * rvM + betaLong * rvL;
      const v = Math.max(predicted, 1e-20);
      variance.push(v);
      history.push(v);
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
   * Get realized variance series (squared returns).
   */
  getRv(): number[] {
    return [...this.rv];
  }
}

/**
 * Convenience function to calibrate HAR-RV from candles or prices.
 */
export function calibrateHarRv(
  data: Candle[] | number[],
  options: HarRvOptions & { maxIter?: number; tol?: number } = {},
): CalibrationResult<HarRvParams> {
  const model = new HarRv(data, options);
  return model.fit(options);
}
