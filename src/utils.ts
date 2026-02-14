import type { Candle, LeverageStats } from './types.js';

/**
 * Calculate log returns from candles
 */
export function calculateReturns(candles: Candle[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    if (!(candles[i].close > 0) || !(candles[i - 1].close > 0)) {
      throw new Error(`Invalid close price at index ${i}`);
    }
    returns.push(Math.log(candles[i].close / candles[i - 1].close));
  }
  return returns;
}

/**
 * Calculate log returns from price array
 */
export function calculateReturnsFromPrices(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    if (!(prices[i] > 0 && Number.isFinite(prices[i])) || !(prices[i - 1] > 0 && Number.isFinite(prices[i - 1]))) {
      throw new Error(`Invalid price at index ${i}`);
    }
    returns.push(Math.log(prices[i] / prices[i - 1]));
  }
  return returns;
}

/**
 * Calculate sample variance (mean-zero assumption)
 */
export function sampleVariance(returns: number[]): number {
  return returns.reduce((sum, r) => sum + r * r, 0) / returns.length;
}

/**
 * Calculate sample variance with mean adjustment
 */
export function sampleVarianceWithMean(returns: number[]): number {
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  return returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
}

/**
 * Check for leverage effect (asymmetry in volatility)
 */
export function checkLeverageEffect(returns: number[]): LeverageStats {
  const negative = returns.filter(r => r < 0);
  const positive = returns.filter(r => r > 0);

  if (negative.length === 0 || positive.length === 0) {
    return {
      negativeVol: 0,
      positiveVol: 0,
      ratio: 1,
      recommendation: 'garch',
    };
  }

  const negativeVol = Math.sqrt(negative.reduce((s, r) => s + r * r, 0) / negative.length);
  const positiveVol = Math.sqrt(positive.reduce((s, r) => s + r * r, 0) / positive.length);
  const ratio = negativeVol / positiveVol;

  return {
    negativeVol,
    positiveVol,
    ratio,
    recommendation: ratio > 1.2 ? 'egarch' : 'garch',
  };
}

/**
 * Garman-Klass (1980) variance estimator using OHLC data.
 *
 * σ²_GK = (1/n) Σ [ 0.5·(ln(H/L))² − (2ln2−1)·(ln(C/O))² ]
 *
 * ~5x more efficient than close-to-close variance.
 */
export function garmanKlassVariance(candles: Candle[]): number {
  const n = candles.length;
  const coeff = 2 * Math.LN2 - 1;
  let sum = 0;

  for (let i = 0; i < n; i++) {
    const { open, high, low, close } = candles[i];
    const hl = Math.log(high / low);
    const co = Math.log(close / open);
    sum += 0.5 * hl * hl - coeff * co * co;
  }

  return sum / n;
}

/**
 * Yang-Zhang (2000) variance estimator using OHLC data.
 *
 * Combines overnight (open vs prev close), open-to-close,
 * and Rogers-Satchell components. More efficient than Garman-Klass
 * and handles overnight gaps (relevant for stocks).
 *
 * σ²_YZ = σ²_overnight + k·σ²_close + (1−k)·σ²_RS
 */
export function yangZhangVariance(candles: Candle[]): number {
  const n = candles.length;
  if (n < 2) return garmanKlassVariance(candles);

  const k = 0.34 / (1.34 + (n + 1) / (n - 1));

  let overnightSum = 0;
  let closeSum = 0;
  let rsSum = 0;
  let count = 0;

  for (let i = 1; i < n; i++) {
    const prevClose = candles[i - 1].close;
    const { open, high, low, close } = candles[i];

    const overnight = Math.log(open / prevClose);
    const co = Math.log(close / open);
    const hc = Math.log(high / close);
    const ho = Math.log(high / open);
    const lc = Math.log(low / close);
    const lo = Math.log(low / open);

    overnightSum += overnight * overnight;
    closeSum += co * co;
    rsSum += ho * hc + lo * lc;
    count++;
  }

  const overnightVar = overnightSum / count;
  const closeVar = closeSum / count;
  const rsVar = rsSum / count;

  return overnightVar + k * closeVar + (1 - k) * rsVar;
}

/**
 * Per-candle Parkinson (1980) realized variance proxy.
 *
 * RV_i = (1/(4·ln2)) · ln(H/L)²
 *
 * ~5× more efficient than squared returns. Falls back to r² when H === L.
 * rv[i] aligned with returns[i], using candles[i+1]'s OHLC.
 */
export function perCandleParkinson(candles: Candle[], returns: number[]): number[] {
  const coeff = 1 / (4 * Math.LN2);
  const rv: number[] = [];
  for (let i = 0; i < returns.length; i++) {
    const c = candles[i + 1];
    const hl = Math.log(c.high / c.low);
    const parkinson = coeff * hl * hl;
    // Fall back to r² if high === low (zero range)
    rv.push(parkinson > 0 ? parkinson : returns[i] * returns[i]);
  }
  return rv;
}

/**
 * Expected value of |Z| where Z ~ N(0,1)
 * E[|Z|] = sqrt(2/π)
 */
export const EXPECTED_ABS_NORMAL = Math.sqrt(2 / Math.PI);

/**
 * Chi-squared survival function approximation (Wilson-Hilferty).
 * P(X > x) where X ~ χ²(df)
 */
export function chi2Survival(x: number, df: number): number {
  if (df <= 0 || x < 0) return 1;
  // Wilson-Hilferty normal approximation
  const z = Math.cbrt(x / df) - (1 - 2 / (9 * df));
  const denom = Math.sqrt(2 / (9 * df));
  const normZ = z / denom;
  // Standard normal CDF via error function approximation
  return 1 - normalCDF(normZ);
}

function normalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327; // 1/sqrt(2π)
  const p = d * Math.exp(-0.5 * x * x) *
    (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))));
  return x >= 0 ? 1 - p : p;
}

/**
 * Ljung-Box test for autocorrelation.
 *
 * Q = n(n+2) Σ(k=1..m) ρ²_k / (n−k)
 *
 * Under H₀ (no autocorrelation), Q ~ χ²(m).
 * Use on squared standardized residuals to test GARCH adequacy.
 */
export function ljungBox(data: number[], maxLag: number): { statistic: number; pValue: number } {
  const n = data.length;
  const mean = data.reduce((s, v) => s + v, 0) / n;
  const gamma0 = data.reduce((s, v) => s + (v - mean) ** 2, 0) / n;

  if (gamma0 === 0) return { statistic: 0, pValue: 1 };

  let Q = 0;
  for (let k = 1; k <= maxLag; k++) {
    let gammaK = 0;
    for (let t = k; t < n; t++) {
      gammaK += (data[t] - mean) * (data[t - k] - mean);
    }
    gammaK /= n;
    const rhoK = gammaK / gamma0;
    Q += (rhoK * rhoK) / (n - k);
  }
  Q *= n * (n + 2);

  return { statistic: Q, pValue: chi2Survival(Q, maxLag) };
}

// ── Student-t distribution helpers ─────────────────────────────

/**
 * Log-Gamma function via Lanczos approximation (g=7, n=9).
 * Accurate to ~15 digits for x > 0.
 */
export function logGamma(x: number): number {
  if (x <= 0) return Infinity;
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  let sum = c[0];
  for (let i = 1; i < g + 2; i++) {
    sum += c[i] / (x - 1 + i);
  }
  const t = x - 1 + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x - 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * Per-observation Student-t negative log-likelihood contribution.
 *
 * For standardized t(df) with variance σ²_t:
 *   -LL_i = 0.5·ln(σ²_t) + ((df+1)/2)·ln(1 + r²_t / ((df-2)·σ²_t))
 *         - lnΓ((df+1)/2) + lnΓ(df/2) + 0.5·ln(π·(df-2))
 *
 * Returns the per-observation neg-LL (without the constant terms).
 * Caller accumulates and adds the constant once.
 */
export function studentTNegLL(
  returns: number[],
  varianceSeries: number[],
  df: number,
): number {
  const n = returns.length;
  // Constant part (same for all observations)
  const halfDfPlus1 = (df + 1) / 2;
  const constant = n * (
    logGamma(df / 2) - logGamma(halfDfPlus1) + 0.5 * Math.log(Math.PI * (df - 2))
  );

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = varianceSeries[i];
    if (v <= 1e-12 || !isFinite(v)) return 1e10;
    sum += 0.5 * Math.log(v) + halfDfPlus1 * Math.log(1 + (returns[i] ** 2) / ((df - 2) * v));
  }

  return sum + constant;
}

/**
 * E[|Z|] where Z follows a standardized Student-t(df) distribution (variance = 1).
 *
 * E[|Z|] = √((df-2)/π) · Γ((df-1)/2) / Γ(df/2)
 *
 * Converges to √(2/π) as df → ∞ (Gaussian limit).
 */
export function expectedAbsStudentT(df: number): number {
  if (df <= 2) return EXPECTED_ABS_NORMAL; // fallback
  return Math.sqrt((df - 2) / Math.PI) * Math.exp(logGamma((df - 1) / 2) - logGamma(df / 2));
}

/**
 * 1D grid search for optimal df that minimizes Student-t neg-LL.
 * Used by HAR-RV and NoVaS where df is profiled after main optimization.
 */
export function profileStudentTDf(
  returns: number[],
  varianceSeries: number[],
): number {
  let bestDf = 30;
  let bestNLL = studentTNegLL(returns, varianceSeries, 30);

  // Coarse grid: 2.5 to 50
  for (let df = 2.5; df <= 50; df += 0.5) {
    const nll = studentTNegLL(returns, varianceSeries, df);
    if (nll < bestNLL) {
      bestNLL = nll;
      bestDf = df;
    }
  }

  // Fine grid around best
  const lo = Math.max(2.1, bestDf - 1);
  const hi = bestDf + 1;
  for (let df = lo; df <= hi; df += 0.05) {
    const nll = studentTNegLL(returns, varianceSeries, df);
    if (nll < bestNLL) {
      bestNLL = nll;
      bestDf = df;
    }
  }

  return bestDf;
}

/**
 * Calculate AIC (Akaike Information Criterion)
 */
export function calculateAIC(logLikelihood: number, numParams: number): number {
  return 2 * numParams - 2 * logLikelihood;
}

/**
 * Calculate BIC (Bayesian Information Criterion)
 */
export function calculateBIC(logLikelihood: number, numParams: number, numObs: number): number {
  return numParams * Math.log(numObs) - 2 * logLikelihood;
}

/**
 * QLIKE loss (Patton 2011) — standard loss function for volatility forecasts.
 *
 * QLIKE = (1/n) · Σ (RV_t / σ²_t − log(RV_t / σ²_t) − 1)
 *
 * Lower = better forecast. Neutral to calibration method — judges only
 * how well the variance series predicts realized variance, regardless
 * of how the model was calibrated (MLE, OLS, D², etc.).
 */
export function qlike(varianceSeries: number[], rv: number[]): number {
  const n = Math.min(varianceSeries.length, rv.length);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (varianceSeries[i] <= 0 || rv[i] <= 0) continue;
    const ratio = rv[i] / varianceSeries[i];
    sum += ratio - Math.log(ratio) - 1;
    count++;
  }
  return count > 0 ? sum / count : Infinity;
}
