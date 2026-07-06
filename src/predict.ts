import type { Candle, VolatilityForecast } from './types.js';
import { Garch } from './garch.js';
import { Egarch } from './egarch.js';
import { GjrGarch } from './gjr-garch.js';
import { HarRv } from './har.js';
import { NoVaS } from './novas.js';
import {
  ljungBox,
  calculateReturns,
  perCandleParkinson,
  qlike,
  studentTProbit,
  profileStudentTDf,
} from './utils.js';

export type CandleInterval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '8h';

const MIN_CANDLES: Record<CandleInterval, number> = {
  '1m': 500,
  '3m': 500,
  '5m': 500,
  '15m': 300,
  '30m': 200,
  '1h': 200,
  '2h': 200,
  '4h': 200,
  '6h': 150,
  '8h': 150,
};

const RECOMMENDED_CANDLES: Record<CandleInterval, number> = {
  '1m': 1500,
  '3m': 1500,
  '5m': 1500,
  '15m': 1000,
  '30m': 1000,
  '1h': 500,
  '2h': 500,
  '4h': 500,
  '6h': 300,
  '8h': 300,
};

const INTERVALS_PER_YEAR: Record<CandleInterval, number> = {
  '1m': 525_600,
  '3m': 175_200,
  '5m': 105_120,
  '15m': 35_040,
  '30m': 17_520,
  '1h': 8_760,
  '2h': 4_380,
  '4h': 2_190,
  '6h': 1_460,
  '8h': 1_095,
};

export interface PredictionResult {
  /** Reference price used to compute the corridor (last close or the value passed as `currentPrice`). */
  currentPrice: number;
  /** One-period (or cumulative) volatility estimate, as a decimal log-return standard deviation (e.g. `0.012` = 1.2%). */
  sigma: number;
  /** Upward expected move in price units: `upperPrice - currentPrice`. */
  move: number;
  /** Upward expected move in percent (0–100 scale, e.g. `1.21` means 1.21%). Equal to `(exp(z·σ) - 1) * 100`. */
  movePercent: number;
  /** Upper price band: `currentPrice · exp(+z·σ)`. */
  upperPrice: number;
  /** Lower price band: `currentPrice · exp(-z·σ)`. Always positive. */
  lowerPrice: number;
  /** Volatility model auto-selected by QLIKE. */
  modelType: 'garch' | 'egarch' | 'gjr-garch' | 'har-rv' | 'novas';
  /** Student-t degrees of freedom used for the corridor quantile (profiled on scale-corrected residuals). */
  df: number;
  /** `true` when the model converged, persistence < 0.999, and Ljung-Box p-value ≥ 0.05. */
  reliable: boolean;
}

function assertMinCandles(candles: Candle[], interval: CandleInterval): void {
  const min = MIN_CANDLES[interval];
  if (candles.length < min) {
    throw new Error(`Need at least ${min} candles for ${interval} interval, got ${candles.length}`);
  }
  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    if (!isFinite(c.close) || c.close <= 0) {
      throw new Error(`Invalid close price at candle ${i}: ${c.close}`);
    }
    if (!isFinite(c.open) || c.open <= 0 || !isFinite(c.high) || c.high <= 0 || !isFinite(c.low) || c.low <= 0) {
      throw new Error(`Invalid OHLC at candle ${i}: open=${c.open} high=${c.high} low=${c.low}`);
    }
    if (c.high < c.low) {
      throw new Error(`Invalid candle ${i}: high (${c.high}) < low (${c.low})`);
    }
  }
  const recommended = RECOMMENDED_CANDLES[interval];
  if (candles.length < recommended) {
    /*console.warn(
      `[garch] ${interval}: ${candles.length} candles provided, recommend ≥${recommended} for reliable results. Check reliable: true in output.`,
    );*/
  }
}

interface FitResult {
  forecast: VolatilityForecast;
  modelType: 'garch' | 'egarch' | 'gjr-garch' | 'har-rv' | 'novas';
  converged: boolean;
  persistence: number;
  varianceSeries: number[];
  returns: number[];
  df: number;
}

// Skip the model burn-in region (HAR long lag = 22, NoVaS lags = 10,
// GARCH initial-variance seeding) when scoring and calibrating.
const BURN_IN = 22;

/**
 * Empirical variance-scale correction.
 *
 * RV-based models (HAR-RV, NoVaS) forecast Parkinson realized variance,
 * which is NOT the same quantity as the close-to-close return variance the
 * price corridor needs — range-based RV is systematically smaller whenever
 * moves happen between closes (gaps, thin wicks). The corridor must satisfy
 * Var(r_t / σ_t) = 1, so rescale by c = E[r²/σ²] measured on the sample.
 *
 * For MLE-calibrated GARCH-family fits c ≈ 1 (the likelihood already
 * self-calibrates the level), so this is a no-op there.
 *
 * z² is capped at 50 to keep a single extreme print from distorting the
 * scale (bias of the cap is <2% even at df = 5).
 */
function varianceScaleCorrection(returns: number[], varianceSeries: number[]): number {
  let sum = 0;
  let count = 0;
  for (let i = BURN_IN; i < returns.length; i++) {
    const v = varianceSeries[i];
    if (!(v > 0) || !isFinite(v)) continue;
    const z2 = (returns[i] * returns[i]) / v;
    if (!isFinite(z2)) continue;
    sum += Math.min(z2, 50);
    count++;
  }
  if (count < 30) return 1;
  const c = sum / count;
  if (!isFinite(c) || c <= 0) return 1;
  // Sanity clamp: beyond this the fit is garbage and `reliable` flags it
  return Math.min(100, Math.max(0.01, c));
}

function applyScale(fit: FitResult, c: number, periodsPerYear: number): void {
  if (c === 1) return;
  fit.varianceSeries = fit.varianceSeries.map(v => v * c);
  const variance = fit.forecast.variance.map(v => v * c);
  fit.forecast = {
    variance,
    volatility: variance.map(v => Math.sqrt(v)),
    annualized: variance.map(v => Math.sqrt(v * periodsPerYear) * 100),
  };
}

function fitGarchFamily(candles: Candle[], periodsPerYear: number, steps: number): FitResult {
  // Fit all three GARCH-family models and pick the best by AIC
  // (AIC is fair here — all three optimize the same Student-t LL)
  const garchModel = new Garch(candles, { periodsPerYear });
  const garchFit = garchModel.fit();
  let bestAic = garchFit.diagnostics.aic;
  let best: FitResult = {
    forecast: garchModel.forecast(garchFit.params, steps),
    modelType: 'garch',
    converged: garchFit.diagnostics.converged,
    persistence: garchFit.params.persistence,
    varianceSeries: garchModel.getVarianceSeries(garchFit.params),
    returns: garchModel.getReturns(),
    df: garchFit.params.df,
  };

  const egarchModel = new Egarch(candles, { periodsPerYear });
  const egarchFit = egarchModel.fit();
  if (egarchFit.diagnostics.aic < bestAic) {
    bestAic = egarchFit.diagnostics.aic;
    best = {
      forecast: egarchModel.forecast(egarchFit.params, steps),
      modelType: 'egarch',
      converged: egarchFit.diagnostics.converged,
      persistence: egarchFit.params.persistence,
      varianceSeries: egarchModel.getVarianceSeries(egarchFit.params),
      returns: egarchModel.getReturns(),
      df: egarchFit.params.df,
    };
  }

  const gjrModel = new GjrGarch(candles, { periodsPerYear });
  const gjrFit = gjrModel.fit();
  if (gjrFit.diagnostics.aic < bestAic) {
    best = {
      forecast: gjrModel.forecast(gjrFit.params, steps),
      modelType: 'gjr-garch',
      converged: gjrFit.diagnostics.converged,
      persistence: gjrFit.params.persistence,
      varianceSeries: gjrModel.getVarianceSeries(gjrFit.params),
      returns: gjrModel.getReturns(),
      df: gjrFit.params.df,
    };
  }

  return best;
}

function fitHarRv(candles: Candle[], periodsPerYear: number, steps: number): FitResult | null {
  try {
    const model = new HarRv(candles, { periodsPerYear });
    const fit = model.fit();

    // Skip HAR-RV if persistence >= 1 (non-stationary) or R² too low
    if (fit.params.persistence >= 1 || fit.params.r2 < 0) return null;

    return {
      forecast: model.forecast(fit.params, steps),
      modelType: 'har-rv',
      converged: fit.diagnostics.converged,
      persistence: fit.params.persistence,
      varianceSeries: model.getVarianceSeries(fit.params),
      returns: model.getReturns(),
      df: fit.params.df,
    };
  } catch {
    return null;
  }
}

function fitNoVaS(candles: Candle[], periodsPerYear: number, steps: number): FitResult | null {
  try {
    const model = new NoVaS(candles, { periodsPerYear });
    const fit = model.fit();

    // Skip if persistence >= 1 (non-stationary)
    if (fit.params.persistence >= 1) return null;

    return {
      forecast: model.forecast(fit.params, steps),
      modelType: 'novas',
      converged: fit.diagnostics.converged,
      persistence: fit.params.persistence,
      varianceSeries: model.getForecastVarianceSeries(fit.params),
      returns: model.getReturns(),
      df: fit.params.df,
    };
  } catch {
    return null;
  }
}

function fitModel(candles: Candle[], periodsPerYear: number, steps: number): FitResult {
  const garchResult = fitGarchFamily(candles, periodsPerYear, steps);
  const harResult = fitHarRv(candles, periodsPerYear, steps);
  const novasResult = fitNoVaS(candles, periodsPerYear, steps);

  // Compute realized variance (Parkinson RV) for QLIKE scoring
  const returns = calculateReturns(candles);
  const rv = perCandleParkinson(candles, returns);

  // Pick model with lowest QLIKE (Patton 2011) — neutral forecast-error metric.
  // Unlike AIC (which favors MLE-calibrated models), QLIKE judges only
  // how well the variance series predicts realized variance.
  let best: FitResult = garchResult;
  let bestScore = qlike(garchResult.varianceSeries, rv);

  if (harResult) {
    const score = qlike(harResult.varianceSeries, rv);
    if (score < bestScore) {
      best = harResult;
      bestScore = score;
    }
  }
  if (novasResult) {
    const score = qlike(novasResult.varianceSeries, rv);
    if (score < bestScore) {
      best = novasResult;
      bestScore = score;
    }
  }

  // Calibrate the winner to the return scale: QLIKE picks the best RV
  // forecaster, but the corridor needs the return-variance scale.
  const c = varianceScaleCorrection(best.returns, best.varianceSeries);
  applyScale(best, c, periodsPerYear);

  // Re-profile tail thickness on the corrected residuals — this df drives
  // the corridor quantile.
  best.df = profileStudentTDf(best.returns, best.varianceSeries);

  return best;
}

function checkReliable(fit: FitResult): boolean {
  if (!fit.converged || fit.persistence >= 0.999) return false;

  // Ljung-Box on squared standardized residuals
  const { returns, varianceSeries } = fit;
  const squared = returns.map((r, i) => {
    const z = r / Math.sqrt(varianceSeries[i]);
    return z * z;
  });
  const lb = ljungBox(squared, 10);
  return lb.pValue >= 0.05;
}

/**
 * Forecast expected price range for t+1 (next candle).
 *
 * Auto-selects the best volatility model via QLIKE, rescales the variance
 * to the return scale (Var(r/σ) = 1), and builds bands with the fitted
 * Student-t quantile: P·exp(±z·σ), where z = studentTProbit(confidence, df).
 * With fat-tailed data this keeps empirical coverage at the requested
 * confidence — a Gaussian z over-covers the center and under-covers tails.
 * @param confidence — two-sided probability in (0,1). Default ≈0.6827 (±1σ).
 *   Common values: 0.90, 0.95, 0.99.
 */
export function predict(
  candles: Candle[],
  interval: CandleInterval,
  currentPrice?: number | null,
  confidence = 0.6827,
): PredictionResult {
  assertMinCandles(candles, interval);

  currentPrice = currentPrice || candles[candles.length - 1].close;

  const fit = fitModel(candles, INTERVALS_PER_YEAR[interval], 1);
  const z = studentTProbit(confidence, fit.df);

  const sigma = fit.forecast.volatility[0];
  const upperPrice = currentPrice * Math.exp(z * sigma);
  const lowerPrice = currentPrice * Math.exp(-z * sigma);

  return {
    modelType: fit.modelType,
    currentPrice,
    sigma,
    df: fit.df,
    move: upperPrice - currentPrice,
    movePercent: (upperPrice / currentPrice - 1) * 100,
    upperPrice,
    lowerPrice,
    reliable: checkReliable(fit),
  };
}

/**
 * Forecast expected price range over multiple candles.
 *
 * Cumulative σ = √(σ₁² + σ₂² + ... + σₙ²) — total expected move over N periods.
 * Uses log-normal price bands: P·exp(±z·σ), where z = studentTProbit(confidence, df).
 * The single-period df is used for multi-step horizons too — aggregated
 * returns are closer to Gaussian, so this errs on the wide (safe) side in tails.
 * @param confidence — two-sided probability in (0,1). Default ≈0.6827 (±1σ).
 */
export function predictRange(
  candles: Candle[],
  interval: CandleInterval,
  steps: number,
  currentPrice?: number | null,
  confidence = 0.6827,
): PredictionResult {
  assertMinCandles(candles, interval);
  const fit = fitModel(candles, INTERVALS_PER_YEAR[interval], steps);
  const z = studentTProbit(confidence, fit.df);

  currentPrice = currentPrice || candles[candles.length - 1].close;

  const cumulativeVariance = fit.forecast.variance.reduce((sum, v) => sum + v, 0);
  const sigma = Math.sqrt(cumulativeVariance);
  const upperPrice = currentPrice * Math.exp(z * sigma);
  const lowerPrice = currentPrice * Math.exp(-z * sigma);

  return {
    modelType: fit.modelType,
    currentPrice,
    sigma,
    df: fit.df,
    move: upperPrice - currentPrice,
    movePercent: (upperPrice / currentPrice - 1) * 100,
    upperPrice,
    lowerPrice,
    reliable: checkReliable(fit),
  };
}

// ── Backtest ──────────────────────────────────────────────────

const BACKTEST_WINDOW_RATIO = 0.75;

export interface BacktestStats {
  /** Number of test candles whose close landed inside the predicted corridor. */
  hits: number;
  /** Number of walk-forward predictions made. */
  total: number;
  /** Empirical coverage in percent (0–100). Compare against `confidence · 100`. */
  hitRate: number;
}

/**
 * Walk-forward calibration statistics for predict.
 *
 * Refits the model at every step on a rolling window (75% of candles,
 * min MIN_CANDLES) and checks whether the next close lands inside the
 * predicted corridor. A well-calibrated tool has hitRate ≈ confidence·100.
 * Throws if not enough candles for the given interval.
 * @param confidence — two-sided probability in (0,1) for the prediction band.
 *   Default ≈0.6827 (±1σ).
 */
export function backtestStats(
  candles: Candle[],
  interval: CandleInterval,
  confidence = 0.6827,
): BacktestStats {
  assertMinCandles(candles, interval);

  const window = Math.max(MIN_CANDLES[interval], Math.floor(candles.length * BACKTEST_WINDOW_RATIO));
  if (candles.length - 1 <= window) {
    throw new Error(
      `Need at least ${window + 2} candles to backtest ${interval} interval, got ${candles.length}`,
    );
  }
  let hits = 0;
  let total = 0;

  for (let i = window; i < candles.length - 1; i++) {
    const slice = candles.slice(i - window, i + 1);
    const predicted = predict(slice, interval, slice[slice.length - 1].close, confidence);
    const actual = candles[i + 1].close;

    if (actual >= predicted.lowerPrice && actual <= predicted.upperPrice) {
      hits++;
    }
    total++;
  }

  return { hits, total, hitRate: (hits / total) * 100 };
}

/**
 * Walk-forward backtest of predict.
 *
 * Window is computed automatically: 75% of candles for fitting, 25% for testing.
 * Throws if not enough candles for the given interval.
 * Returns true if the model's hit rate meets the required threshold.
 * @param confidence — two-sided probability in (0,1) for the prediction band.
 *   Default ≈0.6827 (±1σ).
 * @param requiredPercent — minimum hit rate (0–100) to pass. Default 68.
 */
export function backtest(
  candles: Candle[],
  interval: CandleInterval,
  confidence = 0.6827,
  requiredPercent = 68,
): boolean {
  assertMinCandles(candles, interval);
  if (requiredPercent <= 0) return true;
  if (requiredPercent >= 100) return false;

  return backtestStats(candles, interval, confidence).hitRate >= requiredPercent;
}

