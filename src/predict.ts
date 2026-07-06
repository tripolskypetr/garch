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
  probit,
  studentTProbit,
  profileStudentTDf,
  empiricalQuantile,
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
  /** Student-t degrees of freedom profiled on scale-corrected residuals. */
  df: number;
  /**
   * Corridor multiplier actually used: blend of the empirical |z| quantile
   * of the standardized residuals and the Student-t(df) quantile, weighted
   * by how much data supports the requested tail. Reconstruct bands as
   * `currentPrice · exp(±zScore · sigma)`.
   */
  zScore: number;
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
  /** Structural warm-up of the winning model (its longest lag / seeding region). */
  warmup: number;
  /** Sorted |r_t/σ_t| over the post-warm-up sample — the empirical calibration sample. */
  absZ: number[];
}

/**
 * Candidate HAR lag triples for this interval and sample size.
 *
 * The textbook (1, 5, 22) encodes *daily equity* horizons (day/week/month
 * in trading days) and is wrong for intraday 24/7 markets. Candidates are
 * built from the candle interval itself — one bar / one day / one week in
 * bars — capped by what the sample can support (long lag ≤ n/5 and enough
 * rows for the OLS). The final choice among candidates is made by QLIKE
 * on a common sample, not by convention.
 */
export function selectHarLagCandidates(
  nReturns: number,
  periodsPerYear: number,
): Array<[number, number, number]> {
  const maxLong = Math.floor(Math.min(nReturns / 5, nReturns - 31));
  const candidates: Array<[number, number, number]> = [];

  if (maxLong >= 22) candidates.push([1, 5, 22]);

  const barsPerDay = Math.round(periodsPerYear / 365);
  if (barsPerDay >= 3) {
    const barsPerWeek = 7 * barsPerDay;
    if (barsPerWeek <= maxLong) {
      candidates.push([1, barsPerDay, barsPerWeek]);
    } else if (barsPerDay * 2 <= maxLong) {
      // Week does not fit the sample — use the longest supported horizon
      candidates.push([1, barsPerDay, maxLong]);
    }
  }

  if (candidates.length === 0) {
    // Tiny sample: geometric spacing within what the data supports
    const long = Math.max(3, maxLong);
    candidates.push([1, Math.max(2, Math.round(Math.sqrt(long))), long]);
  }

  // Dedupe
  return candidates.filter((c, i) =>
    candidates.findIndex(d => d[0] === c[0] && d[1] === c[1] && d[2] === c[2]) === i,
  );
}

/**
 * NoVaS lag order grown with the sample (~n^(1/3), the standard
 * lag-order rate), instead of a fixed p = 10. Anchored to p = 10 at
 * n ≈ 500 (where a far-lag ARCH(10) ground truth is recovered best);
 * adapts down for short samples, up for long ones.
 */
export function adaptiveNovasLags(nReturns: number): number {
  return Math.min(20, Math.max(5, Math.round(1.26 * Math.cbrt(nReturns))));
}

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
 *
 * `warmup` is the winning model's own structural warm-up (longest lag /
 * seeding region), not a fixed constant.
 */
function varianceScaleCorrection(returns: number[], varianceSeries: number[], warmup: number): number {
  let sum = 0;
  let count = 0;
  for (let i = warmup; i < returns.length; i++) {
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
    warmup: 0,
    absZ: [],
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
      warmup: 0,
      absZ: [],
    };
  }

  const gjrModel = new GjrGarch(candles, { periodsPerYear });
  const gjrFit = gjrModel.fit();
  if (gjrFit.diagnostics.aic < bestAic) {
    bestAic = gjrFit.diagnostics.aic;
    best = {
      forecast: gjrModel.forecast(gjrFit.params, steps),
      modelType: 'gjr-garch',
      converged: gjrFit.diagnostics.converged,
      persistence: gjrFit.params.persistence,
      varianceSeries: gjrModel.getVarianceSeries(gjrFit.params),
      returns: gjrModel.getReturns(),
      df: gjrFit.params.df,
      warmup: 0,
      absZ: [],
    };
  }

  return best;
}

function fitHarRv(candles: Candle[], periodsPerYear: number, steps: number): FitResult | null {
  const candidates = selectHarLagCandidates(candles.length - 1, periodsPerYear);
  const commonWarmup = Math.max(...candidates.map(c => c[2]));

  let bestModel: HarRv | null = null;
  let bestFit: ReturnType<HarRv['fit']> | null = null;
  let bestLags: [number, number, number] | null = null;
  let bestScore = Infinity;

  for (const [shortLag, mediumLag, longLag] of candidates) {
    try {
      const model = new HarRv(candles, { periodsPerYear, shortLag, mediumLag, longLag });
      const fit = model.fit();

      // Skip candidate if non-stationary or worse than the mean predictor
      if (fit.params.persistence >= 1 || fit.params.r2 < 0) continue;

      // Score candidates on a common sample (past every candidate's warm-up)
      const series = model.getVarianceSeries(fit.params);
      const score = qlike(series.slice(commonWarmup), model.getRv().slice(commonWarmup));
      if (score < bestScore) {
        bestModel = model;
        bestFit = fit;
        bestLags = [shortLag, mediumLag, longLag];
        bestScore = score;
      }
    } catch {
      continue;
    }
  }

  if (!bestModel || !bestFit || !bestLags) return null;

  return {
    forecast: bestModel.forecast(bestFit.params, steps),
    modelType: 'har-rv',
    converged: bestFit.diagnostics.converged,
    persistence: bestFit.params.persistence,
    varianceSeries: bestModel.getVarianceSeries(bestFit.params),
    returns: bestModel.getReturns(),
    df: bestFit.params.df,
    warmup: bestLags[2],
    absZ: [],
  };
}

function fitNoVaS(candles: Candle[], periodsPerYear: number, steps: number): FitResult | null {
  try {
    const lags = adaptiveNovasLags(candles.length - 1);
    const model = new NoVaS(candles, { periodsPerYear, lags });
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
      warmup: lags,
      absZ: [],
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
  // Warm-up comes from the winner's own structure (its longest lag /
  // seeding region), capped so calibration keeps a usable sample.
  const nReturns = best.returns.length;
  const warmup = Math.min(Math.max(best.warmup, 10), Math.floor(nReturns / 4));
  best.warmup = warmup;

  const c = varianceScaleCorrection(best.returns, best.varianceSeries, warmup);
  applyScale(best, c, periodsPerYear);

  // Re-profile tail thickness on the corrected residuals — this df drives
  // the model half of the corridor quantile.
  best.df = profileStudentTDf(best.returns, best.varianceSeries);

  // Empirical calibration sample: |z_t| of the corrected residuals
  const absZ: number[] = [];
  for (let i = warmup; i < nReturns; i++) {
    const v = best.varianceSeries[i];
    if (!(v > 0) || !isFinite(v)) continue;
    const a = Math.abs(best.returns[i]) / Math.sqrt(v);
    if (isFinite(a)) absZ.push(a);
  }
  absZ.sort((a, b) => a - b);
  best.absZ = absZ;

  return best;
}

/**
 * |h-step standardized sum| sample: |Σr| / √(Σσ²) over overlapping windows
 * of the post-warm-up region. This is the h-step analog of fit.absZ — it
 * absorbs whatever the single-period model misses about aggregation
 * (volatility autocorrelation, Jensen bias in EGARCH multi-step, fat tails
 * washing out by CLT).
 */
function horizonAbsZ(fit: FitResult, steps: number): number[] {
  const { returns, varianceSeries, warmup } = fit;
  const out: number[] = [];

  for (let t = warmup; t + steps <= returns.length; t++) {
    let sumR = 0;
    let sumV = 0;
    let ok = true;
    for (let j = 0; j < steps; j++) {
      const v = varianceSeries[t + j];
      if (!(v > 0) || !isFinite(v)) {
        ok = false;
        break;
      }
      sumR += returns[t + j];
      sumV += v;
    }
    if (!ok) continue;
    const z = Math.abs(sumR) / Math.sqrt(sumV);
    if (isFinite(z)) out.push(z);
  }

  out.sort((a, b) => a - b);
  return out;
}

/**
 * Corridor multiplier for a two-sided confidence level at horizon `steps`.
 *
 * Instead of trusting the fitted Student-t shape outright, the multiplier
 * is anchored to the data: the empirical quantile of the |standardized
 * (h-step) return| is used where the sample actually supports the requested
 * tail, and the model quantile takes over as the tail runs out of
 * observations. The blend weight is the expected number of tail exceedances
 * m = n_eff·(1−confidence) shrunk by a prior weight of 10 pseudo-
 * observations; overlapping h-step windows are discounted by 1/steps.
 *
 * The model half relaxes from t(df) toward Gaussian as the horizon grows:
 * the excess kurtosis of a sum of h shocks decays as 1/h, so a single-period
 * fat-tail quantile applied to a multi-step corridor would be too narrow in
 * the center (measured: 68% band covered only ~55% at 10 steps) and too
 * wide in the tails.
 */
function corridorZ(fit: FitResult, confidence: number, steps = 1): number {
  const zGauss = probit(confidence);
  const zT = studentTProbit(confidence, fit.df);
  const zModel = steps === 1 ? zT : zGauss + (zT - zGauss) / steps;

  const absZ = steps === 1 ? fit.absZ : horizonAbsZ(fit, steps);
  const n = absZ.length;
  if (n < 50) return zModel;

  const zEmpirical = empiricalQuantile(absZ, confidence);
  if (!isFinite(zEmpirical)) return zModel;

  const effN = n / steps; // overlap discount
  const tailCount = effN * (1 - confidence);
  const w = tailCount / (tailCount + 10);
  return w * zEmpirical + (1 - w) * zModel;
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
 * to the return scale (Var(r/σ) = 1), and builds bands P·exp(±z·σ) where
 * z is calibrated on the data itself: the empirical |z| quantile of the
 * standardized residuals blended with the fitted Student-t quantile as the
 * tail runs out of observations (see corridorZ). Empirical coverage tracks
 * the requested confidence without assuming a distributional shape.
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
  const z = corridorZ(fit, confidence);

  const sigma = fit.forecast.volatility[0];
  const upperPrice = currentPrice * Math.exp(z * sigma);
  const lowerPrice = currentPrice * Math.exp(-z * sigma);

  return {
    modelType: fit.modelType,
    currentPrice,
    sigma,
    df: fit.df,
    zScore: z,
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
 * Uses log-normal price bands P·exp(±z·σ) where z is calibrated at the
 * requested horizon: the empirical quantile of |h-step standardized sums|
 * from the sample itself, blended with a model quantile that relaxes from
 * t(df) toward Gaussian as aggregation washes the fat tails out.
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
  if (!Number.isFinite(steps) || steps < 1) {
    throw new Error(`steps must be a number >= 1, got ${steps}`);
  }
  steps = Math.floor(steps);
  const fit = fitModel(candles, INTERVALS_PER_YEAR[interval], steps);
  const z = corridorZ(fit, confidence, steps);

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
    zScore: z,
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
 * Refits the model on a rolling window (75% of candles, min MIN_CANDLES)
 * and checks whether the next close lands inside the predicted corridor.
 * A well-calibrated tool has hitRate ≈ confidence·100.
 *
 * Every refit costs a full 5-model calibration, so by default the test
 * points are subsampled to at most ~100 refits (stride grows with the test
 * span). Pass `stride: 1` to evaluate every candle when runtime is not a
 * concern, or any positive stride to control the trade-off yourself.
 * Throws if not enough candles for the given interval.
 * @param confidence — two-sided probability in (0,1) for the prediction band.
 *   Default ≈0.6827 (±1σ).
 */
export function backtestStats(
  candles: Candle[],
  interval: CandleInterval,
  confidence = 0.6827,
  options: { stride?: number } = {},
): BacktestStats {
  assertMinCandles(candles, interval);

  const window = Math.max(MIN_CANDLES[interval], Math.floor(candles.length * BACKTEST_WINDOW_RATIO));
  if (candles.length - 1 <= window) {
    throw new Error(
      `Need at least ${window + 2} candles to backtest ${interval} interval, got ${candles.length}`,
    );
  }

  const testSpan = candles.length - 1 - window;
  const stride = Math.max(1, Math.floor(options.stride ?? Math.ceil(testSpan / 100)));

  let hits = 0;
  let total = 0;

  for (let i = window; i < candles.length - 1; i += stride) {
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
  if (requiredPercent > 100) return false;

  return backtestStats(candles, interval, confidence).hitRate >= requiredPercent;
}

