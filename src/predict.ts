import type {
  Candle,
  VolatilityForecast,
  GarchParams,
  EgarchParams,
  GjrGarchParams,
  RealizedGarchParams,
} from './types.js';
import { Garch } from './garch.js';
import { Egarch } from './egarch.js';
import { GjrGarch } from './gjr-garch.js';
import { RealizedGarch } from './realized-garch.js';
import { HarRv } from './har.js';
import { NoVaS } from './novas.js';
import { NotEnoughDataError, BadDataError, InvalidArgumentError } from './errors.js';
import {
  ljungBox,
  calculateReturns,
  perCandleParkinson,
  qlike,
  probit,
  studentTProbit,
  profileStudentTDf,
  empiricalQuantile,
  validateCandles,
  sampleVariance,
  chi2Survival,
  expectedAbsStudentT,
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

export type WarningCode =
  | 'LOW_SAMPLE'
  | 'NOT_CONVERGED'
  | 'HIGH_PERSISTENCE'
  | 'DEGENERATE_VARIANCE'
  | 'RESIDUAL_AUTOCORRELATION'
  | 'DATA_GAPS'
  | 'INTERVAL_MISMATCH';

export interface PredictionWarning {
  code: WarningCode;
  /** Plain-language explanation with a suggested action. */
  message: string;
  /** true when this warning alone makes the forecast unreliable. */
  critical: boolean;
}

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
  /** Top-weight member of the volatility model combination (selected by out-of-sample QLIKE). */
  modelType: 'garch' | 'egarch' | 'gjr-garch' | 'realized-garch' | 'har-rv' | 'novas';
  /** Student-t degrees of freedom profiled on scale-corrected residuals. */
  df: number;
  /**
   * Average corridor multiplier, (zScoreUp + zScoreDown) / 2 — kept for
   * backward compatibility. The bands themselves are asymmetric; use
   * zScoreUp/zScoreDown to reconstruct them exactly.
   */
  zScore: number;
  /** Upper-tail multiplier: `upperPrice = currentPrice · exp(+zScoreUp · sigma)`. */
  zScoreUp: number;
  /** Lower-tail multiplier: `lowerPrice = currentPrice · exp(−zScoreDown · sigma)`. */
  zScoreDown: number;
  /** `true` when no critical warning fired (model converged, persistence < 0.999, Ljung-Box p ≥ 0.05, non-degenerate variance). */
  reliable: boolean;
  /**
   * Everything the pipeline noticed, in plain language: why `reliable` is
   * false (critical warnings) plus non-critical data quality notes.
   */
  warnings: PredictionWarning[];
  /** Combination weights per model family (out-of-sample QLIKE softmax), summing to 1. */
  modelWeights: Partial<Record<PredictionResult['modelType'], number>>;
  /** `true` when a significant diurnal volatility profile was detected and removed before fitting. */
  seasonalityDetected: boolean;
}

function validateInterval(interval: CandleInterval): void {
  if (!(interval in INTERVALS_PER_YEAR)) {
    throw new InvalidArgumentError(
      `Unknown interval '${interval}' — valid intervals: ${Object.keys(INTERVALS_PER_YEAR).join(', ')}`,
    );
  }
}

function validateConfidence(confidence: number): void {
  if (!(confidence > 0 && confidence < 1)) {
    const hint = confidence > 1 && confidence <= 100
      ? ` (did you pass percent? Use a fraction: ${confidence / 100})`
      : '';
    throw new InvalidArgumentError(`confidence must be in (0, 1), got ${confidence}${hint}`);
  }
}

export interface PredictOptions {
  /** Reference price for the corridor; defaults to the last close. */
  currentPrice?: number | null;
  /** Two-sided probability in (0,1). Default ≈0.6827 (±1σ). */
  confidence?: number;
}

/**
 * Accept both the positional form (currentPrice, confidence) and an options
 * object — `predict(candles, '1h', { confidence: 0.9 })` cannot be confused
 * with a price the way `predict(candles, '1h', 0.9)` can.
 */
function resolvePredictArgs(
  candles: Candle[],
  currentPriceOrOptions: number | null | undefined | PredictOptions,
  confidence: number,
): { currentPrice: number; confidence: number } {
  let price: number | null | undefined;
  let conf = confidence;
  if (currentPriceOrOptions !== null && typeof currentPriceOrOptions === 'object') {
    price = currentPriceOrOptions.currentPrice;
    conf = currentPriceOrOptions.confidence ?? confidence;
  } else {
    price = currentPriceOrOptions;
  }
  validateConfidence(conf);
  if (price !== null && price !== undefined) {
    if (!(Number.isFinite(price) && price > 0)) {
      throw new InvalidArgumentError(`currentPrice must be a positive finite number, got ${price}`);
    }
    return { currentPrice: price, confidence: conf };
  }
  return { currentPrice: candles[candles.length - 1].close, confidence: conf };
}

function assertMinCandles(candles: Candle[], interval: CandleInterval): void {
  validateInterval(interval);
  const min = MIN_CANDLES[interval];
  if (candles.length < min) {
    throw new NotEnoughDataError(`Need at least ${min} candles for ${interval} interval, got ${candles.length}`);
  }
  validateCandles(candles);
}

// ── Data quality ──────────────────────────────────────────────

/** Timestamps in ms (seconds are auto-scaled), or null when any candle lacks one. */
function getTimestampsMs(candles: Candle[]): number[] | null {
  if (!candles.every(c => Number.isFinite(c.timestamp))) return null;
  return candles.map(c => (c.timestamp! < 1e12 ? c.timestamp! * 1000 : c.timestamp!));
}

/** Hard failures on broken timestamp ordering — garbage-in guards for predict. */
function assertTimestampOrder(candles: Candle[]): void {
  const ts = getTimestampsMs(candles);
  if (!ts) return;
  for (let i = 1; i < ts.length; i++) {
    if (ts[i] < ts[i - 1]) {
      throw new BadDataError(
        `Candles are not sorted by timestamp (index ${i}) — sort ascending before calling. Check your data feed.`,
      );
    }
    if (ts[i] === ts[i - 1]) {
      throw new BadDataError(
        `Duplicate candle timestamp at index ${i} — deduplicate your data feed before calling.`,
      );
    }
  }
}

function formatSpacing(ms: number): string {
  if (ms >= 3_600_000) return `${(ms / 3_600_000).toFixed(1)}h`;
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`;
  return `${(ms / 1000).toFixed(0)}s`;
}

/** Non-critical data observations appended to PredictionResult.warnings. */
function collectDataWarnings(
  candles: Candle[],
  interval: CandleInterval,
  warnings: PredictionWarning[],
): void {
  const recommended = RECOMMENDED_CANDLES[interval];
  if (candles.length < recommended) {
    warnings.push({
      code: 'LOW_SAMPLE',
      critical: false,
      message: `${candles.length} candles provided; ≥${recommended} recommended for ${interval} — estimates are noisier on short samples.`,
    });
  }

  const ts = getTimestampsMs(candles);
  if (!ts || ts.length < 3) return;
  const barMs = YEAR_MS / INTERVALS_PER_YEAR[interval];

  const spacings = [];
  for (let i = 1; i < ts.length; i++) spacings.push(ts[i] - ts[i - 1]);
  spacings.sort((a, b) => a - b);
  const median = spacings[Math.floor(spacings.length / 2)];
  if (median > 0 && Math.abs(median / barMs - 1) > 0.25) {
    warnings.push({
      code: 'INTERVAL_MISMATCH',
      critical: false,
      message: `Candle spacing looks like ~${formatSpacing(median)} while interval is '${interval}' — check the interval argument.`,
    });
    return; // gap counting is meaningless against the wrong bar size
  }

  let gaps = 0;
  for (const dt of spacings) {
    if (dt > barMs * 1.5) gaps += Math.round(dt / barMs) - 1;
  }
  const gapPct = (gaps / (candles.length + gaps)) * 100;
  if (gapPct > 1) {
    warnings.push({
      code: 'DATA_GAPS',
      critical: false,
      message: `~${gaps} missing bars (${gapPct.toFixed(1)}%) detected from timestamps — the seasonal profile and lag structure may be distorted. Check your feed for outages.`,
    });
  }
}

export interface DataIssue {
  code:
    | 'TOO_FEW_CANDLES'
    | 'INVALID_OHLC'
    | 'UNSORTED'
    | 'DUPLICATE_TIMESTAMPS'
    | 'LOW_SAMPLE'
    | 'DATA_GAPS'
    | 'INTERVAL_MISMATCH'
    | 'FLAT_CANDLES';
  message: string;
  severity: 'error' | 'warning';
}

export interface DataReport {
  /** false when any error-severity issue is present (predict would throw). */
  ok: boolean;
  issues: DataIssue[];
  recommendedCandles: number;
}

/**
 * Pre-flight data check with plain-language, actionable messages: run it on
 * a new data source before wiring it into predict. Errors are conditions
 * predict would throw on; warnings degrade quality but do not block.
 */
export function checkData(candles: Candle[], interval: CandleInterval): DataReport {
  validateInterval(interval);
  const issues: DataIssue[] = [];
  const recommended = RECOMMENDED_CANDLES[interval];

  if (candles.length < MIN_CANDLES[interval]) {
    issues.push({
      code: 'TOO_FEW_CANDLES',
      severity: 'error',
      message: `Need at least ${MIN_CANDLES[interval]} candles for ${interval}, got ${candles.length} — fetch more history.`,
    });
  }

  try {
    validateCandles(candles);
  } catch (e) {
    issues.push({
      code: 'INVALID_OHLC',
      severity: 'error',
      message: `${(e as Error).message}. Broken OHLC usually means a feed/parsing bug — check the failing candle in your pipeline.`,
    });
  }

  try {
    assertTimestampOrder(candles);
  } catch (e) {
    const msg = (e as Error).message;
    issues.push({
      code: msg.includes('Duplicate') ? 'DUPLICATE_TIMESTAMPS' : 'UNSORTED',
      severity: 'error',
      message: msg,
    });
  }

  const soft: PredictionWarning[] = [];
  collectDataWarnings(candles, interval, soft);
  for (const w of soft) {
    issues.push({ code: w.code as DataIssue['code'], severity: 'warning', message: w.message });
  }

  let flat = 0;
  for (const c of candles) {
    if (c.high === c.low) flat++;
  }
  const flatPct = (flat / Math.max(candles.length, 1)) * 100;
  if (flatPct > 20) {
    issues.push({
      code: 'FLAT_CANDLES',
      severity: 'warning',
      message: `${flatPct.toFixed(0)}% of candles have high === low — range-based estimators degrade to squared returns. Synthetic or illiquid feed?`,
    });
  }

  return {
    ok: !issues.some(i => i.severity === 'error'),
    issues,
    recommendedCandles: recommended,
  };
}

// ── Intraday seasonality ──────────────────────────────────────

const DAY_MS = 86_400_000;
const YEAR_MS = 31_536_000_000;

export interface Seasonality {
  /** Variance factor per intraday bucket, sample-weighted mean 1. */
  factors: number[];
  /** Bucket of return index t (i.e. candle t+1); also valid for future t. */
  bucketOfReturn: (t: number) => number;
}

/**
 * Diurnal variance profile from per-candle Parkinson RV.
 *
 * Intraday markets have a strong time-of-day volatility pattern (sessions,
 * funding, weekends). A GARCH-family fit smears it into average persistence,
 * so corridors are systematically too narrow in active hours and too wide in
 * quiet ones. Factors are estimated per intraday bucket (≤24 per day, bars
 * grouped for sub-hour intervals), circularly smoothed, shrunk toward 1 by
 * bucket support, and gated by a χ² significance test against the RV
 * sampling noise (inflated for volatility clustering) — pure-GARCH data
 * without seasonality returns null and the pipeline is unchanged.
 *
 * Timestamps (ms or s), when present on every candle, anchor buckets to
 * real time of day and survive gaps; otherwise buckets are positional and
 * assume contiguous bars.
 */
export function computeSeasonality(candles: Candle[], interval: CandleInterval): Seasonality | null {
  const periodsPerYear = INTERVALS_PER_YEAR[interval];
  const barsPerDay = Math.round(periodsPerYear / 365);
  if (barsPerDay < 3) return null;
  const buckets = Math.min(24, barsPerDay);

  const returns = calculateReturns(candles);
  const rv = perCandleParkinson(candles, returns);
  const n = returns.length;
  const nCandles = candles.length;

  const hasTs = candles.every(c => Number.isFinite(c.timestamp));
  const ts = hasTs
    ? candles.map(c => (c.timestamp! < 1e12 ? c.timestamp! * 1000 : c.timestamp!))
    : null;
  const barMs = YEAR_MS / periodsPerYear;

  const bucketOfReturn = (t: number): number => {
    const i = t + 1;
    if (ts) {
      const time = i < nCandles ? ts[i] : ts[nCandles - 1] + (i - (nCandles - 1)) * barMs;
      return Math.min(buckets - 1, Math.floor(((time % DAY_MS) / DAY_MS) * buckets));
    }
    return Math.min(buckets - 1, Math.floor(((i % barsPerDay) / barsPerDay) * buckets));
  };

  // Work in log-RV: per-observation RV is heavy-tailed, so level means are
  // dominated by single prints while log means have modest, comparable
  // variance across buckets. Bucket ratios of geometric means equal
  // variance ratios up to a constant (identical noise shape per bucket),
  // which the normalization below removes anyway.
  const logSums = new Array(buckets).fill(0);
  const counts = new Array(buckets).fill(0);
  let totalLog = 0;
  let totalCount = 0;
  for (let t = 0; t < n; t++) {
    if (!(rv[t] > 0)) continue;
    const b = bucketOfReturn(t);
    const lv = Math.log(rv[t]);
    logSums[b] += lv;
    counts[b]++;
    totalLog += lv;
    totalCount++;
  }
  // Every bucket needs support — a sample that does not cover the day
  // (e.g. 500 one-minute candles) cannot identify a diurnal profile
  if (totalCount < buckets * 5 || counts.some(c => c < 5)) return null;
  const overallLog = totalLog / totalCount;

  let varLog = 0;
  for (let t = 0; t < n; t++) {
    if (!(rv[t] > 0)) continue;
    varLog += (Math.log(rv[t]) - overallLog) ** 2;
  }
  varLog /= totalCount;
  if (!(varLog > 0)) return null;

  // Significance gate: χ² of bucket log-means against their sampling noise.
  // The per-observation variance is inflated ×2.25 (≈1.5² for volatility
  // clustering shrinking the effective sample), so the diurnal profile of
  // pure noise does not trigger deseasonalization.
  const rawLog = logSums.map((s, b) => s / counts[b] - overallLog);
  let chi2 = 0;
  for (let b = 0; b < buckets; b++) {
    chi2 += (counts[b] * rawLog[b] * rawLog[b]) / (varLog * 2.25);
  }
  const pValue = chi2Survival(chi2, buckets - 1);

  const smoothedLog = rawLog.map(
    (_, b) => 0.25 * rawLog[(b + buckets - 1) % buckets] + 0.5 * rawLog[b] + 0.25 * rawLog[(b + 1) % buckets],
  );
  // Shrink toward 0 by bucket support so thin buckets cannot inject noise.
  // The prior is light (5 pseudo-obs): the χ² gate is the real protection
  // against fitting noise, and a heavy prior halves genuine profiles on
  // realistic windows (~17 obs/bucket), leaving residual seasonality.
  let factors = smoothedLog.map((v, b) => Math.exp(v * (counts[b] / (counts[b] + 5))));
  // Sample-weighted normalization keeps the overall variance level unchanged
  let m = 0;
  for (let t = 0; t < n; t++) m += factors[bucketOfReturn(t)];
  m /= n;
  factors = factors.map(v => v / m);

  const ratio = Math.max(...factors) / Math.min(...factors);
  if (ratio < 1.25 || pValue > 0.01) return null;

  return { factors, bucketOfReturn };
}

/**
 * Rescale each candle's log-moves by 1/√f(bucket) so the deseasonalized
 * series has a flat diurnal profile. Gaps (open vs prev close) are scaled
 * with the same factor; OHLC ordering is preserved (monotone log map).
 */
export function deseasonalizeCandles(candles: Candle[], season: Seasonality): Candle[] {
  const out: Candle[] = [candles[0]];
  let close = candles[0].close;
  for (let t = 0; t < candles.length - 1; t++) {
    const c = candles[t + 1];
    const prevOriginalClose = candles[t].close;
    const g = 1 / Math.sqrt(season.factors[season.bucketOfReturn(t)]);
    const open = close * Math.pow(c.open / prevOriginalClose, g);
    const newClose = open * Math.pow(c.close / c.open, g);
    const high = open * Math.pow(c.high / c.open, g);
    const low = open * Math.pow(c.low / c.open, g);
    out.push({ ...c, open, high, low, close: newClose });
    close = newClose;
  }
  return out;
}

// ── Model combination ─────────────────────────────────────────

type ModelType = PredictionResult['modelType'];

/** Recursion spec for model-implied horizon simulation (model scale, deseasonalized). */
type SimMember =
  | { kind: 'garch'; weight: number; omega: number; alpha: number; gamma: number; beta: number; v1: number }
  | { kind: 'egarch'; weight: number; omega: number; alpha: number; gamma: number; beta: number; logv1: number; eAbsZ: number; mbar: number }
  | { kind: 'rgarch'; weight: number; omega: number; beta: number; gamma: number; xi: number; tau1: number; tau2: number; sigmaU: number; logv1: number }
  | { kind: 'flat'; weight: number; path: number[] };

/**
 * Cached calibration state threaded between rolling refits: previous
 * optima become warm starts with a reduced multi-start budget, and the
 * HAR spec search collapses to the previously selected configuration.
 */
export interface WarmState {
  garch?: GarchParams;
  garchForget?: GarchParams;
  egarch?: EgarchParams;
  gjr?: GjrGarchParams;
  rgarch?: RealizedGarchParams;
  harLags?: [number, number, number];
  harLog?: boolean;
  novasWeights?: number[];
}

interface MemberFit {
  modelType: ModelType;
  varianceSeries: number[];
  forecast: VolatilityForecast;
  persistence: number;
  converged: boolean;
  warmup: number;
  oosQlike: number;
  weight: number;
  sim: SimMember;
}

interface FitResult {
  forecast: VolatilityForecast;
  modelType: ModelType;
  converged: boolean;
  persistence: number;
  varianceSeries: number[];
  returns: number[];
  df: number;
  /** Structural warm-up of the combination (max over members). */
  warmup: number;
  /** Sorted signed r_t/σ_t over the post-warm-up sample — the empirical calibration sample (both tails). */
  zSorted: number[];
  /** Kept combination members with recursions for horizon simulation. */
  simMembers: SimMember[];
  /** Combination weight per model family, summing to 1. */
  weights: Partial<Record<ModelType, number>>;
}

/**
 * Candidate HAR lag triples for this interval and sample size.
 *
 * The textbook (1, 5, 22) encodes *daily equity* horizons (day/week/month
 * in trading days) and is wrong for intraday 24/7 markets. Candidates are
 * built from the candle interval itself — one bar / one day / one week in
 * bars — capped by what the sample can support (long lag ≤ n/5 and enough
 * rows for the OLS). The final choice among candidates is made by QLIKE
 * on the out-of-sample region, not by convention.
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
 * `warmup` is the combination's own structural warm-up (longest lag /
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
  // simMembers stay on the model scale on purpose: the simulated corridor
  // multiplier is a standardized ratio, so a uniform scale c cancels.
}

const OOS_TRAIN_RATIO = 0.75;

/**
 * Fit all candidate models, score them by out-of-sample QLIKE, and combine.
 *
 * Candidates are calibrated on the first 75% of the sample and their
 * one-step variance forecasts scored by QLIKE on the held-out 25% —
 * in-sample QLIKE favors the OLS-calibrated models (HAR, NoVaS stage 2)
 * exactly as much as they overfit. Final parameters are refitted on the
 * full sample.
 *
 * Combination instead of selection: n·QLIKE behaves like a deviance, so
 * weights w ∝ exp(−0.5·n_eval·ΔQLIKE) collapse to the winner when the gap
 * is decisive and average when candidates are within noise of each other
 * (forecast combination robustly beats picking one model on short samples).
 */
/** Forgetting factor of the adaptive GARCH candidate (half-life ≈ 69 bars). */
const FORGET_LAMBDA = 0.99;

function fitModel(candles: Candle[], periodsPerYear: number, steps: number, warm?: WarmState): FitResult {
  const returns = calculateReturns(candles);
  const rv = perCandleParkinson(candles, returns);
  const nReturns = returns.length;

  const nTrain = Math.floor(candles.length * OOS_TRAIN_RATIO);
  const trainCandles = candles.slice(0, nTrain);
  const evalStart = nTrain - 1; // first out-of-sample index in return space
  const nEval = nReturns - evalStart;

  const members: MemberFit[] = [];

  // ── GARCH (λ = 1) and adaptive GARCH (exponential forgetting) ──
  // Two candidates from the same recursion: the forgetting variant tracks
  // regime shifts, the plain one wins on stationary stretches — the OOS
  // score decides which regime the data is in.
  for (const forgetting of [1, FORGET_LAMBDA]) {
    try {
      const warmStart = forgetting === 1 ? warm?.garch : warm?.garchForget;
      const trainFit = new Garch(trainCandles, { periodsPerYear }).fit({ forgetting, warmStart });
      const model = new Garch(candles, { periodsPerYear });
      const oos = qlike(model.getVarianceSeries(trainFit.params).slice(evalStart), rv.slice(evalStart));
      const fit = model.fit({ forgetting, warmStart });
      const fc = model.forecast(fit.params, steps);
      if (warm) {
        if (forgetting === 1) warm.garch = fit.params;
        else warm.garchForget = fit.params;
      }
      members.push({
        modelType: 'garch',
        varianceSeries: model.getVarianceSeries(fit.params),
        forecast: fc,
        persistence: fit.params.persistence,
        converged: fit.diagnostics.converged,
        warmup: 0,
        oosQlike: oos,
        weight: 0,
        sim: { kind: 'garch', weight: 0, omega: fit.params.omega, alpha: fit.params.alpha, gamma: 0, beta: fit.params.beta, v1: fc.variance[0] },
      });
    } catch { /* degenerate data — other members cover */ }
  }

  // ── EGARCH ──
  try {
    const trainFit = new Egarch(trainCandles, { periodsPerYear }).fit({ warmStart: warm?.egarch });
    const model = new Egarch(candles, { periodsPerYear });
    const oos = qlike(model.getVarianceSeries(trainFit.params).slice(evalStart), rv.slice(evalStart));
    const fit = model.fit({ warmStart: warm?.egarch });
    const fc = model.forecast(fit.params, steps);
    if (warm) warm.egarch = fit.params;
    members.push({
      modelType: 'egarch',
      varianceSeries: model.getVarianceSeries(fit.params),
      forecast: fc,
      persistence: fit.params.persistence,
      converged: fit.diagnostics.converged,
      warmup: 0,
      oosQlike: oos,
      weight: 0,
      sim: {
        kind: 'egarch',
        weight: 0,
        omega: fit.params.omega,
        alpha: fit.params.alpha,
        gamma: fit.params.gamma,
        beta: fit.params.beta,
        logv1: Math.log(Math.max(fc.variance[0], 1e-300)),
        eAbsZ: expectedAbsStudentT(fit.params.df),
        mbar: model.magnitudeDrift(fit.params),
      },
    });
  } catch { /* degenerate data */ }

  // ── GJR-GARCH ──
  try {
    const trainFit = new GjrGarch(trainCandles, { periodsPerYear }).fit({ warmStart: warm?.gjr });
    const model = new GjrGarch(candles, { periodsPerYear });
    const oos = qlike(model.getVarianceSeries(trainFit.params).slice(evalStart), rv.slice(evalStart));
    const fit = model.fit({ warmStart: warm?.gjr });
    const fc = model.forecast(fit.params, steps);
    if (warm) warm.gjr = fit.params;
    members.push({
      modelType: 'gjr-garch',
      varianceSeries: model.getVarianceSeries(fit.params),
      forecast: fc,
      persistence: fit.params.persistence,
      converged: fit.diagnostics.converged,
      warmup: 0,
      oosQlike: oos,
      weight: 0,
      sim: { kind: 'garch', weight: 0, omega: fit.params.omega, alpha: fit.params.alpha, gamma: fit.params.gamma, beta: fit.params.beta, v1: fc.variance[0] },
    });
  } catch { /* degenerate data */ }

  // ── Realized GARCH ──
  try {
    const trainFit = new RealizedGarch(trainCandles, { periodsPerYear }).fit({ warmStart: warm?.rgarch });
    const model = new RealizedGarch(candles, { periodsPerYear });
    const oos = qlike(model.getVarianceSeries(trainFit.params).slice(evalStart), rv.slice(evalStart));
    const fit = model.fit({ warmStart: warm?.rgarch });
    const fc = model.forecast(fit.params, steps);
    if (warm) warm.rgarch = fit.params;
    members.push({
      modelType: 'realized-garch',
      varianceSeries: model.getVarianceSeries(fit.params),
      forecast: fc,
      persistence: fit.params.persistence,
      converged: fit.diagnostics.converged,
      warmup: 0,
      oosQlike: oos,
      weight: 0,
      sim: {
        kind: 'rgarch',
        weight: 0,
        omega: fit.params.omega,
        beta: fit.params.beta,
        gamma: fit.params.gamma,
        xi: fit.params.xi,
        tau1: fit.params.tau1,
        tau2: fit.params.tau2,
        sigmaU: fit.params.sigmaU,
        logv1: Math.log(Math.max(fc.variance[0], 1e-300)),
      },
    });
  } catch { /* degenerate data */ }

  // ── HAR-RV (lag triple AND level/log spec chosen by the same OOS score) ──
  try {
    let bestLags: [number, number, number] | null = null;
    let bestLog = false;
    let bestScore = Infinity;
    // Warm state pins the previously selected configuration — a spec search
    // per rolling refit would mostly rediscover the same one
    const lagCandidates: Array<[number, number, number]> = warm?.harLags
      ? [warm.harLags]
      : selectHarLagCandidates(nTrain - 1, periodsPerYear);
    const specCandidates = warm?.harLags ? [warm.harLog ?? false] : [false, true];
    for (const [shortLag, mediumLag, longLag] of lagCandidates) {
      for (const logSpec of specCandidates) {
        try {
          const trainFit = new HarRv(trainCandles, { periodsPerYear, shortLag, mediumLag, longLag, logSpec }).fit();
          if (trainFit.params.persistence >= 1 || trainFit.params.r2 < 0) continue;
          const full = new HarRv(candles, { periodsPerYear, shortLag, mediumLag, longLag, logSpec });
          const score = qlike(full.getVarianceSeries(trainFit.params).slice(evalStart), rv.slice(evalStart));
          if (score < bestScore) {
            bestScore = score;
            bestLags = [shortLag, mediumLag, longLag];
            bestLog = logSpec;
          }
        } catch {
          continue;
        }
      }
    }
    if (bestLags) {
      const [shortLag, mediumLag, longLag] = bestLags;
      const model = new HarRv(candles, { periodsPerYear, shortLag, mediumLag, longLag, logSpec: bestLog });
      const fit = model.fit();
      if (fit.params.persistence < 1 && fit.params.r2 >= 0) {
        const fc = model.forecast(fit.params, steps);
        if (warm) {
          warm.harLags = bestLags;
          warm.harLog = bestLog;
        }
        members.push({
          modelType: 'har-rv',
          varianceSeries: model.getVarianceSeries(fit.params),
          forecast: fc,
          persistence: fit.params.persistence,
          converged: fit.diagnostics.converged,
          warmup: longLag,
          oosQlike: bestScore,
          weight: 0,
          sim: { kind: 'flat', weight: 0, path: fc.variance },
        });
      }
    }
  } catch { /* degenerate data */ }

  // ── NoVaS ──
  try {
    const lags = adaptiveNovasLags(nReturns);
    const warmWeights = warm?.novasWeights && warm.novasWeights.length === lags + 1
      ? warm.novasWeights
      : undefined;
    const trainFit = new NoVaS(trainCandles, { periodsPerYear, lags }).fit({ warmWeights });
    if (trainFit.params.persistence < 1) {
      const model = new NoVaS(candles, { periodsPerYear, lags });
      const oos = qlike(model.getForecastVarianceSeries(trainFit.params).slice(evalStart), rv.slice(evalStart));
      const fit = model.fit({ warmWeights });
      if (fit.params.persistence < 1) {
        const fc = model.forecast(fit.params, steps);
        if (warm) warm.novasWeights = fit.params.weights;
        members.push({
          modelType: 'novas',
          varianceSeries: model.getForecastVarianceSeries(fit.params),
          forecast: fc,
          persistence: fit.params.persistence,
          converged: fit.diagnostics.converged,
          warmup: lags,
          oosQlike: oos,
          weight: 0,
          sim: { kind: 'flat', weight: 0, path: fc.variance },
        });
      }
    }
  } catch { /* degenerate data */ }

  if (members.length === 0) {
    throw new Error('All volatility models failed to fit');
  }

  // ── Weights from OOS QLIKE ──
  const finiteScores = members.map(m => m.oosQlike).filter(q => isFinite(q));
  const qmin = finiteScores.length > 0 ? Math.min(...finiteScores) : NaN;
  let wsum = 0;
  for (const m of members) {
    const d = isFinite(m.oosQlike) && isFinite(qmin) ? m.oosQlike - qmin : isFinite(qmin) ? Infinity : 0;
    m.weight = Math.exp(-0.5 * nEval * d);
    wsum += m.weight;
  }
  for (const m of members) m.weight /= wsum;
  const kept = members.filter(m => m.weight >= 0.02);
  const keptSum = kept.reduce((s, m) => s + m.weight, 0);
  for (const m of kept) m.weight /= keptSum;

  const top = kept.reduce((a, b) => (b.weight > a.weight ? b : a));

  // ── Combine ──
  const combinedSeries = new Array(nReturns).fill(0);
  for (const m of kept) {
    for (let i = 0; i < nReturns; i++) combinedSeries[i] += m.weight * m.varianceSeries[i];
  }
  const combinedVariance = new Array(steps).fill(0);
  for (const m of kept) {
    for (let h = 0; h < steps; h++) combinedVariance[h] += m.weight * m.forecast.variance[h];
  }

  const best: FitResult = {
    forecast: {
      variance: combinedVariance,
      volatility: combinedVariance.map(v => Math.sqrt(v)),
      annualized: combinedVariance.map(v => Math.sqrt(v * periodsPerYear) * 100),
    },
    modelType: top.modelType,
    converged: top.converged,
    persistence: top.persistence,
    varianceSeries: combinedSeries,
    returns,
    df: 5,
    warmup: Math.max(...kept.map(m => m.warmup)),
    zSorted: [],
    simMembers: kept.map(m => ({ ...m.sim, weight: m.weight })),
    weights: kept.reduce<Partial<Record<ModelType, number>>>((acc, m) => {
      acc[m.modelType] = (acc[m.modelType] ?? 0) + m.weight;
      return acc;
    }, {}),
  };

  // Calibrate the combination to the return scale: QLIKE picks the best RV
  // forecasters, but the corridor needs the return-variance scale.
  // Warm-up comes from the combination's own structure (its longest lag /
  // seeding region), capped so calibration keeps a usable sample.
  const warmup = Math.min(Math.max(best.warmup, 10), Math.floor(nReturns / 4));
  best.warmup = warmup;

  const c = varianceScaleCorrection(best.returns, best.varianceSeries, warmup);
  applyScale(best, c, periodsPerYear);

  // Re-profile tail thickness on the corrected residuals — this df drives
  // the model half of the corridor quantile.
  best.df = profileStudentTDf(best.returns, best.varianceSeries);

  // Empirical calibration sample: signed z_t of the corrected residuals —
  // the two tails are calibrated separately (return distributions are
  // skewed; folding them into |z| hides that)
  const zs: number[] = [];
  for (let i = warmup; i < nReturns; i++) {
    const v = best.varianceSeries[i];
    if (!(v > 0) || !isFinite(v)) continue;
    const z = best.returns[i] / Math.sqrt(v);
    if (isFinite(z)) zs.push(z);
  }
  zs.sort((a, b) => a - b);
  best.zSorted = zs;

  return best;
}

// ── Model-implied horizon quantile by simulation ──────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function simRandn(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Horizon-simulation sizing: two tails × SIM_TAIL_DRAWS target samples,
// bounded from both sides, with a hard cap on total work (B·steps).
const SIM_TAIL_DRAWS = 200;
const SIM_MIN_DRAWS = 500;
const SIM_MAX_DRAWS = 40_000;
const SIM_WORK_BUDGET = 20_000_000;

/**
 * Marsaglia–Tsang gamma sampler (shape a ≥ 1).
 *
 * Rejection sampling accepts >96% of proposals for a ≥ 1, so the caps are
 * never reached on valid inputs — they are a hard termination guarantee:
 * a NaN slipping into the parameters would otherwise make every accept
 * condition false and spin both loops forever.
 */
function gammaSample(a: number, rng: () => number): number {
  const d = a - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (let iter = 0; iter < 100; iter++) {
    let x = simRandn(rng);
    let v = 1 + c * x;
    for (let inner = 0; inner < 100 && v <= 0; inner++) {
      x = simRandn(rng);
      v = 1 + c * x;
    }
    if (!(v > 0)) break;
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
  // Degrades a single draw to the distribution mean — never hangs
  return a;
}

/** Sampler for standardized (unit-variance) Student-t(df); Gaussian for df > 100. */
function makeTSampler(df: number, rng: () => number): () => number {
  if (!isFinite(df) || df > 100) return () => simRandn(rng);
  const scale = Math.sqrt((df - 2) / df);
  return () => {
    const chi2 = 2 * gammaSample(df / 2, rng);
    return (simRandn(rng) / Math.sqrt(chi2 / df)) * scale;
  };
}

/**
 * Model-implied h-step standardized-sum tail quantiles by simulation
 * through the fitted recursions (mixture across combination members).
 *
 * Replaces the old zGauss + (zT − zGauss)/h interpolation: simulation
 * captures volatility feedback within the horizon (a shock raises later
 * σ's), the seasonal σ-path weighting, and the CLT decay of fat tails —
 * all of which the linear interpolation only approximated. Sums are kept
 * signed, so leverage (γ in GJR/EGARCH, τ₁ in Realized GARCH) produces
 * genuinely asymmetric tails. Innovations are standardized t(df) draws
 * with the profiled df; the seed is fixed so results are deterministic.
 */
function simulateHorizonTails(
  fit: FitResult,
  steps: number,
  confidence: number,
  factorPath?: number[],
): { up: number; down: number } | null {
  const members = fit.simMembers;
  if (!members || members.length === 0) return null;
  const f = factorPath && factorPath.length >= steps ? factorPath : new Array(steps).fill(1);
  const rng = mulberry32(0x5eed1e55);
  const drawZ = makeTSampler(fit.df, rng);
  // Draw count: enough that each requested tail holds ≥ SIM_TAIL_DRAWS
  // samples, clamped so B·steps never exceeds SIM_WORK_BUDGET path-steps —
  // runtime stays bounded no matter what horizon or confidence is asked.
  const B = Math.max(
    SIM_MIN_DRAWS,
    Math.min(
      SIM_MAX_DRAWS,
      Math.ceil((2 * SIM_TAIL_DRAWS) / Math.max(1 - confidence, 1e-4)),
      Math.floor(SIM_WORK_BUDGET / steps),
    ),
  );

  const cum: number[] = [];
  let acc = 0;
  for (const m of members) {
    acc += m.weight;
    cum.push(acc);
  }

  const draws: number[] = new Array(B);
  for (let b = 0; b < B; b++) {
    const u = rng() * acc;
    let mi = 0;
    while (mi < cum.length - 1 && u > cum[mi]) mi++;
    const m = members[mi];

    let num = 0;
    let den = 0;
    if (m.kind === 'flat') {
      for (let j = 0; j < steps; j++) {
        const v = m.path[Math.min(j, m.path.length - 1)] * f[j];
        if (!(v > 0)) continue;
        num += Math.sqrt(v) * drawZ();
        den += v;
      }
    } else if (m.kind === 'egarch') {
      let lv = m.logv1;
      for (let j = 0; j < steps; j++) {
        const v = Math.exp(lv) * f[j];
        const z = drawZ();
        num += Math.sqrt(v) * z;
        den += v;
        lv = m.omega + m.alpha * (Math.abs(z) + m.mbar - m.eAbsZ) + m.gamma * z + m.beta * lv;
        lv = Math.max(-50, Math.min(50, lv));
      }
    } else if (m.kind === 'rgarch') {
      let lv = m.logv1;
      for (let j = 0; j < steps; j++) {
        const v = Math.exp(lv) * f[j];
        const z = drawZ();
        num += Math.sqrt(v) * z;
        den += v;
        const lnRvSim = m.xi + lv + m.tau1 * z + m.tau2 * (z * z - 1) + m.sigmaU * simRandn(rng);
        lv = m.omega + m.beta * lv + m.gamma * lnRvSim;
        lv = Math.max(-50, Math.min(50, lv));
      }
    } else {
      let v = m.v1;
      for (let j = 0; j < steps; j++) {
        const vf = v * f[j];
        const z = drawZ();
        num += Math.sqrt(vf) * z;
        den += vf;
        const innov = v * z * z;
        v = m.omega + m.alpha * innov + (z < 0 ? m.gamma * innov : 0) + m.beta * v;
        if (!(v > 0) || !isFinite(v)) v = m.v1;
      }
    }
    draws[b] = den > 0 ? num / Math.sqrt(den) : 0;
  }

  draws.sort((a, b) => a - b);
  const pTail = (1 - confidence) / 2;
  const up = empiricalQuantile(draws, 1 - pTail);
  const down = -empiricalQuantile(draws, pTail);
  if (!isFinite(up) || !isFinite(down) || up <= 0 || down <= 0) return null;
  return { up, down };
}

/**
 * Signed h-step standardized-sum sample: Σr / √(Σσ²) over overlapping
 * windows of the post-warm-up region. This is the h-step analog of
 * fit.zSorted — it absorbs whatever the single-period model misses about
 * aggregation (volatility autocorrelation, Jensen bias in EGARCH
 * multi-step, fat tails washing out by CLT), separately per tail.
 */
function horizonZ(fit: FitResult, steps: number): number[] {
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
    const z = sumR / Math.sqrt(sumV);
    if (isFinite(z)) out.push(z);
  }

  out.sort((a, b) => a - b);
  return out;
}

/**
 * Corridor multipliers (upper and lower, calibrated separately) for a
 * two-sided confidence level at horizon `steps`.
 *
 * Each tail carries (1−confidence)/2 mass. The empirical quantile of the
 * signed standardized (h-step) return anchors each tail where the sample
 * supports it, and the model quantile takes over as that tail runs out of
 * observations. The blend weight per tail is the expected number of tail
 * exceedances m = n_eff·(1−confidence)/2 shrunk by a prior of 5 pseudo-
 * observations; overlapping h-step windows are discounted by 1/steps.
 *
 * The model half is the (symmetric) fitted t(df) quantile at one step and
 * the simulated model-implied tails (simulateHorizonTails — asymmetric
 * through the fitted leverage terms) at longer horizons.
 */
function corridorZBounds(
  fit: FitResult,
  confidence: number,
  steps = 1,
  factorPath?: number[],
): { up: number; down: number } {
  const zGauss = probit(confidence);
  const zT = studentTProbit(confidence, fit.df);
  let modelUp: number;
  let modelDown: number;
  if (steps === 1) {
    modelUp = zT;
    modelDown = zT;
  } else {
    const sim = simulateHorizonTails(fit, steps, confidence, factorPath);
    const fallback = zGauss + (zT - zGauss) / steps;
    modelUp = sim ? sim.up : fallback;
    modelDown = sim ? sim.down : fallback;
  }

  const zs = steps === 1 ? fit.zSorted : horizonZ(fit, steps);
  const n = zs.length;
  if (n < 50) return { up: modelUp, down: modelDown };

  const pTail = (1 - confidence) / 2;
  const empUp = empiricalQuantile(zs, 1 - pTail);
  const empDown = -empiricalQuantile(zs, pTail);

  const effN = n / steps; // overlap discount
  const tailCount = effN * pTail;
  const w = tailCount / (tailCount + 5);

  const up = isFinite(empUp) && empUp > 0 ? w * empUp + (1 - w) * modelUp : modelUp;
  const down = isFinite(empDown) && empDown > 0 ? w * empDown + (1 - w) * modelDown : modelDown;
  return { up, down };
}

/** Reliability checks as explainable warnings; `reliable` = none critical fired. */
function collectFitWarnings(fit: FitResult, warnings: PredictionWarning[]): void {
  if (!fit.converged) {
    warnings.push({
      code: 'NOT_CONVERGED',
      critical: true,
      message: 'The volatility optimizer did not converge — the corridor cannot be trusted. More candles usually helps.',
    });
  }
  if (fit.persistence >= 0.999) {
    warnings.push({
      code: 'HIGH_PERSISTENCE',
      critical: true,
      message: 'Volatility persistence hit the stationarity boundary (≥0.999) — long-run variance is unidentified and the corridor is unstable.',
    });
  }

  // Degenerate forecast: variance collapsed to the numerical clamp
  // (flat market, HAR/NoVaS 1e-20 floor) — a zero-width corridor is never
  // a reliable market forecast. Floor is relative to the sample variance
  // so legitimately low-volatility series are not flagged; a zero-variance
  // (flat) return series is always degenerate.
  const sv = sampleVariance(fit.returns);
  if (!(sv > 0) || !(fit.forecast.variance[0] > sv * 1e-8)) {
    warnings.push({
      code: 'DEGENERATE_VARIANCE',
      critical: true,
      message: 'Forecast variance collapsed to the numerical floor (flat or degenerate market) — a zero-width corridor is not a forecast.',
    });
    return; // Ljung-Box on degenerate residuals is meaningless
  }

  // Ljung-Box on squared standardized residuals
  const { returns, varianceSeries } = fit;
  const squared = returns.map((r, i) => {
    const z = r / Math.sqrt(varianceSeries[i]);
    return z * z;
  });
  const lb = ljungBox(squared, 10);
  if (!(lb.pValue >= 0.05)) {
    warnings.push({
      code: 'RESIDUAL_AUTOCORRELATION',
      critical: true,
      message: `Squared residuals stay autocorrelated (Ljung-Box p=${lb.pValue.toFixed(3)}) — the model did not fully capture volatility clustering; the corridor may understate risk.`,
    });
  }
}

// ── Prediction ────────────────────────────────────────────────

function runPredict(
  candles: Candle[],
  interval: CandleInterval,
  steps: number,
  currentPrice: number,
  confidence: number,
  warm?: WarmState,
): PredictionResult {
  // Hard bound on the horizon: forecasts, seasonal paths, and the horizon
  // simulation are all O(steps) — an unbounded horizon is unbounded work,
  // and a corridor further out than the whole sample is meaningless anyway.
  if (steps > candles.length) {
    throw new InvalidArgumentError(`steps must not exceed the sample length (${candles.length}), got ${steps}`);
  }
  assertTimestampOrder(candles);
  const warnings: PredictionWarning[] = [];
  collectDataWarnings(candles, interval, warnings);

  const periodsPerYear = INTERVALS_PER_YEAR[interval];
  const nReturns = candles.length - 1;

  // Deseasonalize, fit in flat-profile space, reseasonalize the forecast
  const season = computeSeasonality(candles, interval);
  const fitCandles = season ? deseasonalizeCandles(candles, season) : candles;
  const fit = fitModel(fitCandles, periodsPerYear, steps, warm);

  const factorPath: number[] = new Array(steps).fill(1);
  if (season) {
    for (let h = 0; h < steps; h++) {
      factorPath[h] = season.factors[season.bucketOfReturn(nReturns + h)];
    }
    const variance = fit.forecast.variance.map((v, h) => v * factorPath[h]);
    fit.forecast = {
      variance,
      volatility: variance.map(v => Math.sqrt(v)),
      annualized: variance.map(v => Math.sqrt(v * periodsPerYear) * 100),
    };
  }

  const { up: zUp, down: zDown } = corridorZBounds(fit, confidence, steps, factorPath);

  const cumulativeVariance = fit.forecast.variance.reduce((sum, v) => sum + v, 0);
  const sigma = Math.sqrt(cumulativeVariance);
  const upperPrice = currentPrice * Math.exp(zUp * sigma);
  const lowerPrice = currentPrice * Math.exp(-zDown * sigma);

  collectFitWarnings(fit, warnings);

  return {
    modelType: fit.modelType,
    currentPrice,
    sigma,
    df: fit.df,
    zScore: (zUp + zDown) / 2,
    zScoreUp: zUp,
    zScoreDown: zDown,
    move: upperPrice - currentPrice,
    movePercent: (upperPrice / currentPrice - 1) * 100,
    upperPrice,
    lowerPrice,
    reliable: warnings.every(w => !w.critical),
    warnings,
    modelWeights: fit.weights,
    seasonalityDetected: season !== null,
  };
}

/**
 * Stateful predictor for rolling use (bots, backtests): each subsequent
 * predict/predictRange warm-starts every optimizer from the previous
 * window's optimum with a reduced multi-start budget — same math, a
 * fraction of the cost. State is per-instrument: do not share one
 * predictor across symbols.
 */
export function createPredictor(interval: CandleInterval): {
  predict: (candles: Candle[], currentPriceOrOptions?: number | null | PredictOptions, confidence?: number) => PredictionResult;
  predictRange: (candles: Candle[], steps: number, currentPriceOrOptions?: number | null | PredictOptions, confidence?: number) => PredictionResult;
} {
  validateInterval(interval);
  const warm: WarmState = {};
  return {
    predict(candles, currentPriceOrOptions?, confidence = 0.6827) {
      assertMinCandles(candles, interval);
      const args = resolvePredictArgs(candles, currentPriceOrOptions, confidence);
      return runPredict(candles, interval, 1, args.currentPrice, args.confidence, warm);
    },
    predictRange(candles, steps, currentPriceOrOptions?, confidence = 0.6827) {
      assertMinCandles(candles, interval);
      if (!Number.isFinite(steps) || steps < 1) {
        throw new InvalidArgumentError(`steps must be a number >= 1, got ${steps}`);
      }
      const args = resolvePredictArgs(candles, currentPriceOrOptions, confidence);
      return runPredict(candles, interval, Math.floor(steps), args.currentPrice, args.confidence, warm);
    },
  };
}

/**
 * Forecast expected price range for t+1 (next candle).
 *
 * Combines all volatility models weighted by out-of-sample QLIKE,
 * deseasonalizes the diurnal variance profile when one is present,
 * rescales the variance to the return scale (Var(r/σ) = 1), and builds
 * bands P·exp(±z·σ) where z is calibrated on the data itself: the
 * empirical |z| quantile of the standardized residuals blended with the
 * fitted Student-t quantile as the tail runs out of observations (see
 * corridorZ). Empirical coverage tracks the requested confidence without
 * assuming a distributional shape.
 * @param confidence — two-sided probability in (0,1). Default ≈0.6827 (±1σ).
 *   Common values: 0.90, 0.95, 0.99.
 */
export function predict(
  candles: Candle[],
  interval: CandleInterval,
  currentPriceOrOptions?: number | null | PredictOptions,
  confidence = 0.6827,
): PredictionResult {
  assertMinCandles(candles, interval);
  const args = resolvePredictArgs(candles, currentPriceOrOptions, confidence);
  return runPredict(candles, interval, 1, args.currentPrice, args.confidence);
}

/**
 * Forecast expected price range over multiple candles.
 *
 * Cumulative σ = √(σ₁² + σ₂² + ... + σₙ²) — total expected move over N
 * periods, with each step's variance carrying its own seasonal factor.
 * Uses log-normal price bands P·exp(±z·σ) where z is calibrated at the
 * requested horizon: the empirical quantile of |h-step standardized sums|
 * from the sample itself, blended with the model-implied quantile simulated
 * through the fitted recursions (volatility feedback and fat-tail decay
 * included).
 * @param confidence — two-sided probability in (0,1). Default ≈0.6827 (±1σ).
 * @param steps — horizon in candles, 1 ≤ steps ≤ candles.length.
 */
export function predictRange(
  candles: Candle[],
  interval: CandleInterval,
  steps: number,
  currentPriceOrOptions?: number | null | PredictOptions,
  confidence = 0.6827,
): PredictionResult {
  assertMinCandles(candles, interval);
  if (!Number.isFinite(steps) || steps < 1) {
    throw new InvalidArgumentError(`steps must be a number >= 1, got ${steps}`);
  }
  steps = Math.floor(steps);
  const args = resolvePredictArgs(candles, currentPriceOrOptions, confidence);
  return runPredict(candles, interval, steps, args.currentPrice, args.confidence);
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
  /** Statistical judgment of the coverage against the nominal confidence (Kupiec POF test). */
  verdict: 'well-calibrated' | 'too-narrow' | 'too-wide' | 'inconclusive';
  /** Kupiec test p-value: probability of a coverage gap this large under correct calibration. */
  pValue: number;
  /** Plain-language interpretation of the verdict with the numbers filled in. */
  message: string;
}

/**
 * Kupiec (1995) proportion-of-failures test: is the observed hit rate
 * statistically consistent with the nominal confidence, given how many
 * walk-forward points there are? A raw hitRate of 63% vs nominal 68% means
 * nothing without n — this answers "failure or noise" directly.
 */
export function kupiecTest(
  hits: number,
  total: number,
  confidence: number,
): Pick<BacktestStats, 'verdict' | 'pValue' | 'message'> {
  const misses = total - hits;
  const p = 1 - confidence; // nominal failure probability per observation
  const pHat = total > 0 ? misses / total : 0;
  const nominalPct = (confidence * 100).toFixed(1);
  const coveragePct = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';

  // Binomial log-likelihood with the 0·ln(0) := 0 convention
  const ll = (prob: number): number => {
    let v = 0;
    if (misses > 0) v += misses * Math.log(prob);
    if (hits > 0) v += hits * Math.log(1 - prob);
    return v;
  };
  const lr = pHat > 0 && pHat < 1 ? -2 * (ll(p) - ll(pHat)) : -2 * ll(p);
  const pValue = chi2Survival(Math.max(lr, 0), 1);

  if (total < 30) {
    return {
      verdict: 'inconclusive',
      pValue,
      message: `Only ${total} walk-forward points — too few to judge calibration. Aim for ≥30 (more candles or stride: 1).`,
    };
  }
  if (pValue < 0.05) {
    if (pHat > p) {
      return {
        verdict: 'too-narrow',
        pValue,
        message: `Coverage ${coveragePct}% is below the nominal ${nominalPct}% (Kupiec p=${pValue.toFixed(4)}) — the corridor is too narrow: real risk exceeds what it shows.`,
      };
    }
    return {
      verdict: 'too-wide',
      pValue,
      message: `Coverage ${coveragePct}% is above the nominal ${nominalPct}% (Kupiec p=${pValue.toFixed(4)}) — the corridor is too wide: decisions based on it are overly conservative.`,
    };
  }
  return {
    verdict: 'well-calibrated',
    pValue,
    message: `Coverage ${coveragePct}% vs nominal ${nominalPct}% over ${total} points is consistent with a calibrated corridor (Kupiec p=${pValue.toFixed(4)}).`,
  };
}

/**
 * Walk-forward calibration statistics for predict.
 *
 * Refits the model on a rolling window (75% of candles, min MIN_CANDLES)
 * and checks whether the next close lands inside the predicted corridor.
 * A well-calibrated tool has hitRate ≈ confidence·100.
 *
 * Every refit costs a full multi-model calibration, so by default the test
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

  // Rolling refits share warm-start state — same estimates as cold fits up
  // to optimizer tolerance, at a fraction of the multi-start cost
  const warm: WarmState = {};

  for (let i = window; i < candles.length - 1; i += stride) {
    const slice = candles.slice(i - window, i + 1);
    const price = slice[slice.length - 1].close;
    const predicted = runPredict(slice, interval, 1, price, confidence, warm);
    const actual = candles[i + 1].close;

    if (actual >= predicted.lowerPrice && actual <= predicted.upperPrice) {
      hits++;
    }
    total++;
  }

  return { hits, total, hitRate: (hits / total) * 100, ...kupiecTest(hits, total, confidence) };
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
