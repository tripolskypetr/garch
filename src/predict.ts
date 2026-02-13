import type { Candle, VolatilityForecast, GarchParams, EgarchParams } from './types.js';
import { Garch } from './garch.js';
import { Egarch } from './egarch.js';
import { calculateReturnsFromPrices, checkLeverageEffect, ljungBox } from './utils.js';

export type CandleInterval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '8h';

const INTERVAL_MINUTES: Record<CandleInterval, number> = {
  '1m': 1,
  '3m': 3,
  '5m': 5,
  '15m': 15,
  '30m': 30,
  '1h': 60,
  '2h': 120,
  '4h': 240,
  '6h': 360,
  '8h': 480,
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
  currentPrice: number;
  sigma: number;
  move: number;
  upperPrice: number;
  lowerPrice: number;
  modelType: 'garch' | 'egarch';
  reliable: boolean;
}

interface FitResult {
  forecast: VolatilityForecast;
  modelType: 'garch' | 'egarch';
  converged: boolean;
  persistence: number;
  varianceSeries: number[];
  returns: number[];
}

function fitModel(candles: Candle[], periodsPerYear: number, steps: number): FitResult {
  const returns = calculateReturnsFromPrices(candles.map(c => c.close));
  const leverage = checkLeverageEffect(returns);

  if (leverage.recommendation === 'egarch') {
    const model = new Egarch(candles, { periodsPerYear });
    const fit = model.fit();
    return {
      forecast: model.forecast(fit.params, steps),
      modelType: 'egarch',
      converged: fit.diagnostics.converged,
      persistence: fit.params.persistence,
      varianceSeries: model.getVarianceSeries(fit.params),
      returns: model.getReturns(),
    };
  }

  const model = new Garch(candles, { periodsPerYear });
  const fit = model.fit();
  return {
    forecast: model.forecast(fit.params, steps),
    modelType: 'garch',
    converged: fit.diagnostics.converged,
    persistence: fit.params.persistence,
    varianceSeries: model.getVarianceSeries(fit.params),
    returns: model.getReturns(),
  };
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
 * Auto-selects GARCH or EGARCH based on leverage effect.
 * Returns ±1σ price corridor so you can set SL/TP yourself.
 */
export function predict(
  candles: Candle[],
  interval: CandleInterval,
  currentPrice = candles[candles.length - 1].close,
): PredictionResult {
  const fit = fitModel(candles, INTERVALS_PER_YEAR[interval], 1);

  const sigma = fit.forecast.volatility[0];
  const move = currentPrice * sigma;

  return {
    modelType: fit.modelType,
    currentPrice,
    sigma,
    move,
    upperPrice: currentPrice + move,
    lowerPrice: currentPrice - move,
    reliable: checkReliable(fit),
  };
}

/**
 * Forecast expected price range over multiple candles.
 *
 * Cumulative σ = √(σ₁² + σ₂² + ... + σₙ²) — total expected move over N periods.
 * Use for swing trades where you hold across multiple candles.
 */
export function predictRange(
  candles: Candle[],
  interval: CandleInterval,
  steps: number,
  currentPrice = candles[candles.length - 1].close,
): PredictionResult {
  const fit = fitModel(candles, INTERVALS_PER_YEAR[interval], steps);

  const cumulativeVariance = fit.forecast.variance.reduce((sum, v) => sum + v, 0);
  const sigma = Math.sqrt(cumulativeVariance);
  const move = currentPrice * sigma;

  return {
    modelType: fit.modelType,
    currentPrice,
    sigma,
    move,
    upperPrice: currentPrice + move,
    lowerPrice: currentPrice - move,
    reliable: checkReliable(fit),
  };
}

// ── Backtest ──────────────────────────────────────────────────

const BACKTEST_REQUIRED_PERCENT = 68;
const BACKTEST_MIN_WINDOW = 50;
const BACKTEST_MIN_POINTS = 10;
const BACKTEST_WINDOW_RATIO = 0.75;

/**
 * Walk-forward backtest of predict.
 *
 * Window is computed automatically: 75% of candles for fitting, 25% for testing.
 * Throws if fewer than 61 candles (50 min window + 10 test points + 1).
 * Returns true if the model's hit rate meets the required threshold.
 * Default threshold is 68% (±1σ should contain ~68% of moves).
 */
export function backtest(
  candles: Candle[],
  interval: CandleInterval,
  requiredPercent = BACKTEST_REQUIRED_PERCENT,
): boolean {
  const minCandles = BACKTEST_MIN_WINDOW + BACKTEST_MIN_POINTS + 1;
  if (candles.length < minCandles) {
    throw new Error(`Need at least ${minCandles} candles for backtest, got ${candles.length}`);
  }

  const window = Math.max(BACKTEST_MIN_WINDOW, Math.floor(candles.length * BACKTEST_WINDOW_RATIO));
  let hits = 0;
  let total = 0;

  for (let i = window; i < candles.length - 1; i++) {
    const slice = candles.slice(i - window, i + 1);
    const predicted = predict(slice, interval);
    const actual = candles[i + 1].close;

    if (actual >= predicted.lowerPrice && actual <= predicted.upperPrice) {
      hits++;
    }
    total++;
  }

  return (hits / total) * 100 >= requiredPercent;
}

// ── Multi-timeframe ───────────────────────────────────────────

export interface MultiTimeframePrediction {
  primary: PredictionResult;
  secondary: PredictionResult;
  divergence: boolean;
}

/**
 * Compare volatility forecasts across two timeframes.
 *
 * Normalizes both σ to per-hour and checks for divergence.
 * divergence = true when one timeframe sees 2x+ more vol than the other.
 */
export function predictMultiTimeframe(
  primaryCandles: Candle[],
  primaryInterval: CandleInterval,
  secondaryCandles: Candle[],
  secondaryInterval: CandleInterval,
  currentPrice?: number,
): MultiTimeframePrediction {
  const primary = predict(
    primaryCandles,
    primaryInterval,
    currentPrice ?? primaryCandles[primaryCandles.length - 1].close,
  );
  const secondary = predict(
    secondaryCandles,
    secondaryInterval,
    currentPrice ?? secondaryCandles[secondaryCandles.length - 1].close,
  );

  // Normalize σ to hourly: σ_hourly = σ_candle * √(60 / minutes_per_candle)
  const primaryHourly = primary.sigma * Math.sqrt(60 / INTERVAL_MINUTES[primaryInterval]);
  const secondaryHourly = secondary.sigma * Math.sqrt(60 / INTERVAL_MINUTES[secondaryInterval]);

  const ratio = primaryHourly / secondaryHourly;
  const divergence = ratio > 2 || ratio < 0.5;

  return { primary, secondary, divergence };
}
