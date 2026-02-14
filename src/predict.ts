import type { Candle, VolatilityForecast } from './types.js';
import { Garch } from './garch.js';
import { Egarch } from './egarch.js';
import { HarRv } from './har.js';
import { calculateReturnsFromPrices, checkLeverageEffect, ljungBox } from './utils.js';

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
  modelType: 'garch' | 'egarch' | 'har-rv';
  reliable: boolean;
}

function assertMinCandles(candles: Candle[], interval: CandleInterval): void {
  const min = MIN_CANDLES[interval];
  if (candles.length < min) {
    throw new Error(`Need at least ${min} candles for ${interval} interval, got ${candles.length}`);
  }
}

interface FitResult {
  forecast: VolatilityForecast;
  modelType: 'garch' | 'egarch' | 'har-rv';
  converged: boolean;
  persistence: number;
  varianceSeries: number[];
  returns: number[];
}

function fitGarchFamily(candles: Candle[], periodsPerYear: number, steps: number): FitResult & { aic: number } {
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
      aic: fit.diagnostics.aic,
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
    aic: fit.diagnostics.aic,
  };
}

function fitHarRv(candles: Candle[], periodsPerYear: number, steps: number): (FitResult & { aic: number }) | null {
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
      aic: fit.diagnostics.aic,
    };
  } catch {
    return null;
  }
}

function fitModel(candles: Candle[], periodsPerYear: number, steps: number): FitResult {
  const garchResult = fitGarchFamily(candles, periodsPerYear, steps);
  const harResult = fitHarRv(candles, periodsPerYear, steps);

  // Pick model with lower AIC (better fit)
  if (harResult && harResult.aic < garchResult.aic) {
    return harResult;
  }

  return garchResult;
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
  assertMinCandles(candles, interval);
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
  assertMinCandles(candles, interval);
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
const BACKTEST_WINDOW_RATIO = 0.75;

/**
 * Walk-forward backtest of predict.
 *
 * Window is computed automatically: 75% of candles for fitting, 25% for testing.
 * Throws if not enough candles for the given interval.
 * Returns true if the model's hit rate meets the required threshold.
 * Default threshold is 68% (±1σ should contain ~68% of moves).
 */
export function backtest(
  candles: Candle[],
  interval: CandleInterval,
  requiredPercent = BACKTEST_REQUIRED_PERCENT,
): boolean {
  assertMinCandles(candles, interval);

  const window = Math.max(MIN_CANDLES[interval], Math.floor(candles.length * BACKTEST_WINDOW_RATIO));
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

