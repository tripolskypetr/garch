import type { Candle, VolatilityForecast, GarchParams, EgarchParams } from './types.js';
import { Garch } from './garch.js';
import { Egarch } from './egarch.js';
import { calculateReturnsFromPrices, checkLeverageEffect, ljungBox } from './utils.js';

export type CandleInterval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '8h';

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
