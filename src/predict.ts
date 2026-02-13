import type { Candle, VolatilityForecast } from './types.js';
import { Garch } from './garch.js';
import { Egarch } from './egarch.js';
import { calculateReturnsFromPrices, checkLeverageEffect } from './utils.js';

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
  modelType: "garch" | "egarch";
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
  const periodsPerYear = INTERVALS_PER_YEAR[interval];

  const returns = calculateReturnsFromPrices(candles.map(c => c.close));
  const leverage = checkLeverageEffect(returns);

  let forecast: VolatilityForecast;
  let modelType: 'garch' | 'egarch';

  if (leverage.recommendation === 'egarch') {
    const model = new Egarch(candles, { periodsPerYear });
    const fit = model.fit();
    forecast = model.forecast(fit.params, 1);
    modelType = 'egarch';
  } else {
    const model = new Garch(candles, { periodsPerYear });
    const fit = model.fit();
    forecast = model.forecast(fit.params, 1);
    modelType = 'garch';
  }

  const sigma = forecast.volatility[0];
  const move = currentPrice * sigma;

  return {
    modelType,
    currentPrice,
    sigma,
    move,
    upperPrice: currentPrice + move,
    lowerPrice: currentPrice - move,
  };
}
