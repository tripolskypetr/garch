// Models
export { Garch, calibrateGarch, type GarchOptions } from './garch.js';
export { Egarch, calibrateEgarch, type EgarchOptions } from './egarch.js';

// Utilities
export {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  sampleVarianceWithMean,
  checkLeverageEffect,
  garmanKlassVariance,
  yangZhangVariance,
  ljungBox,
  EXPECTED_ABS_NORMAL,
} from './utils.js';

// Prediction
export {
  predict,
  predictRange,
  backtest,
  predictMultiTimeframe,
  type CandleInterval,
  type PredictionResult,
  type MultiTimeframePrediction,
} from './predict.js';

// Optimizer (for advanced usage)
export { nelderMead } from './optimizer.js';

// Types
export type {
  Candle,
  GarchParams,
  EgarchParams,
  CalibrationResult,
  VolatilityForecast,
  LeverageStats,
  OptimizerResult,
} from './types.js';
