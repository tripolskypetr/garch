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
  EXPECTED_ABS_NORMAL,
} from './utils.js';

// Prediction
export { predict, type CandleInterval, type PredictionResult } from './predict.js';

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
