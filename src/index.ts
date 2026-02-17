// Models
export { Garch, calibrateGarch, type GarchOptions } from './garch.js';
export { Egarch, calibrateEgarch, type EgarchOptions } from './egarch.js';
export { HarRv, calibrateHarRv, type HarRvOptions } from './har.js';
export { GjrGarch, calibrateGjrGarch, type GjrGarchOptions } from './gjr-garch.js';
export { NoVaS, calibrateNoVaS, type NoVaSOptions } from './novas.js';

// Utilities
export {
  calculateReturns,
  calculateReturnsFromPrices,
  sampleVariance,
  sampleVarianceWithMean,
  checkLeverageEffect,
  garmanKlassVariance,
  yangZhangVariance,
  perCandleParkinson,
  ljungBox,
  EXPECTED_ABS_NORMAL,
  logGamma,
  expectedAbsStudentT,
  studentTNegLL,
  profileStudentTDf,
  qlike,
  probit,
} from './utils.js';

// Prediction
export {
  predict,
  predictRange,
  backtest,
  type CandleInterval,
  type PredictionResult,
} from './predict.js';

// Optimizer (for advanced usage)
export { nelderMead, nelderMeadMultiStart } from './optimizer.js';

// Types
export type {
  Candle,
  GarchParams,
  EgarchParams,
  GjrGarchParams,
  HarRvParams,
  NoVaSParams,
  CalibrationResult,
  VolatilityForecast,
  LeverageStats,
  OptimizerResult,
} from './types.js';
