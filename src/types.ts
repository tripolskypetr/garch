export interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp?: number;
}

export interface GarchParams {
  omega: number;
  alpha: number;
  beta: number;
  persistence: number;
  unconditionalVariance: number;
  annualizedVol: number;
  df: number;
}

export interface EgarchParams {
  omega: number;
  alpha: number;
  gamma: number;
  beta: number;
  persistence: number;
  unconditionalVariance: number;
  annualizedVol: number;
  leverageEffect: number;
  df: number;
}

export interface GjrGarchParams {
  omega: number;
  alpha: number;
  gamma: number;
  beta: number;
  persistence: number;
  unconditionalVariance: number;
  annualizedVol: number;
  leverageEffect: number;
  df: number;
}

export interface CalibrationResult<T> {
  params: T;
  diagnostics: {
    logLikelihood: number;
    aic: number;
    bic: number;
    iterations: number;
    converged: boolean;
  };
}

export interface VolatilityForecast {
  variance: number[];
  volatility: number[];
  annualized: number[];
}

export interface LeverageStats {
  negativeVol: number;
  positiveVol: number;
  ratio: number;
  recommendation: 'garch' | 'egarch';
}

export interface HarRvParams {
  beta0: number;
  betaShort: number;
  betaMedium: number;
  betaLong: number;
  persistence: number;
  unconditionalVariance: number;
  annualizedVol: number;
  r2: number;
  df: number;
}

export interface NoVaSParams {
  weights: number[];
  forecastWeights: number[];
  lags: number;
  persistence: number;
  unconditionalVariance: number;
  annualizedVol: number;
  dSquared: number;
  r2: number;
  df: number;
}

export interface OptimizerResult {
  x: number[];
  fx: number;
  iterations: number;
  converged: boolean;
}
