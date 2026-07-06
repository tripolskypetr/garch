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
  /** true when the regression runs on ln RV (betas live in log space). */
  logSpec?: boolean;
  /** Residual variance of the log-RV regression (lognormal bias correction). */
  residualLogVar?: number;
}

export interface RealizedGarchParams {
  omega: number;
  beta: number;
  gamma: number;
  /** Measurement-equation intercept: ln RV_t = ξ + ln σ²_t + τ₁z + τ₂(z²−1) + u. */
  xi: number;
  tau1: number;
  tau2: number;
  /** Std of the measurement noise u — how much the model trusts RV. */
  sigmaU: number;
  persistence: number;
  unconditionalVariance: number;
  annualizedVol: number;
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
  /** Iterations of the winning Nelder-Mead run (not summed across multi-start restarts). */
  iterations: number;
  converged: boolean;
}
