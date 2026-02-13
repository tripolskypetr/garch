interface Candle {
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    timestamp?: number;
}
interface GarchParams {
    omega: number;
    alpha: number;
    beta: number;
    persistence: number;
    unconditionalVariance: number;
    annualizedVol: number;
}
interface EgarchParams {
    omega: number;
    alpha: number;
    gamma: number;
    beta: number;
    persistence: number;
    unconditionalVariance: number;
    annualizedVol: number;
    leverageEffect: number;
}
interface CalibrationResult<T> {
    params: T;
    diagnostics: {
        logLikelihood: number;
        aic: number;
        bic: number;
        iterations: number;
        converged: boolean;
    };
}
interface VolatilityForecast {
    variance: number[];
    volatility: number[];
    annualized: number[];
}
interface LeverageStats {
    negativeVol: number;
    positiveVol: number;
    ratio: number;
    recommendation: 'garch' | 'egarch';
}
interface OptimizerResult {
    x: number[];
    fx: number;
    iterations: number;
    converged: boolean;
}

interface GarchOptions {
    periodsPerYear?: number;
    maxIter?: number;
    tol?: number;
}
/**
 * GARCH(1,1) model
 *
 * σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
 *
 * where:
 * - ω (omega) > 0: constant term
 * - α (alpha) ≥ 0: ARCH parameter (reaction to shocks)
 * - β (beta) ≥ 0: GARCH parameter (persistence)
 * - α + β < 1: stationarity condition
 */
declare class Garch {
    private returns;
    private periodsPerYear;
    private initialVariance;
    constructor(data: Candle[] | number[], options?: GarchOptions);
    /**
     * Calibrate GARCH(1,1) parameters using Maximum Likelihood Estimation
     */
    fit(options?: {
        maxIter?: number;
        tol?: number;
    }): CalibrationResult<GarchParams>;
    /**
     * Calculate conditional variance series given parameters
     */
    getVarianceSeries(params: GarchParams): number[];
    /**
     * Forecast variance forward
     */
    forecast(params: GarchParams, steps?: number): VolatilityForecast;
    /**
     * Get the return series
     */
    getReturns(): number[];
    /**
     * Get initial variance estimate
     */
    getInitialVariance(): number;
}
/**
 * Convenience function to calibrate GARCH(1,1) from candles
 */
declare function calibrateGarch(data: Candle[] | number[], options?: GarchOptions): CalibrationResult<GarchParams>;

interface EgarchOptions {
    periodsPerYear?: number;
    maxIter?: number;
    tol?: number;
}
/**
 * EGARCH(1,1) model (Nelson, 1991)
 *
 * ln(σ²ₜ) = ω + α·(|zₜ₋₁| - E[|z|]) + γ·zₜ₋₁ + β·ln(σ²ₜ₋₁)
 *
 * where:
 * - zₜ = εₜ/σₜ (standardized residual)
 * - ω (omega): constant term
 * - α (alpha): magnitude effect
 * - γ (gamma): leverage effect (typically negative)
 * - β (beta): persistence
 * - E[|z|] = √(2/π) for standard normal
 */
declare class Egarch {
    private returns;
    private periodsPerYear;
    private initialVariance;
    constructor(data: Candle[] | number[], options?: EgarchOptions);
    /**
     * Calibrate EGARCH(1,1) parameters using Maximum Likelihood Estimation
     */
    fit(options?: {
        maxIter?: number;
        tol?: number;
    }): CalibrationResult<EgarchParams>;
    /**
     * Calculate conditional variance series given parameters
     */
    getVarianceSeries(params: EgarchParams): number[];
    /**
     * Forecast variance forward
     *
     * Note: EGARCH forecasts are more complex because they depend on
     * the path of shocks. This provides an approximation assuming
     * expected values of future shocks.
     */
    forecast(params: EgarchParams, steps?: number): VolatilityForecast;
    /**
     * Get the return series
     */
    getReturns(): number[];
    /**
     * Get initial variance estimate
     */
    getInitialVariance(): number;
}
/**
 * Convenience function to calibrate EGARCH(1,1) from candles
 */
declare function calibrateEgarch(data: Candle[] | number[], options?: EgarchOptions): CalibrationResult<EgarchParams>;

/**
 * Calculate log returns from candles
 */
declare function calculateReturns(candles: Candle[]): number[];
/**
 * Calculate log returns from price array
 */
declare function calculateReturnsFromPrices(prices: number[]): number[];
/**
 * Calculate sample variance (mean-zero assumption)
 */
declare function sampleVariance(returns: number[]): number;
/**
 * Calculate sample variance with mean adjustment
 */
declare function sampleVarianceWithMean(returns: number[]): number;
/**
 * Check for leverage effect (asymmetry in volatility)
 */
declare function checkLeverageEffect(returns: number[]): LeverageStats;
/**
 * Garman-Klass (1980) variance estimator using OHLC data.
 *
 * σ²_GK = (1/n) Σ [ 0.5·(ln(H/L))² − (2ln2−1)·(ln(C/O))² ]
 *
 * ~5x more efficient than close-to-close variance.
 */
declare function garmanKlassVariance(candles: Candle[]): number;
/**
 * Yang-Zhang (2000) variance estimator using OHLC data.
 *
 * Combines overnight (open vs prev close), open-to-close,
 * and Rogers-Satchell components. More efficient than Garman-Klass
 * and handles overnight gaps (relevant for stocks).
 *
 * σ²_YZ = σ²_overnight + k·σ²_close + (1−k)·σ²_RS
 */
declare function yangZhangVariance(candles: Candle[]): number;
/**
 * Expected value of |Z| where Z ~ N(0,1)
 * E[|Z|] = sqrt(2/π)
 */
declare const EXPECTED_ABS_NORMAL: number;
/**
 * Ljung-Box test for autocorrelation.
 *
 * Q = n(n+2) Σ(k=1..m) ρ²_k / (n−k)
 *
 * Under H₀ (no autocorrelation), Q ~ χ²(m).
 * Use on squared standardized residuals to test GARCH adequacy.
 */
declare function ljungBox(data: number[], maxLag: number): {
    statistic: number;
    pValue: number;
};

type CandleInterval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '8h';
interface PredictionResult {
    currentPrice: number;
    sigma: number;
    move: number;
    upperPrice: number;
    lowerPrice: number;
    modelType: 'garch' | 'egarch';
    reliable: boolean;
}
/**
 * Forecast expected price range for t+1 (next candle).
 *
 * Auto-selects GARCH or EGARCH based on leverage effect.
 * Returns ±1σ price corridor so you can set SL/TP yourself.
 */
declare function predict(candles: Candle[], interval: CandleInterval, currentPrice?: number): PredictionResult;
/**
 * Forecast expected price range over multiple candles.
 *
 * Cumulative σ = √(σ₁² + σ₂² + ... + σₙ²) — total expected move over N periods.
 * Use for swing trades where you hold across multiple candles.
 */
declare function predictRange(candles: Candle[], interval: CandleInterval, steps: number, currentPrice?: number): PredictionResult;
/**
 * Walk-forward backtest of predict.
 *
 * Window is computed automatically: 75% of candles for fitting, 25% for testing.
 * Throws if not enough candles for the given interval.
 * Returns true if the model's hit rate meets the required threshold.
 * Default threshold is 68% (±1σ should contain ~68% of moves).
 */
declare function backtest(candles: Candle[], interval: CandleInterval, requiredPercent?: number): boolean;

declare function nelderMead(fn: (x: number[]) => number, x0: number[], options?: {
    maxIter?: number;
    tol?: number;
    alpha?: number;
    gamma?: number;
    rho?: number;
    sigma?: number;
}): OptimizerResult;

export { EXPECTED_ABS_NORMAL, Egarch, Garch, backtest, calculateReturns, calculateReturnsFromPrices, calibrateEgarch, calibrateGarch, checkLeverageEffect, garmanKlassVariance, ljungBox, nelderMead, predict, predictRange, sampleVariance, sampleVarianceWithMean, yangZhangVariance };
export type { CalibrationResult, Candle, CandleInterval, EgarchOptions, EgarchParams, GarchOptions, GarchParams, LeverageStats, OptimizerResult, PredictionResult, VolatilityForecast };
