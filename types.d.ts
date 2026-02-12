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
 * Expected value of |Z| where Z ~ N(0,1)
 * E[|Z|] = sqrt(2/π)
 */
declare const EXPECTED_ABS_NORMAL: number;

declare function nelderMead(fn: (x: number[]) => number, x0: number[], options?: {
    maxIter?: number;
    tol?: number;
    alpha?: number;
    gamma?: number;
    rho?: number;
    sigma?: number;
}): OptimizerResult;

export { EXPECTED_ABS_NORMAL, Egarch, Garch, calculateReturns, calculateReturnsFromPrices, calibrateEgarch, calibrateGarch, checkLeverageEffect, nelderMead, sampleVariance, sampleVarianceWithMean };
export type { CalibrationResult, Candle, EgarchOptions, EgarchParams, GarchOptions, GarchParams, LeverageStats, OptimizerResult, VolatilityForecast };
