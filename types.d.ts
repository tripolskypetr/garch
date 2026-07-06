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
    df: number;
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
    df: number;
}
interface GjrGarchParams {
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
interface HarRvParams {
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
interface NoVaSParams {
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
    private rv;
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
 * - E[|z|] = expectedAbsStudentT(df) for Student-t(df)
 */
declare class Egarch {
    private returns;
    private rv;
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

interface HarRvOptions {
    periodsPerYear?: number;
    shortLag?: number;
    mediumLag?: number;
    longLag?: number;
}
/**
 * HAR-RV model (Corsi, 2009)
 *
 * RV_{t+1} = β₀ + β₁·RV_short + β₂·RV_medium + β₃·RV_long + ε
 *
 * where:
 * - RV_short  = mean(rv[t-s+1..t])  (default s=1)
 * - RV_medium = mean(rv[t-m+1..t])  (default m=5)
 * - RV_long   = mean(rv[t-l+1..t])  (default l=22)
 * - rv[t] = Parkinson(candle_t) for OHLC data, r[t]² for prices-only
 *
 * Parkinson (1980): RV = (1/(4·ln2))·(ln(H/L))², ~5x more efficient than r².
 *
 * Uses OLS for estimation — closed-form, always converges.
 */
declare class HarRv {
    private returns;
    private rv;
    private periodsPerYear;
    private shortLag;
    private mediumLag;
    private longLag;
    constructor(data: Candle[] | number[], options?: HarRvOptions);
    /**
     * Calibrate HAR-RV via OLS.
     */
    fit(): CalibrationResult<HarRvParams>;
    /**
     * Internal: compute variance series from beta vector.
     */
    private getVarianceSeriesInternal;
    /**
     * Calculate conditional variance series given parameters.
     */
    getVarianceSeries(params: HarRvParams): number[];
    /**
     * Forecast variance forward.
     *
     * Uses iterative substitution: each forecast step feeds back
     * into the rolling RV components for subsequent steps.
     */
    forecast(params: HarRvParams, steps?: number): VolatilityForecast;
    /**
     * Get the return series.
     */
    getReturns(): number[];
    /**
     * Get realized variance series (squared returns).
     */
    getRv(): number[];
}
/**
 * Convenience function to calibrate HAR-RV from candles or prices.
 */
declare function calibrateHarRv(data: Candle[] | number[], options?: HarRvOptions): CalibrationResult<HarRvParams>;

interface GjrGarchOptions {
    periodsPerYear?: number;
    maxIter?: number;
    tol?: number;
}
/**
 * GJR-GARCH(1,1) model (Glosten, Jagannathan & Runkle, 1993)
 *
 * σ²ₜ = ω + α·ε²ₜ₋₁ + γ·ε²ₜ₋₁·I(rₜ₋₁<0) + β·σ²ₜ₋₁
 *
 * where:
 * - ω (omega) > 0: constant term
 * - α (alpha) ≥ 0: symmetric shock response
 * - γ (gamma) ≥ 0: asymmetric leverage coefficient
 * - β (beta) ≥ 0: persistence
 * - I(r<0) = 1 when return is negative, 0 otherwise
 * - Stationarity: α + γ/2 + β < 1
 *
 * With Candle[] input, ε² is replaced by Parkinson per-candle RV.
 * Leverage direction still comes from close-to-close return sign.
 */
declare class GjrGarch {
    private returns;
    private rv;
    private periodsPerYear;
    private initialVariance;
    constructor(data: Candle[] | number[], options?: GjrGarchOptions);
    /**
     * Calibrate GJR-GARCH(1,1) parameters using Maximum Likelihood Estimation
     */
    fit(options?: {
        maxIter?: number;
        tol?: number;
    }): CalibrationResult<GjrGarchParams>;
    /**
     * Calculate conditional variance series given parameters
     */
    getVarianceSeries(params: GjrGarchParams): number[];
    /**
     * Forecast variance forward
     */
    forecast(params: GjrGarchParams, steps?: number): VolatilityForecast;
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
 * Convenience function to calibrate GJR-GARCH(1,1) from candles
 */
declare function calibrateGjrGarch(data: Candle[] | number[], options?: GjrGarchOptions): CalibrationResult<GjrGarchParams>;

interface NoVaSOptions {
    periodsPerYear?: number;
    lags?: number;
    maxIter?: number;
    tol?: number;
}
/**
 * NoVaS (Normalizing and Variance-Stabilizing) model (Politis, 2003)
 *
 * Two-stage calibration:
 *
 * Stage 1 — D² minimization (model-free normality):
 *   σ²_t = a_0 + a_1·X²_{t-1} + a_2·X²_{t-2} + ... + a_p·X²_{t-p}
 *   W_t  = X_t / σ_t
 *   Minimize D² = S² + (K - 3)² where S, K are skewness and kurtosis of {W_t}.
 *
 * Stage 2 — OLS rescaling (forecast-optimal):
 *   RV_{t+1} = β₀ + β₁·σ²_t(D²)
 *   The D²-discovered σ²_t acts as a data-driven smoother over RV lags.
 *   OLS rescales it to minimize forecast error (RSS on RV).
 *   Only 2 parameters → robust on small samples with noisy per-candle RV.
 *
 * D² discovers lag structure (model-free). OLS rescales for prediction accuracy.
 * Both weight sets are stored in params — no identity loss.
 */
declare class NoVaS {
    private returns;
    private rv;
    private periodsPerYear;
    private lags;
    constructor(data: Candle[] | number[], options?: NoVaSOptions);
    /**
     * Calibrate NoVaS weights via two-stage procedure:
     * Stage 1: D² minimization (normality of W_t)
     * Stage 2: OLS rescaling of D²-variance (forecast-optimal)
     */
    fit(options?: {
        maxIter?: number;
        tol?: number;
    }): CalibrationResult<NoVaSParams>;
    /**
     * Internal: compute variance series from D² weight vector.
     */
    private getVarianceSeriesInternal;
    /**
     * Calculate conditional variance series using D² weights (normalization identity).
     */
    getVarianceSeries(params: NoVaSParams): number[];
    /**
     * Calculate forecast variance series using OLS-rescaled D² variance.
     * forecast_σ²_t = β₀ + β₁·σ²_t(D²)
     * Used for QLIKE model comparison — measures forecast quality.
     */
    getForecastVarianceSeries(params: NoVaSParams): number[];
    /**
     * Forecast variance forward using OLS-rescaled D² weights.
     *
     * Step 1: compute D²-based σ²_{t+h} using D² weights
     * Step 2: rescale via β₀ + β₁·σ²_{t+h}
     */
    forecast(params: NoVaSParams, steps?: number): VolatilityForecast;
    /**
     * Get the return series.
     */
    getReturns(): number[];
}
/**
 * Convenience function to calibrate NoVaS from candles or prices.
 */
declare function calibrateNoVaS(data: Candle[] | number[], options?: NoVaSOptions): CalibrationResult<NoVaSParams>;

/**
 * Validate OHLC integrity. Garbage candles (NaN, non-positive prices,
 * high < low) otherwise propagate silently as NaN through every estimator.
 */
declare function validateCandles(candles: Candle[]): void;
/**
 * Linear-interpolation quantile of a pre-sorted (ascending) sample.
 */
declare function empiricalQuantile(sortedAsc: number[], p: number): number;
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
 * Per-candle Parkinson (1980) realized variance proxy.
 *
 * RV_i = (1/(4·ln2)) · ln(H/L)²
 *
 * ~5× more efficient than squared returns. Falls back to r² when H === L.
 * rv[i] aligned with returns[i], using candles[i+1]'s OHLC.
 */
declare function perCandleParkinson(candles: Candle[], returns: number[]): number[];
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
/**
 * Log-Gamma function via Lanczos approximation (g=7, n=9).
 * Accurate to ~15 digits for x > 0.
 */
declare function logGamma(x: number): number;
/**
 * Per-observation Student-t negative log-likelihood contribution.
 *
 * For standardized t(df) with variance σ²_t:
 *   -LL_i = 0.5·ln(σ²_t) + ((df+1)/2)·ln(1 + r²_t / ((df-2)·σ²_t))
 *         - lnΓ((df+1)/2) + lnΓ(df/2) + 0.5·ln(π·(df-2))
 *
 * Returns the per-observation neg-LL (without the constant terms).
 * Caller accumulates and adds the constant once.
 */
declare function studentTNegLL(returns: number[], varianceSeries: number[], df: number): number;
/**
 * E[|Z|] where Z follows a standardized Student-t(df) distribution (variance = 1).
 *
 * E[|Z|] = √((df-2)/π) · Γ((df-1)/2) / Γ(df/2)
 *
 * Converges to √(2/π) as df → ∞ (Gaussian limit).
 */
declare function expectedAbsStudentT(df: number): number;
/**
 * Regularized incomplete beta function I_x(a, b).
 */
declare function incompleteBeta(x: number, a: number, b: number): number;
/**
 * CDF of the (raw, unstandardized) Student-t distribution with df degrees
 * of freedom: P(T ≤ t).
 */
declare function studentTCdf(t: number, df: number): number;
/**
 * Two-sided quantile of the STANDARDIZED Student-t distribution
 * (unit variance). The t-analog of probit(): returns z such that
 * P(|Z| ≤ z) = confidence when Z ~ t(df) scaled to variance 1.
 *
 * This is what price corridors must use when the model was fitted with
 * Student-t innovations: with fat tails (small df) the Gaussian probit
 * makes 68% bands too wide and 99% bands dangerously narrow.
 *
 * Falls back to probit() for df > 200 (Gaussian regime) or df ≤ 2
 * (variance undefined).
 */
declare function studentTProbit(confidence: number, df: number): number;
/**
 * 1D grid search for optimal df that minimizes Student-t neg-LL.
 * Used by HAR-RV and NoVaS where df is profiled after main optimization.
 */
declare function profileStudentTDf(returns: number[], varianceSeries: number[]): number;
/**
 * QLIKE loss (Patton 2011) — standard loss function for volatility forecasts.
 *
 * QLIKE = (1/n) · Σ (RV_t / σ²_t − log(RV_t / σ²_t) − 1)
 *
 * Lower = better forecast. Neutral to calibration method — judges only
 * how well the variance series predicts realized variance, regardless
 * of how the model was calibrated (MLE, OLS, D², etc.).
 */
declare function qlike(varianceSeries: number[], rv: number[]): number;
/**
 * Inverse standard normal CDF (probit function).
 * Converts a two-sided confidence level (e.g. 0.95) to the corresponding
 * z-score (e.g. 1.96).
 *
 * Uses Acklam's rational approximation (max relative error < 1.15e-9).
 */
declare function probit(confidence: number): number;

type CandleInterval = '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '8h';
interface PredictionResult {
    /** Reference price used to compute the corridor (last close or the value passed as `currentPrice`). */
    currentPrice: number;
    /** One-period (or cumulative) volatility estimate, as a decimal log-return standard deviation (e.g. `0.012` = 1.2%). */
    sigma: number;
    /** Upward expected move in price units: `upperPrice - currentPrice`. */
    move: number;
    /** Upward expected move in percent (0–100 scale, e.g. `1.21` means 1.21%). Equal to `(exp(z·σ) - 1) * 100`. */
    movePercent: number;
    /** Upper price band: `currentPrice · exp(+z·σ)`. */
    upperPrice: number;
    /** Lower price band: `currentPrice · exp(-z·σ)`. Always positive. */
    lowerPrice: number;
    /** Volatility model auto-selected by QLIKE. */
    modelType: 'garch' | 'egarch' | 'gjr-garch' | 'har-rv' | 'novas';
    /** Student-t degrees of freedom profiled on scale-corrected residuals. */
    df: number;
    /**
     * Corridor multiplier actually used: blend of the empirical |z| quantile
     * of the standardized residuals and the Student-t(df) quantile, weighted
     * by how much data supports the requested tail. Reconstruct bands as
     * `currentPrice · exp(±zScore · sigma)`.
     */
    zScore: number;
    /** `true` when the model converged, persistence < 0.999, and Ljung-Box p-value ≥ 0.05. */
    reliable: boolean;
}
/**
 * Forecast expected price range for t+1 (next candle).
 *
 * Auto-selects the best volatility model via QLIKE, rescales the variance
 * to the return scale (Var(r/σ) = 1), and builds bands P·exp(±z·σ) where
 * z is calibrated on the data itself: the empirical |z| quantile of the
 * standardized residuals blended with the fitted Student-t quantile as the
 * tail runs out of observations (see corridorZ). Empirical coverage tracks
 * the requested confidence without assuming a distributional shape.
 * @param confidence — two-sided probability in (0,1). Default ≈0.6827 (±1σ).
 *   Common values: 0.90, 0.95, 0.99.
 */
declare function predict(candles: Candle[], interval: CandleInterval, currentPrice?: number | null, confidence?: number): PredictionResult;
/**
 * Forecast expected price range over multiple candles.
 *
 * Cumulative σ = √(σ₁² + σ₂² + ... + σₙ²) — total expected move over N periods.
 * Uses log-normal price bands P·exp(±z·σ) with the same data-calibrated z
 * as predict(). The single-period tail shape is applied to the multi-step
 * horizon too — aggregated returns are closer to Gaussian, so this errs on
 * the wide (safe) side in tails.
 * @param confidence — two-sided probability in (0,1). Default ≈0.6827 (±1σ).
 */
declare function predictRange(candles: Candle[], interval: CandleInterval, steps: number, currentPrice?: number | null, confidence?: number): PredictionResult;
interface BacktestStats {
    /** Number of test candles whose close landed inside the predicted corridor. */
    hits: number;
    /** Number of walk-forward predictions made. */
    total: number;
    /** Empirical coverage in percent (0–100). Compare against `confidence · 100`. */
    hitRate: number;
}
/**
 * Walk-forward calibration statistics for predict.
 *
 * Refits the model at every step on a rolling window (75% of candles,
 * min MIN_CANDLES) and checks whether the next close lands inside the
 * predicted corridor. A well-calibrated tool has hitRate ≈ confidence·100.
 * Throws if not enough candles for the given interval.
 * @param confidence — two-sided probability in (0,1) for the prediction band.
 *   Default ≈0.6827 (±1σ).
 */
declare function backtestStats(candles: Candle[], interval: CandleInterval, confidence?: number): BacktestStats;
/**
 * Walk-forward backtest of predict.
 *
 * Window is computed automatically: 75% of candles for fitting, 25% for testing.
 * Throws if not enough candles for the given interval.
 * Returns true if the model's hit rate meets the required threshold.
 * @param confidence — two-sided probability in (0,1) for the prediction band.
 *   Default ≈0.6827 (±1σ).
 * @param requiredPercent — minimum hit rate (0–100) to pass. Default 68.
 */
declare function backtest(candles: Candle[], interval: CandleInterval, confidence?: number, requiredPercent?: number): boolean;

declare function nelderMead(fn: (x: number[]) => number, x0: number[], options?: {
    maxIter?: number;
    tol?: number;
    alpha?: number;
    gamma?: number;
    rho?: number;
    sigma?: number;
}): OptimizerResult;
declare function nelderMeadMultiStart(fn: (x: number[]) => number, x0: number[], options?: {
    maxIter?: number;
    tol?: number;
    restarts?: number;
}): OptimizerResult;

export { EXPECTED_ABS_NORMAL, Egarch, Garch, GjrGarch, HarRv, NoVaS, backtest, backtestStats, calculateReturns, calculateReturnsFromPrices, calibrateEgarch, calibrateGarch, calibrateGjrGarch, calibrateHarRv, calibrateNoVaS, checkLeverageEffect, empiricalQuantile, expectedAbsStudentT, garmanKlassVariance, incompleteBeta, ljungBox, logGamma, nelderMead, nelderMeadMultiStart, perCandleParkinson, predict, predictRange, probit, profileStudentTDf, qlike, sampleVariance, sampleVarianceWithMean, studentTCdf, studentTNegLL, studentTProbit, validateCandles, yangZhangVariance };
export type { BacktestStats, CalibrationResult, Candle, CandleInterval, EgarchOptions, EgarchParams, GarchOptions, GarchParams, GjrGarchOptions, GjrGarchParams, HarRvOptions, HarRvParams, LeverageStats, NoVaSOptions, NoVaSParams, OptimizerResult, PredictionResult, VolatilityForecast };
