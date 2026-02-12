import type { Candle, LeverageStats } from './types.js';

/**
 * Calculate log returns from candles
 */
export function calculateReturns(candles: Candle[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    if (!(candles[i].close > 0) || !(candles[i - 1].close > 0)) {
      throw new Error(`Invalid close price at index ${i}`);
    }
    returns.push(Math.log(candles[i].close / candles[i - 1].close));
  }
  return returns;
}

/**
 * Calculate log returns from price array
 */
export function calculateReturnsFromPrices(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    if (!(prices[i] > 0 && Number.isFinite(prices[i])) || !(prices[i - 1] > 0 && Number.isFinite(prices[i - 1]))) {
      throw new Error(`Invalid price at index ${i}`);
    }
    returns.push(Math.log(prices[i] / prices[i - 1]));
  }
  return returns;
}

/**
 * Calculate sample variance (mean-zero assumption)
 */
export function sampleVariance(returns: number[]): number {
  return returns.reduce((sum, r) => sum + r * r, 0) / returns.length;
}

/**
 * Calculate sample variance with mean adjustment
 */
export function sampleVarianceWithMean(returns: number[]): number {
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  return returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
}

/**
 * Check for leverage effect (asymmetry in volatility)
 */
export function checkLeverageEffect(returns: number[]): LeverageStats {
  const negative = returns.filter(r => r < 0);
  const positive = returns.filter(r => r > 0);

  if (negative.length === 0 || positive.length === 0) {
    return {
      negativeVol: 0,
      positiveVol: 0,
      ratio: 1,
      recommendation: 'garch',
    };
  }

  const negativeVol = Math.sqrt(negative.reduce((s, r) => s + r * r, 0) / negative.length);
  const positiveVol = Math.sqrt(positive.reduce((s, r) => s + r * r, 0) / positive.length);
  const ratio = negativeVol / positiveVol;

  return {
    negativeVol,
    positiveVol,
    ratio,
    recommendation: ratio > 1.2 ? 'egarch' : 'garch',
  };
}

/**
 * Expected value of |Z| where Z ~ N(0,1)
 * E[|Z|] = sqrt(2/π)
 */
export const EXPECTED_ABS_NORMAL = Math.sqrt(2 / Math.PI);

/**
 * Calculate AIC (Akaike Information Criterion)
 */
export function calculateAIC(logLikelihood: number, numParams: number): number {
  return 2 * numParams - 2 * logLikelihood;
}

/**
 * Calculate BIC (Bayesian Information Criterion)
 */
export function calculateBIC(logLikelihood: number, numParams: number, numObs: number): number {
  return numParams * Math.log(numObs) - 2 * logLikelihood;
}
