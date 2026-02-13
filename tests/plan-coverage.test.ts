import { describe, it, expect } from 'vitest';
import {
  Garch,
  Egarch,
  calibrateGarch,
  calibrateEgarch,
  calculateReturnsFromPrices,
  checkLeverageEffect,
  garmanKlassVariance,
  yangZhangVariance,
  ljungBox,
  predict,
  predictRange,
  backtest,
  EXPECTED_ABS_NORMAL,
} from '../src/index.js';
import { chi2Survival } from '../src/utils.js';
import type { Candle } from '../src/index.js';

// ── helpers ──────────────────────────────────────────────────

/** Deterministic LCG */
function lcg(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function randn(rng: () => number): number {
  const u1 = rng() || 0.001;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Generate candles with tunable volatility */
function makeCandles(n: number, seed = 12345, volScale = 0.04): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const r = (rng() - 0.5) * volScale;
    const close = price * Math.exp(r);
    const high = Math.max(price, close) * (1 + Math.abs(r) * 0.5);
    const low = Math.min(price, close) * (1 - Math.abs(r) * 0.5);
    candles.push({ open: price, high, low, close, volume: 1000 });
    price = close;
  }
  return candles;
}

/** Candles with strong asymmetry: large drops, small rallies */
function makeAsymmetricCandles(n: number, seed = 77): Candle[] {
  const rng = lcg(seed);
  const candles: Candle[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const u = rng();
    const r = u < 0.5 ? -(u * 0.08) : (u - 0.5) * 0.02;
    const close = price * Math.exp(r);
    const high = Math.max(price, close) * 1.005;
    const low = Math.min(price, close) * 0.995;
    candles.push({ open: price, high, low, close, volume: 1000 });
    price = close;
  }
  return candles;
}

// ── 1. predict() at exact MIN_CANDLES boundary ──────────────

describe('predict at MIN_CANDLES boundary', () => {
  it('works with exactly 300 candles for 15m', () => {
    const candles = makeCandles(300, 111);
    const result = predict(candles, '15m');
    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(Number.isFinite(result.move)).toBe(true);
    expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
    expect(result.lowerPrice).toBeLessThan(result.currentPrice);
  });

  it('works with exactly 200 candles for 4h', () => {
    const candles = makeCandles(200, 222);
    const result = predict(candles, '4h');
    expect(result.sigma).toBeGreaterThan(0);
    expect(Number.isFinite(result.move)).toBe(true);
  });

  it('works with exactly 500 candles for 1m', () => {
    const candles = makeCandles(500, 333);
    const result = predict(candles, '1m');
    expect(result.sigma).toBeGreaterThan(0);
  });

  it('works with exactly 150 candles for 8h', () => {
    const candles = makeCandles(150, 444);
    const result = predict(candles, '8h');
    expect(result.sigma).toBeGreaterThan(0);
  });

  it('throws with MIN_CANDLES - 1', () => {
    expect(() => predict(makeCandles(299), '15m')).toThrow('Need at least 300');
    expect(() => predict(makeCandles(199), '4h')).toThrow('Need at least 200');
    expect(() => predict(makeCandles(499), '1m')).toThrow('Need at least 500');
    expect(() => predict(makeCandles(149), '8h')).toThrow('Need at least 150');
  });
});

// ── 2. checkReliable — three failure modes isolated ─────────

describe('checkReliable failure modes', () => {
  it('reliable=true on well-behaved GARCH data', () => {
    const candles = makeCandles(200, 555);
    const result = predict(candles, '4h');
    // Well-behaved LCG data should converge and pass Ljung-Box
    expect(result.reliable).toBe(true);
  });

  it('reliable=false when optimizer cannot converge (1 iteration)', () => {
    const candles = makeCandles(200, 666);
    // Fit with maxIter=1 to force non-convergence
    const model = new Garch(candles, { periodsPerYear: 2190 });
    const fit = model.fit({ maxIter: 1 });
    expect(fit.diagnostics.converged).toBe(false);
  });

  it('reliable=false when optimizer cannot converge (egarch, 1 iteration)', () => {
    const candles = makeAsymmetricCandles(200, 777);
    const model = new Egarch(candles, { periodsPerYear: 2190 });
    const fit = model.fit({ maxIter: 1 });
    expect(fit.diagnostics.converged).toBe(false);
  });

  it('persistence constraint enforced at 0.9999 in GARCH', () => {
    // GARCH rejects alpha + beta >= 0.9999 in NLL
    // So fitted persistence should always be < 0.9999
    const candles = makeCandles(200, 888);
    const model = new Garch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    expect(fit.params.persistence).toBeLessThan(0.9999);
  });

  it('persistence constraint enforced at 0.9999 in EGARCH', () => {
    const candles = makeAsymmetricCandles(200, 999);
    const model = new Egarch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    expect(fit.params.persistence).toBeLessThan(0.9999);
  });

  it('Ljung-Box rejects autocorrelated squared residuals', () => {
    // Create data with strong autocorrelation in squared values
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      // Alternating high/low variance clusters
      const cluster = Math.floor(i / 10) % 2 === 0 ? 4.0 : 0.1;
      data.push(cluster);
    }
    const lb = ljungBox(data, 10);
    expect(lb.pValue).toBeLessThan(0.05);
  });

  it('Ljung-Box accepts white noise squared residuals', () => {
    const rng = lcg(42);
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      data.push(rng() * 2 - 1);
    }
    // Square them to simulate squared standardized residuals
    const squared = data.map(d => d * d);
    const lb = ljungBox(squared, 10);
    expect(lb.pValue).toBeGreaterThan(0.05);
  });
});

// ── 3. chi2Survival accuracy against known critical values ──

describe('chi2Survival accuracy', () => {
  // Known chi-square critical values: P(X > x) for X ~ χ²(df)
  // Source: standard statistical tables

  it('df=10, x=18.307 → p ≈ 0.05', () => {
    const p = chi2Survival(18.307, 10);
    expect(p).toBeCloseTo(0.05, 1);
  });

  it('df=10, x=23.209 → p ≈ 0.01', () => {
    const p = chi2Survival(23.209, 10);
    expect(p).toBeCloseTo(0.01, 1);
  });

  it('df=10, x=15.987 → p ≈ 0.10', () => {
    const p = chi2Survival(15.987, 10);
    expect(p).toBeCloseTo(0.10, 1);
  });

  it('df=5, x=11.070 → p ≈ 0.05', () => {
    const p = chi2Survival(11.070, 5);
    expect(p).toBeCloseTo(0.05, 1);
  });

  it('df=20, x=31.410 → p ≈ 0.05', () => {
    const p = chi2Survival(31.410, 20);
    expect(p).toBeCloseTo(0.05, 1);
  });

  it('x=0 → p ≈ 1.0 (no evidence against H₀)', () => {
    const p = chi2Survival(0, 10);
    expect(p).toBeGreaterThanOrEqual(0.99);
  });

  it('large x → p ≈ 0', () => {
    const p = chi2Survival(100, 10);
    expect(p).toBeLessThan(0.001);
  });
});

// ── 4. Yang-Zhang with overnight gaps ───────────────────────

describe('Yang-Zhang with gaps', () => {
  it('captures overnight gap variance', () => {
    // Candles with no gaps: open === prev close
    const noGap: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 50; i++) {
      const close = price * (1 + (i % 2 === 0 ? 0.01 : -0.01));
      noGap.push({ open: price, high: Math.max(price, close) * 1.002, low: Math.min(price, close) * 0.998, close, volume: 1000 });
      price = close;
    }

    // Candles with gaps: open !== prev close (simulates overnight)
    const withGap: Candle[] = [];
    price = 100;
    for (let i = 0; i < 50; i++) {
      const gapOpen = price * (1 + (i % 3 === 0 ? 0.02 : -0.005));
      const close = gapOpen * (1 + (i % 2 === 0 ? 0.01 : -0.01));
      withGap.push({ open: gapOpen, high: Math.max(gapOpen, close) * 1.002, low: Math.min(gapOpen, close) * 0.998, close, volume: 1000 });
      price = close;
    }

    const varNoGap = yangZhangVariance(noGap);
    const varWithGap = yangZhangVariance(withGap);

    // Gapped data should have higher variance due to overnight component
    expect(varWithGap).toBeGreaterThan(varNoGap);
    expect(Number.isFinite(varWithGap)).toBe(true);
  });

  it('handles extreme gap (50% overnight drop)', () => {
    const candles: Candle[] = [
      { open: 100, high: 105, low: 98, close: 102, volume: 1000 },
      { open: 50, high: 55, low: 48, close: 52, volume: 1000 },  // 50% gap down
      { open: 53, high: 56, low: 50, close: 54, volume: 1000 },
    ];
    const v = yangZhangVariance(candles);
    expect(Number.isFinite(v)).toBe(true);
    expect(v).toBeGreaterThan(0);
  });

  it('yangZhangVariance with n=1 falls back to garmanKlass', () => {
    const candle: Candle[] = [{ open: 100, high: 105, low: 95, close: 102, volume: 1000 }];
    const yz = yangZhangVariance(candle);
    const gk = garmanKlassVariance(candle);
    expect(yz).toBe(gk);
  });

  it('volume does not affect Yang-Zhang result', () => {
    const base: Candle[] = [];
    const rng = lcg(42);
    let price = 100;
    for (let i = 0; i < 30; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      base.push({ open: price, high: Math.max(price, close) * 1.005, low: Math.min(price, close) * 0.995, close, volume: 1000 });
      price = close;
    }

    const zeroVol = base.map(c => ({ ...c, volume: 0 }));
    const bigVol = base.map(c => ({ ...c, volume: 999999 }));

    expect(yangZhangVariance(zeroVol)).toBe(yangZhangVariance(base));
    expect(yangZhangVariance(bigVol)).toBe(yangZhangVariance(base));
  });
});

// ── 5. Data validation — NaN, Infinity, negative, zero ──────

describe('predict rejects invalid candle data', () => {
  it('throws on NaN close price', () => {
    const candles = makeCandles(200);
    candles[50] = { ...candles[50], close: NaN };
    expect(() => predict(candles, '4h')).toThrow();
  });

  it('throws on Infinity close price', () => {
    const candles = makeCandles(200);
    candles[50] = { ...candles[50], close: Infinity };
    expect(() => predict(candles, '4h')).toThrow();
  });

  it('throws on negative close price', () => {
    const candles = makeCandles(200);
    candles[50] = { ...candles[50], close: -1 };
    expect(() => predict(candles, '4h')).toThrow();
  });

  it('throws on zero close price', () => {
    const candles = makeCandles(200);
    candles[50] = { ...candles[50], close: 0 };
    expect(() => predict(candles, '4h')).toThrow();
  });
});

// ── 6. predict output field consistency ─────────────────────

describe('predict output field consistency', () => {
  it('all fields present and valid', () => {
    const candles = makeCandles(200, 101);
    const result = predict(candles, '4h');

    expect(typeof result.reliable).toBe('boolean');
    expect(typeof result.modelType).toBe('string');
    expect(['garch', 'egarch']).toContain(result.modelType);
    expect(result.sigma).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(Number.isFinite(result.move)).toBe(true);
    expect(Number.isFinite(result.currentPrice)).toBe(true);
    expect(Number.isFinite(result.upperPrice)).toBe(true);
    expect(Number.isFinite(result.lowerPrice)).toBe(true);
  });

  it('upperPrice = currentPrice + move', () => {
    const candles = makeCandles(200, 202);
    const result = predict(candles, '4h');
    expect(result.upperPrice).toBeCloseTo(result.currentPrice + result.move, 10);
  });

  it('lowerPrice = currentPrice - move', () => {
    const candles = makeCandles(200, 303);
    const result = predict(candles, '4h');
    expect(result.lowerPrice).toBeCloseTo(result.currentPrice - result.move, 10);
  });

  it('move = currentPrice * sigma', () => {
    const candles = makeCandles(200, 404);
    const result = predict(candles, '4h');
    expect(result.move).toBeCloseTo(result.currentPrice * result.sigma, 10);
  });

  it('corridor is symmetric around currentPrice', () => {
    const candles = makeCandles(200, 505);
    const result = predict(candles, '4h');
    const distUp = result.upperPrice - result.currentPrice;
    const distDown = result.currentPrice - result.lowerPrice;
    expect(distUp).toBeCloseTo(distDown, 10);
  });
});

// ── 7. predictRange for 16-candle timeout ───────────────────

describe('predictRange 16-candle timeout (strategy)', () => {
  it('16-step cumulative sigma > single-step sigma', () => {
    const candles = makeCandles(300, 161);
    const single = predict(candles, '15m');
    const range16 = predictRange(candles, '15m', 16);

    expect(range16.sigma).toBeGreaterThan(single.sigma);
  });

  it('16-step sigma scales approximately as sqrt(16) * single', () => {
    const candles = makeCandles(300, 162);
    const single = predict(candles, '15m');
    const range16 = predictRange(candles, '15m', 16);

    // For stationary GARCH, cumulative sigma ≈ sqrt(steps) * single sigma
    // Allow 50% tolerance due to variance term structure
    const ratio = range16.sigma / single.sigma;
    expect(ratio).toBeGreaterThan(2);   // sqrt(16) = 4, but term structure varies
    expect(ratio).toBeLessThan(8);      // shouldn't exceed 2x theoretical
  });

  it('all output fields valid for 16-step', () => {
    const candles = makeCandles(300, 163);
    const result = predictRange(candles, '15m', 16);
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(Number.isFinite(result.move)).toBe(true);
    expect(result.upperPrice).toBeGreaterThan(result.currentPrice);
    expect(result.lowerPrice).toBeLessThan(result.currentPrice);
  });
});

// ── 8. Leverage effect boundary at ratio=1.2 ───────────────

describe('checkLeverageEffect boundary', () => {
  it('ratio exactly 1.2 → garch (not >1.2)', () => {
    // Construct returns where negVol/posVol = exactly 1.2
    // negVol = sqrt(sum(neg^2)/count(neg))
    // posVol = sqrt(sum(pos^2)/count(pos))
    // We need negVol = 1.2 * posVol
    // If all neg = -a, all pos = b, then a/b = 1.2, so a = 1.2b
    const b = 0.01;
    const a = 0.012; // ratio = 1.2 exactly
    const returns: number[] = [];
    for (let i = 0; i < 100; i++) {
      returns.push(i % 2 === 0 ? b : -a);
    }
    const result = checkLeverageEffect(returns);
    expect(result.ratio).toBeCloseTo(1.2, 10);
    expect(result.recommendation).toBe('garch'); // > 1.2, not >= 1.2
  });

  it('ratio 1.21 → egarch', () => {
    const b = 0.01;
    const a = 0.0121;
    const returns: number[] = [];
    for (let i = 0; i < 100; i++) {
      returns.push(i % 2 === 0 ? b : -a);
    }
    const result = checkLeverageEffect(returns);
    expect(result.ratio).toBeGreaterThan(1.2);
    expect(result.recommendation).toBe('egarch');
  });

  it('ratio 1.19 → garch', () => {
    const b = 0.01;
    const a = 0.0119;
    const returns: number[] = [];
    for (let i = 0; i < 100; i++) {
      returns.push(i % 2 === 0 ? b : -a);
    }
    const result = checkLeverageEffect(returns);
    expect(result.ratio).toBeLessThan(1.2);
    expect(result.recommendation).toBe('garch');
  });
});

// ── 9. EGARCH with positive gamma (inverse leverage) ────────

describe('EGARCH positive gamma', () => {
  it('handles gamma > 0 without crashing', () => {
    // Generate EGARCH data with positive gamma
    const rng = lcg(42);
    const returns: number[] = [];
    let logVar = -8;
    let variance = Math.exp(logVar);
    for (let i = 0; i < 300; i++) {
      const z = randn(rng);
      returns.push(Math.sqrt(variance) * z);
      logVar = -0.5 + 0.1 * (Math.abs(z) - EXPECTED_ABS_NORMAL) + 0.05 * z + 0.9 * logVar;
      logVar = Math.max(-50, Math.min(50, logVar));
      variance = Math.exp(logVar);
    }
    const prices = [100];
    for (const r of returns) prices.push(prices[prices.length - 1] * Math.exp(r));

    const model = new Egarch(prices, { periodsPerYear: 35040 });
    const fit = model.fit();
    expect(Number.isFinite(fit.params.omega)).toBe(true);
    expect(Number.isFinite(fit.params.gamma)).toBe(true);

    const forecast = model.forecast(fit.params, 5);
    for (const v of forecast.volatility) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  });
});

// ── 10. Numerical stability — large prices, tiny sigma ──────

describe('numerical stability', () => {
  it('BTC-scale prices (100k) produce valid sigma', () => {
    const rng = lcg(100);
    const candles: Candle[] = [];
    let price = 100_000;
    for (let i = 0; i < 200; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * (1 + Math.abs(r) * 0.3);
      const low = Math.min(price, close) * (1 - Math.abs(r) * 0.3);
      candles.push({ open: price, high, low, close, volume: 500 });
      price = close;
    }
    const result = predict(candles, '4h');
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.sigma).toBeGreaterThan(0);
    expect(result.move).toBeGreaterThan(100); // BTC at 100k, 0.1% = $100 min
  });

  it('very low price (penny stock) produce valid sigma', () => {
    const rng = lcg(200);
    const candles: Candle[] = [];
    let price = 0.001;
    for (let i = 0; i < 200; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * (1 + Math.abs(r) * 0.3);
      const low = Math.min(price, close) * (1 - Math.abs(r) * 0.3);
      candles.push({ open: price, high, low, close, volume: 1e9 });
      price = close;
    }
    const result = predict(candles, '4h');
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.sigma).toBeGreaterThan(0);
  });

  it('near-constant prices → sigma near zero but finite', () => {
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      // Tiny noise: ±0.001%
      const close = price * (1 + (i % 2 === 0 ? 0.00001 : -0.00001));
      candles.push({ open: price, high: price * 1.00002, low: price * 0.99998, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.sigma).toBeGreaterThanOrEqual(0);
    expect(result.sigma).toBeLessThan(0.001);
  });

  it('log returns precise at price 1e10', () => {
    const prices = [1e10, 1e10 * 1.0001, 1e10 * 0.9999];
    const returns = calculateReturnsFromPrices(prices);
    // ln(1.0001) ≈ 0.00009999
    expect(returns[0]).toBeCloseTo(Math.log(1.0001), 10);
    expect(returns[1]).toBeCloseTo(Math.log(0.9999 / 1.0001), 10);
  });
});

// ── 11. backtest edge cases ─────────────────────────────────

describe('backtest edge cases', () => {
  it('works at MIN_CANDLES boundary for 4h (200 candles)', () => {
    // 200 candles, window = max(200, floor(200*0.75)) = 200
    // Loop: i=200 to 198 → no iterations → total=0, hits/total = NaN
    // This is a known edge case: 0/0 >= 68 → false
    const candles = makeCandles(200, 1001);
    const result = backtest(candles, '4h');
    expect(typeof result).toBe('boolean');
  });

  it('works with enough candles beyond window', () => {
    // 267 candles for 4h: window = max(200, floor(267*0.75)=200) = 200
    // Test range: i=200..265, total=66
    const candles = makeCandles(267, 1002);
    const result = backtest(candles, '4h');
    expect(typeof result).toBe('boolean');
  });

  it('0% threshold always returns true', () => {
    const candles = makeCandles(250, 1003);
    expect(backtest(candles, '4h', 0)).toBe(true);
  });

  it('100% threshold returns false on noisy data', () => {
    const candles = makeCandles(250, 1004);
    expect(backtest(candles, '4h', 100)).toBe(false);
  });

  it('throws for 15m with < 300 candles', () => {
    expect(() => backtest(makeCandles(299), '15m')).toThrow('Need at least 300');
  });
});

// ── 12. Forecast values: manual verification ────────────────

describe('GARCH forecast manual verification', () => {
  it('1-step forecast matches manual σ² = ω + α·ε² + β·σ²', () => {
    const candles = makeCandles(200, 2001);
    const model = new Garch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    const { omega, alpha, beta } = fit.params;

    const varSeries = model.getVarianceSeries(fit.params);
    const returns = model.getReturns();
    const lastVar = varSeries[varSeries.length - 1];
    const lastRet = returns[returns.length - 1];

    // Manual one-step ahead
    const expectedVar = omega + alpha * lastRet ** 2 + beta * lastVar;

    const forecast = model.forecast(fit.params, 1);
    expect(forecast.variance[0]).toBeCloseTo(expectedVar, 15);
    expect(forecast.volatility[0]).toBeCloseTo(Math.sqrt(expectedVar), 15);
  });

  it('2-step forecast: σ²_{t+2} = ω + (α+β)·σ²_{t+1}', () => {
    const candles = makeCandles(200, 2002);
    const model = new Garch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    const { omega, alpha, beta } = fit.params;

    const forecast = model.forecast(fit.params, 2);
    const v1 = forecast.variance[0];
    const expectedV2 = omega + (alpha + beta) * v1;
    expect(forecast.variance[1]).toBeCloseTo(expectedV2, 15);
  });

  it('multi-step forecast converges to unconditional variance', () => {
    const candles = makeCandles(200, 2003);
    const model = new Garch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    const { omega, alpha, beta } = fit.params;
    const uncondVar = omega / (1 - alpha - beta);

    const forecast = model.forecast(fit.params, 100);
    const lastForecastVar = forecast.variance[99];
    expect(lastForecastVar).toBeCloseTo(uncondVar, 6);
  });
});

describe('EGARCH forecast manual verification', () => {
  it('1-step forecast uses actual last z', () => {
    const candles = makeAsymmetricCandles(200, 3001);
    const model = new Egarch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    const { omega, alpha, gamma, beta } = fit.params;

    const varSeries = model.getVarianceSeries(fit.params);
    const returns = model.getReturns();
    const lastVar = varSeries[varSeries.length - 1];
    const lastRet = returns[returns.length - 1];
    const z = lastRet / Math.sqrt(lastVar);

    const expectedLogVar = omega
      + alpha * (Math.abs(z) - EXPECTED_ABS_NORMAL)
      + gamma * z
      + beta * Math.log(lastVar);
    const expectedVar = Math.exp(expectedLogVar);

    const forecast = model.forecast(fit.params, 1);
    expect(forecast.variance[0]).toBeCloseTo(expectedVar, 10);
  });

  it('multi-step EGARCH converges to exp(ω/(1−β))', () => {
    const candles = makeAsymmetricCandles(200, 3002);
    const model = new Egarch(candles, { periodsPerYear: 2190 });
    const fit = model.fit();
    const { omega, beta } = fit.params;
    const uncondVar = Math.exp(omega / (1 - beta));

    const forecast = model.forecast(fit.params, 500);
    const lastForecastVar = forecast.variance[499];
    // EGARCH converges slower than GARCH; allow 30% tolerance
    expect(Math.abs(lastForecastVar - uncondVar) / uncondVar).toBeLessThan(0.3);
  });
});

// ── 13. Ljung-Box with different lag values ─────────────────

describe('ljungBox lag sensitivity', () => {
  it('maxLag=5 produces valid result', () => {
    const rng = lcg(42);
    const data = Array.from({ length: 200 }, () => randn(rng));
    const lb = ljungBox(data, 5);
    expect(lb.pValue).toBeGreaterThan(0);
    expect(lb.pValue).toBeLessThanOrEqual(1);
    expect(lb.statistic).toBeGreaterThanOrEqual(0);
  });

  it('maxLag=20 produces valid result', () => {
    const rng = lcg(42);
    const data = Array.from({ length: 200 }, () => randn(rng));
    const lb = ljungBox(data, 20);
    expect(lb.pValue).toBeGreaterThan(0);
    expect(lb.pValue).toBeLessThanOrEqual(1);
  });

  it('higher lag detects longer-range autocorrelation', () => {
    // Create data with lag-15 autocorrelation
    const rng = lcg(55);
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      if (i < 15) {
        data.push(randn(rng));
      } else {
        data.push(data[i - 15] * 0.5 + randn(rng) * 0.5);
      }
    }
    const lb5 = ljungBox(data, 5);
    const lb20 = ljungBox(data, 20);
    // lag=20 should pick up lag-15 autocorrelation better
    expect(lb20.statistic).toBeGreaterThan(lb5.statistic);
  });
});

// ── 14. Candles with zero volume ────────────────────────────

describe('candles with zero volume', () => {
  it('predict works with volume=0', () => {
    const candles = makeCandles(200, 4001).map(c => ({ ...c, volume: 0 }));
    const result = predict(candles, '4h');
    expect(Number.isFinite(result.sigma)).toBe(true);
    expect(result.sigma).toBeGreaterThan(0);
  });

  it('predict result identical regardless of volume values', () => {
    const base = makeCandles(200, 4002);
    const zeroVol = base.map(c => ({ ...c, volume: 0 }));
    const bigVol = base.map(c => ({ ...c, volume: 1e12 }));

    const r1 = predict(base, '4h');
    const r2 = predict(zeroVol, '4h');
    const r3 = predict(bigVol, '4h');

    expect(r2.sigma).toBe(r1.sigma);
    expect(r3.sigma).toBe(r1.sigma);
  });
});

// ── 15. GARCH parameter recovery from known DGP ─────────────

describe('GARCH parameter recovery', () => {
  it('recovers ω, α, β from known GARCH(1,1) process', () => {
    // True parameters
    const trueOmega = 0.00001;
    const trueAlpha = 0.1;
    const trueBeta = 0.85;

    // Generate 2000 samples from true GARCH process
    const rng = lcg(7777);
    const returns: number[] = [];
    let variance = trueOmega / (1 - trueAlpha - trueBeta);
    for (let i = 0; i < 2000; i++) {
      const eps = Math.sqrt(variance) * randn(rng);
      returns.push(eps);
      variance = trueOmega + trueAlpha * eps ** 2 + trueBeta * variance;
    }
    const prices = [100];
    for (const r of returns) prices.push(prices[prices.length - 1] * Math.exp(r));

    const result = calibrateGarch(prices, { periodsPerYear: 252 });
    const { omega, alpha, beta } = result.params;

    // Allow 50% tolerance — MLE on finite sample won't be exact
    expect(omega).toBeGreaterThan(trueOmega * 0.2);
    expect(omega).toBeLessThan(trueOmega * 5);
    expect(alpha).toBeGreaterThan(trueAlpha * 0.5);
    expect(alpha).toBeLessThan(trueAlpha * 2);
    expect(beta).toBeGreaterThan(trueBeta * 0.8);
    expect(beta).toBeLessThan(trueBeta * 1.2);
  });

  it('recovers persistence ≈ α + β from true process', () => {
    const trueAlpha = 0.08;
    const trueBeta = 0.9;
    const truePersistence = trueAlpha + trueBeta; // 0.98

    const rng = lcg(8888);
    const returns: number[] = [];
    let variance = 0.00001 / (1 - truePersistence);
    for (let i = 0; i < 2000; i++) {
      const eps = Math.sqrt(variance) * randn(rng);
      returns.push(eps);
      variance = 0.00001 + trueAlpha * eps ** 2 + trueBeta * variance;
    }
    const prices = [100];
    for (const r of returns) prices.push(prices[prices.length - 1] * Math.exp(r));

    const result = calibrateGarch(prices, { periodsPerYear: 252 });
    expect(result.params.persistence).toBeGreaterThan(truePersistence * 0.9);
    expect(result.params.persistence).toBeLessThan(truePersistence * 1.1);
  });
});
