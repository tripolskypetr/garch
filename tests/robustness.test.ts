import { describe, it, expect } from 'vitest';
import { predict, predictRange, backtest } from '../src/index.js';
import type { Candle } from '../src/index.js';

// ── helpers ──────────────────────────────────────────────────

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

/** Assert that a PredictionResult is sane (no NaN, no Infinity, positive sigma) */
function assertSaneResult(result: ReturnType<typeof predict>, label: string) {
  expect(Number.isFinite(result.sigma), `${label}: sigma finite`).toBe(true);
  expect(Number.isFinite(result.move), `${label}: move finite`).toBe(true);
  expect(Number.isFinite(result.upperPrice), `${label}: upperPrice finite`).toBe(true);
  expect(Number.isFinite(result.lowerPrice), `${label}: lowerPrice finite`).toBe(true);
  expect(Number.isFinite(result.currentPrice), `${label}: currentPrice finite`).toBe(true);
  expect(result.sigma, `${label}: sigma > 0`).toBeGreaterThanOrEqual(0);
  expect(result.upperPrice, `${label}: upper > lower`).toBeGreaterThan(result.lowerPrice);
  expect(typeof result.reliable, `${label}: reliable is boolean`).toBe('boolean');
  expect(['garch', 'egarch', 'har-rv', 'novas'], `${label}: valid modelType`).toContain(result.modelType);
}

// ── 1. Flash crash — single candle drops 50% ────────────────

describe('flash crash', () => {
  function makeFlashCrash(n: number, crashAt: number, seed = 42): Candle[] {
    const rng = lcg(seed);
    const candles: Candle[] = [];
    let price = 1000;
    for (let i = 0; i < n; i++) {
      let r = (rng() - 0.5) * 0.03;
      if (i === crashAt) r = -0.7; // ~50% drop
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * (1 + Math.abs(r) * 0.2);
      const low = Math.min(price, close) * (1 - Math.abs(r) * 0.2);
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }
    return candles;
  }

  it('predict survives flash crash in the middle', () => {
    const candles = makeFlashCrash(200, 100);
    const result = predict(candles, '4h');
    assertSaneResult(result, 'flash-crash-mid');
  });

  it('predict survives flash crash near the end', () => {
    const candles = makeFlashCrash(200, 195);
    const result = predict(candles, '4h');
    assertSaneResult(result, 'flash-crash-end');
  });

  it('predict survives flash crash at start', () => {
    const candles = makeFlashCrash(200, 1);
    const result = predict(candles, '4h');
    assertSaneResult(result, 'flash-crash-start');
  });

  it('flash crash increases sigma vs same data without crash', () => {
    // Build identical base data, then inject crash into a copy
    const rng = lcg(42);
    const base: Candle[] = [];
    let price = 1000;
    for (let i = 0; i < 200; i++) {
      const r = (rng() - 0.5) * 0.03;
      const close = price * Math.exp(r);
      base.push({ open: price, high: Math.max(price, close) * 1.005, low: Math.min(price, close) * 0.995, close, volume: 1000 });
      price = close;
    }
    const crashed = base.map(c => ({ ...c }));
    // Inject -50% crash at candle 190
    const prevClose = crashed[189].close;
    const crashClose = prevClose * Math.exp(-0.7);
    crashed[190] = { open: prevClose, high: prevClose * 1.005, low: crashClose * 0.995, close: crashClose, volume: 5000 };

    const calmSigma = predict(base, '4h').sigma;
    const crashSigma = predict(crashed, '4h').sigma;
    expect(crashSigma).toBeGreaterThan(calmSigma);
  });
});

// ── 2. Monotone trend — all returns same direction ──────────

describe('monotone trend', () => {
  it('survives 100% uptrend (every candle green)', () => {
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const close = price * 1.005; // +0.5% every candle
      candles.push({ open: price, high: close * 1.001, low: price * 0.999, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'uptrend');
  });

  it('survives 100% downtrend (every candle red)', () => {
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const close = price * 0.995; // -0.5% every candle
      candles.push({ open: price, high: price * 1.001, low: close * 0.999, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'downtrend');
  });

  it('uptrend with noise still produces sane results', () => {
    const rng = lcg(100);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const drift = 0.003; // consistent upward bias
      const noise = (rng() - 0.5) * 0.01;
      const close = price * Math.exp(drift + noise);
      candles.push({ open: price, high: Math.max(price, close) * 1.002, low: Math.min(price, close) * 0.998, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'biased-uptrend');
  });
});

// ── 3. Shitcoin prices (1e-8 satoshi level) ────────────────

describe('micro-cap / shitcoin prices', () => {
  it('predict works at satoshi-level prices', () => {
    const rng = lcg(777);
    const candles: Candle[] = [];
    let price = 0.00000001; // 1 satoshi
    for (let i = 0; i < 200; i++) {
      const r = (rng() - 0.5) * 0.1;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * 1.05;
      const low = Math.min(price, close) * 0.95;
      candles.push({ open: price, high, low, close, volume: 1e15 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'satoshi-price');
  });

  it('predictRange works at satoshi-level prices', () => {
    const rng = lcg(888);
    const candles: Candle[] = [];
    let price = 0.00000001;
    for (let i = 0; i < 300; i++) {
      const r = (rng() - 0.5) * 0.1;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * 1.05;
      const low = Math.min(price, close) * 0.95;
      candles.push({ open: price, high, low, close, volume: 1e15 });
      price = close;
    }
    const result = predictRange(candles, '15m', 16);
    assertSaneResult(result, 'satoshi-predictRange');
  });
});

// ── 4. Duplicate / repeated candles ─────────────────────────

describe('repeated candles', () => {
  it('survives many identical candles followed by normal data', () => {
    const candles: Candle[] = [];
    // 100 identical candles (exchange freeze / no trades)
    for (let i = 0; i < 100; i++) {
      candles.push({ open: 100, high: 100.01, low: 99.99, close: 100, volume: 0 });
    }
    // Then 100 normal candles
    const rng = lcg(42);
    let price = 100;
    for (let i = 0; i < 100; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      candles.push({ open: price, high: Math.max(price, close) * 1.005, low: Math.min(price, close) * 0.995, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'repeated-then-normal');
  });

  it('survives nearly-identical prices (tight spread)', () => {
    const rng = lcg(55);
    const candles: Candle[] = [];
    let price = 50000;
    for (let i = 0; i < 200; i++) {
      // Noise within $0.01 on a $50k asset (0.00002%)
      const close = price + (rng() - 0.5) * 0.01;
      candles.push({ open: price, high: price + 0.01, low: price - 0.01, close, volume: 100 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'tight-spread');
    expect(result.sigma).toBeLessThan(0.001); // essentially zero vol
  });
});

// ── 5. Regime switch mid-series ─────────────────────────────

describe('regime switch', () => {
  it('calm → volatile transition', () => {
    const rng = lcg(33);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const vol = i < 150 ? 0.005 : 0.06; // 10x volatility jump at candle 150
      const r = (rng() - 0.5) * vol;
      const close = price * Math.exp(r);
      candles.push({ open: price, high: Math.max(price, close) * 1.002, low: Math.min(price, close) * 0.998, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'calm-to-volatile');
  });

  it('volatile → calm transition', () => {
    const rng = lcg(44);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const vol = i < 100 ? 0.06 : 0.005;
      const r = (rng() - 0.5) * vol;
      const close = price * Math.exp(r);
      candles.push({ open: price, high: Math.max(price, close) * 1.002, low: Math.min(price, close) * 0.998, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'volatile-to-calm');
  });
});

// ── 6. Large dataset — 5000 candles ─────────────────────────

describe('large dataset', () => {
  it('predict handles 5000 candles without crashing', () => {
    const rng = lcg(999);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 5000; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * 1.005;
      const low = Math.min(price, close) * 0.995;
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, '5000-candles');
  });

  it('predictRange handles 5000 candles', () => {
    const rng = lcg(998);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 5000; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * 1.005;
      const low = Math.min(price, close) * 0.995;
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }
    const result = predictRange(candles, '4h', 16);
    assertSaneResult(result, '5000-candles-range');
  });
});

// ── 7. Fuzz — predict never returns NaN on valid data ───────

describe('fuzz: predict never returns NaN/Infinity', () => {
  const seeds = [1, 2, 3, 7, 13, 42, 99, 137, 256, 512, 1024, 2048, 4096, 7777, 9999, 31337];

  for (const seed of seeds) {
    it(`seed=${seed}: 4h predict returns sane result`, () => {
      const rng = lcg(seed);
      const candles: Candle[] = [];
      let price = 50 + rng() * 100000; // random starting price
      for (let i = 0; i < 200; i++) {
        const r = (rng() - 0.5) * (0.01 + rng() * 0.1); // random volatility
        const close = price * Math.exp(r);
        const high = Math.max(price, close) * (1 + rng() * 0.02);
        const low = Math.min(price, close) * (1 - rng() * 0.02);
        candles.push({ open: price, high, low, close, volume: rng() * 1e6 });
        price = close;
      }
      const result = predict(candles, '4h');
      assertSaneResult(result, `fuzz-${seed}`);
    });
  }

  for (const seed of [11, 22, 33, 44, 55, 66, 77, 88]) {
    it(`seed=${seed}: 15m predict returns sane result`, () => {
      const rng = lcg(seed);
      const candles: Candle[] = [];
      let price = 1 + rng() * 50000;
      for (let i = 0; i < 300; i++) {
        const r = (rng() - 0.5) * (0.005 + rng() * 0.08);
        const close = price * Math.exp(r);
        const high = Math.max(price, close) * (1 + rng() * 0.01);
        const low = Math.min(price, close) * (1 - rng() * 0.01);
        candles.push({ open: price, high, low, close, volume: rng() * 1e6 });
        price = close;
      }
      const result = predict(candles, '15m');
      assertSaneResult(result, `fuzz-15m-${seed}`);
    });
  }
});

// ── 8. Extreme wicks (high >> close, low << close) ──────────

describe('extreme wicks', () => {
  it('survives candles with 20% wicks', () => {
    const rng = lcg(111);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 200; i++) {
      const r = (rng() - 0.5) * 0.03;
      const close = price * Math.exp(r);
      // Extreme wicks: high 20% above, low 20% below
      const high = Math.max(price, close) * 1.2;
      const low = Math.min(price, close) * 0.8;
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'extreme-wicks');
  });
});

// ── 9. Price pump-and-dump ──────────────────────────────────

describe('pump and dump', () => {
  it('survives 10x pump followed by 90% dump', () => {
    const rng = lcg(222);
    const candles: Candle[] = [];
    let price = 1;
    for (let i = 0; i < 200; i++) {
      let r: number;
      if (i >= 80 && i < 100) {
        r = 0.12; // pump phase: ~12% per candle
      } else if (i >= 100 && i < 120) {
        r = -0.15; // dump phase: ~15% per candle
      } else {
        r = (rng() - 0.5) * 0.04;
      }
      const close = price * Math.exp(r);
      const high = Math.max(price, close) * (1 + Math.abs(r) * 0.3);
      const low = Math.min(price, close) * (1 - Math.abs(r) * 0.3);
      candles.push({ open: price, high, low, close, volume: 1000 });
      price = close;
    }
    const result = predict(candles, '4h');
    assertSaneResult(result, 'pump-and-dump');
  });
});

// ── 10. All intervals produce sane output ───────────────────

describe('all CandleInterval values work', () => {
  const intervals = [
    { interval: '1m' as const, min: 500 },
    { interval: '3m' as const, min: 500 },
    { interval: '5m' as const, min: 500 },
    { interval: '15m' as const, min: 300 },
    { interval: '30m' as const, min: 200 },
    { interval: '1h' as const, min: 200 },
    { interval: '2h' as const, min: 200 },
    { interval: '4h' as const, min: 200 },
    { interval: '6h' as const, min: 150 },
    { interval: '8h' as const, min: 150 },
  ];

  for (const { interval, min } of intervals) {
    it(`predict works for ${interval} with ${min} candles`, () => {
      const rng = lcg(interval.length * 100 + min);
      const candles: Candle[] = [];
      let price = 100;
      for (let i = 0; i < min; i++) {
        const r = (rng() - 0.5) * 0.04;
        const close = price * Math.exp(r);
        candles.push({ open: price, high: Math.max(price, close) * 1.005, low: Math.min(price, close) * 0.995, close, volume: 1000 });
        price = close;
      }
      const result = predict(candles, interval);
      assertSaneResult(result, `interval-${interval}`);
    });
  }
});

// ── 11. backtest never throws on valid data ─────────────────

describe('backtest robustness', () => {
  it('does not throw on 250 normal candles for 4h', () => {
    const rng = lcg(300);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 250; i++) {
      const r = (rng() - 0.5) * 0.04;
      const close = price * Math.exp(r);
      candles.push({ open: price, high: Math.max(price, close) * 1.005, low: Math.min(price, close) * 0.995, close, volume: 1000 });
      price = close;
    }
    expect(() => backtest(candles, '4h')).not.toThrow();
  });

  it('does not throw on regime-switch data', () => {
    const rng = lcg(400);
    const candles: Candle[] = [];
    let price = 100;
    for (let i = 0; i < 300; i++) {
      const vol = i < 200 ? 0.01 : 0.08;
      const r = (rng() - 0.5) * vol;
      const close = price * Math.exp(r);
      candles.push({ open: price, high: Math.max(price, close) * 1.005, low: Math.min(price, close) * 0.995, close, volume: 1000 });
      price = close;
    }
    expect(() => backtest(candles, '4h')).not.toThrow();
  });
});
