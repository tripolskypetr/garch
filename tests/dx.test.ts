import { describe, it, expect } from 'vitest';
import {
  predict,
  predictRange,
  checkData,
  kupiecTest,
  backtestStats,
  type CandleInterval,
} from '../src/predict.js';
import { NotEnoughDataError, BadDataError, InvalidArgumentError, GarchError } from '../src/errors.js';
import type { Candle } from '../src/types.js';

// ── helpers ──────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeCandles(n: number, seed = 42, withTs = true, hourMs = 3_600_000): Candle[] {
  const rng = mulberry32(seed);
  const t0 = Date.UTC(2026, 0, 1);
  let close = 100;
  const candles: Candle[] = [];
  let v = 4e-4;
  let rPrev = Math.sqrt(v) * randn(rng);
  for (let i = 0; i < n; i++) {
    v = 4e-4 * 0.04 + 0.08 * rPrev * rPrev + 0.88 * v;
    const r = Math.sqrt(v) * randn(rng);
    rPrev = r;
    const open = close;
    close = open * Math.exp(r);
    const w = (Math.abs(r) / 2) * (0.5 + rng());
    candles.push({
      open,
      high: Math.max(open, close) * Math.exp(w),
      low: Math.min(open, close) * Math.exp(-w),
      close,
      volume: 1,
      ...(withTs ? { timestamp: t0 + i * hourMs } : {}),
    });
  }
  return candles;
}

// ── typed errors ─────────────────────────────────────────────

describe('typed errors', () => {
  it('too few candles → NotEnoughDataError with the actionable message kept', () => {
    const candles = makeCandles(100);
    try {
      predict(candles, '1h');
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(NotEnoughDataError);
      expect(e).toBeInstanceOf(GarchError);
      expect((e as Error).message).toContain('Need at least 200');
    }
  });

  it('unknown interval → InvalidArgumentError listing valid intervals', () => {
    const candles = makeCandles(300);
    try {
      predict(candles, '7h' as CandleInterval);
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(InvalidArgumentError);
      expect((e as Error).message).toContain('1m');
      expect((e as Error).message).toContain('8h');
    }
  });

  it('percent confidence → InvalidArgumentError with a fraction hint', () => {
    const candles = makeCandles(300);
    try {
      predict(candles, '1h', { confidence: 90 });
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(InvalidArgumentError);
      expect((e as Error).message).toContain('0.9');
    }
  });

  it('non-positive currentPrice → InvalidArgumentError (no silent fallback)', () => {
    const candles = makeCandles(300);
    expect(() => predict(candles, '1h', 0)).toThrow(InvalidArgumentError);
    expect(() => predict(candles, '1h', -5)).toThrow(InvalidArgumentError);
    expect(() => predict(candles, '1h', NaN)).toThrow(InvalidArgumentError);
  });

  it('unsorted timestamps → BadDataError', () => {
    const candles = makeCandles(300);
    [candles[50], candles[51]] = [candles[51], candles[50]];
    const swapped = candles.map((c, i) => ({ ...c, timestamp: makeCandles(300)[i].timestamp }));
    // Swap only timestamps to keep OHLC chain valid
    const ts50 = swapped[50].timestamp!;
    swapped[50] = { ...swapped[50], timestamp: swapped[51].timestamp };
    swapped[51] = { ...swapped[51], timestamp: ts50 };
    expect(() => predict(swapped, '1h')).toThrow(BadDataError);
  });

  it('duplicate timestamps → BadDataError', () => {
    const candles = makeCandles(300);
    candles[100] = { ...candles[100], timestamp: candles[99].timestamp };
    expect(() => predict(candles, '1h')).toThrow(BadDataError);
  });
});

// ── options object ───────────────────────────────────────────

describe('options-object API', () => {
  const candles = makeCandles(300, 7);

  it('predict(candles, interval, { confidence }) equals the positional form', () => {
    const a = predict(candles, '1h', null, 0.9);
    const b = predict(candles, '1h', { confidence: 0.9 });
    expect(b.upperPrice).toBe(a.upperPrice);
    expect(b.lowerPrice).toBe(a.lowerPrice);
  });

  it('options object carries currentPrice too', () => {
    const a = predict(candles, '1h', 123.45, 0.9);
    const b = predict(candles, '1h', { currentPrice: 123.45, confidence: 0.9 });
    expect(b.upperPrice).toBe(a.upperPrice);
    expect(b.currentPrice).toBe(123.45);
  });

  it('predictRange accepts the options object as 4th argument', () => {
    const a = predictRange(candles, '1h', 5, null, 0.9);
    const b = predictRange(candles, '1h', 5, { confidence: 0.9 });
    expect(b.upperPrice).toBe(a.upperPrice);
  });
});

// ── warnings and transparency ────────────────────────────────

describe('prediction warnings and transparency', () => {
  it('healthy data: reliable, no critical warnings, weights sum to 1', () => {
    const res = predict(makeCandles(600, 11), '1h');
    expect(res.warnings.every(w => !w.critical)).toBe(res.reliable);
    const sum = Object.values(res.modelWeights).reduce((s, w) => s + (w ?? 0), 0);
    expect(sum).toBeCloseTo(1, 6);
    expect(typeof res.seasonalityDetected).toBe('boolean');
  });

  it('short sample carries a LOW_SAMPLE warning (non-critical)', () => {
    const res = predict(makeCandles(220, 12), '1h'); // < 500 recommended
    const w = res.warnings.find(w => w.code === 'LOW_SAMPLE');
    expect(w).toBeDefined();
    expect(w!.critical).toBe(false);
    expect(w!.message).toContain('500');
  });

  it('flat market: reliable=false explained by a critical warning', () => {
    const flat: Candle[] = Array.from({ length: 300 }, () => ({
      open: 100, high: 100, low: 100, close: 100, volume: 1,
    }));
    const res = predict(flat, '1h');
    expect(res.reliable).toBe(false);
    expect(res.warnings.some(w => w.critical)).toBe(true);
  });

  it('gappy feed carries a DATA_GAPS warning', () => {
    const candles = makeCandles(360, 13).filter((_, i) => i % 30 !== 7); // drop ~3.3% of bars
    const res = predict(candles, '1h');
    const w = res.warnings.find(w => w.code === 'DATA_GAPS');
    expect(w).toBeDefined();
    expect(w!.critical).toBe(false);
  });

  it('5m candles passed as 1h carry an INTERVAL_MISMATCH warning', () => {
    const res = predict(makeCandles(300, 14, true, 300_000), '1h');
    expect(res.warnings.some(w => w.code === 'INTERVAL_MISMATCH')).toBe(true);
  });
});

// ── checkData ────────────────────────────────────────────────

describe('checkData', () => {
  it('clean data: ok with no error issues', () => {
    const report = checkData(makeCandles(600, 21), '1h');
    expect(report.ok).toBe(true);
    expect(report.issues.filter(i => i.severity === 'error')).toHaveLength(0);
    expect(report.recommendedCandles).toBe(500);
  });

  it('reports too few candles as an error without throwing', () => {
    const report = checkData(makeCandles(50, 22), '1h');
    expect(report.ok).toBe(false);
    expect(report.issues.some(i => i.code === 'TOO_FEW_CANDLES' && i.severity === 'error')).toBe(true);
  });

  it('reports broken OHLC with the candle index', () => {
    const candles = makeCandles(300, 23);
    candles[42] = { ...candles[42], high: candles[42].low * 0.5 };
    const report = checkData(candles, '1h');
    expect(report.ok).toBe(false);
    const issue = report.issues.find(i => i.code === 'INVALID_OHLC');
    expect(issue).toBeDefined();
    expect(issue!.message).toContain('42');
  });

  it('reports unsorted and duplicate timestamps as errors', () => {
    const unsorted = makeCandles(300, 24);
    const ts = unsorted[10].timestamp!;
    unsorted[10] = { ...unsorted[10], timestamp: unsorted[11].timestamp };
    unsorted[11] = { ...unsorted[11], timestamp: ts };
    expect(checkData(unsorted, '1h').issues.some(i => i.code === 'UNSORTED')).toBe(true);

    const dup = makeCandles(300, 25);
    dup[100] = { ...dup[100], timestamp: dup[99].timestamp };
    expect(checkData(dup, '1h').issues.some(i => i.code === 'DUPLICATE_TIMESTAMPS')).toBe(true);
  });

  it('reports gaps, interval mismatch, and flat candles as warnings', () => {
    const gappy = makeCandles(360, 26).filter((_, i) => i % 25 !== 3);
    expect(checkData(gappy, '1h').issues.some(i => i.code === 'DATA_GAPS' && i.severity === 'warning')).toBe(true);

    const wrong = makeCandles(300, 27, true, 300_000);
    expect(checkData(wrong, '1h').issues.some(i => i.code === 'INTERVAL_MISMATCH')).toBe(true);

    const flat = makeCandles(300, 28).map((c, i) =>
      i % 3 === 0 ? { ...c, high: c.close, low: c.close, open: c.close } : c,
    );
    expect(checkData(flat, '1h').issues.some(i => i.code === 'FLAT_CANDLES')).toBe(true);
  });
});

// ── Kupiec verdict ───────────────────────────────────────────

describe('kupiecTest', () => {
  it('coverage matching nominal → well-calibrated', () => {
    const r = kupiecTest(68, 100, 0.68);
    expect(r.verdict).toBe('well-calibrated');
    expect(r.pValue).toBeGreaterThan(0.05);
    expect(r.message).toContain('consistent');
  });

  it('massive under-coverage → too-narrow with tiny p-value', () => {
    const r = kupiecTest(50, 100, 0.9);
    expect(r.verdict).toBe('too-narrow');
    expect(r.pValue).toBeLessThan(0.001);
    expect(r.message).toContain('too narrow');
  });

  it('perfect coverage at 68% over many points → too-wide', () => {
    const r = kupiecTest(200, 200, 0.6827);
    expect(r.verdict).toBe('too-wide');
    expect(r.message).toContain('too wide');
  });

  it('small samples → inconclusive regardless of rate', () => {
    const r = kupiecTest(2, 10, 0.9);
    expect(r.verdict).toBe('inconclusive');
    expect(r.message).toContain('30');
  });

  it('borderline gap on a small-but-sufficient sample is not flagged', () => {
    // 63% observed vs 68% nominal on n=100 — within noise, must not scream
    const r = kupiecTest(63, 100, 0.6827);
    expect(r.verdict).toBe('well-calibrated');
  });

  it('backtestStats carries the verdict fields end-to-end', () => {
    const stats = backtestStats(makeCandles(300, 31), '1h', 0.6827);
    expect(['well-calibrated', 'too-narrow', 'too-wide', 'inconclusive']).toContain(stats.verdict);
    expect(stats.pValue).toBeGreaterThanOrEqual(0);
    expect(stats.pValue).toBeLessThanOrEqual(1);
    expect(stats.message.length).toBeGreaterThan(20);
  }, 900_000);
});
