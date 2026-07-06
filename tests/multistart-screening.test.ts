import { describe, it, expect } from 'vitest';
import { nelderMeadMultiStart } from '../src/optimizer.js';
import { NoVaS } from '../src/novas.js';

// ── extraStarts: shape-changing basins are unreachable from x0 ──
// The multi-start perturbation scales x0 multiplicatively (x0[i]·(1±0.5)),
// which preserves x0's shape: a component near zero stays near zero. A
// global minimum whose shape differs from x0 needs an explicit extra start.

describe('nelderMeadMultiStart extraStarts', () => {
  // Two basins: shallow at (1, 0), deep at (0, 10). Multiplicative
  // perturbation of x0 = (1, 0) can never move the second coordinate
  // beyond ±0.0005, so the deep basin is unreachable without a seed.
  const twoBasin = (x: number[]): number => {
    const shallow = (x[0] - 1) ** 2 + x[1] ** 2;
    const deep = (x[0]) ** 2 + (x[1] - 10) ** 2 - 5;
    return Math.min(shallow, deep);
  };

  it('without extraStarts stays in the x0-shaped basin', () => {
    const res = nelderMeadMultiStart(twoBasin, [1, 0], { restarts: 6 });
    expect(res.fx).toBeCloseTo(0, 5);
  });

  it('with extraStarts escapes to the deeper basin', () => {
    const res = nelderMeadMultiStart(twoBasin, [1, 0], {
      restarts: 6,
      extraStarts: [[0.5, 8]],
    });
    expect(res.fx).toBeCloseTo(-5, 5);
    expect(res.x[1]).toBeCloseTo(10, 3);
  });

  it('bad extraStarts do not degrade the result', () => {
    const sphere = (x: number[]) => x.reduce((s, v) => s + v * v, 0);
    const base = nelderMeadMultiStart(sphere, [1, 1], { restarts: 3 });
    const withBad = nelderMeadMultiStart(sphere, [1, 1], {
      restarts: 3,
      extraStarts: [[1e6, -1e6], [42, 42]],
    });
    expect(withBad.fx).toBeLessThanOrEqual(base.fx + 1e-12);
  });
});

// ── NoVaS screening: far-lag structure recovery ─────────────────
// Ground truth: ARCH with weight spikes on lags 1 and 9. The exp-decay x0
// puts ~0.011 on lag 9; multiplicative perturbation cannot grow that into
// the true ~0.45 spike. Before screening the fit landed at D² ≈ 3.9e-2;
// quasi-random screening + extra starts reaches ≈ 6e-3 with the weight
// mass on the correct lags.

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function synthArchPrices(n: number, seed: number, omega: number, lagW: number[]): number[] {
  const rng = mulberry32(seed);
  const p = lagW.length;
  const rs: number[] = [];
  let price = 100;
  const prices = [price];
  const uncond = omega / (1 - lagW.reduce((s, w) => s + w, 0));
  for (let i = 0; i < n; i++) {
    let v = omega;
    for (let j = 1; j <= p; j++) {
      const r2 = i - j >= 0 ? rs[i - j] ** 2 : uncond;
      v += lagW[j - 1] * r2;
    }
    const r = Math.sqrt(v) * gaussian(rng);
    rs.push(r);
    price *= Math.exp(r);
    prices.push(price);
  }
  return prices;
}

describe('NoVaS far-lag recovery via screening', () => {
  it('two-spike ARCH (lags 1 and 9): D² escapes the exp-decay local minimum', () => {
    // seed 7 is the measured failure case: D² was 3.9e-2 without screening
    const lagW = [0.45, 0, 0, 0, 0, 0, 0, 0, 0.45, 0];
    const prices = synthArchPrices(800, 7, 4e-6, lagW);
    const fit = new NoVaS(prices, { lags: 10 }).fit();

    expect(fit.params.dSquared).toBeLessThan(2e-2);
    // weight mass must reach the far spike at lag 9
    expect(fit.params.weights[9]).toBeGreaterThan(0.05);
  });

  it('far-lag ARCH (lags 8-10): weight mass lands on the far lags', () => {
    const lagW = [0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3];
    const prices = synthArchPrices(800, 7, 4e-6, lagW);
    const fit = new NoVaS(prices, { lags: 10 }).fit();

    const farMass = fit.params.weights[8] + fit.params.weights[9] + fit.params.weights[10];
    const nearMass = fit.params.weights[1] + fit.params.weights[2] + fit.params.weights[3];
    expect(farMass).toBeGreaterThan(nearMass);
    expect(farMass).toBeGreaterThan(0.3);
  });

  it('screening is deterministic: same input, same weights', () => {
    const lagW = [0.45, 0, 0, 0, 0, 0, 0, 0, 0.45, 0];
    const prices = synthArchPrices(800, 7, 4e-6, lagW);
    const a = new NoVaS(prices, { lags: 10 }).fit();
    const b = new NoVaS(prices, { lags: 10 }).fit();
    expect(a.params.weights).toEqual(b.params.weights);
    expect(a.params.dSquared).toBe(b.params.dSquared);
  });
});
