import { predict } from '../build/index.mjs';

// ── RNG ──
function lcg(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff; };
}
function randn(rng) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// helper: return → candle with Brownian-bridge H/L
function priceToCandle(open, r, sigma, rng) {
  const close = open * Math.exp(r);
  const drift = Math.abs(randn(rng)) * sigma * 0.35;
  const high = Math.max(open, close) * (1 + drift);
  const low = Math.min(open, close) * (1 - Math.abs(randn(rng)) * sigma * 0.35);
  return { open, high: Math.max(high, open, close), low: Math.min(low, open, close), close, volume: 1000 };
}

// ═══════════════════════════════════════════════════════════
// DGP 1: GARCH(1,1) — symmetric vol clustering, high alpha
// Strong shock reaction, no leverage → pure GARCH territory
// ═══════════════════════════════════════════════════════════
function dgpGarch(n, seed) {
  const rng = lcg(seed);
  const omega = 0.000005, alpha = 0.25, beta = 0.70;
  let sigma2 = omega / (1 - alpha - beta);
  let price = 100;
  const candles = [];
  for (let i = 0; i < n; i++) {
    const z = randn(rng);
    const r = Math.sqrt(sigma2) * z;
    candles.push(priceToCandle(price, r, Math.sqrt(sigma2), rng));
    sigma2 = omega + alpha * r * r + beta * sigma2;
    price = candles[candles.length - 1].close;
  }
  return candles;
}

// ═══════════════════════════════════════════════════════════
// DGP 2: EGARCH — strong leverage (negative returns → vol explosion)
// gamma = -0.15 makes bad news amplify variance much more
// ═══════════════════════════════════════════════════════════
function dgpEgarch(n, seed) {
  const rng = lcg(seed);
  const omega = -0.15, alphaE = 0.2, gamma = -0.25, beta = 0.98;
  const EabsZ = Math.sqrt(2 / Math.PI); // E[|Z|] for N(0,1)
  let lnSigma2 = omega / (1 - beta);
  let price = 100;
  const candles = [];
  for (let i = 0; i < n; i++) {
    const sigma2 = Math.exp(lnSigma2);
    const sigma = Math.sqrt(sigma2);
    const z = randn(rng);
    const r = sigma * z;
    candles.push(priceToCandle(price, r, sigma, rng));
    lnSigma2 = omega + alphaE * (Math.abs(z) - EabsZ) + gamma * z + beta * lnSigma2;
    price = candles[candles.length - 1].close;
  }
  return candles;
}

// ═══════════════════════════════════════════════════════════
// DGP 3: GJR-GARCH — moderate leverage via indicator I(r<0)
// gamma adds extra variance only on negative returns
// ═══════════════════════════════════════════════════════════
function dgpGjr(n, seed) {
  const rng = lcg(seed);
  const omega = 0.000003, alpha = 0.05, gammaGjr = 0.20, beta = 0.72;
  let sigma2 = omega / (1 - alpha - gammaGjr / 2 - beta);
  let price = 100;
  const candles = [];
  for (let i = 0; i < n; i++) {
    const z = randn(rng);
    const r = Math.sqrt(sigma2) * z;
    const ind = r < 0 ? 1 : 0;
    candles.push(priceToCandle(price, r, Math.sqrt(sigma2), rng));
    sigma2 = omega + alpha * r * r + gammaGjr * r * r * ind + beta * sigma2;
    price = candles[candles.length - 1].close;
  }
  return candles;
}

// ═══════════════════════════════════════════════════════════
// DGP 4: HAR-RV — multi-scale volatility (short + medium + long memory)
// Variance depends on rolling averages at 3 horizons
// ═══════════════════════════════════════════════════════════
function dgpHar(n, seed) {
  const rng = lcg(seed);
  const b0 = 0.000002, b1 = 0.3, b2 = 0.35, b3 = 0.25;
  const rvHist = [];
  let price = 100;
  const candles = [];
  // warm-up RV history
  for (let i = 0; i < 22; i++) rvHist.push(0.0001);
  for (let i = 0; i < n; i++) {
    const rvShort = rvHist[rvHist.length - 1];
    const rvMed = rvHist.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const rvLong = rvHist.slice(-22).reduce((a, b) => a + b, 0) / 22;
    const sigma2 = Math.max(b0 + b1 * rvShort + b2 * rvMed + b3 * rvLong, 1e-10);
    const sigma = Math.sqrt(sigma2);
    const z = randn(rng);
    const r = sigma * z;
    candles.push(priceToCandle(price, r, sigma, rng));
    // realized variance = r^2 + noise (simulating Parkinson-like measurement)
    rvHist.push(r * r * (0.8 + 0.4 * rng()));
    price = candles[candles.length - 1].close;
  }
  return candles;
}

// ═══════════════════════════════════════════════════════════
// DGP 5: NoVaS — far-lag ARCH(10) with non-standard lag structure
// Variance depends on lags 1, 4, 7, 10 — no parametric model fits this cleanly
// ═══════════════════════════════════════════════════════════
function dgpNovas(n, seed) {
  const rng = lcg(seed);
  const a0 = 0.00001;
  const weights = [0, 0.15, 0, 0, 0.15, 0, 0, 0.20, 0, 0, 0.15]; // lags 1,4,7,10
  const rvHist = [];
  for (let i = 0; i < 12; i++) rvHist.push(0.0001);
  let price = 100;
  const candles = [];
  for (let i = 0; i < n; i++) {
    let sigma2 = a0;
    for (let j = 1; j <= 10; j++) {
      sigma2 += weights[j] * rvHist[rvHist.length - j];
    }
    sigma2 = Math.max(sigma2, 1e-10);
    const sigma = Math.sqrt(sigma2);
    const z = randn(rng);
    const r = sigma * z;
    candles.push(priceToCandle(price, r, sigma, rng));
    rvHist.push(r * r);
    price = candles[candles.length - 1].close;
  }
  return candles;
}

// ═══════════════════════════════════════════════════════════
// Run each DGP across 20 seeds, count model wins
// ═══════════════════════════════════════════════════════════
const dgps = [
  { name: 'GARCH(1,1) — symmetric clustering', fn: dgpGarch, expect: 'garch' },
  { name: 'EGARCH — strong leverage', fn: dgpEgarch, expect: 'egarch' },
  { name: 'GJR-GARCH — indicator leverage', fn: dgpGjr, expect: 'gjr-garch' },
  { name: 'HAR-RV — multi-scale memory', fn: dgpHar, expect: 'har-rv' },
  { name: 'NoVaS — far-lag ARCH(10)', fn: dgpNovas, expect: 'novas' },
];

const N = 500;
const SEEDS = 20;
let totalPassed = 0, totalFailed = 0;

for (const { name, fn, expect } of dgps) {
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`DGP: ${name}`);
  console.log(`Expected winner: ${expect}`);
  console.log(`${'─'.repeat(60)}`);

  const wins = {};
  for (let seed = 1; seed <= SEEDS; seed++) {
    const candles = fn(N, seed * 7 + 3);
    const result = predict(candles, '4h');
    wins[result.modelType] = (wins[result.modelType] || 0) + 1;
  }

  // sort by wins descending
  const sorted = Object.entries(wins).sort((a, b) => b[1] - a[1]);
  for (const [model, count] of sorted) {
    const bar = '█'.repeat(count) + '░'.repeat(SEEDS - count);
    const pct = ((count / SEEDS) * 100).toFixed(0);
    const marker = model === expect ? ' ◄ expected' : '';
    console.log(`  ${model.padEnd(12)} ${bar} ${count}/${SEEDS} (${pct}%)${marker}`);
  }

  // the expected model should win at least once
  const expectedWins = wins[expect] || 0;
  if (expectedWins > 0) {
    totalPassed++;
    console.log(`  ✓ ${expect} won ${expectedWins}/${SEEDS} times`);
  } else {
    totalFailed++;
    console.log(`  ✗ ${expect} never won! Top: ${sorted[0][0]} (${sorted[0][1]}/${SEEDS})`);
  }

  // the expected model should be in top-3 by win count
  const top3 = sorted.slice(0, 3).map(s => s[0]);
  if (top3.includes(expect)) {
    totalPassed++;
    console.log(`  ✓ ${expect} is in top-3 models`);
  } else {
    totalFailed++;
    console.log(`  ✗ ${expect} not in top-3: [${top3.join(', ')}]`);
  }
}

// ═══════════════════════════════════════════════════════════
// Verify all 5 modelTypes appeared at least once across all DGPs
// ═══════════════════════════════════════════════════════════
console.log(`\n${'═'.repeat(60)}`);
console.log('Coverage: checking all 5 modelTypes appeared');
const allModels = new Set();
for (const { fn } of dgps) {
  for (let seed = 1; seed <= SEEDS; seed++) {
    const result = predict(fn(N, seed * 7 + 3), '4h');
    allModels.add(result.modelType);
  }
}
const required = ['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas'];
for (const m of required) {
  if (allModels.has(m)) {
    totalPassed++;
    console.log(`  ✓ ${m} appeared`);
  } else {
    totalFailed++;
    console.log(`  ✗ ${m} never appeared in any DGP`);
  }
}

console.log(`\n${'═'.repeat(60)}`);
console.log(`TOTAL: ${totalPassed + totalFailed} checks — ${totalPassed} passed, ${totalFailed} failed`);
if (totalFailed > 0) process.exit(1);
else console.log('ALL CHECKS PASSED');
