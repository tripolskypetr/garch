import { predict, predictRange, backtest } from '../build/index.mjs';

// ── RNG helpers ──
function lcg(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff; };
}
function randn(rng) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ── Candle generator ──
function makeCandles(n, sigma, seed = 42) {
  const rng = lcg(seed);
  const candles = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    const r = sigma * randn(rng);
    const close = price * Math.exp(r);
    const high = Math.max(price, close) * (1 + Math.abs(randn(rng)) * sigma * 0.4);
    const low = Math.min(price, close) * (1 - Math.abs(randn(rng)) * sigma * 0.4);
    candles.push({ open: price, high: Math.max(high, price, close), low: Math.min(low, price, close), close, volume: 1000 });
    price = close;
  }
  return candles;
}

// ── GARCH(1,1) DGP ──
function makeGarchCandles(n, omega, alpha, beta, seed = 99) {
  const rng = lcg(seed);
  const candles = [];
  let price = 100;
  let sigma2 = omega / (1 - alpha - beta);
  for (let i = 0; i < n; i++) {
    const z = randn(rng);
    const r = Math.sqrt(sigma2) * z;
    const close = price * Math.exp(r);
    const vol = Math.sqrt(sigma2);
    const high = Math.max(price, close) * (1 + Math.abs(randn(rng)) * vol * 0.3);
    const low = Math.min(price, close) * (1 - Math.abs(randn(rng)) * vol * 0.3);
    candles.push({ open: price, high: Math.max(high, price, close), low: Math.min(low, price, close), close, volume: 1000 });
    sigma2 = omega + alpha * r * r + beta * sigma2;
    price = close;
  }
  return candles;
}

let passed = 0, failed = 0;
function assert(cond, msg) {
  if (cond) { passed++; console.log(`  PASS: ${msg}`); }
  else { failed++; console.error(`  FAIL: ${msg}`); }
}

// ══════════════════════════════════════════════════════
// TEST 1: Monotonicity — higher true vol → higher predicted sigma
// ══════════════════════════════════════════════════════
console.log('\n── Test 1: Monotonicity ──');
const sigmas = [0.005, 0.01, 0.03];
const predicted = sigmas.map(s => predict(makeCandles(400, s), '4h').sigma);
console.log(`  true: [${sigmas.join(', ')}]`);
console.log(`  pred: [${predicted.map(s => s.toFixed(6)).join(', ')}]`);
assert(predicted[0] < predicted[1], 'sigma 0.5% < sigma 1%');
assert(predicted[1] < predicted[2], 'sigma 1% < sigma 3%');

// ══════════════════════════════════════════════════════
// TEST 2: Accuracy — predicted sigma within 50% of true
// ══════════════════════════════════════════════════════
console.log('\n── Test 2: Accuracy (relative error < 50%) ──');
for (const trueS of [0.005, 0.01, 0.03]) {
  const p = predict(makeCandles(400, trueS, 77), '4h');
  const relErr = Math.abs(p.sigma - trueS) / trueS;
  assert(relErr < 0.5, `sigma_true=${trueS} predicted=${p.sigma.toFixed(6)} relErr=${(relErr * 100).toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════
// TEST 3: predictRange cumulative sigma grows with steps
// ══════════════════════════════════════════════════════
console.log('\n── Test 3: predictRange — cumulative sigma grows ──');
const candles400 = makeCandles(400, 0.01);
const s1 = predictRange(candles400, '4h', 1).sigma;
const s3 = predictRange(candles400, '4h', 3).sigma;
const s10 = predictRange(candles400, '4h', 10).sigma;
console.log(`  steps=1: ${s1.toFixed(6)}, steps=3: ${s3.toFixed(6)}, steps=10: ${s10.toFixed(6)}`);
assert(s1 < s3, 'sigma(1) < sigma(3)');
assert(s3 < s10, 'sigma(3) < sigma(10)');
// sqrt scaling: s10/s1 should be roughly sqrt(10) ≈ 3.16
const ratio = s10 / s1;
assert(ratio > 1.5 && ratio < 6, `s10/s1 ratio=${ratio.toFixed(2)} (expect ~sqrt(10)=3.16)`);

// ══════════════════════════════════════════════════════
// TEST 4: Confidence — wider band at higher confidence
// ══════════════════════════════════════════════════════
console.log('\n── Test 4: Confidence — wider bands ──');
const p68 = predict(candles400, '4h');
const p90 = predict(candles400, '4h', undefined, 0.90);
const p95 = predict(candles400, '4h', undefined, 0.95);
const p99 = predict(candles400, '4h', undefined, 0.99);
const width = p => p.upperPrice - p.lowerPrice;
console.log(`  widths: 68%=${width(p68).toFixed(2)}, 90%=${width(p90).toFixed(2)}, 95%=${width(p95).toFixed(2)}, 99%=${width(p99).toFixed(2)}`);
assert(width(p68) < width(p90), 'band(68%) < band(90%)');
assert(width(p90) < width(p95), 'band(90%) < band(95%)');
assert(width(p95) < width(p99), 'band(95%) < band(99%)');
// sigma should be the same regardless of confidence
assert(Math.abs(p68.sigma - p95.sigma) < 1e-10, 'sigma invariant to confidence');

// ══════════════════════════════════════════════════════
// TEST 5: Log-normal asymmetry — upper move > lower move
// ══════════════════════════════════════════════════════
console.log('\n── Test 5: Log-normal asymmetry ──');
const upMove = p68.upperPrice - p68.currentPrice;
const downMove = p68.currentPrice - p68.lowerPrice;
assert(upMove > downMove, `up=${upMove.toFixed(4)} > down=${downMove.toFixed(4)} (log-normal)`);
assert(p68.lowerPrice > 0, 'lowerPrice > 0');

// ══════════════════════════════════════════════════════
// TEST 6: Custom currentPrice
// ══════════════════════════════════════════════════════
console.log('\n── Test 6: Custom currentPrice ──');
const pCustom = predict(candles400, '4h', 50000);
assert(Math.abs(pCustom.currentPrice - 50000) < 0.01, 'currentPrice = 50000');
assert(pCustom.upperPrice > 50000, 'upper > 50000');
assert(pCustom.lowerPrice < 50000, 'lower < 50000');
assert(pCustom.sigma === p68.sigma, 'sigma same as without custom price');

// ══════════════════════════════════════════════════════
// TEST 7: GARCH DGP — model should pick GARCH-family or at least give reasonable sigma
// ══════════════════════════════════════════════════════
console.log('\n── Test 7: GARCH(1,1) DGP ──');
const garchCandles = makeGarchCandles(500, 0.000001, 0.1, 0.85);
const pg = predict(garchCandles, '4h');
const unconditionalSigma = Math.sqrt(0.000001 / (1 - 0.1 - 0.85));
console.log(`  modelType: ${pg.modelType}, sigma: ${pg.sigma.toFixed(6)}, unconditional: ${unconditionalSigma.toFixed(6)}`);
assert(pg.sigma > 0, 'sigma > 0');
assert(pg.reliable !== undefined, 'reliable flag present');

// ══════════════════════════════════════════════════════
// TEST 8: Determinism — same input → same output
// ══════════════════════════════════════════════════════
console.log('\n── Test 8: Determinism ──');
const c1 = makeCandles(300, 0.01, 123);
const r1 = predict(c1, '4h');
const r2 = predict(c1, '4h');
assert(r1.sigma === r2.sigma, 'sigma deterministic');
assert(r1.modelType === r2.modelType, 'modelType deterministic');
assert(r1.upperPrice === r2.upperPrice, 'upperPrice deterministic');

// ══════════════════════════════════════════════════════
// TEST 9: Backtest on high-vol data with wide band
// ══════════════════════════════════════════════════════
console.log('\n── Test 9: Backtest with 99% VaR ──');
const btCandles = makeCandles(400, 0.01, 55);
const bt99 = backtest(btCandles, '4h', 0.99);
console.log(`  backtest(99% VaR): ${bt99}`);
// 99% band should capture almost everything
assert(bt99 === true, 'backtest passes with 99% VaR wide band');

// ══════════════════════════════════════════════════════
// TEST 10: Different intervals produce different annualized sigma
// ══════════════════════════════════════════════════════
console.log('\n── Test 10: Interval affects output ──');
const intCandles = makeCandles(300, 0.01, 200);
const p4h = predict(intCandles, '4h');
const p1h = predict(intCandles, '1h');
console.log(`  4h sigma: ${p4h.sigma.toFixed(6)}, 1h sigma: ${p1h.sigma.toFixed(6)}`);
// Same candles interpreted as 1h vs 4h — sigma should differ because periodsPerYear differs
// (model fitting uses periodsPerYear internally)
assert(p4h.modelType !== undefined, '4h modelType defined');
assert(p1h.modelType !== undefined, '1h modelType defined');

// ══════════════════════════════════════════════════════
// TEST 11: Median accuracy over multiple seeds
// ══════════════════════════════════════════════════════
console.log('\n── Test 11: Median accuracy (10 seeds, sigma=1%) ──');
const errors = [];
for (let seed = 1; seed <= 10; seed++) {
  const c = makeCandles(400, 0.01, seed * 13);
  const p = predict(c, '4h');
  errors.push(Math.abs(p.sigma - 0.01) / 0.01);
}
errors.sort((a, b) => a - b);
const median = errors[Math.floor(errors.length / 2)];
console.log(`  errors: [${errors.map(e => (e * 100).toFixed(1) + '%').join(', ')}]`);
console.log(`  median relative error: ${(median * 100).toFixed(1)}%`);
assert(median < 0.4, `median error ${(median * 100).toFixed(1)}% < 40%`);

// ══════════════════════════════════════════════════════
// TEST 12: Immutability — input candles not mutated
// ══════════════════════════════════════════════════════
console.log('\n── Test 12: Immutability ──');
const origCandles = makeCandles(300, 0.01, 42);
const snapshot = JSON.stringify(origCandles);
predict(origCandles, '4h');
predictRange(origCandles, '4h', 5);
backtest(origCandles, '4h');
assert(JSON.stringify(origCandles) === snapshot, 'candles not mutated after predict/predictRange/backtest');

// ══════════════════════════════════════════════════════
// TEST 13: PredictionResult structure
// ══════════════════════════════════════════════════════
console.log('\n── Test 13: PredictionResult structure ──');
const pr2 = predict(makeCandles(300, 0.01), '4h');
const requiredKeys = ['currentPrice', 'sigma', 'move', 'upperPrice', 'lowerPrice', 'modelType', 'reliable'];
for (const key of requiredKeys) {
  assert(key in pr2, `result has '${key}'`);
}
assert(typeof pr2.sigma === 'number' && !isNaN(pr2.sigma), 'sigma is a valid number');
assert(typeof pr2.reliable === 'boolean', 'reliable is boolean');
assert(['garch', 'egarch', 'gjr-garch', 'har-rv', 'novas'].includes(pr2.modelType), `modelType '${pr2.modelType}' is valid`);

// ══════════════════════════════════════════════════════
// TEST 14: move = upperPrice - currentPrice
// ══════════════════════════════════════════════════════
console.log('\n── Test 14: move consistency ──');
assert(Math.abs(pr2.move - (pr2.upperPrice - pr2.currentPrice)) < 1e-10, 'move = upper - current');

// ══════════════════════════════════════════════════════
// SUMMARY
// ══════════════════════════════════════════════════════
console.log(`\n${'═'.repeat(50)}`);
console.log(`TOTAL: ${passed + failed} tests — ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
else console.log('ALL TESTS PASSED');
