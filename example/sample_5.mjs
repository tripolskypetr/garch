import { predict, predictRange } from 'garch';

function lcg(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff; };
}
function randn(rng) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

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

let passed = 0, failed = 0;
function assert(cond, msg) {
  if (cond) { passed++; console.log(`  PASS: ${msg}`); }
  else { failed++; console.error(`  FAIL: ${msg}`); }
}

// ══════════════════════════════════════════════════════════════
//  PART A: reliable flag
// ══════════════════════════════════════════════════════════════
console.log('╔══════════════════════════════════════════════════════╗');
console.log('║  PART A: reliable flag                              ║');
console.log('╚══════════════════════════════════════════════════════╝');

// A1: Collect reliable=true and reliable=false across many seeds on good data
console.log('\n── A1: reliable can be both true and false across seeds (good data) ──');
{
  let trueCount = 0, falseCount = 0;
  for (let seed = 1; seed <= 30; seed++) {
    const r = predict(makeCandles(400, 0.01, seed * 11), '4h');
    if (r.reliable) trueCount++; else falseCount++;
  }
  console.log(`  reliable=true: ${trueCount}/30, reliable=false: ${falseCount}/30`);
  assert(trueCount > 0, `reliable=true appeared at least once (${trueCount}/30)`);
  assert(falseCount > 0, `reliable=false appeared at least once (${falseCount}/30)`);
  // on good data majority should still be reliable
  assert(trueCount >= 10, `at least 1/3 reliable on good data: ${trueCount}/30`);
}

// A2: Large dataset → almost always reliable
console.log('\n── A2: Large dataset (800 candles) → mostly reliable ──');
{
  let trueCount = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const r = predict(makeCandles(800, 0.01, seed * 7), '4h');
    if (r.reliable) trueCount++;
  }
  console.log(`  reliable=true: ${trueCount}/10`);
  assert(trueCount >= 7, `at least 70% reliable on 800 candles: ${trueCount}/10`);
}

// A3: Degenerate data → reliable=false
console.log('\n── A3: Degenerate data → reliable=false ──');
{
  // constant prices
  const constCandles = [];
  for (let i = 0; i < 300; i++) {
    constCandles.push({ open: 100, high: 100, low: 100, close: 100, volume: 1000 });
  }
  try {
    const r = predict(constCandles, '4h');
    assert(r.reliable === false, `constant prices: reliable=${r.reliable}`);
  } catch (e) {
    assert(true, `threw on constant prices (acceptable): ${e.message}`);
  }

  // IGARCH-like: random walk in variance → near unit root
  const rng = lcg(44);
  const igarchCandles = [];
  let price = 100, sigma2 = 0.0001;
  for (let i = 0; i < 400; i++) {
    sigma2 = 0.0000001 + 0.15 * (sigma2 * randn(rng) ** 2) + 0.85 * sigma2;
    sigma2 = Math.max(sigma2, 1e-10);
    const r = Math.sqrt(sigma2) * randn(rng);
    const close = price * Math.exp(r);
    const high = Math.max(price, close) * (1 + Math.abs(randn(rng)) * 0.005);
    const low = Math.min(price, close) * (1 - Math.abs(randn(rng)) * 0.005);
    igarchCandles.push({ open: price, high: Math.max(high, price, close), low: Math.min(low, price, close), close, volume: 1000 });
    price = close;
  }
  const rI = predict(igarchCandles, '4h');
  console.log(`  IGARCH-like: reliable=${rI.reliable}, model=${rI.modelType}`);
  assert(rI.reliable === false, `IGARCH-like: reliable=${rI.reliable} (expect false)`);
}

// A4: reliable=true on clean GARCH(1,1) DGP with moderate persistence
console.log('\n── A4: Clean GARCH DGP → reliable=true ──');
{
  const rng = lcg(55);
  const omega = 0.000005, alpha = 0.08, beta = 0.85;
  let price = 100, s2 = omega / (1 - alpha - beta);
  const candles = [];
  for (let i = 0; i < 500; i++) {
    const z = randn(rng);
    const r = Math.sqrt(s2) * z;
    const close = price * Math.exp(r);
    const vol = Math.sqrt(s2);
    const high = Math.max(price, close) * (1 + Math.abs(randn(rng)) * vol * 0.3);
    const low = Math.min(price, close) * (1 - Math.abs(randn(rng)) * vol * 0.3);
    candles.push({ open: price, high: Math.max(high, price, close), low: Math.min(low, price, close), close, volume: 1000 });
    s2 = omega + alpha * r * r + beta * s2;
    price = close;
  }
  const r = predict(candles, '4h');
  console.log(`  reliable=${r.reliable}, model=${r.modelType}`);
  assert(r.reliable === true, `clean GARCH DGP: reliable=${r.reliable}`);
}

// A5: predictRange inherits reliable from predict
console.log('\n── A5: predictRange.reliable matches predict.reliable ──');
{
  // test on two datasets: one likely reliable, one likely not
  for (const seed of [7, 1]) {
    const candles = makeCandles(400, 0.01, seed);
    const p = predict(candles, '4h');
    const pr = predictRange(candles, '4h', 5);
    console.log(`  seed=${seed}: predict.reliable=${p.reliable}, predictRange.reliable=${pr.reliable}`);
    assert(pr.reliable === p.reliable, `seed=${seed}: predictRange matches predict`);
  }
}

// ══════════════════════════════════════════════════════════════
//  PART B: Forecast convergence
// ══════════════════════════════════════════════════════════════
console.log('\n╔══════════════════════════════════════════════════════╗');
console.log('║  PART B: Forecast convergence                       ║');
console.log('╚══════════════════════════════════════════════════════╝');

// B1: Cumulative variance grows — each step adds positive marginal variance
console.log('\n── B1: Cumulative variance monotonically grows ──');
{
  const candles = makeCandles(400, 0.01, 50);
  const steps = [1, 2, 3, 4, 5, 10, 20, 50];
  const cumSigmas = steps.map(s => predictRange(candles, '4h', s).sigma);

  console.log('  step | cumSigma');
  for (let i = 0; i < steps.length; i++) {
    console.log(`  ${String(steps[i]).padStart(4)} | ${cumSigmas[i].toFixed(6)}`);
  }

  // each step should have higher cumulative sigma
  let mono = true;
  for (let i = 1; i < cumSigmas.length; i++) {
    if (cumSigmas[i] <= cumSigmas[i - 1]) mono = false;
  }
  assert(mono, 'cumulative sigma strictly increasing with steps');

  // growth rate decelerates: ratio(step 50 / step 20) < ratio(step 20 / step 1)
  const r20_1 = cumSigmas[steps.indexOf(20)] / cumSigmas[0];
  const r50_20 = cumSigmas[steps.indexOf(50)] / cumSigmas[steps.indexOf(20)];
  assert(r50_20 < r20_1, `deceleration: ratio(50/20)=${r50_20.toFixed(2)} < ratio(20/1)=${r20_1.toFixed(2)}`);
}

// B2: Cumulative sigma grows sublinearly (sqrt-scaling)
console.log('\n── B2: Cumulative sigma grows sublinearly ──');
{
  const candles = makeCandles(400, 0.01, 60);
  const s1 = predictRange(candles, '4h', 1).sigma;
  const s50 = predictRange(candles, '4h', 50).sigma;
  const ratio = s50 / s1;
  // if sigma were iid: ratio = sqrt(50) ≈ 7.07
  // with mean reversion: ratio < sqrt(50)
  // without: ratio ≈ sqrt(50)
  console.log(`  s1=${s1.toFixed(6)}, s50=${s50.toFixed(6)}, ratio=${ratio.toFixed(2)} (sqrt(50)=${Math.sqrt(50).toFixed(2)})`);
  assert(ratio > 2, `ratio ${ratio.toFixed(2)} > 2 (grows with horizon)`);
  assert(ratio < 12, `ratio ${ratio.toFixed(2)} < 12 (sublinear, not explosive)`);
}

// B3: Very long horizon — sigma should plateau (not explode)
console.log('\n── B3: Long horizon convergence (100 steps) ──');
{
  const candles = makeCandles(400, 0.01, 70);
  const s50 = predictRange(candles, '4h', 50).sigma;
  const s100 = predictRange(candles, '4h', 100).sigma;
  const growth = s100 / s50;
  console.log(`  s50=${s50.toFixed(6)}, s100=${s100.toFixed(6)}, growth=${growth.toFixed(3)}`);
  // growth from 50→100 should be less than growth from 1→50
  // (decelerating accumulation)
  assert(growth < 2.0, `growth 50→100 = ${growth.toFixed(3)} < 2.0`);
  assert(Number.isFinite(s100), 's100 is finite');
}

// B4: Convergence rate — diff between step N and N+1 should shrink
console.log('\n── B4: Convergence rate — sigma increments shrink ──');
{
  const candles = makeCandles(400, 0.01, 80);
  const sigs = [];
  for (let s = 1; s <= 30; s++) {
    sigs.push(predictRange(candles, '4h', s).sigma);
  }
  // compute increments
  const increments = [];
  for (let i = 1; i < sigs.length; i++) {
    increments.push(sigs[i] - sigs[i - 1]);
  }
  // first 5 increments vs last 5 — last should be smaller on average
  const avgFirst5 = increments.slice(0, 5).reduce((a, b) => a + b) / 5;
  const avgLast5 = increments.slice(-5).reduce((a, b) => a + b) / 5;
  console.log(`  avg increment (steps 2-6): ${avgFirst5.toFixed(6)}`);
  console.log(`  avg increment (steps 26-30): ${avgLast5.toFixed(6)}`);
  assert(avgLast5 < avgFirst5, 'later increments smaller than early (convergence)');
}

// B5: predictRange(1) ≈ predict()
console.log('\n── B5: predictRange(steps=1) ≈ predict() ──');
{
  const candles = makeCandles(400, 0.01, 90);
  const p = predict(candles, '4h');
  const pr = predictRange(candles, '4h', 1);
  const sigDiff = Math.abs(p.sigma - pr.sigma);
  console.log(`  predict.sigma=${p.sigma.toFixed(8)}, predictRange(1).sigma=${pr.sigma.toFixed(8)}, diff=${sigDiff.toExponential(2)}`);
  assert(sigDiff < 1e-10, `sigma diff ${sigDiff.toExponential(2)} < 1e-10`);
  assert(p.modelType === pr.modelType, `same model: ${p.modelType} = ${pr.modelType}`);
  assert(Math.abs(p.upperPrice - pr.upperPrice) < 1e-8, 'upperPrice matches');
  assert(Math.abs(p.lowerPrice - pr.lowerPrice) < 1e-8, 'lowerPrice matches');
}

// B6: Higher vol data — same convergence pattern
console.log('\n── B6: Convergence at higher vol (3%) ──');
{
  const candles = makeCandles(400, 0.03, 100);
  const s1 = predictRange(candles, '4h', 1).sigma;
  const s10 = predictRange(candles, '4h', 10).sigma;
  const s50 = predictRange(candles, '4h', 50).sigma;
  console.log(`  s1=${s1.toFixed(6)}, s10=${s10.toFixed(6)}, s50=${s50.toFixed(6)}`);
  assert(s1 < s10, 's1 < s10');
  assert(s10 < s50, 's10 < s50');
  const r10 = s10 / s1;
  const r50 = s50 / s10;
  assert(r50 < r10, `deceleration: ratio(10/1)=${r10.toFixed(2)} > ratio(50/10)=${r50.toFixed(2)}`);
}

// B7: GARCH DGP — forecast converges toward unconditional
console.log('\n── B7: GARCH DGP — convergence toward unconditional ──');
{
  const rng = lcg(55);
  const omega = 0.000005, alpha = 0.10, beta = 0.85;
  const unconditional = Math.sqrt(omega / (1 - alpha - beta));
  let price = 100, sigma2 = omega / (1 - alpha - beta);
  const candles = [];
  for (let i = 0; i < 500; i++) {
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

  // per-step sigma at long horizon should approach unconditional
  const sLong = predictRange(candles, '4h', 50).sigma / Math.sqrt(50);
  console.log(`  unconditional σ: ${unconditional.toFixed(6)}`);
  console.log(`  per-step σ at h=50: ${sLong.toFixed(6)}`);
  const relErr = Math.abs(sLong - unconditional) / unconditional;
  console.log(`  relative error: ${(relErr * 100).toFixed(1)}%`);
  // should be in the right ballpark (within 80%)
  assert(relErr < 0.8, `per-step sigma at h=50 within 80% of unconditional`);
}

// ══════════════════════════════════════════════════════════════
//  SUMMARY
// ══════════════════════════════════════════════════════════════
console.log(`\n${'═'.repeat(55)}`);
console.log(`TOTAL: ${passed + failed} tests — ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
else console.log('ALL TESTS PASSED');
