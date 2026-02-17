import { predict, predictRange, backtest } from '../build/index.mjs';

   // Generate synthetic candles with known volatility
   function lcg(seed) { let s = seed; return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff; }; }
   function randn(rng) { const u1 = rng(), u2 = rng(); return Math.sqrt(-2*Math.log(u1||1e-10))*Math.cos(2*Math.PI*u2); }

   const rng = lcg(42);
   const sigma = 0.01; // 1% per-period vol
   const candles = [];
   let price = 100;

   for (let i = 0; i < 300; i++) {
     const r = sigma * randn(rng);
     const close = price * Math.exp(r);
     const mid = (price + close) / 2;
     const high = Math.max(price, close) * (1 + Math.abs(randn(rng)) * sigma * 0.3);
     const low = Math.min(price, close) * (1 - Math.abs(randn(rng)) * sigma * 0.3);
     candles.push({ open: price, high, low, close, volume: 1000 });
     price = close;
   }

   console.log('=== predict ===');
   const p = predict(candles, '4h');
   console.log('sigma:', p.sigma.toFixed(6));
   console.log('modelType:', p.modelType);
   console.log('reliable:', p.reliable);
   console.log('upperPrice:', p.upperPrice.toFixed(2));
   console.log('lowerPrice:', p.lowerPrice.toFixed(2));
   console.log('currentPrice:', p.currentPrice.toFixed(2));

   console.log();
   console.log('=== predictRange (5 steps) ===');
   const pr = predictRange(candles, '4h', 5);
   console.log('sigma:', pr.sigma.toFixed(6));
   console.log('modelType:', pr.modelType);
   console.log('reliable:', pr.reliable);
   console.log('upperPrice:', pr.upperPrice.toFixed(2));
   console.log('lowerPrice:', pr.lowerPrice.toFixed(2));

   console.log();
   console.log('=== backtest ===');
   const bt = backtest(candles, '4h');
   console.log('backtest passed:', bt);

   console.log();
   console.log('=== predict with 95% confidence ===');
   const p95 = predict(candles, '4h', undefined, 0.95);
   console.log('sigma:', p95.sigma.toFixed(6));
   console.log('upperPrice:', p95.upperPrice.toFixed(2));
   console.log('lowerPrice:', p95.lowerPrice.toFixed(2));

   console.log();
   console.log('All functions work OK');