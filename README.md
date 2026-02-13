<p align="center">
  <img src="https://github.com/tripolskypetr/garch/raw/master/assets/logo.png" height="115px" alt="garch" />
</p>

<p align="center">
  <strong>Missing GARCH/EGARCH forecast for NodeJS</strong><br>
  GARCH and EGARCH volatility models for TypeScript. Zero dependencies.
</p>

## Installation

```bash
npm install garch
```

## Usage

### GARCH(1,1)

```typescript
import { calibrateGarch, Garch } from 'garch';

// From price array
const prices = [100, 101, 99, 102, 98, ...];
const result = calibrateGarch(prices, { periodsPerYear: 252 });

console.log(result.params);
// {
//   omega: 0.000012,
//   alpha: 0.08,
//   beta: 0.89,
//   persistence: 0.97,
//   unconditionalVariance: 0.0004,
//   annualizedVol: 31.7
// }

// Or use the class for more control
const model = new Garch(prices, { periodsPerYear: 252 });
const fit = model.fit();

// Get variance series
const variance = model.getVarianceSeries(fit.params);

// Forecast 10 periods ahead
const forecast = model.forecast(fit.params, 10);
console.log(forecast.annualized); // [32.1, 31.9, 31.8, ...]
```

### EGARCH(1,1)

EGARCH captures asymmetric volatility (leverage effect):

```typescript
import { calibrateEgarch, Egarch, checkLeverageEffect } from 'garch';

// Check if EGARCH is warranted
const returns = calculateReturnsFromPrices(prices);
const leverage = checkLeverageEffect(returns);
console.log(leverage);
// { negativeVol: 0.021, positiveVol: 0.015, ratio: 1.4, recommendation: 'egarch' }

// Fit EGARCH
const result = calibrateEgarch(prices, { periodsPerYear: 365 }); // crypto = 365

console.log(result.params);
// {
//   omega: -0.12,
//   alpha: 0.15,
//   gamma: -0.08,  // negative = leverage effect
//   beta: 0.95,
//   persistence: 0.95,
//   annualizedVol: 45.2,
//   leverageEffect: -0.08
// }
```

### From OHLCV Candles

```typescript
import { Candle, calibrateGarch } from 'garch';

const candles: Candle[] = [
  { open: 100, high: 102, low: 99, close: 101, volume: 1000 },
  { open: 101, high: 103, low: 100, close: 99, volume: 1200 },
  // ...
];

const result = calibrateGarch(candles);
```

### Model Selection

```typescript
import { calibrateGarch, calibrateEgarch } from 'garch';

const garch = calibrateGarch(prices);
const egarch = calibrateEgarch(prices);

// Compare using AIC (lower is better)
if (egarch.diagnostics.aic < garch.diagnostics.aic) {
  console.log('EGARCH fits better');
}
```

## API

### `calibrateGarch(data, options?)`

Calibrate GARCH(1,1) model.

**Parameters:**
- `data`: `Candle[]` or `number[]` (prices)
- `options.periodsPerYear`: Annualization factor (default: 252)
- `options.maxIter`: Maximum optimizer iterations (default: 1000)
- `options.tol`: Convergence tolerance (default: 1e-8)

**Returns:** `CalibrationResult<GarchParams>`

### `calibrateEgarch(data, options?)`

Calibrate EGARCH(1,1) model.

**Parameters:** Same as `calibrateGarch`

**Returns:** `CalibrationResult<EgarchParams>`

### `checkLeverageEffect(returns)`

Check for asymmetric volatility.

**Returns:** `{ negativeVol, positiveVol, ratio, recommendation }`

### Classes

`Garch` and `Egarch` classes provide:
- `.fit(options?)` - Calibrate parameters
- `.getVarianceSeries(params)` - Compute conditional variance
- `.forecast(params, steps)` - Multi-step variance forecast
- `.getReturns()` - Get computed returns

## Timeframes

The library works with any candle timeframe. The only thing that changes is the `periodsPerYear` option, which controls annualization of volatility.

| Timeframe | `periodsPerYear` | Notes |
|-----------|-----------------|-------|
| **1d** | `252` (default) | Trading days per year |
| **4h** | `1512` | 252 × 6 |
| **1h** | `6048` (crypto) / `1638` (stocks) | Crypto trades 24/7, stocks ~6.5h/day |
| **15m** | `24192` (crypto) / `6552` (stocks) | 96 or 26 bars per day × 252 |
| **1m** | `362880` (crypto) / `393120` (stocks) | 1440 or 390 bars per day × 252 |

```typescript
// Daily candles (default)
calibrateGarch(prices);

// 4-hour candles
calibrateGarch(prices, { periodsPerYear: 1512 });

// 15-minute candles (crypto, 24/7 market)
calibrateGarch(prices, { periodsPerYear: 24192 });
```

**Minimum data:** 50 candles are required for stable parameter estimation.

**Recommended timeframes:** 1d and 4h are the most reliable for GARCH models. Lower timeframes (15m, 1m) contain more microstructure noise which can degrade calibration quality — use larger datasets to compensate.

## Predict

The `predict` function forecasts the expected price range for the next candle (t+1). It auto-selects GARCH or EGARCH, fits the model, and returns a ±1σ price corridor. You decide SL/TP yourself based on the forecast.

```typescript
import { predict } from 'garch';
import type { Candle } from 'garch';

const candles: Candle[] = await fetchCandles('BTCUSDT', '4h', 200);

const result = predict(candles, '4h');
// {
//   currentPrice: 97500,
//   sigma: 0.012,          // 1.2% expected move
//   move: 1170,            // ±$1170 price range
//   upperPrice: 98670,     // ceiling for next candle
//   lowerPrice: 96330,     // floor for next candle
//   modelType: 'egarch',
//   reliable: true          // false if model didn't converge or is inadequate
// }

// EMA says "long", sigma says price can move ~1.2%
// You set TP = +1%, SL = -0.5% manually
```

The third argument `currentPrice` defaults to the last candle close. You can pass a VWAP or any other reference price to center the corridor around it:

```typescript
// VWAP from last 5 candles as the reference price
const recent = candles.slice(-5);
const vwap = recent.reduce((sum, c) => sum + c.close * c.volume, 0)
            / recent.reduce((sum, c) => sum + c.volume, 0);

const result = predict(candles, '4h', vwap);
// corridor is now centered around VWAP, not last close
```

### Supported intervals

| Interval | Periods/year | Recommended candles | Min | Coverage |
|----------|-------------|---------------------|-----|----------|
| 1m | 525,600 | 500–1000 | 50 | ~8–16 hours |
| 3m | 175,200 | 500 | 50 | ~25 hours |
| 5m | 105,120 | 500 | 50 | ~1.7 days |
| 15m | 35,040 | 300 | 50 | ~3 days |
| 30m | 17,520 | 200 | 50 | ~4 days |
| 1h | 8,760 | 200 | 50 | ~8 days |
| 2h | 4,380 | 200 | 50 | ~17 days |
| 4h | 2,190 | 200 | 50 | ~33 days |
| 6h | 1,460 | 150 | 50 | ~37 days |
| 8h | 1,095 | 150 | 50 | ~50 days |

Lower timeframes contain more microstructure noise — use larger datasets to compensate. Too few candles and the model won't capture volatility clustering; too many and you fit stale regimes that no longer apply.

### predictRange

`predictRange` forecasts the cumulative expected move over N candles. The cumulative σ = √(σ₁² + σ₂² + ... + σₙ²) — total expected range, not per-candle. Use for swing trades where you hold a position across multiple periods.

```typescript
import { predictRange } from 'garch';

const candles = await fetchCandles('BTCUSDT', '4h', 200);

// Expected range over next 5 candles (20 hours)
const range = predictRange(candles, '4h', 5);
// {
//   currentPrice: 97500,
//   sigma: 0.027,           // cumulative ~2.7% over 5 candles
//   move: 2632,             // ±$2632 total range
//   upperPrice: 100132,
//   lowerPrice: 94868,
//   modelType: 'egarch',
//   reliable: true
// }

// Also accepts VWAP as 4th argument
const range = predictRange(candles, '4h', 5, vwap);
```

### backtest

Walk-forward validation of `predict`. Slides a window across historical candles, calls predict at each step, checks if the next candle's close landed within ±1σ corridor. Hit rate should be ~68% if the model is well-calibrated.

```typescript
import { backtest } from 'garch';

const candles = await fetchCandles('BTCUSDT', '4h', 500);

const result = backtest(candles, '4h', 200); // window = 200 candles per fit
// {
//   total: 299,       // number of predictions
//   hits: 210,        // times actual was within corridor
//   hitRate: 0.702,   // ~70% — model is well-calibrated
//   predictions: [{ predicted: PredictionResult, actual: number }, ...]
// }
```

### predictMultiTimeframe

Compare volatility forecasts across two timeframes. Normalizes both σ to per-hour and detects divergence (one timeframe sees 2x+ more vol than the other). Separate entry point — does not modify `predict` or `predictRange`.

```typescript
import { predictMultiTimeframe } from 'garch';

const candles4h = await fetchCandles('BTCUSDT', '4h', 200);
const candles15m = await fetchCandles('BTCUSDT', '15m', 300);

const result = predictMultiTimeframe(candles4h, '4h', candles15m, '15m');
// {
//   primary: PredictionResult,    // 4h forecast
//   secondary: PredictionResult,  // 15m forecast
//   divergence: true              // timeframes disagree on vol
// }

// divergence = true → one tf sees calm, the other sees storm
// useful as a filter: skip entry when timeframes diverge
```

## Model Details

### GARCH(1,1)

```
σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
```

- `ω` (omega) > 0: constant term
- `α` (alpha) ≥ 0: reaction to shocks
- `β` (beta) ≥ 0: persistence
- Stationarity: α + β < 1

### EGARCH(1,1)

```
ln(σ²ₜ) = ω + α·(|zₜ₋₁| - E[|z|]) + γ·zₜ₋₁ + β·ln(σ²ₜ₋₁)
```

- `γ` (gamma) < 0: leverage effect (negative returns increase vol more)
- No positivity constraints needed (models log-variance)
- `|β|` < 1 for stationarity

### Model Selection

```typescript
import { calibrateGarch, calibrateEgarch } from 'garch';

const garch = calibrateGarch(prices);
const egarch = calibrateEgarch(prices);

// Compare using AIC (lower is better)
if (egarch.diagnostics.aic < garch.diagnostics.aic) {
  console.log('EGARCH fits better — leverage effect is significant');
}
```

## License

MIT
