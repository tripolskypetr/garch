<p align="center">
  <img src="./assets/logo.png" height="115px" alt="garch" />
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
