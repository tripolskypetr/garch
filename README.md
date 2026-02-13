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

## API

### `predict(candles, interval, currentPrice?)`

Forecast expected price range for the next candle (t+1). Auto-selects GARCH or EGARCH based on leverage effect. Returns a +-1 sigma price corridor.

```typescript
import { predict } from 'garch';
import type { Candle } from 'garch';

const candles: Candle[] = await fetchCandles('BTCUSDT', '4h', 200);

const result = predict(candles, '4h');
// {
//   currentPrice: 97500,
//   sigma: 0.012,          // 1.2% expected move
//   move: 1170,            // +/-$1170 price range
//   upperPrice: 98670,     // ceiling for next candle
//   lowerPrice: 96330,     // floor for next candle
//   modelType: 'egarch',
//   reliable: true
// }

// Pass VWAP or any reference price as 3rd argument
const result = predict(candles, '4h', vwap);
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candles` | `Candle[]` | required | OHLCV candle data |
| `interval` | `CandleInterval` | required | Candle timeframe |
| `currentPrice` | `number` | last close | Reference price to center the corridor |

**Returns:** `PredictionResult`

```typescript
interface PredictionResult {
  currentPrice: number;           // Reference price
  sigma: number;                  // One-period volatility (decimal, e.g. 0.012 = 1.2%)
  move: number;                   // +/- price move = currentPrice * sigma
  upperPrice: number;             // currentPrice + move
  lowerPrice: number;             // currentPrice - move
  modelType: 'garch' | 'egarch'; // Auto-selected model
  reliable: boolean;              // Quality flag (convergence + persistence + Ljung-Box)
}
```

---

### `predictRange(candles, interval, steps, currentPrice?)`

Forecast cumulative expected price range over multiple candles. Cumulative sigma = sqrt(sigma_1^2 + sigma_2^2 + ... + sigma_n^2). Use for swing trades where you hold across multiple periods.

```typescript
import { predictRange } from 'garch';

const range = predictRange(candles, '4h', 5);
// {
//   currentPrice: 97500,
//   sigma: 0.027,           // cumulative ~2.7% over 5 candles
//   move: 2632,             // +/-$2632 total range
//   upperPrice: 100132,
//   lowerPrice: 94868,
//   modelType: 'egarch',
//   reliable: true
// }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candles` | `Candle[]` | required | OHLCV candle data |
| `interval` | `CandleInterval` | required | Candle timeframe |
| `steps` | `number` | required | Number of candles to forecast over |
| `currentPrice` | `number` | last close | Reference price |

**Returns:** `PredictionResult` (same structure as `predict`)

---

### `backtest(candles, interval, requiredPercent?)`

Walk-forward validation of `predict`. Uses 75% of candles for fitting, 25% for testing. Checks if the model's +-1 sigma corridor captures actual price moves at the required hit rate.

```typescript
import { backtest } from 'garch';

backtest(candles, '4h');     // true  -- hit rate >= 68% (default)
backtest(candles, '4h', 50); // true  -- hit rate >= 50% (custom)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candles` | `Candle[]` | required | OHLCV candle data |
| `interval` | `CandleInterval` | required | Candle timeframe |
| `requiredPercent` | `number` | `68` | Minimum hit rate (+-1 sigma ~ 68% theoretically) |

**Returns:** `boolean`

---

## Supported Intervals

| Interval | Min Candles | Periods/Year | Coverage |
|----------|-------------|--------------|----------|
| `1m` | 500 | 525,600 | ~8-16 hours |
| `3m` | 500 | 175,200 | ~25 hours |
| `5m` | 500 | 105,120 | ~1.7 days |
| `15m` | 300 | 35,040 | ~3 days |
| `30m` | 200 | 17,520 | ~4 days |
| `1h` | 200 | 8,760 | ~8 days |
| `2h` | 200 | 4,380 | ~17 days |
| `4h` | 200 | 2,190 | ~33 days |
| `6h` | 150 | 1,460 | ~37 days |
| `8h` | 150 | 1,095 | ~50 days |

## Timeframes

The `periodsPerYear` value controls annualization of volatility. When using `predict`/`predictRange`/`backtest`, this is handled automatically via the `interval` parameter. When using `Garch`/`Egarch` classes directly, pass `periodsPerYear` manually.

| Timeframe | `periodsPerYear` | Notes |
|-----------|-----------------|-------|
| **1m** | `525,600` | 1440/day x 365 |
| **3m** | `175,200` | 480/day x 365 |
| **5m** | `105,120` | 288/day x 365 |
| **15m** | `35,040` | 96/day x 365 |
| **30m** | `17,520` | 48/day x 365 |
| **1h** | `8,760` | 24/day x 365 |
| **2h** | `4,380` | 12/day x 365 |
| **4h** | `2,190` | 6/day x 365 |
| **6h** | `1,460` | 4/day x 365 |
| **8h** | `1,095` | 3/day x 365 |

Lower timeframes contain more microstructure noise — use larger datasets to compensate.

## Math

### GARCH(1,1)

Conditional variance model (Bollerslev, 1986):

```
sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
```

- **omega** > 0 — long-run variance anchor
- **alpha** >= 0 — shock reaction (how much yesterday's surprise matters)
- **beta** >= 0 — persistence (memory of past variance)
- Stationarity constraint: **alpha + beta < 1**
- Unconditional variance: **E[sigma^2] = omega / (1 - alpha - beta)**

Parameter estimation via **Gaussian MLE** (maximum likelihood):

```
LL = -0.5 * sum[ ln(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]
```

Multi-step forecast converges to unconditional variance:

```
sigma_{t+h}^2 = omega + (alpha + beta) * sigma_{t+h-1}^2
```

### EGARCH(1,1)

Exponential GARCH (Nelson, 1991). Models log-variance, capturing asymmetric volatility:

```
ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - sqrt(2/pi)) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
```

Where **z_t = epsilon_t / sigma_t** is the standardized residual, **sqrt(2/pi) ~ 0.7979**.

- **gamma** < 0 — leverage effect (negative returns increase vol more than positive)
- No positivity constraints needed (log-variance is always real)
- Stationarity: **|beta| < 1**
- Unconditional variance: **E[sigma^2] ~ exp(omega / (1 - beta))**

### Model Auto-Selection

`predict` and `predictRange` automatically choose between GARCH and EGARCH:

1. Compute volatility of negative returns vs. positive returns
2. If ratio > 1.2 — use EGARCH (significant leverage effect detected)
3. Otherwise — use simpler GARCH

### Variance Estimators

**Yang-Zhang** (used as initial variance for model fitting):

```
sigma^2_YZ = sigma^2_overnight + k * sigma^2_close + (1-k) * sigma^2_RS
```

Combines overnight gaps, open-to-close moves, and Rogers-Satchell intraday range. More robust than close-to-close for OHLC data.

**Garman-Klass** (fallback):

```
sigma^2_GK = (1/n) * sum[ 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2 ]
```

~5x more efficient than close-to-close variance.

### Reliability Check

The `reliable` flag in `PredictionResult` is `true` when all three conditions hold:

1. Optimizer converged
2. Persistence < 0.999 (not near unit root)
3. Ljung-Box test on squared standardized residuals: p-value >= 0.05 (no residual autocorrelation)

### Optimization

Parameters are estimated via **Nelder-Mead** simplex method (derivative-free). Default: 1000 iterations, tolerance 1e-8. Model comparison uses **AIC** (2k - 2LL) and **BIC** (k*ln(n) - 2LL).

## Tests

**400 tests** across **17 test files**. All passing.

| Category | Files | Tests | What's covered |
|----------|-------|-------|----------------|
| Mathematical formulas | `math.test.ts` | 45 | GARCH/EGARCH variance recursion, log-likelihood, forecast formulas, AIC/BIC, Yang-Zhang, Garman-Klass, Ljung-Box, chi-squared |
| Full pipeline coverage | `plan-coverage.test.ts` | 73 | End-to-end: fit, forecast, predict, predictRange, backtest, model selection |
| GARCH unit | `garch.test.ts` | 10 | Parameter estimation, variance series, forecast convergence, candle vs price input |
| EGARCH unit | `egarch.test.ts` | 11 | Leverage detection, asymmetric volatility, model comparison via AIC |
| Optimizer | `optimizer.test.ts`, `optimizer-shrink.test.ts` | 16 | Nelder-Mead on Rosenbrock/quadratic/parabolic, convergence, shrinking |
| Statistical properties | `properties.test.ts` | 13 | Parameter recovery from synthetic data, local LL maximum, unconditional variance |
| Regression | `regression.test.ts` | 9 | Parameter recovery, deterministic outputs |
| Stability | `stability.test.ts` | 10 | Long-term forecast behavior, variance convergence |
| Robustness | `robustness.test.ts` | 53 | Extreme moves, stress scenarios |
| Edge cases | `edge-cases.test.ts`, `coverage-gaps*.test.ts` | 148 | Insufficient data, near-unit-root, zero returns, constant prices, negative prices, overflow/underflow, trending data, 10K+ data points |
| Miscellaneous | `misc.test.ts` | 12 | Integration scenarios, different intervals |

```bash
npm test              # run all tests
npm run test:coverage # run with coverage report
```

## License

MIT
