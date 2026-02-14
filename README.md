<p align="center">
  <img src="https://github.com/tripolskypetr/garch/raw/master/assets/logo.png" height="115px" alt="garch" />
</p>

<p align="center">
  <strong>Missing volatility forecast for NodeJS</strong><br>
  Realized GARCH, Realized EGARCH, Realized GJR-GARCH, HAR-RV and NoVaS<br>
  models for TypeScript. Zero dependencies.
</p>

## Installation

```bash
npm install garch
```

## API

### `predict(candles, interval, currentPrice?)`

Forecast expected price range for the next candle (t+1). Auto-selects GARCH, EGARCH, GJR-GARCH, HAR-RV or NoVaS based on leverage effect and AIC comparison. Returns a +-1 sigma price corridor.

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
  modelType: 'garch' | 'egarch' | 'gjr-garch' | 'har-rv' | 'novas'; // Auto-selected model
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

Conditional variance model (Bollerslev, 1986). Input type determines the innovation term automatically:

**Candle[] input** — Realized GARCH (Hansen & Huang, 2016). Uses Parkinson (1980) per-candle realized variance proxy (~5× more efficient than squared returns):

```
sigma_t^2 = omega + alpha * RV_{t-1} + beta * sigma_{t-1}^2
RV_t = (1 / (4·ln2)) · ln(H/L)^2     (Parkinson estimator)
```

**number[] input** — Classical GARCH. Uses squared returns:

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

Multi-step forecast converges to unconditional variance (E[RV] = sigma^2, so recursion is identical):

```
sigma_{t+h}^2 = omega + (alpha + beta) * sigma_{t+h-1}^2
```

### EGARCH(1,1)

Exponential GARCH (Nelson, 1991). Models log-variance, capturing asymmetric volatility. Input type determines the magnitude term automatically:

**Candle[] input** — Realized EGARCH. Magnitude uses Parkinson RV, leverage keeps directional return:

```
ln(sigma_t^2) = omega + alpha * (sqrt(RV_{t-1} / sigma_{t-1}^2) - sqrt(2/pi)) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
```

**number[] input** — Classical EGARCH. Magnitude uses |z|:

```
ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - sqrt(2/pi)) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
```

Where **z_t = epsilon_t / sigma_t** is the standardized residual, **sqrt(2/pi) ~ 0.7979**.

- **sqrt(RV/sigma^2)** is a more efficient estimate of |z| from OHLC data
- **gamma** < 0 — leverage effect (negative returns increase vol more than positive)
- No positivity constraints needed (log-variance is always real)
- Stationarity: **|beta| < 1**
- Unconditional variance: **E[sigma^2] ~ exp(omega / (1 - beta))**

### GJR-GARCH(1,1)

Threshold GARCH (Glosten, Jagannathan & Runkle, 1993). Captures leverage effect in variance space (unlike EGARCH which uses log-variance). Input type determines the innovation term automatically:

**Candle[] input** — Realized GJR-GARCH. Uses Parkinson RV as innovation, return sign as leverage indicator:

```
sigma_t^2 = omega + alpha * RV_{t-1} + gamma * RV_{t-1} * I(r_{t-1}<0) + beta * sigma_{t-1}^2
```

**number[] input** — Classical GJR-GARCH. Uses squared returns:

```
sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + gamma * epsilon_{t-1}^2 * I(r_{t-1}<0) + beta * sigma_{t-1}^2
```

Where **I(r<0) = 1** when return is negative, **0** otherwise.

- **gamma** >= 0 — leverage coefficient (bad news amplifies variance by extra gamma)
- **omega** > 0, **alpha** >= 0, **beta** >= 0
- Stationarity: **alpha + gamma/2 + beta < 1** (half of innovations are negative on average)
- Unconditional variance: **E[sigma^2] = omega / (1 - alpha - gamma/2 - beta)**

Multi-step forecast uses effective persistence = alpha + gamma/2 + beta:

```
sigma_{t+h}^2 = omega + (alpha + gamma/2 + beta) * sigma_{t+h-1}^2
```

### HAR-RV

Heterogeneous Autoregressive model of Realized Variance (Corsi, 2009). Captures multi-scale volatility clustering via three overlapping horizons:

```
RV_{t+1} = beta_0 + beta_1 * RV_short + beta_2 * RV_medium + beta_3 * RV_long + epsilon
```

Where **RV_t** is the per-candle realized variance proxy:

- **OHLC input** — Parkinson (1980): **RV_t = (1 / (4·ln2)) · ln(H/L)^2** (~5× more efficient than close-to-close)
- **Prices-only input** — squared return: **RV_t = r_t^2** (fallback when no OHLC available)

Classic HAR-RV uses sum of intraday squared returns, but that requires tick/minute data. Parkinson uses the high-low range of each candle as a variance proxy — no intraday data needed.

- **RV_short** = mean(RV_t) — last 1 period (default)
- **RV_medium** = mean(RV_{t-4} ... RV_t) — last 5 periods
- **RV_long** = mean(RV_{t-21} ... RV_t) — last 22 periods

Parameter estimation via **OLS** (closed-form, always converges):

```
beta = (X'X)^{-1} X'y
```

- Persistence: **beta_1 + beta_2 + beta_3 < 1** for stationarity
- Unconditional variance: **E[RV] = beta_0 / (1 - beta_1 - beta_2 - beta_3)**
- **R^2** measures explained variance in the regression

Multi-step forecast via iterative substitution: each predicted RV feeds back into the rolling components for subsequent steps.

### NoVaS

Normalizing and Variance-Stabilizing transformation (Politis, 2003). Model-free approach using the ARCH frame. Input type determines the innovation term automatically:

**Candle[] input** — Realized NoVaS. Uses Parkinson (1980) per-candle RV as innovation (~5× more efficient than squared returns):

```
sigma_t^2 = a_0 + a_1 * RV_{t-1} + a_2 * RV_{t-2} + ... + a_p * RV_{t-p}
RV_t = (1 / (4·ln2)) · ln(H/L)^2     (Parkinson estimator)
W_t  = X_t / sigma_t
```

**number[] input** — Classical NoVaS. Uses squared returns:

```
sigma_t^2 = a_0 + a_1 * X_{t-1}^2 + a_2 * X_{t-2}^2 + ... + a_p * X_{t-p}^2
W_t       = X_t / sigma_t
```

Parameters **a_0, ..., a_p** are chosen to minimize the non-normality of the transformed series {W_t}:

```
D^2 = S^2 + (K - 3)^2
```

where **S** = skewness and **K** = kurtosis of {W_t}. For perfect normality D^2 = 0.

- **a_0** > 0 — baseline variance
- **a_j** >= 0 — weight on j-th lagged innovation (RV or squared return)
- Stationarity: **sum(a_1 ... a_p) < 1**
- Unconditional variance: **E[sigma^2] = a_0 / (1 - sum(a_1 ... a_p))**
- Default lags: **p = 10** (configurable)
- Parkinson RV is less noisy than r², so Candle[] typically achieves lower D^2

Key difference from GARCH: parameters are found via **normality criterion** (D^2 minimization), not MLE. No distributional assumptions on the return series — truly model-free. Uses Nelder-Mead for optimization.

Multi-step forecast: replace future innovations with sigma^2 (since E[RV] = E[X^2] = sigma^2):

```
sigma_{t+h}^2 = a_0 + sum_j a_j * E[innovation_{t+h-j}]
```

### Model Auto-Selection

`predict` and `predictRange` fit three pipelines in parallel and pick the winner by AIC:

1. **GARCH-family pipeline**: compute leverage ratio (negative vol / positive vol). If ratio > 1.2 — fit both EGARCH and GJR-GARCH, return lower AIC. Otherwise fit GARCH
2. **HAR-RV pipeline**: fit HAR-RV via OLS. Skip if persistence >= 1 or R^2 < 0
3. **NoVaS pipeline**: fit NoVaS via D^2 minimization. Skip if persistence >= 1
4. Compare AIC of all pipelines. Lowest AIC wins

```
fitModel()
  |-- fitGarchFamily() --> GARCH or min(EGARCH, GJR-GARCH) --> AIC_1
  |-- fitHarRv()       --> HAR-RV (OLS)                     --> AIC_2
  |-- fitNoVaS()       --> NoVaS (D²)                       --> AIC_3
  \-- return min(AIC_1, AIC_2, AIC_3)
```

When given `Candle[]`, all five OHLC-aware models (GARCH, EGARCH, GJR-GARCH, HAR-RV, NoVaS) use Parkinson per-candle RV instead of squared returns, extracting ~5× more information from the same data.

GARCH/EGARCH/GJR-GARCH tends to win on data with pronounced shock reactions or leverage effects. HAR-RV tends to win on data with strong multi-scale clustering (e.g. crypto, FX). NoVaS tends to win on short, volatile data where parametric assumptions break down.

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

GARCH/EGARCH/GJR-GARCH parameters are estimated via **Nelder-Mead** simplex method (derivative-free). Default: 1000 iterations, tolerance 1e-8. HAR-RV uses **OLS** (exact solution in one step). NoVaS uses **Nelder-Mead** with 2000 iterations to minimize D^2. Model comparison uses **AIC** (2k - 2LL) and **BIC** (k*ln(n) - 2LL).

For AIC comparison across all models, log-likelihoods are computed using the same Gaussian formulation over the return series:

```
LL = -0.5 * sum[ ln(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]
```

where sigma_t^2 comes from the GARCH conditional variance, HAR-RV fitted variance, or NoVaS variance.

## Tests

**778 tests** across **21 test files**. All passing.

| Category | Files | Tests | What's covered |
|----------|-------|-------|----------------|
| Mathematical formulas | `math.test.ts` | 45 | GARCH/EGARCH variance recursion, log-likelihood, forecast formulas, AIC/BIC, Yang-Zhang, Garman-Klass, Ljung-Box, chi-squared |
| Math coverage | `math-coverage.test.ts` | 79 | Parkinson formula verification, rv↔returns alignment, H=L fallback, Parkinson-based forecast, candle validation, reliable flag cascade, backtest validity, numerical precision, cross-model consistency, Realized GARCH/EGARCH/GJR-GARCH Candle[] vs number[], perCandleParkinson shared function |
| Full pipeline coverage | `plan-coverage.test.ts` | 73 | End-to-end: fit, forecast, predict, predictRange, backtest, model selection |
| GARCH unit | `garch.test.ts` | 10 | Parameter estimation, variance series, forecast convergence, candle vs price input |
| EGARCH unit | `egarch.test.ts` | 11 | Leverage detection, asymmetric volatility, model comparison via AIC |
| HAR-RV unit | `har.test.ts` | 138 | OLS regression, R^2, Parkinson RV proxy, forecast convergence, multi-step iterative substitution, rolling RV components, edge cases, fuzz, integration with predict, OLS orthogonality, TSS=RSS+ESS, normal equations, regression snapshots, mutation safety |
| NoVaS unit | `novas.test.ts` | 109 | D^2 minimization, normality improvement, variance series, forecast convergence, edge cases, fuzz, integration with predict, determinism, scale invariance |
| Optimizer | `optimizer.test.ts`, `optimizer-shrink.test.ts` | 16 | Nelder-Mead on Rosenbrock/quadratic/parabolic, convergence, shrinking |
| Statistical properties | `properties.test.ts` | 13 | Parameter recovery from synthetic data, local LL maximum, unconditional variance |
| Regression | `regression.test.ts` | 9 | Parameter recovery, deterministic outputs |
| Stability | `stability.test.ts` | 10 | Long-term forecast behavior, variance convergence |
| Robustness | `robustness.test.ts` | 53 | Extreme moves, stress scenarios |
| Realized models | `realized-garch.test.ts` | 52 | Candle[] vs number[] for GARCH/EGARCH/NoVaS, Parkinson RV edge cases, flat candles, extreme H/L, scale invariance, all-identical OHLC, minimum-length boundary, D² comparison, predict fallback when NoVaS fails |
| Edge cases | `edge-cases.test.ts`, `coverage-gaps*.test.ts` | 148 | Insufficient data, near-unit-root, zero returns, constant prices, negative prices, overflow/underflow, trending data, 10K+ data points |
| Miscellaneous | `misc.test.ts` | 12 | Integration scenarios, different intervals |

```bash
npm test        # run all tests
```

## License

MIT
