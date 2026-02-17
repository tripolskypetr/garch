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

### `predict(candles, interval, currentPrice?, confidence?)`

Forecast expected price range for the next candle (t+1). Auto-selects the best model (GARCH, EGARCH, GJR-GARCH, HAR-RV or NoVaS) by QLIKE forecast-error comparison.

Uses **log-normal price bands**: `P·exp(±z·σ)`, where `z = probit(confidence)`. This correctly maps log-return volatility back to price space — the corridor is asymmetric (upside > downside in absolute terms) and `lowerPrice` can never go negative.

```typescript
import { predict } from 'garch';
import type { Candle } from 'garch';

const candles: Candle[] = await fetchCandles('BTCUSDT', '4h', 200);

// Default: ±1σ band (~68% coverage)
const result = predict(candles, '4h');
// {
//   currentPrice: 97500,
//   sigma: 0.012,          // 1.2% per-period volatility
//   move: 1177,            // upward expected move (upper - current)
//   upperPrice: 98677,     // P·exp(+σ) — ceiling
//   lowerPrice: 96337,     // P·exp(-σ) — floor
//   modelType: 'egarch',
//   reliable: true
// }

// 95% VaR band (z ≈ 1.96)
const var95 = predict(candles, '4h', undefined, 0.95);

// Custom reference price (e.g. VWAP)
const result = predict(candles, '4h', vwap);
```

**Confidence → z mapping:**

| `confidence` | z-score | Meaning |
|-------------|---------|---------|
| 0.6827 (default) | 1.00 | ±1σ, ~68% of moves captured |
| 0.90 | 1.645 | Moderate VaR |
| 0.95 | 1.96 | 95% VaR (standard) |
| 0.99 | 2.576 | Conservative VaR |

Any value in (0, 1) is valid — the table above lists common choices, but `probit` computes z for arbitrary confidence.

Higher confidence = wider corridor. `sigma` stays the same (it's the model's volatility estimate), only the z-multiplier changes. Example with sigma=1.2% and P=$97,500:

| `confidence` | z | upperPrice | lowerPrice | Corridor width |
|-------------|---|-----------|-----------|----------------|
| 0.6827 | 1.00 | $98,677 | $96,337 | $2,340 |
| 0.95 | 1.96 | $99,808 | $95,222 | $4,586 |
| 0.99 | 2.58 | $100,545 | $94,520 | $6,025 |

**When to use which:**

- **±1σ (default)** — typical expected move for the next candle. Good for scalping SL/TP targets and assessing whether a move is "normal" or significant
- **95% VaR** — worst reasonable scenario. Good for risk management, position sizing, and stop-losses that shouldn't be triggered by noise
- **99% VaR** — extreme tail risk. Good for stress testing and margin calculations

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candles` | `Candle[]` | required | OHLCV candle data |
| `interval` | `CandleInterval` | required | Candle timeframe |
| `currentPrice` | `number` | last close | Reference price to center the corridor |
| `confidence` | `number` | `0.6827` | Two-sided probability in (0,1). Controls band width via `z = probit(confidence)` |

**Returns:** `PredictionResult`

```typescript
interface PredictionResult {
  currentPrice: number;           // Reference price
  sigma: number;                  // One-period volatility (decimal, e.g. 0.012 = 1.2%)
  move: number;                   // Upward price move = upperPrice - currentPrice
  upperPrice: number;             // P · exp(+z·σ)
  lowerPrice: number;             // P · exp(-z·σ)
  modelType: 'garch' | 'egarch' | 'gjr-garch' | 'har-rv' | 'novas'; // Auto-selected model
  reliable: boolean;              // Quality flag (convergence + persistence + Ljung-Box)
}
```

---

### `predictRange(candles, interval, steps, currentPrice?, confidence?)`

Forecast cumulative expected price range over multiple candles. Cumulative sigma = sqrt(sigma_1^2 + sigma_2^2 + ... + sigma_n^2). Uses the same log-normal bands as `predict`. Use for swing trades where you hold across multiple periods.

```typescript
import { predictRange } from 'garch';

const range = predictRange(candles, '4h', 5);
// {
//   currentPrice: 97500,
//   sigma: 0.027,           // cumulative ~2.7% over 5 candles
//   move: 2669,             // upward expected move
//   upperPrice: 100169,     // P·exp(+z·σ)
//   lowerPrice: 94901,      // P·exp(-z·σ)
//   modelType: 'egarch',
//   reliable: true
// }

// 95% VaR over 5 candles
const var95 = predictRange(candles, '4h', 5, undefined, 0.95);
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candles` | `Candle[]` | required | OHLCV candle data |
| `interval` | `CandleInterval` | required | Candle timeframe |
| `steps` | `number` | required | Number of candles to forecast over |
| `currentPrice` | `number` | last close | Reference price |
| `confidence` | `number` | `0.6827` | Two-sided probability in (0,1) |

**Returns:** `PredictionResult` (same structure as `predict`)

---

### `backtest(candles, interval, confidence?, requiredPercent?)`

Walk-forward validation of `predict`. Uses 75% of candles for fitting, 25% for testing. Checks if the model's price corridor captures actual price moves at the required hit rate.

`confidence` and `requiredPercent` are independent: `confidence` controls the **band width** (via `probit`), `requiredPercent` controls the **pass/fail threshold**.

```typescript
import { backtest } from 'garch';

backtest(candles, '4h');                   // ±1σ band, hit rate >= 68%
backtest(candles, '4h', 0.95);            // 95% VaR band, hit rate >= 68%
backtest(candles, '4h', 0.95, 90);        // 95% VaR band, hit rate >= 90%
backtest(candles, '4h', undefined, 50);   // ±1σ band, hit rate >= 50%
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candles` | `Candle[]` | required | OHLCV candle data |
| `interval` | `CandleInterval` | required | Candle timeframe |
| `confidence` | `number` | `0.6827` | Two-sided probability in (0,1) for the prediction band |
| `requiredPercent` | `number` | `68` | Minimum hit rate (0–100) to pass |

**Returns:** `boolean`

---

## Supported Intervals

| Interval | Min Candles | Recommended | Periods/Year | Coverage |
|----------|-------------|-------------|--------------|----------|
| `1m` | 500 | 1,500 | 525,600 | ~8-16 hours |
| `3m` | 500 | 1,500 | 175,200 | ~25 hours |
| `5m` | 500 | 1,500 | 105,120 | ~1.7 days |
| `15m` | 300 | 1,000 | 35,040 | ~3 days |
| `30m` | 200 | 1,000 | 17,520 | ~4 days |
| `1h` | 200 | 500 | 8,760 | ~8 days |
| `2h` | 200 | 500 | 4,380 | ~17 days |
| `4h` | 200 | 500 | 2,190 | ~33 days |
| `6h` | 150 | 300 | 1,460 | ~37 days |
| `8h` | 150 | 300 | 1,095 | ~50 days |

For intervals below 1h, per-candle Parkinson RV is noisier — more data helps OLS and QLIKE model selection. A `console.warn` is emitted when candle count is below the recommended value. Always check `reliable: true` in the output.

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

Parameter estimation via **Student-t MLE** (maximum likelihood) with multi-start Nelder-Mead optimization. The Student-t distribution captures fat tails observed in financial returns:

```
LL = n·[lnΓ((df+1)/2) - lnΓ(df/2) - 0.5·ln(π·(df-2))]
     - 0.5 · sum[ ln(sigma_t^2) + (df+1)·ln(1 + r_t^2 / ((df-2)·sigma_t^2)) ]
```

- **df** > 2 — degrees of freedom (estimated jointly with omega, alpha, beta via multi-start Nelder-Mead)

Multi-step forecast converges to unconditional variance (E[RV] = sigma^2, so recursion is identical):

```
sigma_{t+h}^2 = omega + (alpha + beta) * sigma_{t+h-1}^2
```

### EGARCH(1,1)

Exponential GARCH (Nelson, 1991). Models log-variance, capturing asymmetric volatility. Input type determines the magnitude term automatically:

**Candle[] input** — Realized EGARCH. Magnitude uses Parkinson RV, leverage keeps directional return:

```
ln(sigma_t^2) = omega + alpha * (sqrt(RV_{t-1} / sigma_{t-1}^2) - E[|Z|]) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
```

**number[] input** — Classical EGARCH. Magnitude uses |z|:

```
ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - E[|Z|]) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
```

Where **z_t = epsilon_t / sigma_t** is the standardized residual and **E[|Z|]** is the expected absolute value of a standardized Student-t(df) variable:

```
E[|Z|] = sqrt((df-2)/pi) · Γ((df-1)/2) / Γ(df/2)
```

- **sqrt(RV/sigma^2)** is a more efficient estimate of |z| from OHLC data
- **gamma** < 0 — leverage effect (negative returns increase vol more than positive)
- No positivity constraints needed (log-variance is always real)
- Stationarity: **|beta| < 1**
- Unconditional variance: **E[sigma^2] ~ exp(omega / (1 - beta))**
- **df** > 2 — degrees of freedom (estimated jointly via multi-start Nelder-Mead)

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
- **df** > 2 — degrees of freedom (estimated jointly via multi-start Nelder-Mead)

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

After OLS, **df** (degrees of freedom) is profiled via grid search over the Student-t log-likelihood with the OLS-fitted variance series.

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

Key difference from GARCH: parameters are found via **normality criterion** (D^2 minimization), not MLE. No distributional assumptions on the return series — truly model-free. Uses multi-start Nelder-Mead for optimization (6 restarts to escape local minima in the 11-dimensional D^2 landscape). After fitting, **df** is profiled via grid search over the Student-t log-likelihood (same as HAR-RV).

After D^2 optimization, weights are **rescaled via OLS** on the realized variance series to minimize forecast error. This keeps NoVaS model-free (D^2 selects lag structure) while making it QLIKE-competitive with HAR-RV.

**Two-stage calibration:**

- **Stage 1** — D^2 minimization: discovers lag structure via normality criterion (model-free). Produces `weights` (a_0, ..., a_p).
- **Stage 2** — OLS rescaling: regresses RV_{t+1} on sigma_t^2(D^2) to produce forecast-optimal weights. Produces `forecastWeights` = [beta_0, beta_1].

```
forecast_sigma_t^2 = beta_0 + beta_1 * sigma_t^2(D^2)
```

D^2 acts as a data-driven smoother over RV lags — more flexible than HAR-RV's fixed rolling means (1, 5, 22). OLS rescaling adjusts for bias with only 2 parameters, keeping the model robust on small samples with noisy per-candle RV.

Multi-step forecast: replace future innovations with sigma^2 (since E[RV] = E[X^2] = sigma^2), then rescale:

```
d2_{t+h} = a_0 + sum_j a_j * E[innovation_{t+h-j}]
sigma_{t+h}^2 = beta_0 + beta_1 * d2_{t+h}
```

### Model Auto-Selection

`predict` and `predictRange` fit all five models and pick the winner by **QLIKE** (Patton, 2011) — a neutral forecast-error metric that judges how well each model's variance series predicts realized variance:

```
QLIKE = (1/n) · sum[ RV_t / sigma_t^2 - ln(RV_t / sigma_t^2) - 1 ]
```

Lower QLIKE = better forecast. Unlike AIC (which favors MLE-calibrated models), QLIKE is neutral to calibration method — OLS, D^2-minimization, and MLE all compete on equal footing.

1. **GARCH-family pipeline**: fit GARCH, EGARCH, GJR-GARCH — pick best by AIC (fair since all three optimize Student-t LL)
2. **HAR-RV pipeline**: fit HAR-RV via OLS. Skip if persistence >= 1 or R^2 < 0
3. **NoVaS pipeline**: fit NoVaS via D^2 minimization + OLS rescaling. Skip if persistence >= 1
4. Compute Parkinson RV from candles. Score each pipeline's variance series by QLIKE. Lowest QLIKE wins

```
fitModel(candles)
  |-- fitGarchFamily() --> min_AIC(GARCH, EGARCH, GJR-GARCH) --> QLIKE_1
  |-- fitHarRv()       --> HAR-RV (OLS)                       --> QLIKE_2
  |-- fitNoVaS()       --> NoVaS (D²)                         --> QLIKE_3
  |-- rv = perCandleParkinson(candles)
  \-- return min(QLIKE_1, QLIKE_2, QLIKE_3)
```

When given `Candle[]`, all five OHLC-aware models (GARCH, EGARCH, GJR-GARCH, HAR-RV, NoVaS) use Parkinson per-candle RV instead of squared returns, extracting ~5× more information from the same data.

### When Each Model Wins

The library fits all five models on every call and picks the best by QLIKE. Each model uses its own calibration method (MLE, OLS, or D^2) but they all compete on the same forecast-error metric. Here's what patterns in data favor each:

- **GARCH** — Volatility spikes after any big move (up or down equally), then gradually fades back to normal. No difference between bullish and bearish shocks. Classic symmetric mean-reverting vol clustering.

- **EGARCH** — Drops hit harder than pumps. A -5% candle causes way more volatility than a +5% candle. Strong "fear > greed" asymmetry. Works through a log-variance model so the asymmetry coefficient `gamma * z` directly amplifies negative shocks. Typical for stocks and BTC during risk-off periods.

- **GJR-GARCH** — Same idea as EGARCH (red candles increase vol more than green ones) but the effect is milder and simpler: when the return is negative, a bonus term `gamma * epsilon^2` is added to variance. When positive — nothing extra. A binary switch rather than a continuous asymmetry. Common in altcoins and less panic-prone markets.

- **HAR-RV** — Volatility has memory at multiple horizons. The model takes a single timeframe of candles and internally builds three scales: last candle's RV (short, lag 1), rolling average over 5 candles (medium), and rolling average over 22 candles (long). These three components are combined via OLS regression. Works well when different types of participants (scalpers, swing traders, institutions) all influence the same market at different speeds. If your asset has visible "rhythm" across day/week/month — HAR-RV will likely beat GARCH family (sideways, daily patterns).

- **NoVaS** — Volatility drifts or cycles without clear shock-and-decay patterns. Slow trend changes, regime shifts, compression/expansion phases, or "breathing" patterns that don't fit any parametric formula. Model-free: finds weights `a_0...a_p` that make the normalized series as close to Gaussian as possible (minimizes D^2 = skewness^2 + (kurtosis - 3)^2). No assumptions about the distribution shape.

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

### Log-Normal Price Bands

GARCH models volatility of **log-returns**, not absolute price changes. The correct mapping from log-return volatility back to price space uses the exponential:

```
upperPrice = P · exp(+z · sigma)
lowerPrice = P · exp(-z · sigma)
```

where `z = probit(confidence)` is the z-score corresponding to the desired two-sided confidence level. This produces **asymmetric** bands (upside > downside in absolute terms) and guarantees `lowerPrice > 0`.

The previous linear approximation `P · (1 ± sigma)` is a first-order Taylor expansion of `exp(±sigma)`. The difference grows with sigma:

| sigma | Linear `1 + sigma` | Exact `exp(sigma)` | Error |
|-------|--------------------|--------------------|-------|
| 0.02 | 1.0200 | 1.0202 | ~0.01% |
| 0.10 | 1.1000 | 1.1052 | ~0.5% |
| 0.30 | 1.3000 | 1.3499 | ~3.8% |

### Probit (Inverse Normal CDF)

`probit(confidence)` computes the inverse of the standard normal CDF (Phi^{-1}). It converts a two-sided probability to a z-score:

```
confidence = P(-z < Z < z),  Z ~ N(0,1)
z = probit(confidence)
```

There is no closed-form solution for Phi^{-1} — it is a transcendental equation. The implementation uses **Acklam's rational approximation** (Peter J. Acklam, 2002) with max relative error < 1.15 x 10^{-9}.

**Step 1** — convert two-sided confidence to upper-tail probability:

```
p = (1 + confidence) / 2
```

**Step 2** — piecewise rational approximation over three regions:

**Central region** (0.02425 <= p <= 0.97575) — covers ~95% of inputs:

```
q = p - 0.5
r = q^2
z = (a1·r^5 + a2·r^4 + a3·r^3 + a4·r^2 + a5·r + a6) · q
    / (b1·r^5 + b2·r^4 + b3·r^3 + b4·r^2 + b5·r + 1)
```

**Tails** (p < 0.02425 or p > 0.97575):

```
q = sqrt(-2·ln(p))          // left tail
z = (c1·q^5 + ... + c6) / (d1·q^4 + ... + 1)
```

For the right tail: `q = sqrt(-2·ln(1-p))`, result is negated. The 16 coefficients (a1–a6, b1–b5, c1–c6, d1–d4) are minimax-optimal rational approximation constants.

| `confidence` | `p` | `z` | Meaning |
|-------------|-----|-----|---------|
| 0.6827 | 0.8413 | 1.000 | ±1 sigma (default) |
| 0.90 | 0.9500 | 1.645 | Moderate VaR |
| 0.95 | 0.9750 | 1.960 | 95% VaR |
| 0.99 | 0.9950 | 2.576 | Conservative VaR |

### Student-t Log-Likelihood

All five models use a **Student-t** distribution for log-likelihood computation. Financial return series exhibit fat tails (excess kurtosis), and the Student-t captures this with an additional **degrees of freedom (df)** parameter:

- **df ~ 3–10**: Heavy tails (common in crypto/equity returns)
- **df ~ 20–50**: Mild excess kurtosis
- **df → ∞**: Thin tails

For GARCH, EGARCH, and GJR-GARCH, **df** is optimized jointly with the other parameters via Nelder-Mead. For HAR-RV and NoVaS, **df** is profiled via a two-pass grid search (coarse 2.5–50, then fine ±1 around the best) after the main optimization.

Helper functions exported from the library:

- `logGamma(x)` — Lanczos approximation (g=7, n=9), ~15-digit accuracy
- `studentTNegLL(returns, varianceSeries, df)` — Full negative log-likelihood
- `expectedAbsStudentT(df)` — E[|Z|] for standardized t(df), used in EGARCH centering
- `profileStudentTDf(returns, varianceSeries)` — Grid search for optimal df
- `qlike(varianceSeries, rv)` — QLIKE loss (Patton, 2011) for volatility forecast evaluation

### Reliability Check

The `reliable` flag in `PredictionResult` is `true` when all three conditions hold:

1. Optimizer converged
2. Persistence < 0.999 (not near unit root)
3. Ljung-Box test on squared standardized residuals: p-value >= 0.05 (no residual autocorrelation)

### Optimization

GARCH/EGARCH/GJR-GARCH parameters (including df) are estimated via **multi-start Nelder-Mead** simplex method (derivative-free). Each model runs Nelder-Mead from multiple deterministic starting points (golden-ratio quasi-random perturbation) and keeps the best result — this escapes local minima that single-start NM would get stuck in, especially important for higher-dimensional problems like NoVaS (11 parameters).

| Model | Parameters | Restarts | Total NM runs |
|-------|-----------|----------|---------------|
| GARCH | 4 (omega, alpha, beta, df) | 3 | 4 |
| EGARCH | 5 (omega, alpha, gamma, beta, df) | 4 | 5 |
| GJR-GARCH | 5 (omega, alpha, gamma, beta, df) | 4 | 5 |
| NoVaS | p+1 (a_0...a_p, default p=10) | 6 | 7 |
| HAR-RV | 4 (beta_0...beta_3) | — | OLS (closed-form) |

Default per run: 1000 iterations (2000 for NoVaS D^2), tolerance 1e-8. Initial simplex uses 20% perturbation from x0 for broader exploration. HAR-RV uses **OLS** (exact solution in one step) + df profiling.

Within the GARCH family, model comparison uses **AIC** (2k - 2LL) — fair since all three optimize the same Student-t LL objective. Across model families (GARCH vs HAR-RV vs NoVaS), comparison uses **QLIKE** (Patton, 2011):

```
QLIKE = (1/n) · sum[ RV_t / sigma_t^2 - ln(RV_t / sigma_t^2) - 1 ]
```

where RV_t is Parkinson per-candle realized variance and sigma_t^2 is the model's fitted conditional variance. QLIKE is the standard loss function for volatility forecast evaluation — it is neutral to calibration method (MLE, OLS, D^2 all compete fairly).

## Tests

**932 tests** across **22 test files**. All passing.

| Category | Files | Tests | What's covered |
|----------|-------|-------|----------------|
| Mathematical formulas | `math.test.ts` | 45 | GARCH/EGARCH variance recursion, log-likelihood, forecast formulas, AIC/BIC, QLIKE, Yang-Zhang, Garman-Klass, Ljung-Box, chi-squared |
| Math coverage | `math-coverage.test.ts` | 79 | Parkinson formula verification, rv↔returns alignment, H=L fallback, Parkinson-based forecast, candle validation, reliable flag cascade, backtest validity, numerical precision, cross-model consistency, Realized GARCH/EGARCH/GJR-GARCH Candle[] vs number[], perCandleParkinson shared function |
| Full pipeline coverage | `plan-coverage.test.ts` | 82 | End-to-end: fit, forecast, predict, predictRange, backtest, model selection |
| GARCH unit | `garch.test.ts` | 10 | Parameter estimation, variance series, forecast convergence, candle vs price input |
| EGARCH unit | `egarch.test.ts` | 11 | Leverage detection, asymmetric volatility, model comparison |
| GJR-GARCH unit | `gjr-garch.test.ts` | 86 | Variance recursion (r² and Parkinson), indicator function I(r<0), forecast formula (one-step + multi-step), constraint barriers, computed fields, AIC/BIC numParams=5, estimation properties (perturbation, determinism), numerical stability, degenerate params, Realized path (Candle[] vs number[], flat candles, bad OHLC), options forwarding, immutability, instance isolation, cross-model consistency, scale invariance, property-based fuzz, predict/predictRange/backtest integration |
| HAR-RV unit | `har.test.ts` | 138 | OLS regression, R^2, Parkinson RV proxy, forecast convergence, multi-step iterative substitution, rolling RV components, edge cases, fuzz, integration with predict, OLS orthogonality, TSS=RSS+ESS, normal equations, regression snapshots, mutation safety |
| NoVaS unit | `novas.test.ts` | 109 | D^2 minimization, normality improvement, variance series, forecast convergence, edge cases, fuzz, integration with predict, determinism, scale invariance |
| Optimizer | `optimizer.test.ts`, `optimizer-shrink.test.ts` | 20 | Nelder-Mead on Rosenbrock/quadratic/parabolic, convergence, shrinking, multi-start (Rastrigin escape, high-dimensional, equivalence) |
| Statistical properties | `properties.test.ts` | 15 | Parameter recovery from synthetic data, local LL maximum, unconditional variance, GJR-GARCH forecast convergence and model selection |
| Regression | `regression.test.ts` | 11 | Parameter recovery, deterministic outputs, cross-model consistency for GARCH/EGARCH/GJR-GARCH |
| Stability | `stability.test.ts` | 12 | Long-term forecast behavior, variance convergence, GJR-GARCH near-constant and outlier handling |
| Robustness | `robustness.test.ts` | 53 | Extreme moves, stress scenarios |
| Realized models | `realized-garch.test.ts` | 83 | Candle[] vs number[] for GARCH/EGARCH/GJR-GARCH/NoVaS, Parkinson RV edge cases, flat candles, extreme H/L, scale invariance, all-identical OHLC, minimum-length boundary, D² comparison, predict fallback, **ground-truth volatility recovery** (see below) |
| Edge cases | `edge-cases.test.ts`, `coverage-gaps*.test.ts` | 165 | Insufficient data, near-unit-root, zero returns, constant prices, negative prices, overflow/underflow, trending data, 10K+ data points, GJR-GARCH immutability and instance isolation, EGARCH df≤2 fallback, logGamma/expectedAbsStudentT/qlike edge cases |
| Miscellaneous | `misc.test.ts` | 13 | Integration scenarios, different intervals, immutability |

### Ground-Truth Volatility Test

The test suite includes an end-to-end integration test that verifies `predict` recovers known volatility from synthetic data. This is the strongest correctness guarantee — it proves the entire pipeline (data generation, model fitting, auto-selection, forecasting) produces numerically accurate results.

**How it works:**

1. Generate 500 OHLC candles with **known constant per-period volatility** sigma_true (returns are iid N(0, sigma_true^2), high/low simulated via Brownian bridge noise)
2. Run `predict()` on these candles — auto-selects the best model, fits parameters, produces a 1-step forecast
3. Verify predicted sigma matches sigma_true

**What is verified:**

| Test | Assertion |
|------|-----------|
| sigma_true = 0.2%, 1%, 3% | Relative error < 50% for each |
| Monotonicity | sigma_true_1 < sigma_true_2 < sigma_true_3 implies predicted_1 < predicted_2 < predicted_3 |
| Median accuracy (20 seeds) | Median relative error < 30% across 20 independent runs (sigma_true = 1%) |
| +-1 sigma hit rate (Monte Carlo) | Out-of-sample +-1 sigma corridor captures 45–90% of actual next moves across 30 trials (theoretical: 68.27%) |
| Proportionality | 2x sigma_true produces ~2x predicted sigma (ratio between 1.2x and 3.5x) |

```bash
npm test        # run all tests
```

## License

MIT
