
## Trading Integration: Pine Direction + JS Vol Regime

The primary use case: **Pine Script** generates directional signals (long/short/flat), **JS** classifies the volatility regime and computes dynamic SL/TP. The two combine into a final trading signal.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TradingView                      │
│  Pine Script indicator/strategy                     │
│  ─ EMA crossover, RSI, MACD, structure, etc.       │
│  ─ Outputs: direction = 1 (long) / -1 (short) / 0  │
│  ─ Sends signal via webhook JSON                    │
└──────────────────────┬──────────────────────────────┘
                       │ webhook POST
                       ▼
┌─────────────────────────────────────────────────────┐
│                  Node.js Server                     │
│                                                     │
│  1. Receive Pine direction signal                   │
│  2. Fetch latest OHLCV candles (exchange API)       │
│  3. Fit GARCH/EGARCH → vol forecast                 │
│  4. Classify vol regime (low / normal / high)       │
│  5. Combine: Pine direction + vol regime → final    │
│  6. Compute SL/TP from forecast volatility          │
│  7. Execute order on exchange                       │
└─────────────────────────────────────────────────────┘
```

### Step 1: Fit Model and Forecast

```typescript
import {
  Garch, Egarch,
  checkLeverageEffect,
  calculateReturnsFromPrices,
  type Candle,
  type VolatilityForecast,
  type CalibrationResult,
  type GarchParams,
  type EgarchParams,
} from 'garch';

function fitAndForecast(
  candles: Candle[],
  periodsPerYear: number,
  steps: number
): { forecast: VolatilityForecast; params: GarchParams | EgarchParams } {
  // Auto-select model based on leverage effect
  const returns = calculateReturnsFromPrices(candles.map(c => c.close));
  const leverage = checkLeverageEffect(returns);

  if (leverage.recommendation === 'egarch') {
    const model = new Egarch(candles, { periodsPerYear });
    const fit = model.fit();
    return { forecast: model.forecast(fit.params, steps), params: fit.params };
  }

  const model = new Garch(candles, { periodsPerYear });
  const fit = model.fit();
  return { forecast: model.forecast(fit.params, steps), params: fit.params };
}
```

### Step 2: Classify Volatility Regime

```typescript
type VolRegime = 'low' | 'normal' | 'high' | 'extreme';

function classifyVolRegime(
  currentAnnualizedVol: number,
  unconditionalVol: number
): VolRegime {
  // unconditionalVol = long-run average annualized vol
  // currentAnnualizedVol = forecast annualized vol for next period
  const ratio = currentAnnualizedVol / unconditionalVol;

  if (ratio < 0.7) return 'low';       // vol well below average
  if (ratio < 1.3) return 'normal';    // vol near average
  if (ratio < 2.0) return 'high';      // vol elevated
  return 'extreme';                     // vol spike / crisis
}
```

### Step 3: Pine Direction + Vol Regime -> Final Signal

```typescript
type Direction = 1 | -1 | 0; // long | short | flat
type FinalSignal = {
  action: 'long' | 'short' | 'close' | 'skip';
  size: number;      // position size multiplier (0.0 - 1.0)
  reason: string;
};

function combineSignal(
  pineDirection: Direction,
  regime: VolRegime,
  persistence: number
): FinalSignal {
  // No Pine signal => nothing to do
  if (pineDirection === 0) {
    return { action: 'close', size: 0, reason: 'Pine signal flat' };
  }

  const action = pineDirection === 1 ? 'long' : 'short';

  switch (regime) {
    case 'extreme':
      // Extreme vol: skip new entries, too risky
      return { action: 'skip', size: 0, reason: `Vol regime extreme — skip ${action}` };

    case 'high':
      // High vol: reduce size, only trade if persistence is moderate
      // (high persistence = vol won't revert soon, riskier)
      if (persistence > 0.98) {
        return { action: 'skip', size: 0, reason: 'High vol + high persistence — skip' };
      }
      return { action, size: 0.5, reason: `High vol — half size ${action}` };

    case 'normal':
      // Normal vol: full size
      return { action, size: 1.0, reason: `Normal vol — full size ${action}` };

    case 'low':
      // Low vol: full size, potentially wider TP (mean reversion in vol expected)
      return { action, size: 1.0, reason: `Low vol — full size ${action}, expect vol expansion` };
  }
}
```

### Step 4: Dynamic SL/TP from Vol Forecast

```typescript
type StopLevels = {
  stopLoss: number;    // absolute price
  takeProfit: number;  // absolute price
  slDistance: number;   // SL distance in price units
  tpDistance: number;   // TP distance in price units
  slPercent: number;    // SL distance as % of price
  tpPercent: number;    // TP distance as % of price
};

function calculateSLTP(
  currentPrice: number,
  direction: 1 | -1,
  forecast: VolatilityForecast,
  riskRewardRatio: number = 2.0,
  slMultiplier: number = 2.0
): StopLevels {
  // Use 1-step ahead volatility (daily standard deviation)
  // forecast.volatility[0] = σ (one-period std dev)
  const onePeriodVol = forecast.volatility[0];

  // SL = N standard deviations away from entry
  // Typical: 2σ gives ~95% confidence the move is not noise
  const slDistance = currentPrice * onePeriodVol * slMultiplier;
  const tpDistance = slDistance * riskRewardRatio;

  const stopLoss = direction === 1
    ? currentPrice - slDistance
    : currentPrice + slDistance;

  const takeProfit = direction === 1
    ? currentPrice + tpDistance
    : currentPrice - tpDistance;

  return {
    stopLoss,
    takeProfit,
    slDistance,
    tpDistance,
    slPercent: (slDistance / currentPrice) * 100,
    tpPercent: (tpDistance / currentPrice) * 100,
  };
}
```

### Full Example: Webhook Handler

```typescript
import {
  Garch, Egarch,
  checkLeverageEffect,
  calculateReturnsFromPrices,
  type Candle,
} from 'garch';

// Webhook from TradingView Pine Script
async function handlePineWebhook(body: {
  symbol: string;
  direction: 1 | -1 | 0;
  price: number;
}) {
  const { symbol, direction, price } = body;

  // 1. Fetch candles from exchange (e.g. Binance, Bybit)
  const candles: Candle[] = await fetchCandles(symbol, '4h', 200);
  const periodsPerYear = 1512; // 4h candles

  // 2. Auto-select model and forecast
  const returns = calculateReturnsFromPrices(candles.map(c => c.close));
  const leverage = checkLeverageEffect(returns);

  let forecast, params;
  if (leverage.recommendation === 'egarch') {
    const model = new Egarch(candles, { periodsPerYear });
    const fit = model.fit();
    forecast = model.forecast(fit.params, 5);
    params = fit.params;
  } else {
    const model = new Garch(candles, { periodsPerYear });
    const fit = model.fit();
    forecast = model.forecast(fit.params, 5);
    params = fit.params;
  }

  // 3. Classify vol regime
  const currentVol = forecast.annualized[0];
  const longRunVol = params.annualizedVol;
  const regime = classifyVolRegime(currentVol, longRunVol);

  // 4. Combine Pine direction + vol regime
  const signal = combineSignal(direction, regime, params.persistence);

  if (signal.action === 'skip' || signal.action === 'close') {
    console.log(`${symbol}: ${signal.reason}`);
    return;
  }

  // 5. Calculate dynamic SL/TP
  const dir = signal.action === 'long' ? 1 : -1;
  const levels = calculateSLTP(price, dir, forecast, 2.0, 2.0);

  console.log(`${symbol} ${signal.action.toUpperCase()} x${signal.size}`);
  console.log(`  Vol regime: ${regime} | Current vol: ${currentVol.toFixed(1)}% | Long-run: ${longRunVol.toFixed(1)}%`);
  console.log(`  SL: ${levels.stopLoss.toFixed(2)} (-${levels.slPercent.toFixed(2)}%)`);
  console.log(`  TP: ${levels.takeProfit.toFixed(2)} (+${levels.tpPercent.toFixed(2)}%)`);

  // 6. Place order on exchange
  // await placeOrder(symbol, signal.action, signal.size, levels);
}
```

### Multi-Step Forecast for Swing Trades

For trades held over multiple periods, use the forecast array to set time-based exits:

```typescript
const model = new Garch(candles, { periodsPerYear: 1512 }); // 4h
const fit = model.fit();
const forecast = model.forecast(fit.params, 20); // 20 bars = ~3.3 days

// forecast.variance    — raw variance per step [σ²₁, σ²₂, ..., σ²₂₀]
// forecast.volatility  — std dev per step      [σ₁,  σ₂,  ..., σ₂₀]
// forecast.annualized  — annualized vol (%)    [v₁,  v₂,  ..., v₂₀]

// Cumulative expected move over N periods
// Total variance = sum of individual variances (independent increments)
const cumulativeVariance = forecast.variance.reduce((sum, v) => sum + v, 0);
const expectedMove = Math.sqrt(cumulativeVariance); // total std dev over N bars
const expectedMovePercent = expectedMove * 100;

console.log(`Expected ±${expectedMovePercent.toFixed(2)}% move over 20 bars`);
// Use this for swing trade SL/TP sizing
```

### Vol Regime Filtering Strategies

| Vol Regime | Entry | Size | SL Width | TP Width | Notes |
|---|---|---|---|---|---|
| **low** | Yes | 100% | Tight (1.5σ) | Wide (3σ+) | Low vol often precedes breakouts. Tight SL ok — noise is low |
| **normal** | Yes | 100% | Normal (2σ) | Normal (4σ) | Standard conditions |
| **high** | Selective | 50% | Wide (3σ) | Wide (6σ) | Only high-conviction signals. Wider SL to avoid noise |
| **extreme** | No | 0% | — | — | Stay flat. Re-enter when vol normalizes |
