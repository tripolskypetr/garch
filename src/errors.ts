/**
 * Typed error hierarchy so bot code can branch on error class instead of
 * parsing message strings.
 *
 *   try { predict(candles, '1h') }
 *   catch (e) {
 *     if (e instanceof NotEnoughDataError) await fetchMoreCandles();
 *     else if (e instanceof BadDataError) alertDataPipeline(e.message);
 *     else throw e;
 *   }
 */
export class GarchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The sample is too short for the requested interval/model. Fetch more candles. */
export class NotEnoughDataError extends GarchError {}

/** The candles themselves are broken: invalid OHLC, unsorted or duplicated timestamps. Fix the data pipeline. */
export class BadDataError extends GarchError {}

/** A call argument is out of range or of the wrong shape (interval, confidence, steps, currentPrice). Fix the call site. */
export class InvalidArgumentError extends GarchError {}
