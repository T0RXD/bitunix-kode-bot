"""
Advanced Trading Strategy for Bitunix Futures

Implements a LONG-only breakout strategy using:
  - Bollinger Bands (Period 50, StdDev 1.5)
  - EMA 9 and EMA 21
  - Parabolic SAR (Start 0.25, Step 0.07, Max 0.1)
  - ATR (Period 14)

IMPORTANT — Next-candle entry logic:
  The signal is evaluated on the LAST FULLY CLOSED candle (the "volume candle").
  If all conditions are met, the symbol is flagged for entry on the NEXT candle's
  open.  The caller (main loop) is responsible for:
    1. Calling analyze_symbol() once a new candle has closed.
    2. Storing the pending signal.
    3. Executing the trade at the open of the following candle.

Signal conditions (all must be true on the volume candle):
  1. Volume > 1.8 * previous candle volume
  2. Close > Upper Bollinger Band
  3. Current BB Width > Previous BB Width (bands opening)
  4. EMA 9 > EMA 21
  5. PSAR < Close
  6. PSAR flip: PSAR was above price on the prior candle and is now below,
     and price > EMA 21

Q-Learning Integration:
  analyze_symbol now also returns a state_tuple representing the market condition:
    (EMA_cross, BB_breakout, Vol_spike, PSAR_pos)
  where each is 1 if the condition is met, 0 otherwise.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

StateTuple = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Indicator calculations (pure pandas, no TA-Lib)
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_bollinger_bands(
    close: pd.Series,
    period: int = 50,
    std_dev: float = 1.5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def calc_parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.25,
    af_step: float = 0.07,
    af_max: float = 0.1,
) -> pd.Series:
    """Manual Parabolic SAR implementation."""
    length = len(high)
    sar = pd.Series(index=high.index, dtype=float)
    if length < 2:
        return sar

    high_vals = high.values
    low_vals = low.values
    close_vals = close.values
    sar_vals = [0.0] * length

    is_bullish = close_vals[1] >= close_vals[0]

    af = af_start
    if is_bullish:
        sar_vals[0] = low_vals[0]
        ep = high_vals[0]
    else:
        sar_vals[0] = high_vals[0]
        ep = low_vals[0]

    for i in range(1, length):
        prev_sar = sar_vals[i - 1]

        if is_bullish:
            sar_val = prev_sar + af * (ep - prev_sar)
            if i >= 2:
                sar_val = min(sar_val, low_vals[i - 1], low_vals[i - 2])
            else:
                sar_val = min(sar_val, low_vals[i - 1])

            if low_vals[i] < sar_val:
                is_bullish = False
                sar_val = ep
                ep = low_vals[i]
                af = af_start
            else:
                if high_vals[i] > ep:
                    ep = high_vals[i]
                    af = min(af + af_step, af_max)
        else:
            sar_val = prev_sar + af * (ep - prev_sar)
            if i >= 2:
                sar_val = max(sar_val, high_vals[i - 1], high_vals[i - 2])
            else:
                sar_val = max(sar_val, high_vals[i - 1])

            if high_vals[i] > sar_val:
                is_bullish = True
                sar_val = ep
                ep = high_vals[i]
                af = af_start
            else:
                if low_vals[i] < ep:
                    ep = low_vals[i]
                    af = min(af + af_step, af_max)

        sar_vals[i] = sar_val

    sar = pd.Series(sar_vals, index=high.index, dtype=float)
    return sar


# ---------------------------------------------------------------------------
# Add all indicators to a DataFrame
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all strategy indicators to an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume
    Adds: ema9, ema21, bb_upper, bb_mid, bb_lower, bb_width,
          psar, atr
    """
    df = df.copy()

    df["ema9"] = calc_ema(df["close"], 9)
    df["ema21"] = calc_ema(df["close"], 21)

    df["bb_upper"], df["bb_mid"], df["bb_lower"] = calc_bollinger_bands(
        df["close"], period=50, std_dev=1.5
    )
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df["atr"] = calc_atr(df["high"], df["low"], df["close"], period=14)

    df["psar"] = calc_parabolic_sar(
        df["high"], df["low"], df["close"],
        af_start=0.25, af_step=0.07, af_max=0.1,
    )

    return df


# ---------------------------------------------------------------------------
# Pending signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class PendingSignal:
    """Represents a confirmed signal on a closed volume candle, awaiting next-candle entry."""
    symbol: str
    signal_candle_close: float
    atr: float
    signal_candle_ts: float
    state_tuple: StateTuple = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# State extraction for Q-Learning
# ---------------------------------------------------------------------------

def extract_state_tuple(df: pd.DataFrame) -> StateTuple:
    """
    Extract a binary state tuple from an indicator-enriched OHLCV DataFrame.

    Returns (EMA_cross, BB_breakout, Vol_spike, PSAR_pos):
      - EMA_cross:   1 if EMA9 > EMA21 on the last candle, else 0
      - BB_breakout: 1 if close > upper Bollinger Band, else 0
      - Vol_spike:   1 if volume > 1.8 * previous candle volume, else 0
      - PSAR_pos:    1 if PSAR < close (bullish), else 0

    Requires at least 52 rows and indicators already computed via add_indicators().
    Returns (0, 0, 0, 0) if data is insufficient.
    """
    if len(df) < 52:
        return (0, 0, 0, 0)

    df_ind = add_indicators(df)
    curr = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]

    required = ["ema9", "ema21", "bb_upper", "psar"]
    for col in required:
        if pd.isna(curr[col]):
            return (0, 0, 0, 0)

    ema_cross = 1 if curr["ema9"] > curr["ema21"] else 0
    bb_breakout = 1 if curr["close"] > curr["bb_upper"] else 0
    vol_spike = 1 if curr["volume"] > 1.8 * prev["volume"] else 0
    psar_pos = 1 if curr["psar"] < curr["close"] else 0

    return (ema_cross, bb_breakout, vol_spike, psar_pos)


# ---------------------------------------------------------------------------
# Signal analysis (updated to also return state_tuple)
# ---------------------------------------------------------------------------

def analyze_symbol(df: pd.DataFrame) -> Tuple[bool, StateTuple]:
    """
    Analyze an OHLCV DataFrame and return:
      - bool: True if the last fully closed candle triggers a long entry signal.
      - StateTuple: The market state as (EMA_cross, BB_breakout, Vol_spike, PSAR_pos).

    IMPORTANT: The caller must treat this as a *pending* signal.  The actual
    entry should happen at the OPEN of the next candle.

    The DataFrame should contain only CLOSED candles.
    Requires at least 52 rows (50 for BB period + 2 for prev-candle checks).
    """
    state = (0, 0, 0, 0)

    if len(df) < 52:
        logger.warning("Not enough data for analysis (need >= 52 rows, got %d)", len(df))
        return False, state

    df = add_indicators(df)

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    required = ["bb_upper", "bb_width", "ema9", "ema21", "psar", "atr"]
    for col in required:
        if pd.isna(curr[col]) or pd.isna(prev.get(col, float("nan"))):
            logger.debug("NaN in indicator '%s', skipping signal", col)
            return False, state

    # --- Build state tuple from individual conditions ---
    ema_cross = 1 if curr["ema9"] > curr["ema21"] else 0
    bb_breakout = 1 if curr["close"] > curr["bb_upper"] else 0
    vol_spike = 1 if curr["volume"] > 1.8 * prev["volume"] else 0
    psar_pos = 1 if curr["psar"] < curr["close"] else 0

    state = (ema_cross, bb_breakout, vol_spike, psar_pos)

    # --- Condition 1: Volume spike ---
    volume_spike = curr["volume"] > 1.8 * prev["volume"]

    # --- Condition 2: Close above upper Bollinger Band ---
    close_above_bb = curr["close"] > curr["bb_upper"]

    # --- Condition 3: BB bands opening (width increasing) ---
    bb_opening = curr["bb_width"] > prev["bb_width"]

    # --- Condition 4: EMA 9 > EMA 21 (bullish trend) ---
    ema_bullish = curr["ema9"] > curr["ema21"]

    # --- Condition 5: PSAR below close (bullish) ---
    psar_bullish = curr["psar"] < curr["close"]

    # --- Condition 6: PSAR flip check ---
    psar_was_above = prev["psar"] > prev["close"]
    psar_now_below = curr["psar"] < curr["close"]
    price_above_ema21 = curr["close"] > curr["ema21"]
    psar_flip = psar_was_above and psar_now_below and price_above_ema21

    signal_long = all([
        volume_spike,
        close_above_bb,
        bb_opening,
        ema_bullish,
        psar_bullish,
        psar_flip,
    ])

    if signal_long:
        logger.info(
            "LONG SIGNAL on volume candle: close=%.4f bb_upper=%.4f ema9=%.4f "
            "ema21=%.4f psar=%.4f atr=%.4f vol=%.2f prev_vol=%.2f  "
            "[entry deferred to next candle open]",
            curr["close"], curr["bb_upper"], curr["ema9"], curr["ema21"],
            curr["psar"], curr["atr"], curr["volume"], prev["volume"],
        )

    return signal_long, state
