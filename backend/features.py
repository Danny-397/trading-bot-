"""
Centralized feature engineering layer.

All three existing strategies consume this module instead of computing
indicators inline.  When the ML strategy is added, it will call
get_feature_df() to get a ready-made feature matrix — no duplication,
no inconsistency between training data and live signals.

Public API
----------
get_feature_df(ticker, period, start, end)  → DataFrame | None
    Main entry point.  Returns OHLCV + all indicator columns, NaN rows
    (indicator warmup period) already dropped.

fetch_ohlcv(ticker, period, start, end)     → DataFrame | None
    Raw OHLCV download from yfinance.  Used by the backtest engine when
    it needs data without indicator computation.

compute_features(df)                         → DataFrame
    Adds indicator columns to any OHLCV DataFrame in-place copy.
    Called by get_feature_df() and by the backtest engine.

FEATURE_COLS
    Ordered list of every feature column an ML model should consume.
    Keep this list in sync whenever new indicators are added.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Feature column registry ────────────────────────────────────────────────────
# When you build the ML model, use df[FEATURE_COLS] as the input matrix.
# Add new indicators here and to compute_features() together.
FEATURE_COLS = [
    'close',
    'volume',
    'sma20',
    'sma50',
    'rsi14',
    'macd_line',
    'macd_signal',
    'macd_hist',
    'vol_ma20',
    'return_1d',
    'return_5d',
]

# Minimum bars needed for all indicators to be fully warmed up (SMA-50 is longest)
MIN_BARS = 55


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = '6mo', interval: str = '1d',
                start: str = None, end: str = None) -> pd.DataFrame | None:
    """Download OHLCV data from yfinance and normalise the column index."""
    try:
        if start and end:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
        if df.empty:
            return None
        # yfinance ≥ 0.2.40 returns MultiIndex columns even for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        logger.error('yfinance error for %s: %s', ticker, exc)
        return None


# ── Indicator computation ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicator columns to a raw OHLCV DataFrame.

    Does NOT drop NaN rows — callers decide whether to dropna or keep
    the warmup period for other purposes.
    """
    df = df.copy()
    close  = df['Close']
    volume = df['Volume']

    # ── Trend: simple moving averages ─────────────────────────────────────
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()

    # ── Momentum: RSI (14-period, Wilder EMA method) ──────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=True, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=True, min_periods=14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))

    # ── Momentum: MACD (12 / 26 / 9) ─────────────────────────────────────
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df['macd_line']   = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd_line'] - df['macd_signal']

    # ── Volume: rolling average ────────────────────────────────────────────
    df['vol_ma20'] = volume.rolling(20).mean()

    # ── Price returns ──────────────────────────────────────────────────────
    df['return_1d'] = close.pct_change(1)
    df['return_5d'] = close.pct_change(5)

    # ── Lowercase aliases (convenience for ML feature matrix) ─────────────
    df['close']  = close
    df['volume'] = volume

    return df


# ── Main entry point ───────────────────────────────────────────────────────────

def get_feature_df(ticker: str, period: str = '6mo',
                   start: str = None, end: str = None) -> pd.DataFrame | None:
    """
    Fetch OHLCV data and return a fully-featured DataFrame ready for signal
    generation or ML training.

    NaN rows (indicator warmup period) are dropped before returning.

    Usage
    -----
    Strategy signals:
        df = features.get_feature_df('AAPL')
        signal = 'BUY' if df['rsi14'].iloc[-1] < 30 else 'HOLD'

    ML training:
        df = features.get_feature_df('AAPL', start='2022-01-01', end='2024-01-01')
        X = df[features.FEATURE_COLS].values
        y = build_labels(df)          # your label function
        model.fit(X, y)

    ML live prediction:
        df = features.get_feature_df('AAPL')
        row = df[features.FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        signal = model.predict(row)[0]
    """
    raw = fetch_ohlcv(ticker, period=period, start=start, end=end)
    if raw is None or len(raw) < MIN_BARS:
        return None
    df = compute_features(raw)
    df.dropna(inplace=True)
    if df.empty:
        return None
    return df
