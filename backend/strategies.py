"""
Three trading strategies, each returning (signal, price).
Indicators are computed from scratch with pandas — no pandas-ta dependency,
which makes the maths fully transparent and auditable.

signal values:  'BUY' | 'SELL' | 'HOLD' | None  (None = data error)
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'JPM', 'SPY']


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch(ticker: str, period: str = '6mo', interval: str = '1d') -> pd.DataFrame | None:
    """Download OHLCV data from yfinance and normalise the column index."""
    try:
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


# ── Indicator maths ────────────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Strategy 1 — Moving Average Crossover ────────────────────────────────────

def ma_crossover_signal(ticker: str) -> tuple[str | None, float | None]:
    """
    BUY  when 20-day SMA crosses above 50-day SMA (golden cross).
    SELL when 20-day SMA crosses below 50-day SMA (death cross).
    """
    df = _fetch(ticker, period='6mo')
    if df is None or len(df) < 52:
        return None, None

    df = df.copy()
    df['sma20'] = _sma(df['Close'], 20)
    df['sma50'] = _sma(df['Close'], 50)
    df.dropna(inplace=True)

    if len(df) < 2:
        return None, None

    price     = float(df['Close'].iloc[-1])
    prev20    = float(df['sma20'].iloc[-2])
    prev50    = float(df['sma50'].iloc[-2])
    curr20    = float(df['sma20'].iloc[-1])
    curr50    = float(df['sma50'].iloc[-1])

    if prev20 <= prev50 and curr20 > curr50:
        return 'BUY', price
    if prev20 >= prev50 and curr20 < curr50:
        return 'SELL', price
    return 'HOLD', price


# ── Strategy 2 — RSI Mean Reversion ──────────────────────────────────────────

def rsi_signal(ticker: str) -> tuple[str | None, float | None]:
    """
    BUY  when RSI < 30 (oversold).
    SELL when RSI > 70 (overbought).
    """
    df = _fetch(ticker, period='3mo')
    if df is None or len(df) < 16:
        return None, None

    df = df.copy()
    df['rsi'] = _rsi(df['Close'], 14)
    df.dropna(inplace=True)

    if df.empty:
        return None, None

    price   = float(df['Close'].iloc[-1])
    rsi_val = float(df['rsi'].iloc[-1])

    if rsi_val < 30:
        return 'BUY', price
    if rsi_val > 70:
        return 'SELL', price
    return 'HOLD', price


# ── Strategy 3 — MACD Momentum ────────────────────────────────────────────────

def macd_signal(ticker: str) -> tuple[str | None, float | None]:
    """
    BUY  when MACD crosses above signal line AND volume > 20-day average.
    SELL when MACD crosses below signal line.
    """
    df = _fetch(ticker, period='6mo')
    if df is None or len(df) < 35:
        return None, None

    df = df.copy()
    df['macd'], df['signal_line'], df['hist'] = _macd(df['Close'])
    df.dropna(inplace=True)

    if len(df) < 2:
        return None, None

    price      = float(df['Close'].iloc[-1])
    avg_vol    = float(df['Volume'].tail(20).mean())
    curr_vol   = float(df['Volume'].iloc[-1])
    volume_ok  = curr_vol > avg_vol

    prev_m = float(df['macd'].iloc[-2])
    prev_s = float(df['signal_line'].iloc[-2])
    curr_m = float(df['macd'].iloc[-1])
    curr_s = float(df['signal_line'].iloc[-1])

    if prev_m <= prev_s and curr_m > curr_s and volume_ok:
        return 'BUY', price
    if prev_m >= prev_s and curr_m < curr_s:
        return 'SELL', price
    return 'HOLD', price


# ── Dispatcher ────────────────────────────────────────────────────────────────

def get_signal(strategy: str, ticker: str) -> tuple[str | None, float | None]:
    dispatch = {
        'ma_crossover': ma_crossover_signal,
        'rsi':          rsi_signal,
        'macd':         macd_signal,
    }
    fn = dispatch.get(strategy)
    if fn is None:
        logger.error('Unknown strategy: %s', strategy)
        return None, None
    return fn(ticker)


# ── Indicator snapshot for dashboard display ──────────────────────────────────

def get_indicator_data(ticker: str, strategy: str) -> dict:
    """Returns current indicator values for the watchlist table."""
    df = _fetch(ticker, period='6mo')
    if df is None or df.empty:
        return {}

    result = {'price': round(float(df['Close'].iloc[-1]), 2)}

    if strategy == 'ma_crossover':
        df = df.copy()
        df['sma20'] = _sma(df['Close'], 20)
        df['sma50'] = _sma(df['Close'], 50)
        df.dropna(inplace=True)
        if len(df):
            result['sma20'] = round(float(df['sma20'].iloc[-1]), 2)
            result['sma50'] = round(float(df['sma50'].iloc[-1]), 2)

    elif strategy == 'rsi':
        df = df.copy()
        df['rsi'] = _rsi(df['Close'], 14)
        df.dropna(inplace=True)
        if len(df):
            result['rsi'] = round(float(df['rsi'].iloc[-1]), 2)

    elif strategy == 'macd':
        df = df.copy()
        df['macd'], df['signal_line'], df['hist'] = _macd(df['Close'])
        df.dropna(inplace=True)
        if len(df):
            result['macd']      = round(float(df['macd'].iloc[-1]), 4)
            result['signal']    = round(float(df['signal_line'].iloc[-1]), 4)
            result['histogram'] = round(float(df['hist'].iloc[-1]), 4)

    return result
