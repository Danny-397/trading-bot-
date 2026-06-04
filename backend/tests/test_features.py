"""
Unit tests for features.py.

Uses synthetic OHLCV data — no internet, no yfinance calls.
Verifies that indicator maths produce correct values on known inputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest
import features as feat


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    idx = pd.date_range('2023-01-01', periods=n, freq='B')
    return pd.DataFrame({
        'Open':   prices,
        'High':   [p * 1.01 for p in prices],
        'Low':    [p * 0.99 for p in prices],
        'Close':  prices,
        'Volume': [1_000_000] * n,
    }, index=idx)


def _flat_prices(n: int, price: float = 100.0) -> list[float]:
    return [price] * n


def _rising_prices(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


# ── SMA tests ─────────────────────────────────────────────────────────────────

class TestSMA:
    def test_sma20_equals_mean_of_last_20_bars(self):
        prices = _flat_prices(60, 100.0)
        prices[40:60] = [200.0] * 20   # last 20 bars are 200
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        # sma20 at last bar should be mean of last 20 closes = 200
        assert df['sma20'].iloc[-1] == pytest.approx(200.0)

    def test_sma50_equals_mean_of_last_50_bars(self):
        prices = _flat_prices(60, 100.0)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        assert df['sma50'].iloc[-1] == pytest.approx(100.0)

    def test_sma20_less_than_sma50_in_downtrend(self):
        # Recent prices lower than older prices → SMA20 < SMA50
        prices = list(reversed(_rising_prices(60, start=60.0, step=1.0)))
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        assert df['sma20'].iloc[-1] < df['sma50'].iloc[-1]


# ── RSI tests ──────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_is_between_0_and_100(self):
        prices = _rising_prices(60)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        assert (df['rsi14'] >= 0).all()
        assert (df['rsi14'] <= 100).all()

    def test_rsi_near_100_for_pure_uptrend(self):
        # Prices that only go up → RSI should be close to 100
        prices = _rising_prices(60, step=5.0)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        assert df['rsi14'].iloc[-1] > 90

    def test_rsi_near_0_for_pure_downtrend(self):
        prices = list(reversed(_rising_prices(60, start=300.0, step=5.0)))
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        assert df['rsi14'].iloc[-1] < 10

    def test_rsi_near_50_for_flat_prices(self):
        # Flat prices have no gains or losses → RSI is undefined, but near 50
        prices = _flat_prices(60)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        # With all zeros in diff, RSI should be NaN or 50-ish; either is acceptable
        last = df['rsi14'].iloc[-1]
        assert pd.isna(last) or (40 <= last <= 60)


# ── MACD tests ─────────────────────────────────────────────────────────────────

class TestMACD:
    def test_macd_line_is_ema12_minus_ema26(self):
        prices = _rising_prices(60)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        close = pd.Series(prices, dtype=float)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        expected_macd = float(ema12.iloc[-1] - ema26.iloc[-1])
        assert df['macd_line'].iloc[-1] == pytest.approx(expected_macd, rel=1e-4)

    def test_histogram_is_macd_minus_signal(self):
        prices = _rising_prices(60)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        last = df.iloc[-1]
        expected = last['macd_line'] - last['macd_signal']
        assert last['macd_hist'] == pytest.approx(expected, rel=1e-6)

    def test_macd_positive_in_uptrend(self):
        prices = _rising_prices(80, step=2.0)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        assert df['macd_line'].iloc[-1] > 0


# ── Feature column registry ────────────────────────────────────────────────────

class TestFeatureCols:
    def test_all_feature_cols_present_after_compute(self):
        prices = _rising_prices(60)
        df = feat.compute_features(_make_ohlcv(prices))
        df.dropna(inplace=True)
        for col in feat.FEATURE_COLS:
            assert col in df.columns, f'Missing feature column: {col}'

    def test_get_feature_df_returns_none_for_insufficient_data(self, monkeypatch):
        short_df = _make_ohlcv(_rising_prices(10))
        monkeypatch.setattr(feat, 'fetch_ohlcv', lambda *a, **kw: short_df)
        result = feat.get_feature_df('FAKE')
        assert result is None

    def test_get_feature_df_has_no_nan_rows(self, monkeypatch):
        long_df = _make_ohlcv(_rising_prices(100))
        monkeypatch.setattr(feat, 'fetch_ohlcv', lambda *a, **kw: long_df)
        result = feat.get_feature_df('FAKE')
        assert result is not None
        assert result.isnull().sum().sum() == 0
