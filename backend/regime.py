"""
Market regime detection — pure math, no ML.

Classifies the current market into one of four states using three
independent indicators computed from OHLCV data only:

  ADX  (Average Directional Index)   — trend strength, not direction
  BB Width (Bollinger Band Width)     — volatility compression/expansion
  30-day Realized Volatility          — annualized standard deviation of returns

Regime states
-------------
TRENDING_UP      ADX > 25, +DI > -DI   → MA Crossover (follow the trend)
TRENDING_DOWN    ADX > 25, -DI > +DI   → MA Crossover (respect the downtrend)
RANGING          ADX < 20              → RSI Mean Reversion (fade extremes)
HIGH_VOLATILITY  vol > 25%             → RSI with reduced position size

When the ML course is done
--------------------------
Replace the rule-based detect_regime() with a trained classifier that
predicts regime from the same input indicators.  The RegimeResult dataclass,
the strategy map, and compute_regime_series() stay exactly the same —
only the internal classification logic changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

# ── Thresholds ─────────────────────────────────────────────────────────────────
ADX_TRENDING   = 25     # ADX >= this → trending market
ADX_RANGING    = 20     # ADX <  this → ranging / sideways
VOL_HIGH_PCT   = 25.0   # 30-day annualized vol (%) above this → high-volatility
ADX_PERIOD     = 14     # Wilder's smoothing period for ADX

# ── Strategy mapping ───────────────────────────────────────────────────────────
REGIME_STRATEGY = {
    'TRENDING_UP':    'ma_crossover',
    'TRENDING_DOWN':  'ma_crossover',
    'RANGING':        'rsi',
    'HIGH_VOLATILITY': 'rsi',
}

REGIME_LABELS = {
    'TRENDING_UP':    'Trending Up',
    'TRENDING_DOWN':  'Trending Down',
    'RANGING':        'Ranging / Sideways',
    'HIGH_VOLATILITY': 'High Volatility',
}

REGIME_DESCRIPTIONS = {
    'TRENDING_UP':    f'Strong uptrend (ADX > {ADX_TRENDING}, momentum is bullish)',
    'TRENDING_DOWN':  f'Strong downtrend (ADX > {ADX_TRENDING}, momentum is bearish)',
    'RANGING':        f'No clear trend (ADX < {ADX_RANGING}, price oscillating)',
    'HIGH_VOLATILITY': f'Elevated volatility (30d vol > {VOL_HIGH_PCT}%, unstable)',
}


@dataclass
class RegimeResult:
    regime:      str
    label:       str
    description: str
    strategy:    str
    adx:         float
    plus_di:     float
    minus_di:    float
    bb_width:    float
    vol_30d:     float


# ── Indicator computations ─────────────────────────────────────────────────────

def _adx(df: pd.DataFrame, period: int = ADX_PERIOD):
    """
    Wilder's ADX with +DI and -DI.

    Returns (adx_series, plus_di_series, minus_di_series).
    ADX measures trend STRENGTH only.  +DI > -DI → bullish direction.
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # True Range — largest of the three measures
    hl  = high - low
    hcp = (high - close.shift(1)).abs()
    lcp = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)

    # Directional Movement
    up   = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm  = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    # Wilder's smoothing: alpha = 1/period
    alpha = 1.0 / period
    tr_s    = tr.ewm(alpha=alpha,     adjust=False, min_periods=period).mean()
    pdm_s   = plus_dm.ewm(alpha=alpha,  adjust=False, min_periods=period).mean()
    mdm_s   = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di  = 100 * pdm_s  / tr_s.replace(0, np.nan)
    minus_di = 100 * mdm_s  / tr_s.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx, plus_di, minus_di


def _bb_width(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Bollinger Band Width = (Upper - Lower) / Middle.

    Narrow = consolidation (often precedes a breakout or ranging market).
    Wide   = expanding volatility (often trend or high-vol regime).
    """
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    width = (sma + 2 * std - (sma - 2 * std)) / sma.replace(0, np.nan)
    return width


def _realized_vol(close: pd.Series, period: int = 30) -> float:
    """30-day annualized realized volatility as a percentage."""
    rets = close.pct_change().dropna()
    if len(rets) < period:
        return 0.0
    return float(rets.tail(period).std() * np.sqrt(252) * 100)


# ── Point-in-time detection ────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> RegimeResult:
    """
    Classify the current market regime from an OHLCV DataFrame.

    Requires at least 60 bars (50-bar warmup + 10 buffer).
    Uses the final row for the current reading.

    When the ML model is ready, replace the if/elif classification block
    below with: regime = ml_classifier.predict(feature_row)[0]
    Everything else — the dataclass, strategy map, callers — stays the same.
    """
    adx_s, pdi_s, mdi_s = _adx(df)
    bbw_s = _bb_width(df['Close'])

    adx     = float(adx_s.dropna().iloc[-1])  if not adx_s.dropna().empty  else 0.0
    pdi     = float(pdi_s.dropna().iloc[-1])  if not pdi_s.dropna().empty  else 0.0
    mdi     = float(mdi_s.dropna().iloc[-1])  if not mdi_s.dropna().empty  else 0.0
    bbw     = float(bbw_s.dropna().iloc[-1])  if not bbw_s.dropna().empty  else 0.0
    vol_30d = _realized_vol(df['Close'])

    # ── Rule-based classification ──────────────────────────────────────────
    # ML hook: replace these four lines with a model prediction
    if vol_30d > VOL_HIGH_PCT and adx < ADX_TRENDING:
        regime = 'HIGH_VOLATILITY'
    elif adx >= ADX_TRENDING:
        regime = 'TRENDING_UP' if pdi > mdi else 'TRENDING_DOWN'
    else:
        regime = 'RANGING'
    # ──────────────────────────────────────────────────────────────────────

    return RegimeResult(
        regime      = regime,
        label       = REGIME_LABELS[regime],
        description = REGIME_DESCRIPTIONS[regime],
        strategy    = REGIME_STRATEGY[regime],
        adx         = round(adx,     2),
        plus_di     = round(pdi,     2),
        minus_di    = round(mdi,     2),
        bb_width    = round(bbw,     4),
        vol_30d     = round(vol_30d, 2),
    )


# ── Time-series regime labelling (for backtest) ────────────────────────────────

def compute_regime_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of regime labels, one per row, for the full DataFrame.

    Each label reflects only information available up to that date
    (no look-ahead) because all indicators use trailing windows.

    Used by the backtest engine to tag each trade with its regime context.
    """
    adx_s, pdi_s, mdi_s = _adx(df)
    vol_s = df['Close'].pct_change().rolling(30).std() * np.sqrt(252) * 100

    high_vol  = (vol_s > VOL_HIGH_PCT) & (adx_s < ADX_TRENDING)
    trend_up  = (adx_s >= ADX_TRENDING) & (pdi_s > mdi_s)
    trend_dn  = (adx_s >= ADX_TRENDING) & (pdi_s <= mdi_s)

    regimes = np.select(
        [high_vol, trend_up, trend_dn],
        ['HIGH_VOLATILITY', 'TRENDING_UP', 'TRENDING_DOWN'],
        default='RANGING',
    )

    return pd.Series(regimes, index=df.index, dtype=str)


# ── Risk-tolerance-aware strategy selection ────────────────────────────────────

def get_regime_strategy(regime: str, risk_tolerance: str = 'moderate') -> str:
    """
    Map a detected regime and user risk tolerance to the best strategy.

    CONSERVATIVE  Avoids high-vol conditions entirely; prefers mean-reversion.
    MODERATE      Uses the default regime→strategy map.
    AGGRESSIVE    Uses MACD momentum in trending markets for faster signals.
    """
    if risk_tolerance == 'conservative':
        # Sit out HIGH_VOLATILITY; always favour RSI as the safer strategy
        if regime == 'HIGH_VOLATILITY':
            return 'hold'           # signal: do nothing, preserve capital
        if 'TRENDING' in regime:
            return 'ma_crossover'
        return 'rsi'

    if risk_tolerance == 'aggressive':
        # MACD in trends (faster entry/exit), RSI everywhere else
        return 'macd' if 'TRENDING' in regime else 'rsi'

    # moderate (default)
    return REGIME_STRATEGY.get(regime, 'ma_crossover')
