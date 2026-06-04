"""
Backtesting engine.

Simulates a strategy against historical OHLCV data using the same signal
logic as the live bot.  Enforces identical risk rules: 5% stop-loss,
15% take-profit, 10% max position size, 20% cash reserve.

Walk-forward mode
-----------------
Pass walk_forward=True to run an out-of-sample evaluation.

The date range is split 70 / 30.  The first 70% is the "in-sample"
context window used only for indicator warmup.  Actual trades are only
recorded in the final 30% — the out-of-sample test period.

This matters most for the future ML strategy: when the model is added,
it will be trained on the 70% portion and evaluated on the 30% portion,
with this same infrastructure.  Non-ML strategies can use it now to
guard against favourable in-sample curve-fitting.

Returns a structured dict with:
  metrics       — performance statistics
  equity_curve  — [{date, value}] portfolio value over time
  spy_curve     — SPY buy-and-hold benchmark scaled to same capital
  trades        — last 200 closed trades
  walk_forward  — {enabled, split_date} metadata (always present)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import features as feat

logger = logging.getLogger(__name__)


# ── Signal generation ──────────────────────────────────────────────────────────

def _add_signals(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Compute all indicators via features.compute_features(), then append a
    'signal' column: +1 = BUY, -1 = SELL, 0 = HOLD.

    When the ML strategy is added, replace the 'ml' branch stub with:
        df['signal'] = model.predict(df[feat.FEATURE_COLS].values)
    """
    df = feat.compute_features(df)
    df.dropna(inplace=True)

    if strategy == 'ma_crossover':
        buy  = (df['sma20'].shift(1) <= df['sma50'].shift(1)) & (df['sma20'] > df['sma50'])
        sell = (df['sma20'].shift(1) >= df['sma50'].shift(1)) & (df['sma20'] < df['sma50'])

    elif strategy == 'rsi':
        buy  = df['rsi14'] < 30
        sell = df['rsi14'] > 70

    elif strategy == 'macd':
        buy  = (
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &
            (df['macd_line'] > df['macd_signal']) &
            (df['Volume'] > df['vol_ma20'])
        )
        sell = (
            (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) &
            (df['macd_line'] < df['macd_signal'])
        )

    elif strategy == 'ml':
        # ── ML backtest hook ───────────────────────────────────────────────
        # When the model is ready, replace these two lines with:
        #   preds = _ml_model.predict(df[feat.FEATURE_COLS].values)
        #   buy  = pd.Series(preds == 1,  index=df.index)
        #   sell = pd.Series(preds == -1, index=df.index)
        # ──────────────────────────────────────────────────────────────────
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)

    else:
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)

    df['signal'] = 0
    df.loc[buy,  'signal'] = 1
    df.loc[sell, 'signal'] = -1
    return df


# ── Main engine ────────────────────────────────────────────────────────────────

def run_backtest(strategy: str, tickers: list[str],
                 start_date: str, end_date: str,
                 initial_capital: float = 100_000.0,
                 walk_forward: bool = False) -> dict:
    """
    Run a daily-bar backtest simulation.

    Parameters
    ----------
    strategy        : 'ma_crossover' | 'rsi' | 'macd' | 'ml'
    tickers         : list of ticker symbols
    start_date      : 'YYYY-MM-DD' — download data from this date
    end_date        : 'YYYY-MM-DD' — download data until this date
    initial_capital : starting cash
    walk_forward    : if True, only record trades in the final 30% of the
                      date range (out-of-sample test period)
    """
    # ── Walk-forward split ─────────────────────────────────────────────────
    split_date = None
    if walk_forward:
        s = datetime.strptime(start_date, '%Y-%m-%d')
        e = datetime.strptime(end_date,   '%Y-%m-%d')
        split_date = (s + timedelta(days=int((e - s).days * 0.70))).strftime('%Y-%m-%d')

    # ── Download and signal-label data ─────────────────────────────────────
    data = {}
    for ticker in tickers:
        raw = feat.fetch_ohlcv(ticker, start=start_date, end=end_date)
        if raw is None or len(raw) < 30:
            logger.warning('Skipping %s — insufficient data', ticker)
            continue
        data[ticker] = _add_signals(raw, strategy)

    if not data:
        return {'error': 'No sufficient data for the selected tickers and date range.'}

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    # ── Simulation loop ────────────────────────────────────────────────────
    cash      = float(initial_capital)
    positions = {}   # ticker -> {'shares': int, 'entry': float}
    trades    = []
    port_hist = []

    for date in all_dates:
        # In walk-forward mode, skip the in-sample (training) window
        recording = (split_date is None) or (date.strftime('%Y-%m-%d') >= split_date)

        port_val = cash
        for tkr, pos in positions.items():
            if tkr in data and date in data[tkr].index:
                port_val += pos['shares'] * float(data[tkr].loc[date, 'Close'])

        if recording:
            port_hist.append({'date': date.strftime('%Y-%m-%d'), 'value': round(port_val, 2)})

        for ticker, df in data.items():
            if date not in df.index:
                continue

            row    = df.loc[date]
            price  = float(row['Close'])
            signal = int(row.get('signal', 0))

            # ── Stop-loss / take-profit on open position ───────────────────
            if ticker in positions:
                pos    = positions[ticker]
                entry  = pos['entry']
                reason = None

                if price <= entry * 0.95:
                    reason = 'stop_loss'
                elif price >= entry * 1.15:
                    reason = 'take_profit'
                elif signal == -1:
                    reason = 'sell_signal'

                if reason:
                    shares  = pos['shares']
                    pnl     = (price - entry) * shares
                    pnl_pct = (price - entry) / entry * 100
                    cash   += shares * price
                    if recording:
                        trades.append({
                            'date':    date.strftime('%Y-%m-%d'),
                            'ticker':  ticker,
                            'action':  'SELL',
                            'price':   round(price,   2),
                            'shares':  shares,
                            'pnl':     round(pnl,     2),
                            'pnl_pct': round(pnl_pct, 2),
                            'reason':  reason,
                        })
                    del positions[ticker]

            # ── Buy signal ─────────────────────────────────────────────────
            elif signal == 1 and price > 0 and recording:
                max_spend   = port_val * 0.10
                usable_cash = cash - port_val * 0.20
                if usable_cash > price:
                    spend  = min(max_spend, usable_cash)
                    shares = int(spend / price)
                    if shares > 0:
                        cash -= shares * price
                        positions[ticker] = {'shares': shares, 'entry': price}
                        trades.append({
                            'date':    date.strftime('%Y-%m-%d'),
                            'ticker':  ticker,
                            'action':  'BUY',
                            'price':   round(price, 2),
                            'shares':  shares,
                            'pnl':     None,
                            'pnl_pct': None,
                            'reason':  'buy_signal',
                        })

    # ── Performance metrics ────────────────────────────────────────────────
    final_value  = port_hist[-1]['value'] if port_hist else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100

    sell_trades = [t for t in trades if t['action'] == 'SELL']
    wins        = [t for t in sell_trades if t['pnl'] and t['pnl'] > 0]
    losses      = [t for t in sell_trades if t['pnl'] and t['pnl'] <= 0]
    pnls        = [t['pnl'] for t in sell_trades if t['pnl'] is not None]

    # Max drawdown
    values = [p['value'] for p in port_hist]
    max_dd = 0.0
    peak   = values[0] if values else initial_capital
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualised, daily returns, 4% risk-free)
    sharpe = 0.0
    if len(values) > 2:
        arr  = np.array(values, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        std  = rets.std()
        if std > 0:
            sharpe = round((rets.mean() - 0.04 / 252) / std * np.sqrt(252), 2)

    # Calmar ratio (annualised return / max drawdown)
    calmar = 0.0
    if max_dd > 0:
        years  = max(len(values) / 252, 1 / 252)
        ann_r  = (final_value / initial_capital) ** (1 / years) - 1
        calmar = round(ann_r / (max_dd / 100), 2)

    # SPY buy-and-hold benchmark
    spy_df           = feat.fetch_ohlcv('SPY', start=start_date, end=end_date)
    benchmark_return = 0.0
    spy_curve        = []
    if spy_df is not None and len(spy_df) > 1:
        # Align SPY to the simulation window if walk_forward
        if walk_forward and split_date:
            spy_df = spy_df[spy_df.index >= pd.Timestamp(split_date)]
        if len(spy_df) > 1:
            spy_start        = float(spy_df['Close'].iloc[0])
            spy_end          = float(spy_df['Close'].iloc[-1])
            benchmark_return = (spy_end - spy_start) / spy_start * 100
            spy_curve        = [
                {'date': d.strftime('%Y-%m-%d'),
                 'value': round(initial_capital * float(v) / spy_start, 2)}
                for d, v in spy_df['Close'].items()
            ]

    return {
        'metrics': {
            'total_return':     round(total_return, 2),
            'win_rate':         round(len(wins) / len(sell_trades) * 100, 1) if sell_trades else 0,
            'max_drawdown':     round(max_dd, 2),
            'sharpe_ratio':     sharpe,
            'calmar_ratio':     calmar,
            'total_trades':     len(sell_trades),
            'winning_trades':   len(wins),
            'losing_trades':    len(losses),
            'avg_win':          round(sum(t['pnl'] for t in wins)   / len(wins),   2) if wins   else 0,
            'avg_loss':         round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0,
            'best_trade':       round(max(pnls), 2) if pnls else 0,
            'worst_trade':      round(min(pnls), 2) if pnls else 0,
            'final_value':      round(final_value, 2),
            'initial_capital':  initial_capital,
            'benchmark_return': round(benchmark_return, 2),
        },
        'equity_curve': port_hist,
        'spy_curve':    spy_curve,
        'trades':       trades[-200:],
        'walk_forward': {
            'enabled':    walk_forward,
            'split_date': split_date,
        },
    }
