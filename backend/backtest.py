"""
Backtesting engine.

Simulates a strategy against historical OHLCV data using the same signal logic
as the live bot.  Enforces identical risk rules: 5% stop-loss, 15% take-profit,
10% max position size, 20% cash reserve.

Returns a structured result dict with performance metrics, equity curve, SPY
benchmark, and full trade log.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception as exc:
        logger.error('fetch error %s: %s', ticker, exc)
        return None


def _sma(s, w):        return s.rolling(w).mean()

def _rsi(s, p=14):
    delta = s.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=p - 1, adjust=True, min_periods=p).mean()
    al    = loss.ewm(com=p - 1, adjust=True, min_periods=p).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig,  adjust=False).mean()
    return ml, sl


def _add_signals(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Append a 'signal' column: +1 = BUY, -1 = SELL, 0 = HOLD."""
    df = df.copy()
    close = df['Close']

    if strategy == 'ma_crossover':
        df['sma20'] = _sma(close, 20)
        df['sma50'] = _sma(close, 50)
        df.dropna(subset=['sma20', 'sma50'], inplace=True)
        buy  = (df['sma20'].shift(1) <= df['sma50'].shift(1)) & (df['sma20'] > df['sma50'])
        sell = (df['sma20'].shift(1) >= df['sma50'].shift(1)) & (df['sma20'] < df['sma50'])

    elif strategy == 'rsi':
        df['rsi'] = _rsi(close)
        df.dropna(subset=['rsi'], inplace=True)
        buy  = df['rsi'] < 30
        sell = df['rsi'] > 70

    elif strategy == 'macd':
        df['macd_line'], df['signal_line'] = _macd(close)
        df.dropna(subset=['macd_line', 'signal_line'], inplace=True)
        avg_vol = df['Volume'].rolling(20).mean()
        buy  = (
            (df['macd_line'].shift(1) <= df['signal_line'].shift(1)) &
            (df['macd_line'] > df['signal_line']) &
            (df['Volume'] > avg_vol)
        )
        sell = (
            (df['macd_line'].shift(1) >= df['signal_line'].shift(1)) &
            (df['macd_line'] < df['signal_line'])
        )
    else:
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)

    df['signal'] = 0
    df.loc[buy,  'signal'] =  1
    df.loc[sell, 'signal'] = -1
    return df


# ── main engine ───────────────────────────────────────────────────────────────

def run_backtest(strategy: str, tickers: list[str],
                 start_date: str, end_date: str,
                 initial_capital: float = 100_000.0) -> dict:
    """
    Runs a daily-bar backtest simulation.

    Risk rules enforced identically to the live bot:
      • 5%  stop-loss per position
      • 15% take-profit per position
      • 10% max portfolio allocation per ticker
      • 20% minimum cash reserve
    """
    cash      = float(initial_capital)
    positions = {}   # ticker -> {'shares': int, 'entry': float}
    trades    = []
    port_hist = []   # [{date, value}]

    # ── Prepare data ──────────────────────────────────────────────────────────
    data = {}
    for ticker in tickers:
        df = _fetch(ticker, start_date, end_date)
        if df is None or len(df) < 30:
            logger.warning('Skipping %s — insufficient data', ticker)
            continue
        data[ticker] = _add_signals(df, strategy)

    if not data:
        return {'error': 'No sufficient data for the selected tickers and date range.'}

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    # ── Simulation loop ───────────────────────────────────────────────────────
    for date in all_dates:
        # Current portfolio value (cash + mark-to-market positions)
        port_val = cash
        for tkr, pos in positions.items():
            if tkr in data and date in data[tkr].index:
                port_val += pos['shares'] * float(data[tkr].loc[date, 'Close'])

        port_hist.append({'date': date.strftime('%Y-%m-%d'), 'value': round(port_val, 2)})

        for ticker, df in data.items():
            if date not in df.index:
                continue

            row    = df.loc[date]
            price  = float(row['Close'])
            signal = int(row.get('signal', 0))

            # ── Check stop-loss / take-profit on open position ────────────
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
                    trades.append({
                        'date':    date.strftime('%Y-%m-%d'),
                        'ticker':  ticker,
                        'action':  'SELL',
                        'price':   round(price, 2),
                        'shares':  shares,
                        'pnl':     round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'reason':  reason,
                    })
                    del positions[ticker]

            # ── Buy signal ────────────────────────────────────────────────
            elif signal == 1 and price > 0:
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

    # ── Compute metrics ───────────────────────────────────────────────────────
    final_value  = port_hist[-1]['value'] if port_hist else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100

    sell_trades = [t for t in trades if t['action'] == 'SELL']
    wins        = [t for t in sell_trades if t['pnl'] and t['pnl'] > 0]
    losses      = [t for t in sell_trades if t['pnl'] and t['pnl'] <= 0]
    pnls        = [t['pnl'] for t in sell_trades if t['pnl'] is not None]

    # Max drawdown
    values  = [p['value'] for p in port_hist]
    max_dd  = 0.0
    peak    = values[0] if values else initial_capital
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

    # SPY buy-and-hold benchmark
    spy_df           = _fetch('SPY', start_date, end_date)
    benchmark_return = 0.0
    spy_curve        = []
    if spy_df is not None and len(spy_df) > 1:
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
        'trades':       trades[-200:],   # last 200 for the table
    }
