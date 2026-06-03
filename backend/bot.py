"""
Core trading bot.

Runs in a background daemon thread.  Every 5 minutes during market hours it:
  1. Checks risk gates (market open, daily trade limit, cash reserve)
  2. Enforces stop-loss / take-profit on every open position
  3. Generates signals for every watchlist ticker
  4. Executes BUY / SELL market orders via Alpaca Paper Trading
  5. Persists a portfolio snapshot to SQLite
"""

import logging
import os
import threading
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

import database
import risk
import strategies

load_dotenv()

logger        = logging.getLogger(__name__)
_bot_thread   = None
_stop_event   = threading.Event()
_activity_log = []   # in-memory ring buffer shown on the dashboard


# ── Alpaca client ─────────────────────────────────────────────────────────────

def _make_client() -> TradingClient:
    api_key    = os.getenv('ALPACA_API_KEY',    '').strip()
    secret_key = os.getenv('ALPACA_SECRET_KEY', '').strip()
    if not api_key or not secret_key:
        raise ValueError('ALPACA_API_KEY and ALPACA_SECRET_KEY are not set in .env')
    return TradingClient(api_key, secret_key, paper=True)


# ── Activity log ──────────────────────────────────────────────────────────────

def _log(msg: str):
    ts    = datetime.utcnow().strftime('%H:%M:%S UTC')
    entry = f'[{ts}] {msg}'
    _activity_log.insert(0, entry)
    if len(_activity_log) > 200:
        _activity_log.pop()
    logger.info(msg)


def get_activity_log() -> list[str]:
    return _activity_log[:50]


# ── Portfolio snapshot ────────────────────────────────────────────────────────

def get_portfolio_summary() -> dict:
    """Fetches live account data from Alpaca and returns a dashboard-ready dict."""
    try:
        client   = _make_client()
        account  = client.get_account()
        all_pos  = client.get_all_positions()

        port_val = float(account.portfolio_value)
        cash     = float(account.cash)
        equity   = float(account.equity)

        positions = []
        for pos in all_pos:
            entry   = float(pos.avg_entry_price)
            current = float(pos.current_price) if pos.current_price else entry
            pnl     = float(pos.unrealized_pl)     if pos.unrealized_pl     else 0.0
            pnl_pct = float(pos.unrealized_plpc)   if pos.unrealized_plpc   else 0.0
            positions.append({
                'ticker':        pos.symbol,
                'shares':        float(pos.qty),
                'entry_price':   round(entry,   2),
                'current_price': round(current, 2),
                'pnl':           round(pnl,     2),
                'pnl_pct':       round(pnl_pct * 100, 2),
                'stop_loss':     risk.calculate_stop_loss(entry),
                'take_profit':   risk.calculate_take_profit(entry),
            })

        state         = database.get_bot_state()
        initial_val   = state['initial_value'] if state else 100_000
        total_return  = (port_val - initial_val) / initial_val * 100 if initial_val else 0

        return {
            'portfolio_value':  round(port_val, 2),
            'cash':             round(cash,     2),
            'equity':           round(equity,   2),
            'total_return':     round(total_return, 2),
            'positions':        positions,
            'active_positions': len(positions),
        }

    except ValueError as exc:
        return {'error': str(exc), 'positions': [], 'active_positions': 0}
    except Exception as exc:
        logger.error('portfolio fetch error: %s', exc)
        return {'error': str(exc), 'positions': [], 'active_positions': 0}


# ── Order execution ───────────────────────────────────────────────────────────

def _buy(client: TradingClient, ticker: str, shares: int,
         price: float, strategy: str) -> bool:
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares,
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
        risk.increment_trade_count()
        database.log_trade(ticker, 'BUY', shares, price, strategy,
                           order_id=str(order.id))
        _log(f'BUY  {shares:>4} {ticker:<5} @ ${price:>9.2f}  [{strategy}]')
        return True
    except Exception as exc:
        _log(f'BUY ERROR {ticker}: {exc}')
        return False


def _sell(client: TradingClient, ticker: str, shares: float,
          price: float, entry: float, strategy: str, reason: str) -> bool:
    try:
        order   = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares,
            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        ))
        risk.increment_trade_count()
        pnl     = (price - entry) * shares
        pnl_pct = (price - entry) / entry * 100
        database.log_trade(ticker, 'SELL', shares, price, strategy,
                           order_id=str(order.id),
                           entry_price=entry,
                           pnl=round(pnl, 2),
                           pnl_pct=round(pnl_pct, 2))
        _log(f'SELL {shares:>4} {ticker:<5} @ ${price:>9.2f}  PnL ${pnl:>+9.2f} ({pnl_pct:>+.1f}%)  [{reason}]')
        return True
    except Exception as exc:
        _log(f'SELL ERROR {ticker}: {exc}')
        return False


# ── Trading cycle ─────────────────────────────────────────────────────────────

def _trading_cycle(strategy_name: str):
    try:
        client   = _make_client()
        account  = client.get_account()
        port_val = float(account.portfolio_value)
        cash     = float(account.cash)

        # Global risk gate
        ok, reason = risk.can_trade(port_val, cash)
        if not ok:
            _log(f'Skipping cycle — {reason}')
            database.log_portfolio_snapshot(
                port_val, cash, float(account.equity), strategy_name)
            return

        positions = {p.symbol: p for p in client.get_all_positions()}

        # ── Stop-loss / take-profit sweep ─────────────────────────────────
        for ticker, pos in list(positions.items()):
            current = float(pos.current_price) if pos.current_price else float(pos.avg_entry_price)
            entry   = float(pos.avg_entry_price)
            trigger = risk.check_stop_take(current, entry)
            if trigger:
                _sell(client, ticker, float(pos.qty),
                      current, entry, strategy_name, reason=trigger)

        # Refresh after sells
        positions = {p.symbol: p for p in client.get_all_positions()}
        account   = client.get_account()
        port_val  = float(account.portfolio_value)
        cash      = float(account.cash)

        # ── Signal-driven trading ─────────────────────────────────────────
        for ticker in strategies.WATCHLIST:
            if _stop_event.is_set():
                break

            signal, price = strategies.get_signal(strategy_name, ticker)
            if signal is None or price is None:
                continue

            if signal == 'BUY' and ticker not in positions:
                shares = risk.calculate_position_size(port_val, price, cash)
                if shares > 0:
                    ok, _ = risk.can_trade(port_val, cash)
                    if ok:
                        _buy(client, ticker, shares, price, strategy_name)
                        cash -= shares * price   # optimistic local update

            elif signal == 'SELL' and ticker in positions:
                pos   = positions[ticker]
                entry = float(pos.avg_entry_price)
                curr  = float(pos.current_price) if pos.current_price else price
                _sell(client, ticker, float(pos.qty),
                      curr, entry, strategy_name, reason='signal')

        # Snapshot after cycle
        account = client.get_account()
        database.log_portfolio_snapshot(
            float(account.portfolio_value),
            float(account.cash),
            float(account.equity),
            strategy_name,
        )

    except Exception as exc:
        _log(f'Cycle error: {exc}')
        logger.exception('Unhandled error in trading cycle')


# ── Bot loop ──────────────────────────────────────────────────────────────────

def _bot_loop():
    _log('Bot started')
    while not _stop_event.is_set():
        state    = database.get_bot_state()
        strategy = state['strategy'] if state else 'ma_crossover'
        _trading_cycle(strategy)
        _stop_event.wait(300)   # sleep 5 minutes between cycles
    _log('Bot stopped')


# ── Public controls ───────────────────────────────────────────────────────────

def start_bot() -> tuple[bool, str]:
    global _bot_thread
    if _bot_thread and _bot_thread.is_alive():
        return False, 'Bot is already running'

    _stop_event.clear()
    _bot_thread = threading.Thread(target=_bot_loop, daemon=True, name='tradebot')
    _bot_thread.start()
    database.update_bot_state(is_running=True, started_at=datetime.utcnow().isoformat())

    # Record starting portfolio value for return calculation
    try:
        client  = _make_client()
        account = client.get_account()
        database.update_bot_state(initial_value=float(account.portfolio_value))
    except Exception:
        pass

    return True, 'Bot started'


def stop_bot() -> tuple[bool, str]:
    _stop_event.set()
    database.update_bot_state(is_running=False)
    return True, 'Stop signal sent'


def is_running() -> bool:
    return _bot_thread is not None and _bot_thread.is_alive()
