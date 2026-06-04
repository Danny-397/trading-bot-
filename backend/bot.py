"""
Core trading bot.

Every 5-minute cycle during market hours:
  1. Detects the current market regime from SPY (broad market proxy)
  2. Selects the optimal strategy for that regime + the user's risk tolerance
  3. Checks risk gates (market open, daily trade limit, cash reserve)
  4. Enforces stop-loss / take-profit on every open position
  5. Generates signals for every watchlist ticker using the selected strategy
  6. Executes BUY / SELL market orders via Alpaca Paper Trading
  7. Persists a portfolio snapshot to SQLite
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
import features
import regime as reg
import risk
import strategies

load_dotenv()

logger        = logging.getLogger(__name__)
_bot_thread   = None
_stop_event   = threading.Event()
_activity_log = []

# Cache last detected regime so the dashboard can read it between cycles
_last_regime: reg.RegimeResult | None = None


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


def get_last_regime() -> dict | None:
    if _last_regime is None:
        return None
    return {
        'regime':      _last_regime.regime,
        'label':       _last_regime.label,
        'description': _last_regime.description,
        'strategy':    _last_regime.strategy,
        'adx':         _last_regime.adx,
        'plus_di':     _last_regime.plus_di,
        'minus_di':    _last_regime.minus_di,
        'bb_width':    _last_regime.bb_width,
        'vol_30d':     _last_regime.vol_30d,
    }


# ── Regime detection (using SPY as market proxy) ──────────────────────────────

def _detect_market_regime() -> reg.RegimeResult | None:
    global _last_regime
    try:
        spy_df = features.fetch_ohlcv('SPY', period='6mo')
        if spy_df is not None and len(spy_df) >= 60:
            result = reg.detect_regime(spy_df)
            _last_regime = result
            return result
    except Exception as exc:
        logger.error('Regime detection error: %s', exc)
    return _last_regime  # return cached if detection fails


# ── Portfolio snapshot ────────────────────────────────────────────────────────

def get_portfolio_summary() -> dict:
    try:
        client  = _make_client()
        account = client.get_account()
        all_pos = client.get_all_positions()

        port_val = float(account.portfolio_value)
        cash     = float(account.cash)
        equity   = float(account.equity)

        state       = database.get_bot_state()
        risk_tol    = state.get('risk_tolerance', 'moderate') if state else 'moderate'
        profile     = risk.get_risk_profile(risk_tol)

        positions = []
        for pos in all_pos:
            entry   = float(pos.avg_entry_price)
            current = float(pos.current_price) if pos.current_price else entry
            pnl     = float(pos.unrealized_pl)   if pos.unrealized_pl   else 0.0
            pnl_pct = float(pos.unrealized_plpc) if pos.unrealized_plpc else 0.0
            positions.append({
                'ticker':        pos.symbol,
                'shares':        float(pos.qty),
                'entry_price':   round(entry,   2),
                'current_price': round(current, 2),
                'pnl':           round(pnl,     2),
                'pnl_pct':       round(pnl_pct * 100, 2),
                'stop_loss':     risk.calculate_stop_loss(entry,   profile),
                'take_profit':   risk.calculate_take_profit(entry, profile),
            })

        initial_val  = state['initial_value'] if state else 100_000
        total_return = (port_val - initial_val) / initial_val * 100 if initial_val else 0

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

def _buy(client, ticker, shares, price, strategy_name, regime_name, profile):
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares,
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
        risk.increment_trade_count()
        database.log_trade(ticker, 'BUY', shares, price, strategy_name,
                           order_id=str(order.id), regime=regime_name)
        _log(f'BUY  {shares:>4} {ticker:<5} @ ${price:>9.2f}  '
             f'[{strategy_name}] [{regime_name}]')
        return True
    except Exception as exc:
        _log(f'BUY ERROR {ticker}: {exc}')
        return False


def _sell(client, ticker, shares, price, entry, strategy_name, regime_name, reason, profile):
    try:
        order   = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares,
            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        ))
        risk.increment_trade_count()
        pnl     = (price - entry) * shares
        pnl_pct = (price - entry) / entry * 100
        database.log_trade(ticker, 'SELL', shares, price, strategy_name,
                           order_id=str(order.id),
                           entry_price=entry,
                           pnl=round(pnl, 2),
                           pnl_pct=round(pnl_pct, 2),
                           regime=regime_name)
        _log(f'SELL {shares:>4} {ticker:<5} @ ${price:>9.2f}  '
             f'PnL ${pnl:>+9.2f} ({pnl_pct:>+.1f}%)  [{reason}]')
        return True
    except Exception as exc:
        _log(f'SELL ERROR {ticker}: {exc}')
        return False


# ── Trading cycle ─────────────────────────────────────────────────────────────

def _trading_cycle(configured_strategy: str):
    try:
        client   = _make_client()
        account  = client.get_account()
        port_val = float(account.portfolio_value)
        cash     = float(account.cash)

        # ── Regime detection ──────────────────────────────────────────────
        state       = database.get_bot_state()
        risk_tol    = state.get('risk_tolerance', 'moderate') if state else 'moderate'
        profile     = risk.get_risk_profile(risk_tol)

        regime_result = _detect_market_regime()
        regime_name   = regime_result.regime if regime_result else 'RANGING'

        # Adaptive mode: let regime pick the strategy
        if configured_strategy == 'adaptive':
            active_strategy = reg.get_regime_strategy(regime_name, risk_tol)
            _log(f'Regime: {regime_name} | Risk: {risk_tol} → Strategy: {active_strategy}')
        else:
            active_strategy = configured_strategy

        # Conservative profile: sit out high-vol entirely
        if regime_name == 'HIGH_VOLATILITY' and not profile['trade_high_vol']:
            _log(f'Skipping cycle — HIGH_VOLATILITY regime, {risk_tol} profile avoids it')
            database.log_portfolio_snapshot(
                port_val, cash, float(account.equity), active_strategy)
            return

        # Position size multiplier for HIGH_VOLATILITY
        size_mult = profile['vol_size_mult'] if regime_name == 'HIGH_VOLATILITY' else 1.0

        # ── Global risk gate ──────────────────────────────────────────────
        ok, reason = risk.can_trade(port_val, cash, profile)
        if not ok:
            _log(f'Skipping cycle — {reason}')
            database.log_portfolio_snapshot(
                port_val, cash, float(account.equity), active_strategy)
            return

        # ── Stop-loss / take-profit sweep ─────────────────────────────────
        positions = {p.symbol: p for p in client.get_all_positions()}
        for ticker, pos in list(positions.items()):
            current = float(pos.current_price) if pos.current_price else float(pos.avg_entry_price)
            entry   = float(pos.avg_entry_price)
            trigger = risk.check_stop_take(current, entry, profile)
            if trigger:
                _sell(client, ticker, float(pos.qty),
                      current, entry, active_strategy, regime_name, trigger, profile)

        # Refresh after sells
        positions = {p.symbol: p for p in client.get_all_positions()}
        account   = client.get_account()
        port_val  = float(account.portfolio_value)
        cash      = float(account.cash)

        # ── Signal-driven trading ─────────────────────────────────────────
        for ticker in strategies.WATCHLIST:
            if _stop_event.is_set():
                break

            signal, price = strategies.get_signal(active_strategy, ticker)
            if signal is None or price is None:
                continue

            if signal == 'BUY' and ticker not in positions:
                # Kelly Criterion sizing — falls back to fixed sizing if
                # fewer than 10 closed trades exist in the database
                kelly = database.compute_kelly_fraction(active_strategy)
                base_shares = risk.calculate_position_size_kelly(
                    port_val, price, cash, kelly, profile)
                shares = max(int(base_shares * size_mult), 0)
                if shares > 0:
                    ok, _ = risk.can_trade(port_val, cash, profile)
                    if ok:
                        _buy(client, ticker, shares, price,
                             active_strategy, regime_name, profile)
                        cash -= shares * price

            elif signal == 'SELL' and ticker in positions:
                pos   = positions[ticker]
                entry = float(pos.avg_entry_price)
                curr  = float(pos.current_price) if pos.current_price else price
                _sell(client, ticker, float(pos.qty),
                      curr, entry, active_strategy, regime_name, 'signal', profile)

        # Snapshot
        account = client.get_account()
        database.log_portfolio_snapshot(
            float(account.portfolio_value),
            float(account.cash),
            float(account.equity),
            active_strategy,
        )

    except Exception as exc:
        _log(f'Cycle error: {exc}')
        logger.exception('Unhandled error in trading cycle')


# ── Bot loop ──────────────────────────────────────────────────────────────────

def _bot_loop():
    _log('Bot started')
    while not _stop_event.is_set():
        state    = database.get_bot_state()
        strategy = state['strategy'] if state else 'adaptive'
        _trading_cycle(strategy)
        _stop_event.wait(300)
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
