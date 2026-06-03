from datetime import datetime
import pytz

# ── Risk constants ────────────────────────────────────────────────────────────
STOP_LOSS_PCT       = 0.05   # 5 %  — exit if position drops this much
TAKE_PROFIT_PCT     = 0.15   # 15 % — exit if position gains this much
MAX_POSITION_PCT    = 0.10   # 10 % — max portfolio allocation per stock
MAX_DAILY_TRADES    = 10     # hard cap on orders per calendar day
MIN_CASH_RESERVE    = 0.20   # 20 % — cash that must always be kept free
MARKET_OPEN_H       = 9
MARKET_OPEN_M       = 30
MARKET_CLOSE_H      = 16
MARKET_CLOSE_M      = 0

_daily_trade_count = 0
_last_trade_date   = None


def is_market_open() -> bool:
    est  = pytz.timezone('America/New_York')
    now  = datetime.now(est)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    open_  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_ = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_ <= now < close_


def _reset_if_new_day():
    global _daily_trade_count, _last_trade_date
    today = datetime.now().date()
    if _last_trade_date != today:
        _daily_trade_count = 0
        _last_trade_date   = today


def check_daily_trade_limit() -> bool:
    _reset_if_new_day()
    return _daily_trade_count < MAX_DAILY_TRADES


def increment_trade_count():
    _reset_if_new_day()
    global _daily_trade_count
    _daily_trade_count += 1


def get_daily_trade_count() -> int:
    _reset_if_new_day()
    return _daily_trade_count


def calculate_position_size(portfolio_value: float, price: float, cash: float) -> int:
    """Returns number of whole shares to buy, respecting all size limits."""
    max_by_pct  = portfolio_value * MAX_POSITION_PCT
    usable_cash = cash - (portfolio_value * MIN_CASH_RESERVE)
    if usable_cash <= 0 or price <= 0:
        return 0
    spend  = min(max_by_pct, usable_cash)
    return max(int(spend / price), 0)


def calculate_stop_loss(entry_price: float) -> float:
    return round(entry_price * (1 - STOP_LOSS_PCT), 2)


def calculate_take_profit(entry_price: float) -> float:
    return round(entry_price * (1 + TAKE_PROFIT_PCT), 2)


def check_stop_take(current_price: float, entry_price: float):
    """Returns 'stop_loss', 'take_profit', or None."""
    if current_price <= entry_price * (1 - STOP_LOSS_PCT):
        return 'stop_loss'
    if current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
        return 'take_profit'
    return None


def can_trade(portfolio_value: float, cash: float) -> tuple[bool, str]:
    """Master gate — returns (allowed, reason_string)."""
    if not is_market_open():
        return False, 'Market is closed'
    if not check_daily_trade_limit():
        return False, f'Daily trade limit ({MAX_DAILY_TRADES}) reached'
    if portfolio_value > 0 and (cash / portfolio_value) < MIN_CASH_RESERVE:
        return False, f'Cash below {int(MIN_CASH_RESERVE*100)}% reserve minimum'
    return True, 'OK'
