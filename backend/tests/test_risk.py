"""
Unit tests for risk.py.

All tests are pure Python — no network, no Alpaca, no database.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import risk


class TestStopTakeCalculations:
    def test_stop_loss_is_five_percent_below_entry(self):
        assert risk.calculate_stop_loss(100.0) == 95.0

    def test_take_profit_is_fifteen_percent_above_entry(self):
        assert risk.calculate_take_profit(100.0) == 115.0

    def test_stop_loss_rounds_to_two_decimals(self):
        result = risk.calculate_stop_loss(99.99)
        assert result == round(99.99 * 0.95, 2)

    def test_take_profit_rounds_to_two_decimals(self):
        result = risk.calculate_take_profit(99.99)
        assert result == round(99.99 * 1.15, 2)


class TestCheckStopTake:
    def test_triggers_stop_loss_when_at_threshold(self):
        assert risk.check_stop_take(95.0, 100.0) == 'stop_loss'

    def test_triggers_stop_loss_when_below_threshold(self):
        assert risk.check_stop_take(90.0, 100.0) == 'stop_loss'

    def test_triggers_take_profit_when_at_threshold(self):
        assert risk.check_stop_take(115.0, 100.0) == 'take_profit'

    def test_triggers_take_profit_when_above_threshold(self):
        assert risk.check_stop_take(120.0, 100.0) == 'take_profit'

    def test_returns_none_when_between_levels(self):
        assert risk.check_stop_take(100.0, 100.0) is None
        assert risk.check_stop_take(105.0, 100.0) is None
        assert risk.check_stop_take(95.01, 100.0) is None

    def test_trailing_stop_triggers_from_high_not_entry(self):
        # Entry 100, price ran to 120, then dropped 5% from high → fires at 114
        assert risk.check_stop_take(114.0, 100.0, high_since_entry=120.0) == 'stop_loss'

    def test_trailing_stop_does_not_trigger_above_trail_level(self):
        # Price rose to 110, current is 106 — only 3.6% below high, within 5% trail
        assert risk.check_stop_take(106.0, 100.0, high_since_entry=110.0) is None

    def test_trailing_stop_matches_fixed_stop_at_entry(self):
        # When high equals entry the trailing stop is identical to the old fixed stop
        assert risk.check_stop_take(95.0, 100.0, high_since_entry=100.0) == 'stop_loss'

    def test_trailing_stop_level_displayed_correctly(self):
        # calculate_stop_loss should use high_price when provided
        assert risk.calculate_stop_loss(100.0, high_price=120.0) == round(120.0 * 0.95, 2)

    def test_trailing_stop_level_falls_back_to_entry(self):
        # Without high_price, falls back to entry-based stop
        assert risk.calculate_stop_loss(100.0) == 95.0


class TestPositionSizing:
    def test_respects_max_position_pct(self):
        # Portfolio $100k, price $100, cash $100k → max spend 10% = $10k → 100 shares
        shares = risk.calculate_position_size(100_000, 100.0, 100_000)
        assert shares == 100

    def test_respects_cash_reserve(self):
        # Portfolio $100k, price $100, cash just at reserve ($20k) → 0 shares
        shares = risk.calculate_position_size(100_000, 100.0, 20_000)
        assert shares == 0

    def test_returns_zero_when_price_is_zero(self):
        assert risk.calculate_position_size(100_000, 0.0, 50_000) == 0

    def test_returns_zero_when_portfolio_is_zero(self):
        assert risk.calculate_position_size(0.0, 100.0, 0.0) == 0

    def test_returns_whole_shares_only(self):
        # Should truncate, not round
        shares = risk.calculate_position_size(100_000, 333.0, 100_000)
        assert isinstance(shares, int)
        assert shares == int(100_000 * 0.10 / 333.0)


class TestDailyTradeLimit:
    def test_limit_not_reached_initially(self):
        # Reset state before test
        risk._daily_trade_count = 0
        risk._last_trade_date = None
        assert risk.check_daily_trade_limit() is True

    def test_limit_reached_after_max_trades(self):
        risk._daily_trade_count = 0
        risk._last_trade_date = None
        for _ in range(risk.MAX_DAILY_TRADES):
            risk.increment_trade_count()
        assert risk.check_daily_trade_limit() is False

    def test_count_increments_correctly(self):
        risk._daily_trade_count = 0
        risk._last_trade_date = None
        risk.increment_trade_count()
        risk.increment_trade_count()
        assert risk.get_daily_trade_count() == 2


class TestCanTrade:
    def test_rejects_when_market_closed(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: False)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda: True)
        ok, reason = risk.can_trade(100_000, 50_000)
        assert ok is False
        assert 'closed' in reason.lower()

    def test_rejects_when_daily_limit_hit(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: True)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda *a, **kw: False)
        ok, reason = risk.can_trade(100_000, 50_000)
        assert ok is False
        assert 'limit' in reason.lower()

    def test_rejects_when_cash_below_reserve(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: True)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda *a, **kw: True)
        # Cash is 10% of portfolio — below the 20% reserve
        ok, reason = risk.can_trade(100_000, 10_000)
        assert ok is False
        assert 'reserve' in reason.lower()

    def test_allows_trade_when_all_gates_pass(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: True)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda *a, **kw: True)
        ok, reason = risk.can_trade(100_000, 50_000)
        assert ok is True
        assert reason == 'OK'
