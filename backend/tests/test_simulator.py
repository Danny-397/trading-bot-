"""
Unit tests for simulator.py — internal paper-trading engine.

Uses an in-memory SQLite database for every test so nothing
touches the real tradebot.db file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import simulator
import database


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Point the database module at a temp file for every test."""
    db_path = str(tmp_path / 'test.db')
    monkeypatch.setattr(database, 'DB_PATH', db_path)
    database.init_db()
    simulator.init_simulator()
    yield


@pytest.fixture
def mock_price(monkeypatch):
    """Patch _latest_price so tests never hit the network."""
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'SPY': 500.0}

    def _price(ticker):
        return prices.get(ticker, 100.0)

    monkeypatch.setattr(simulator, '_latest_price', _price)
    return prices


# ── init_simulator ─────────────────────────────────────────────────────────────

class TestInitSimulator:
    def test_account_seeded_with_initial_value(self):
        account = simulator.get_account()
        assert account.portfolio_value == 100_000.0

    def test_no_positions_on_fresh_start(self):
        assert simulator.get_all_positions() == []

    def test_init_is_idempotent(self):
        simulator.init_simulator()
        simulator.init_simulator()
        account = simulator.get_account()
        assert account.cash == 100_000.0


# ── submit_buy ─────────────────────────────────────────────────────────────────

class TestSubmitBuy:
    def test_buy_deducts_cash(self, mock_price):
        simulator.submit_buy('AAPL', 10, 150.0)
        assert simulator._cash() == pytest.approx(100_000.0 - 1_500.0)

    def test_buy_creates_position(self, mock_price):
        simulator.submit_buy('AAPL', 10, 150.0)
        positions = simulator.get_all_positions()
        assert len(positions) == 1
        assert positions[0].symbol == 'AAPL'
        assert positions[0].qty == 10

    def test_buy_records_entry_price(self, mock_price):
        simulator.submit_buy('AAPL', 5, 150.0)
        pos = simulator.get_all_positions()[0]
        assert pos.avg_entry_price == 150.0

    def test_buy_raises_on_insufficient_cash(self):
        with pytest.raises(ValueError, match='Insufficient cash'):
            simulator.submit_buy('AAPL', 10_000, 150.0)

    def test_buy_returns_order_with_id(self):
        order = simulator.submit_buy('AAPL', 1, 100.0)
        assert order.id is not None
        assert len(order.id) > 0

    def test_two_buys_same_ticker_averages_entry(self):
        simulator.submit_buy('AAPL', 10, 100.0)
        simulator.submit_buy('AAPL', 10, 200.0)
        pos = simulator.get_all_positions()[0]
        assert pos.qty == 20
        assert pos.avg_entry_price == pytest.approx(150.0)


# ── submit_sell ────────────────────────────────────────────────────────────────

class TestSubmitSell:
    def test_sell_adds_proceeds_to_cash(self, mock_price):
        simulator.submit_buy('AAPL', 10, 100.0)
        simulator.submit_sell('AAPL', 10, 150.0)
        assert simulator._cash() == pytest.approx(100_000.0 - 1_000.0 + 1_500.0)

    def test_sell_removes_position(self, mock_price):
        simulator.submit_buy('AAPL', 10, 100.0)
        simulator.submit_sell('AAPL', 10, 150.0)
        assert simulator.get_all_positions() == []

    def test_sell_raises_when_no_position(self):
        with pytest.raises(ValueError, match='No open position'):
            simulator.submit_sell('AAPL', 10, 150.0)


# ── get_account ────────────────────────────────────────────────────────────────

class TestGetAccount:
    def test_portfolio_value_includes_position_value(self, mock_price):
        # Buy 10 AAPL at 100, current price is 150 → equity = 1500
        simulator.submit_buy('AAPL', 10, 100.0)
        account = simulator.get_account()
        assert account.equity == pytest.approx(1_500.0)
        assert account.portfolio_value == pytest.approx(100_000.0 - 1_000.0 + 1_500.0)

    def test_cash_only_when_no_positions(self):
        account = simulator.get_account()
        assert account.portfolio_value == account.cash
        assert account.equity == 0.0

    def test_attributes_match_expected_names(self):
        account = simulator.get_account()
        assert hasattr(account, 'portfolio_value')
        assert hasattr(account, 'cash')
        assert hasattr(account, 'equity')


# ── get_all_positions ──────────────────────────────────────────────────────────

class TestGetAllPositions:
    def test_unrealised_pnl_positive_when_price_up(self, mock_price):
        simulator.submit_buy('AAPL', 10, 100.0)   # entry 100, current 150
        pos = simulator.get_all_positions()[0]
        assert pos.unrealized_pl == pytest.approx(500.0)
        assert pos.unrealized_plpc == pytest.approx(0.5)

    def test_unrealised_pnl_negative_when_price_down(self, mock_price):
        simulator.submit_buy('MSFT', 10, 400.0)   # entry 400, current 300
        pos = simulator.get_all_positions()[0]
        assert pos.unrealized_pl == pytest.approx(-1_000.0)

    def test_position_attributes_match_expected_names(self, mock_price):
        simulator.submit_buy('AAPL', 5, 100.0)
        pos = simulator.get_all_positions()[0]
        assert hasattr(pos, 'symbol')
        assert hasattr(pos, 'qty')
        assert hasattr(pos, 'avg_entry_price')
        assert hasattr(pos, 'current_price')
        assert hasattr(pos, 'unrealized_pl')
        assert hasattr(pos, 'unrealized_plpc')


# ── reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_positions(self):
        simulator.submit_buy('AAPL', 10, 100.0)
        simulator.reset(50_000.0)
        assert simulator.get_all_positions() == []

    def test_reset_sets_cash_to_new_capital(self):
        simulator.reset(50_000.0)
        assert simulator._cash() == 50_000.0
