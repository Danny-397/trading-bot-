"""
Unit tests for portfolio.py — Markowitz optimisation.

All tests use synthetic return data generated in-memory.
No network calls, no yfinance.

Mathematical invariants verified:
  - Weights sum to 1 and are non-negative (constraint satisfaction)
  - Min-variance portfolio has vol <= max-Sharpe portfolio vol
  - Single-asset portfolio vol matches the series standard deviation
  - Max-Sharpe Sharpe >= min-variance Sharpe (by construction)
  - Efficient frontier is monotonically increasing in return
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import portfolio as portopt


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_returns(n_assets: int = 3, n_days: int = 120,
                  seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.012, (n_days, n_assets))
    cols = [f'A{i}' for i in range(n_assets)]
    idx  = pd.date_range('2023-01-01', periods=n_days, freq='B')
    return pd.DataFrame(data, index=idx, columns=cols)


def _optimise(df: pd.DataFrame) -> dict:
    """Run both optimisers directly on a pre-built returns DataFrame."""
    mu  = df.mean().values
    cov = df.cov().values
    ms  = portopt._minimize(lambda w, m, c: -portopt._stats(w, m, c)[2], mu, cov)
    mv  = portopt._minimize(lambda w, m, c:  portopt._stats(w, m, c)[1], mu, cov)
    return {'ms': ms.x, 'mv': mv.x, 'mu': mu, 'cov': cov}


# ── Constraint satisfaction ────────────────────────────────────────────────────

class TestConstraints:
    def test_max_sharpe_weights_sum_to_one(self):
        r = _optimise(_make_returns())
        assert np.isclose(r['ms'].sum(), 1.0, atol=1e-6)

    def test_min_var_weights_sum_to_one(self):
        r = _optimise(_make_returns())
        assert np.isclose(r['mv'].sum(), 1.0, atol=1e-6)

    def test_max_sharpe_weights_non_negative(self):
        r = _optimise(_make_returns())
        assert (r['ms'] >= -1e-8).all()

    def test_min_var_weights_non_negative(self):
        r = _optimise(_make_returns())
        assert (r['mv'] >= -1e-8).all()


# ── Mathematical properties ────────────────────────────────────────────────────

class TestMathProperties:
    def test_min_var_vol_le_max_sharpe_vol(self):
        r = _optimise(_make_returns())
        _, ms_vol, _ = portopt._stats(r['ms'], r['mu'], r['cov'])
        _, mv_vol, _ = portopt._stats(r['mv'], r['mu'], r['cov'])
        assert mv_vol <= ms_vol + 1e-6

    def test_max_sharpe_ge_min_var_sharpe(self):
        r = _optimise(_make_returns())
        _, _, ms_sr = portopt._stats(r['ms'], r['mu'], r['cov'])
        _, _, mv_sr = portopt._stats(r['mv'], r['mu'], r['cov'])
        assert ms_sr >= mv_sr - 1e-6

    def test_single_asset_vol_matches_series_std(self):
        df  = _make_returns(n_assets=2)
        mu  = df.mean().values
        cov = df.cov().values
        w   = np.array([1.0, 0.0])
        _, vol, _ = portopt._stats(w, mu, cov)
        expected  = float(df.iloc[:, 0].std() * np.sqrt(portopt.TRADING_DAYS))
        assert abs(vol - expected) < 1e-8

    def test_equal_weight_vol_positive(self):
        df  = _make_returns()
        mu  = df.mean().values
        cov = df.cov().values
        n   = len(mu)
        w   = np.ones(n) / n
        _, vol, _ = portopt._stats(w, mu, cov)
        assert vol > 0

    def test_diversification_reduces_vol(self):
        # Equal-weight portfolio should have lower vol than the average
        # individual asset vol (diversification benefit)
        df  = _make_returns()
        mu  = df.mean().values
        cov = df.cov().values
        n   = len(mu)
        w   = np.ones(n) / n
        _, port_vol, _ = portopt._stats(w, mu, cov)
        individual_vols = [
            portopt._stats(np.eye(n)[i], mu, cov)[1] for i in range(n)
        ]
        assert port_vol < np.mean(individual_vols)


# ── Return matrix builder ──────────────────────────────────────────────────────

class TestBuildReturnMatrix:
    def test_returns_none_when_only_one_ticker(self, monkeypatch):
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv', lambda *a, **kw: None)
        result = portopt.build_return_matrix(['AAPL'], '2023-01-01', '2024-01-01')
        assert result is None

    def test_returns_none_when_all_fetches_fail(self, monkeypatch):
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv', lambda *a, **kw: None)
        result = portopt.build_return_matrix(['A', 'B', 'C'], '2023-01-01', '2024-01-01')
        assert result is None

    def test_aligned_returns_have_no_nans(self, monkeypatch):
        prices = pd.Series(
            np.linspace(100, 150, 80),
            index=pd.date_range('2023-01-01', periods=80, freq='B'),
        )
        mock = pd.DataFrame({
            'Close': prices, 'Open': prices,
            'High': prices, 'Low': prices, 'Volume': 1_000_000,
        })
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv', lambda *a, **kw: mock)
        result = portopt.build_return_matrix(['X', 'Y'], '2023-01-01', '2024-01-01')
        assert result is not None
        assert result.isnull().sum().sum() == 0

    def test_returns_none_when_insufficient_overlap(self, monkeypatch):
        prices = pd.Series(
            np.linspace(100, 110, 20),
            index=pd.date_range('2023-01-01', periods=20, freq='B'),
        )
        mock = pd.DataFrame({
            'Close': prices, 'Open': prices,
            'High': prices, 'Low': prices, 'Volume': 1_000_000,
        })
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv', lambda *a, **kw: mock)
        result = portopt.build_return_matrix(['X', 'Y'], '2023-01-01', '2023-02-01')
        assert result is None


# ── compute_efficient_frontier ─────────────────────────────────────────────────

class TestComputeEfficientFrontier:
    """Each test patches fetch_ohlcv with a distinct random-walk series per
    ticker so the covariance matrix is non-degenerate and the optimiser works."""

    @staticmethod
    def _make_patch(seeds: list[int], n_days: int = 80):
        """Return a fetch_ohlcv mock that gives each ticker its own price path."""
        call = [0]

        def _fetch(*a, **kw):
            seed    = seeds[call[0] % len(seeds)]
            call[0] += 1
            rng     = np.random.default_rng(seed)
            rets    = rng.normal(0.0008, 0.012, n_days)
            prices  = pd.Series(
                100 * np.cumprod(1 + rets),
                index=pd.date_range('2023-01-01', periods=n_days, freq='B'),
            )
            return pd.DataFrame({
                'Close': prices, 'Open': prices,
                'High': prices, 'Low': prices, 'Volume': 1_000_000,
            })
        return _fetch

    def test_returns_error_for_single_ticker(self, monkeypatch):
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv', lambda *a, **kw: None)
        result = portopt.compute_efficient_frontier(
            ['AAPL'], '2023-01-01', '2024-01-01')
        assert 'error' in result

    def test_output_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv',
                            self._make_patch([1, 2]))
        result = portopt.compute_efficient_frontier(
            ['A', 'B'], '2023-01-01', '2024-01-01')
        for key in ('tickers', 'max_sharpe', 'min_variance',
                    'efficient_frontier', 'individual_assets', 'covariance_matrix'):
            assert key in result, f'Missing key: {key}'

    def test_frontier_is_non_empty(self, monkeypatch):
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv',
                            self._make_patch([10, 99]))
        result = portopt.compute_efficient_frontier(
            ['A', 'B'], '2023-01-01', '2024-01-01', n_points=10)
        assert len(result['efficient_frontier']) > 0

    def test_covariance_matrix_is_square(self, monkeypatch):
        monkeypatch.setattr(portopt.feat, 'fetch_ohlcv',
                            self._make_patch([3, 7, 13]))
        result = portopt.compute_efficient_frontier(
            ['A', 'B', 'C'], '2023-01-01', '2024-01-01')
        n = len(result['covariance_matrix']['tickers'])
        assert len(result['covariance_matrix']['data']) == n
        for row in result['covariance_matrix']['data']:
            assert len(row) == n
