"""
Flask REST API for TradeBot.

All endpoints return JSON.  The frontend polls these on a 10-second interval.
"""

import logging
import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

import backtest as backtester
import bot
import database
import risk
import strategies

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)-20s  %(levelname)s  %(message)s',
)

app = Flask(__name__)
CORS(app)
database.init_db()


# ── Status / control ──────────────────────────────────────────────────────────

@app.route('/api/status')
def get_status():
    state    = database.get_bot_state()
    strategy = state['strategy'] if state else 'ma_crossover'
    portfolio = bot.get_portfolio_summary()
    metrics  = database.get_performance_metrics(strategy)

    return jsonify({
        'is_running':   bot.is_running(),
        'strategy':     strategy,
        'started_at':   state.get('started_at') if state else None,
        'market_open':  risk.is_market_open(),
        'daily_trades': risk.get_daily_trade_count(),
        'max_daily':    risk.MAX_DAILY_TRADES,
        'portfolio':    portfolio,
        'metrics':      metrics,
    })


@app.route('/api/start', methods=['POST'])
def start():
    data     = request.get_json() or {}
    strategy = data.get('strategy', 'ma_crossover')
    if strategy not in ('ma_crossover', 'rsi', 'macd'):
        return jsonify({'error': 'Invalid strategy'}), 400
    database.update_bot_state(strategy=strategy)
    success, msg = bot.start_bot()
    return jsonify({'success': success, 'message': msg})


@app.route('/api/stop', methods=['POST'])
def stop():
    success, msg = bot.stop_bot()
    return jsonify({'success': success, 'message': msg})


@app.route('/api/strategy', methods=['POST'])
def set_strategy():
    data     = request.get_json() or {}
    strategy = data.get('strategy')
    if strategy not in ('ma_crossover', 'rsi', 'macd'):
        return jsonify({'error': 'Invalid strategy'}), 400
    database.update_bot_state(strategy=strategy)
    return jsonify({'success': True, 'strategy': strategy})


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.route('/api/trades')
def get_trades():
    limit    = request.args.get('limit', 20, type=int)
    strategy = request.args.get('strategy')
    return jsonify(database.get_trades(limit=limit, strategy=strategy))


@app.route('/api/portfolio/history')
def portfolio_history():
    strategy = request.args.get('strategy')
    limit    = request.args.get('limit', 500, type=int)
    return jsonify(database.get_portfolio_history(strategy=strategy, limit=limit))


@app.route('/api/activity')
def activity():
    return jsonify(bot.get_activity_log())


@app.route('/api/indicators')
def indicators():
    """Current indicator values for all watchlist tickers."""
    strategy = request.args.get('strategy', 'ma_crossover')
    if strategy not in ('ma_crossover', 'rsi', 'macd'):
        return jsonify({'error': 'Invalid strategy'}), 400
    result = {}
    for ticker in strategies.WATCHLIST:
        result[ticker] = strategies.get_indicator_data(ticker, strategy)
    return jsonify(result)


@app.route('/api/watchlist')
def get_watchlist():
    return jsonify(strategies.WATCHLIST)


# ── Backtesting ───────────────────────────────────────────────────────────────

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    data            = request.get_json() or {}
    strategy        = data.get('strategy', 'ma_crossover')
    tickers         = data.get('tickers', ['AAPL', 'MSFT', 'SPY'])
    start_date      = data.get('start_date',
                               (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date        = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    initial_capital = float(data.get('initial_capital', 100_000))

    if strategy not in ('ma_crossover', 'rsi', 'macd'):
        return jsonify({'error': 'Invalid strategy'}), 400
    if not (1_000 <= initial_capital <= 10_000_000):
        return jsonify({'error': 'Capital must be between $1,000 and $10,000,000'}), 400
    if not tickers:
        return jsonify({'error': 'At least one ticker required'}), 400

    result = backtester.run_backtest(
        strategy, tickers, start_date, end_date, initial_capital
    )
    return jsonify(result)


# ── Health check ──────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ts': datetime.utcnow().isoformat()})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
