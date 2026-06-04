# Gunicorn production configuration for TradeBot
#
# CRITICAL — workers must be 1.
# The trading bot runs in a background thread inside the Flask process.
# Multiple workers = multiple bot instances trading the same Alpaca account
# simultaneously, which will cause duplicate orders.
#
# Render start command: gunicorn app:app
# (gunicorn auto-discovers this file when it's in the working directory)

import os

workers   = 1      # one process only — see note above
threads   = 4      # handle concurrent API requests within the single process
timeout   = 120    # backtests can take 60-90s with yfinance downloads
keepalive = 5
# Render passes the assigned port via $PORT — must be respected or the
# health check fails and the service never comes up.
bind      = f'0.0.0.0:{os.getenv("PORT", "5000")}'
accesslog = '-'   # log to stdout (captured by Render)
errorlog  = '-'
loglevel  = 'info'
