/* ── Config ──────────────────────────────────────────────────────────────── */
// Update API_BASE to your Render URL after deployment
const API_BASE =
  (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
    ? 'http://localhost:5000'
    : 'https://YOUR-RENDER-APP.onrender.com';   // ← update after deploy

const STRATEGY_LABELS = {
  adaptive:     'Adaptive (Regime-Driven)',
  ma_crossover: 'Moving Average Crossover',
  rsi:          'RSI Mean Reversion',
  macd:         'MACD Momentum',
  ml:           'ML Model',
};

const RISK_NOTES = {
  conservative: 'Conservative: 3% stop-loss, 10% target, 5% max position. Sits out high-volatility regimes entirely.',
  moderate:     'Moderate: 5% stop-loss, 15% target, 10% max position. Reduces size 50% in high-volatility regimes.',
  aggressive:   'Aggressive: 7% stop-loss, 20% target, 15% max position. Trades at full size in all regimes.',
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */
async function api(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function fmt$(n)   { return '$' + (parseFloat(n) || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
function fmtPct(n) { const v = parseFloat(n) || 0; return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'; }
function colorClass(n) { return parseFloat(n) > 0 ? 'positive' : parseFloat(n) < 0 ? 'negative' : ''; }
function el(id)    { return document.getElementById(id); }
function setText(id, val) { const e = el(id); if (e) e.textContent = val; }
function setClass(id, cls) { const e = el(id); if (e) { e.className = e.className.replace(/positive|negative|neutral/g, '').trim() + (cls ? ' ' + cls : ''); }}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   LIVE DASHBOARD  (index.html)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
if (document.body.dataset.page === 'dashboard') {
  let equityChart   = null;
  let botRunning    = false;
  let refreshTimer  = null;

  /* ── Bot start/stop ─────────────────────────────────────────────────── */
  el('bot-toggle').addEventListener('click', async () => {
    const btn = el('bot-toggle');
    btn.disabled = true;
    try {
      if (botRunning) {
        await api('/api/stop', { method: 'POST' });
      } else {
        const strategy = el('strategy-select').value;
        await api('/api/start', {
          method: 'POST',
          body: JSON.stringify({ strategy }),
        });
      }
    } catch (e) {
      // swallow — refreshAll will reflect the true state
    } finally {
      btn.disabled = false;
      await refreshAll();
    }
  });

  /* ── Strategy change ────────────────────────────────────────────────── */
  el('strategy-select').addEventListener('change', async (e) => {
    await api('/api/strategy', {
      method: 'POST',
      body:   JSON.stringify({ strategy: e.target.value }),
    });
  });

  /* ── Main refresh ───────────────────────────────────────────────────── */
  async function refreshAll() {
    try {
      const status = await api('/api/status');
      updateControlBar(status);
      updateStats(status);
      updatePositions(status.portfolio?.positions || []);
    } catch (e) { /* API not ready yet */ }

    try {
      const trades = await api('/api/trades?limit=20');
      updateTrades(trades);
    } catch (_) {}

    try {
      const history = await api('/api/portfolio/history?limit=500');
      updateEquityChart(history);
    } catch (_) {}

    try {
      const log = await api('/api/activity');
      updateActivityLog(log);
    } catch (_) {}
  }

  function updateControlBar(status) {
    botRunning = status.is_running;

    const dot  = el('bot-dot');
    const text = el('bot-status-text');
    const btn  = el('bot-toggle');

    if (botRunning) {
      dot.className  = 'status-dot running';
      text.textContent = 'BOT RUNNING';
      btn.textContent  = 'STOP BOT';
      btn.className    = 'btn btn-stop';
    } else {
      dot.className  = 'status-dot stopped';
      text.textContent = 'BOT STOPPED';
      btn.textContent  = 'START BOT';
      btn.className    = 'btn btn-start';
    }

    const mp = el('market-status');
    if (mp) {
      mp.textContent = status.market_open ? 'MARKET OPEN' : 'MARKET CLOSED';
      mp.className   = 'market-pill ' + (status.market_open ? 'open' : 'closed');
    }

    const sel = el('strategy-select');
    if (sel && status.strategy) sel.value = status.strategy;

    updateRegimePanel(status.regime);
    updateRiskButtons(status.risk_tolerance);
  }

  function updateStats(status) {
    const p  = status.portfolio    || {};
    const m  = status.metrics      || {};
    const lm = status.live_metrics || {};

    setText('stat-value', fmt$(p.portfolio_value ?? 0));
    const ret = parseFloat(p.total_return ?? 0);
    setText('stat-return', fmtPct(ret));
    setClass('stat-return', colorClass(ret));

    const wr = parseFloat(m.win_rate ?? 0);
    setText('stat-winrate', wr.toFixed(1) + '%');
    setClass('stat-winrate', wr >= 50 ? 'positive' : wr > 0 ? 'neutral' : '');

    setText('stat-positions', p.active_positions ?? 0);

    const dt  = status.daily_trades ?? 0;
    const max = status.max_daily    ?? 10;
    const sub = el('stat-positions-sub');
    if (sub) sub.textContent = `${dt} / ${max} daily trades used`;

    const sharpe = parseFloat(lm.sharpe_ratio ?? 0);
    setText('stat-sharpe', sharpe.toFixed(2));
    setClass('stat-sharpe', colorClass(sharpe));

    const dd = parseFloat(lm.max_drawdown ?? 0);
    setText('stat-drawdown', '-' + dd.toFixed(2) + '%');
    setClass('stat-drawdown', dd > 0 ? 'negative' : '');
  }

  function updatePositions(positions) {
    const tbody = el('positions-tbody');
    const badge = el('positions-count');
    if (!tbody) return;

    if (badge) badge.textContent = positions.length;

    if (!positions.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No open positions</td></tr>';
      return;
    }

    tbody.innerHTML = positions.map(p => {
      const pnlClass = colorClass(p.pnl);
      return `
        <tr>
          <td><strong>${p.ticker}</strong></td>
          <td>${p.shares}</td>
          <td>${fmt$(p.entry_price)}</td>
          <td>${fmt$(p.current_price)}</td>
          <td class="${pnlClass}">${fmtPct(p.pnl_pct)}</td>
          <td class="negative">${fmt$(p.stop_loss)}</td>
          <td class="positive">${fmt$(p.take_profit)}</td>
        </tr>`;
    }).join('');
  }

  function updateTrades(trades) {
    const feed = el('trades-feed');
    if (!feed) return;
    if (!trades.length) {
      feed.innerHTML = '<div class="empty-state">No trades yet</div>';
      return;
    }
    feed.innerHTML = trades.map(t => {
      const ts     = t.timestamp ? t.timestamp.slice(11, 19) : '—';
      const action = t.action === 'BUY'
        ? '<span class="badge badge-buy">BUY</span>'
        : '<span class="badge badge-sell">SELL</span>';
      const pnl = t.pnl != null
        ? `<span class="${colorClass(t.pnl)}">${fmtPct(t.pnl_pct)}</span>`
        : '<span class="neutral">—</span>';
      return `
        <div class="trade-row">
          <span class="trade-time">${ts}</span>
          ${action}
          <strong>${t.ticker}</strong>
          <span>${fmt$(t.price)}</span>
          <span>${t.shares}</span>
          ${pnl}
        </div>`;
    }).join('');
  }

  function updateEquityChart(history) {
    if (!history.length) return;
    const ctx = el('equity-chart');
    if (!ctx) return;

    const labels = history.map(h => h.timestamp ? h.timestamp.slice(0, 16).replace('T', ' ') : '');
    const values = history.map(h => h.portfolio_value);

    if (equityChart) {
      equityChart.data.labels   = labels;
      equityChart.data.datasets[0].data = values;
      equityChart.update('none');
      return;
    }

    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label:           'Portfolio Value',
          data:            values,
          borderColor:     '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.08)',
          borderWidth:     2,
          pointRadius:     0,
          tension:         0.3,
          fill:            true,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        interaction:         { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#161b22',
            borderColor:     '#21262d',
            borderWidth:     1,
            titleColor:      '#7d8590',
            bodyColor:       '#e6edf3',
            callbacks: { label: ctx => ' ' + fmt$(ctx.parsed.y) },
          },
        },
        scales: {
          x: {
            ticks:  { color: '#7d8590', maxTicksLimit: 8, font: { family: 'monospace', size: 10 } },
            grid:   { color: '#21262d' },
          },
          y: {
            ticks:  { color: '#7d8590', callback: v => '$' + v.toLocaleString(), font: { family: 'monospace', size: 10 } },
            grid:   { color: '#21262d' },
          },
        },
      },
    });
  }

  function updateActivityLog(lines) {
    const log = el('activity-log');
    if (!log) return;
    if (!lines.length) {
      log.innerHTML = '<div class="log-empty">No activity yet</div>';
      return;
    }
    log.innerHTML = lines.map(l => `<div class="log-line">${escHtml(l)}</div>`).join('');
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ── Risk tolerance buttons ───────────────────────────────────────────────
  document.querySelectorAll('.risk-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const tol = btn.dataset.tol;
      try {
        await api('/api/risk_tolerance', {
          method: 'POST',
          body: JSON.stringify({ tolerance: tol }),
        });
        document.querySelectorAll('.risk-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const note = el('risk-note');
        if (note) note.textContent = RISK_NOTES[tol] || '';
      } catch (_) {}
    });
  });

  function updateRegimePanel(regime) {
    if (!regime) return;
    const badge   = el('regime-badge');
    const label   = el('regime-label');
    const desc    = el('regime-desc');
    const strat   = el('regime-strategy');
    const adxEl   = el('ri-adx');
    const volEl   = el('ri-vol');
    const bbwEl   = el('ri-bbw');

    if (badge) {
      badge.textContent = regime.label || regime.regime;
      badge.className   = 'regime-badge ' + (regime.regime || '');
    }
    if (label) {
      label.textContent = regime.label || regime.regime;
      label.className   = 'regime-label ' + (regime.regime || '');
    }
    if (desc)  desc.textContent  = regime.description || '—';
    if (strat) strat.textContent = STRATEGY_LABELS[regime.strategy] || regime.strategy || '—';
    if (adxEl) adxEl.textContent = regime.adx != null ? regime.adx.toFixed(1) : '—';
    if (volEl) volEl.textContent = regime.vol_30d != null ? regime.vol_30d.toFixed(1) + '%' : '—';
    if (bbwEl) bbwEl.textContent = regime.bb_width != null ? regime.bb_width.toFixed(3) : '—';
  }

  function updateRiskButtons(tolerance) {
    if (!tolerance) return;
    document.querySelectorAll('.risk-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tol === tolerance);
    });
    const note = el('risk-note');
    if (note) note.textContent = RISK_NOTES[tolerance] || '';
  }

  // Kick off
  refreshAll();
  refreshTimer = setInterval(refreshAll, 10_000);
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   BACKTEST PAGE  (backtest.html)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
if (document.body.dataset.page === 'backtest') {
  let btChart = null;

  // Ticker chip selection
  document.querySelectorAll('.ticker-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      chip.classList.toggle('selected');
    });
  });

  // Default date range: last 1 year
  const today    = new Date();
  const oneYrAgo = new Date(today);
  oneYrAgo.setFullYear(oneYrAgo.getFullYear() - 1);
  const fmt = d => d.toISOString().slice(0, 10);
  const startEl = el('bt-start');
  const endEl   = el('bt-end');
  if (startEl) startEl.value = fmt(oneYrAgo);
  if (endEl)   endEl.value   = fmt(today);

  el('bt-run')?.addEventListener('click', runBacktest);

  async function runBacktest() {
    const selectedTickers = [...document.querySelectorAll('.ticker-chip.selected')]
      .map(c => c.dataset.ticker);

    if (!selectedTickers.length) {
      alert('Select at least one ticker.');
      return;
    }

    const btn = el('bt-run');
    btn.disabled     = true;
    btn.innerHTML    = '<span class="spinner"></span> RUNNING…';

    const riskRadio = document.querySelector('input[name="bt-risk"]:checked');
    const payload = {
      strategy:        el('bt-strategy').value,
      tickers:         selectedTickers,
      start_date:      el('bt-start').value,
      end_date:        el('bt-end').value,
      initial_capital: parseFloat(el('bt-capital').value) || 100_000,
      walk_forward:    el('bt-walkforward')?.checked ?? false,
      risk_tolerance:  riskRadio ? riskRadio.value : 'moderate',
      commission_pct:  (parseFloat(el('bt-commission')?.value) || 0.10) / 100,
      slippage_pct:    (parseFloat(el('bt-slippage')?.value)   || 0.05) / 100,
    };

    try {
      const result = await api('/api/backtest', {
        method: 'POST',
        body:   JSON.stringify(payload),
      });

      if (result.error) {
        alert('Backtest error: ' + result.error);
        return;
      }

      renderResults(result, payload);
    } catch (e) {
      alert('Request failed. Is the backend running?');
    } finally {
      btn.disabled  = false;
      btn.innerHTML = 'RUN BACKTEST';
    }
  }

  function renderResults(result, payload) {
    const m = result.metrics;

    // Show results section
    el('bt-results').classList.add('visible');

    // Scroll to results
    el('bt-results').scrollIntoView({ behavior: 'smooth' });

    // Stat cards
    const retClass = colorClass(m.total_return);
    el('r-return').innerHTML      = `<span class="${retClass}">${fmtPct(m.total_return)}</span>`;
    el('r-final').textContent     = fmt$(m.final_value);
    el('r-winrate').innerHTML     = `<span class="${parseFloat(m.win_rate) >= 50 ? 'positive' : 'negative'}">${m.win_rate}%</span>`;
    el('r-trades').textContent    = m.total_trades;
    el('r-drawdown').innerHTML    = `<span class="negative">-${m.max_drawdown}%</span>`;
    el('r-sharpe').innerHTML      = `<span class="${colorClass(m.sharpe_ratio)}">${m.sharpe_ratio}</span>`;
    el('r-avgwin').innerHTML      = `<span class="positive">${fmt$(m.avg_win)}</span>`;
    el('r-avgloss').innerHTML     = `<span class="negative">${fmt$(m.avg_loss)}</span>`;
    el('r-best').innerHTML        = `<span class="positive">${fmt$(m.best_trade)}</span>`;
    el('r-worst').innerHTML       = `<span class="negative">${fmt$(m.worst_trade)}</span>`;
    el('r-benchmark').innerHTML   = `<span class="${colorClass(m.benchmark_return)}">${fmtPct(m.benchmark_return)}</span>`;
    el('r-calmar').innerHTML  = `<span class="${colorClass(m.calmar_ratio)}">${(m.calmar_ratio ?? 0).toFixed(2)}</span>`;

    // Kelly fraction
    const kellyEl = el('r-kelly');
    if (kellyEl) {
      if (m.kelly_fraction != null) {
        kellyEl.innerHTML = `<span class="${colorClass(m.kelly_fraction)}">${m.kelly_fraction.toFixed(1)}%</span>`;
      } else {
        kellyEl.innerHTML = '<span class="neutral">N/A</span>';
        const ksub = kellyEl.nextElementSibling;
        if (ksub) ksub.textContent = '< 10 trades (using fixed sizing)';
      }
    }

    // Gross return vs net
    const grossRetClass = colorClass(m.gross_return);
    el('r-gross').innerHTML = `<span class="${grossRetClass}">${fmtPct(m.gross_return)}</span>`;
    const costsSub = el('r-costs-sub');
    if (costsSub && m.total_costs != null) {
      costsSub.textContent = `Costs: ${fmt$(m.total_costs)} total`;
    }

    // Walk-forward banner
    const wf = result.walk_forward || {};
    const banner = el('wf-banner');
    if (banner) {
      if (wf.enabled && wf.split_date) {
        el('wf-split-date').textContent = wf.split_date;
        banner.style.display = 'block';
      } else {
        banner.style.display = 'none';
      }
    }

    // Chart
    renderBtChart(result.equity_curve, result.spy_curve, payload.initial_capital);

    // Monte Carlo
    renderMonteCarlo(result.monte_carlo, result.equity_curve, payload.initial_capital);

    // Regime breakdown
    renderRegimeBreakdown(result.regime_breakdown || {});

    // Trade table
    renderTradeTable(result.trades || []);
  }

  function renderBtChart(curve, spyCurve, initialCapital) {
    const ctx = el('bt-chart');
    if (!ctx) return;

    const labels = curve.map(p => p.date);
    const stratData = curve.map(p => p.value);

    // Align spy curve to same labels
    const spyMap = {};
    (spyCurve || []).forEach(p => { spyMap[p.date] = p.value; });
    const spyData = labels.map(d => spyMap[d] ?? null);

    // Initial capital reference line
    const initLine = labels.map(() => initialCapital);

    if (btChart) {
      btChart.destroy();
    }

    btChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label:           'Strategy',
            data:            stratData,
            borderColor:     '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.07)',
            borderWidth:     2,
            pointRadius:     0,
            tension:         0.2,
            fill:            true,
          },
          {
            label:       'SPY Buy & Hold',
            data:        spyData,
            borderColor: '#7d8590',
            borderWidth: 1.5,
            pointRadius: 0,
            borderDash:  [5, 4],
            tension:     0.2,
          },
          {
            label:       'Starting Capital',
            data:        initLine,
            borderColor: '#30363d',
            borderWidth: 1,
            pointRadius: 0,
            borderDash:  [3, 3],
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        interaction:         { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display:  true,
            position: 'top',
            labels:   { color: '#7d8590', font: { family: 'monospace', size: 11 }, boxWidth: 20 },
          },
          tooltip: {
            backgroundColor: '#161b22',
            borderColor:     '#21262d',
            borderWidth:     1,
            titleColor:      '#7d8590',
            bodyColor:       '#e6edf3',
            callbacks: { label: c => ` ${c.dataset.label}: ${fmt$(c.parsed.y)}` },
          },
        },
        scales: {
          x: {
            ticks: { color: '#7d8590', maxTicksLimit: 10, font: { family: 'monospace', size: 10 } },
            grid:  { color: '#21262d' },
          },
          y: {
            ticks: { color: '#7d8590', callback: v => '$' + v.toLocaleString(), font: { family: 'monospace', size: 10 } },
            grid:  { color: '#21262d' },
          },
        },
      },
    });
  }

  let mcChart = null;

  function renderMonteCarlo(mc, equityCurve, initialCapital) {
    const panel = el('mc-panel');
    if (!panel || !mc || !mc.enabled) {
      if (panel) panel.style.display = 'none';
      return;
    }
    panel.style.display = 'block';

    // ── Headline ────────────────────────────────────────────────────────────
    const pct     = mc.actual_percentile;
    const sPct    = mc.sharpe_percentile;
    const rankEl  = el('mc-rank-badge');
    const headEl  = el('mc-headline');

    if (rankEl) {
      rankEl.textContent = `${pct.toFixed(0)}th percentile`;
      rankEl.className   = 'mc-rank-badge ' +
        (pct >= 75 ? 'strong' : pct >= 40 ? 'moderate' : 'weak');
    }

    if (headEl) {
      const strength = pct >= 75
        ? 'statistically meaningful'
        : pct >= 40 ? 'within normal random variation' : 'weaker than most random paths';
      headEl.innerHTML =
        `Your strategy's return ranks in the <strong>${pct.toFixed(0)}th percentile</strong> of ` +
        `${mc.n_simulations.toLocaleString()} random resampled paths — ` +
        `<strong>${strength}</strong>. ` +
        `Its Sharpe ratio ranks in the <strong>${sPct.toFixed(0)}th percentile</strong> of simulated Sharpe ratios.`;
    }

    // ── Fan chart ────────────────────────────────────────────────────────────
    const fan   = mc.fan_chart;
    const ctx   = el('mc-chart');
    if (!ctx || !fan) return;

    // Align actual equity curve to fan chart dates
    const actualMap = {};
    (equityCurve || []).forEach(p => { actualMap[p.date] = p.value; });
    const actualData = fan.dates.map(d => actualMap[d] ?? null);

    if (mcChart) { mcChart.destroy(); mcChart = null; }

    mcChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: fan.dates,
        datasets: [
          // Outer band (P95 fills down to P5)
          {
            label: 'P95',
            data:  fan.p95,
            fill:  1,
            borderColor: 'transparent',
            backgroundColor: 'rgba(88,166,255,0.06)',
            pointRadius: 0,
          },
          {
            label: 'P5',
            data:  fan.p5,
            fill:  false,
            borderColor: 'transparent',
            pointRadius: 0,
          },
          // Inner band (P75 fills down to P25)
          {
            label: 'P75',
            data:  fan.p75,
            fill:  3,
            borderColor: 'transparent',
            backgroundColor: 'rgba(88,166,255,0.11)',
            pointRadius: 0,
          },
          {
            label: 'P25',
            data:  fan.p25,
            fill:  false,
            borderColor: 'transparent',
            pointRadius: 0,
          },
          // Median
          {
            label: 'Median (P50)',
            data:  fan.p50,
            fill:  false,
            borderColor: 'rgba(88,166,255,0.35)',
            borderDash: [5, 4],
            borderWidth: 1.5,
            pointRadius: 0,
          },
          // Actual strategy
          {
            label: 'Your Strategy',
            data:  actualData,
            fill:  false,
            borderColor: '#58a6ff',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              filter: item => ['Median (P50)', 'Your Strategy', 'P95', 'P25'].includes(item.text),
              color: '#7d8590',
              font: { family: 'monospace', size: 10 },
              boxWidth: 16,
            },
          },
          tooltip: {
            backgroundColor: '#161b22',
            borderColor: '#21262d',
            borderWidth: 1,
            titleColor: '#7d8590',
            bodyColor: '#e6edf3',
            callbacks: {
              label: c => {
                if (['P5', 'P75'].includes(c.dataset.label)) return null;
                return ` ${c.dataset.label}: ${fmt$(c.parsed.y)}`;
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: '#7d8590', maxTicksLimit: 8, font: { family: 'monospace', size: 10 } }, grid: { color: '#21262d' } },
          y: { ticks: { color: '#7d8590', callback: v => '$' + v.toLocaleString(), font: { family: 'monospace', size: 10 } }, grid: { color: '#21262d' } },
        },
      },
    });

    // ── Distribution tables ──────────────────────────────────────────────────
    const dist = mc.return_distribution;
    const shar = mc.sharpe_distribution;
    const pctiles = [['5th', 'p5'], ['25th', 'p25'], ['50th (median)', 'p50'], ['75th', 'p75'], ['95th', 'p95']];

    const rTbody = el('mc-return-tbody');
    const sTbody = el('mc-sharpe-tbody');

    if (rTbody) rTbody.innerHTML = pctiles.map(([lbl, key]) =>
      `<tr><td>${lbl}</td><td class="${colorClass(dist[key])}">${fmtPct(dist[key])}</td></tr>`
    ).join('');

    if (sTbody) sTbody.innerHTML = pctiles.map(([lbl, key]) =>
      `<tr><td>${lbl}</td><td class="${colorClass(shar[key])}">${(shar[key] ?? 0).toFixed(2)}</td></tr>`
    ).join('');
  }

  function renderRegimeBreakdown(breakdown) {
    const panel = el('regime-breakdown-panel');
    const tbody = el('regime-breakdown-tbody');
    if (!panel || !tbody) return;

    const rows = Object.entries(breakdown);
    if (!rows.length) {
      panel.style.display = 'none';
      return;
    }

    panel.style.display = 'block';
    tbody.innerHTML = rows
      .sort((a, b) => b[1].trade_count - a[1].trade_count)
      .map(([regimeName, d]) => {
        const wrClass  = parseFloat(d.win_rate) >= 50 ? 'positive' : 'negative';
        const pnlClass = colorClass(d.total_pnl);
        return `
          <tr>
            <td><span class="regime-badge ${regimeName}">${d.label}</span></td>
            <td>${d.trade_count}</td>
            <td class="${wrClass}">${d.win_rate}%</td>
            <td class="${pnlClass}">${fmt$(d.total_pnl)}</td>
            <td class="${colorClass(d.avg_pnl)}">${fmt$(d.avg_pnl)}</td>
            <td class="positive">${fmt$(d.best_trade)}</td>
            <td class="negative">${fmt$(d.worst_trade)}</td>
          </tr>`;
      }).join('');
  }

  function renderTradeTable(trades) {
    const tbody = el('bt-trades-tbody');
    if (!tbody) return;
    if (!trades.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No completed trades in this period</td></tr>';
      return;
    }
    tbody.innerHTML = trades
      .filter(t => t.action === 'SELL')
      .reverse()
      .map(t => {
        const pnlClass = colorClass(t.pnl);
        const badge    = '<span class="badge badge-sell">SELL</span>';
        return `
          <tr>
            <td>${t.date}</td>
            <td><strong>${t.ticker}</strong></td>
            <td>${badge}</td>
            <td>${fmt$(t.price)}</td>
            <td>${t.shares}</td>
            <td class="${pnlClass}">${fmt$(t.pnl)}</td>
            <td class="${pnlClass}">${fmtPct(t.pnl_pct)}</td>
          </tr>`;
      }).join('');
  }
}
