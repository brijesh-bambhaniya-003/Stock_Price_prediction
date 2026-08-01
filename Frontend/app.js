/* ═══════════════════════════════════════════════════════
   Tesla StockAI — Frontend App (Black · Red · White)
   Chart.js 3.x + chartjs-chart-financial 0.1.1
   ═══════════════════════════════════════════════════════ */

const API = '';

// ── Color tokens (match CSS) ─────────────────────────────
const C = {
  red:     '#e63946',
  redLt:   '#ff6b6b',
  redDim:  'rgba(230,57,70,0.18)',
  up:      '#22c55e',
  down:    '#e63946',
  white:   '#ffffff',
  white60: 'rgba(255,255,255,0.6)',
  white10: 'rgba(255,255,255,0.10)',
  grid:    'rgba(255,255,255,0.07)',
  text:    'rgba(255,255,255,0.55)',
};

// ── Chart.js 3.x global defaults ─────────────────────────
Chart.defaults.color            = C.text;
Chart.defaults.borderColor      = C.grid;
Chart.defaults.font.family      = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size        = 11;

// ── State ─────────────────────────────────────────────────
let historyData  = [];
let forecastData = [];
let dashDays     = 90;      // current dashboard range in days
let fcHorizon    = 7;

// chart instances
let dashCandleChart = null;
let dashLineChart   = null;
let predictChart    = null;
let forecastChart   = null;
let avpChart        = null;

// table state
let tableData     = [];
let tableSorted   = [];
let tableFiltered = [];
let tablePage     = 1;
const PAGE_SIZE   = 25;
let sortKey       = 'date';
let sortAsc       = false;

// ═══════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════
const fmt    = v => v != null ? `$${(+v).toFixed(2)}`       : '—';
const fmtN   = (v, d=4) => v != null ? (+v).toFixed(d)      : '—';
const toTs   = dateStr => new Date(dateStr).getTime();        // ms timestamp

/** Filter data to the last `days` calendar days from dataset's LAST date */
function filterByDays(data, days) {
  if (!days || !data.length) return data;
  const lastDate = new Date(data[data.length - 1].date);
  const cutoff   = new Date(lastDate);
  cutoff.setDate(cutoff.getDate() - days);
  return data.filter(d => new Date(d.date) >= cutoff);
}

/** Human-readable date range label, e.g. "Oct 2025 – Dec 2025" */
function makeDateLabel(data) {
  if (!data.length) return '';
  const opts = { month: 'short', year: 'numeric' };
  const first = new Date(data[0].date).toLocaleDateString('en-GB', opts);
  const last  = new Date(data[data.length - 1].date).toLocaleDateString('en-GB', opts);
  return first === last ? first : `${first} – ${last}`;
}

/** Next business day after a date string */
function nextBusinessDay(dateStr) {
  if (!dateStr) return new Date().toISOString().slice(0, 10);
  const d = new Date(dateStr);
  d.setDate(d.getDate() + 1);
  while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}

// ═══════════════════════════════════════════════════════════
//  SHARED CHART OPTIONS
// ═══════════════════════════════════════════════════════════
const timeX = (unit = 'month') => ({
  type: 'time',
  time: { unit, tooltipFormat: 'dd MMM yyyy' },
  grid: { color: C.grid },
  ticks: { color: C.text, maxTicksLimit: 8 },
});

const priceY = {
  position: 'right',
  grid: { color: C.grid },
  ticks: { color: C.text, callback: v => '$' + v.toFixed(0) },
};

// ═══════════════════════════════════════════════════════════
//  DASHBOARD — CANDLESTICK CHART
// ═══════════════════════════════════════════════════════════
function drawCandleChart(data) {
  const loader = document.getElementById('candle-loader');
  if (dashCandleChart) { dashCandleChart.destroy(); dashCandleChart = null; }
  if (!data.length) { if (loader) loader.classList.add('hidden'); return; }

  const ctx = document.getElementById('dashCandleChart').getContext('2d');
  const spanDays = (new Date(data[data.length-1].date) - new Date(data[0].date)) / 864e5;
  const unit = spanDays <= 35 ? 'day' : spanDays <= 200 ? 'week' : 'month';

  try {
    dashCandleChart = new Chart(ctx, {
      type: 'candlestick',
      data: {
        datasets: [{
          label: 'TSLA',
          data: data.map(d => ({
            x: toTs(d.date),
            o: +d.open, h: +d.high, l: +d.low, c: +d.close,
          })),
          color:       { up: C.up, down: C.down, unchanged: C.white60 },
          borderColor: { up: C.up, down: C.down, unchanged: C.white60 },
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 350 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#111', borderColor: C.red, borderWidth: 1,
            callbacks: {
              title: items => new Date(items[0].parsed.x).toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' }),
              label: ctx => { const r = ctx.raw; return [`Open: $${r.o.toFixed(2)}`, `High: $${r.h.toFixed(2)}`, `Low: $${r.l.toFixed(2)}`, `Close: $${r.c.toFixed(2)}`]; }
            }
          }
        },
        scales: { x: timeX(unit), y: priceY }
      }
    });
  } catch (err) {
    console.error('[CandleChart] Error:', err);
  }
  if (loader) loader.classList.add('hidden');
}


// ═══════════════════════════════════════════════════════════
//  DASHBOARD — LINE CHART
// ═══════════════════════════════════════════════════════════
function drawLineChart(data) {
  if (dashLineChart) { dashLineChart.destroy(); dashLineChart = null; }
  if (!data.length) return;

  const ctx = document.getElementById('dashLineChart').getContext('2d');
  const spanDays = (new Date(data[data.length-1].date) - new Date(data[0].date)) / 864e5;
  const unit = spanDays <= 35 ? 'day' : spanDays <= 200 ? 'week' : 'month';

  const makeDs = (key, label, color, dash=[]) => ({
    label,
    data: data.filter(d => d[key] != null).map(d => ({ x: toTs(d.date), y: +d[key] })),
    borderColor:     color,
    backgroundColor: 'transparent',
    borderWidth:     key === 'close' ? 2 : 1.5,
    pointRadius:     0,
    pointHoverRadius: key === 'close' ? 5 : 3,
    tension:         0.3,
    fill:            false,
    borderDash:      dash,
  });

  dashLineChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        makeDs('close', 'Close',  C.white),
        makeDs('ma_7',  'MA 7',   C.red),
        makeDs('ma_30', 'MA 30',  C.redLt,  [4,4]),
        makeDs('ma_90', 'MA 90',  '#666666', [4,4]),
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111',
          borderColor: C.red,
          borderWidth: 1,
          callbacks: {
            title: items => new Date(items[0].parsed.x).toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' }),
            label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`,
          }
        }
      },
      scales: {
        x: timeX(unit),
        y: priceY,
      }
    }
  });

  document.getElementById('line-loader').classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════
//  DASHBOARD — SET RANGE
// ═══════════════════════════════════════════════════════════
function setDashRange(days, btn) {
  dashDays = days;

  // Update active button
  document.querySelectorAll('.range-bar .range-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const slice = filterByDays(historyData, days);

  // Date range badge
  const badge = document.getElementById('dash-range-label');
  if (days === 0) {
    badge.classList.remove('visible');
  } else {
    const label = makeDateLabel(slice);
    badge.textContent = label;
    badge.classList.add('visible');
  }

  // Subtitle on each chart
  const sub = days === 0 ? 'All time · 2010–2025' : makeDateLabel(slice);
  document.getElementById('candle-subtitle').textContent = sub;
  document.getElementById('line-subtitle').textContent   = sub;

  drawCandleChart(slice);
  drawLineChart(slice);

  // Keep table filtered to same range
  populateTable(slice);
}

// ═══════════════════════════════════════════════════════════
//  HISTORY TABLE
// ═══════════════════════════════════════════════════════════
function populateTable(data) {
  tableData     = [...data].reverse();
  tableSorted   = [...tableData];
  tableFiltered = [...tableData];
  tablePage     = 1;
  renderTable();
}

function sortTable(key) {
  if (sortKey === key) sortAsc = !sortAsc;
  else { sortKey = key; sortAsc = true; }
  tableSorted = [...tableFiltered].sort((a, b) => {
    const va = isNaN(+a[key]) ? a[key] : +a[key];
    const vb = isNaN(+b[key]) ? b[key] : +b[key];
    return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });
  tablePage = 1;
  renderTable();
}

function filterTable() {
  const q = document.getElementById('table-search').value.toLowerCase();
  tableFiltered = tableData.filter(d =>
    d.date.includes(q) ||
    String(+d.close).includes(q) ||
    String(+d.open).includes(q)
  );
  tableSorted = [...tableFiltered];
  tablePage = 1;
  renderTable();
}

function changePage(dir) {
  const maxPage = Math.ceil(tableSorted.length / PAGE_SIZE);
  tablePage = Math.max(1, Math.min(maxPage, tablePage + dir));
  renderTable();
}

function renderTable() {
  const tbody = document.getElementById('history-tbody');
  const start = (tablePage - 1) * PAGE_SIZE;
  const rows  = tableSorted.slice(start, start + PAGE_SIZE);

  tbody.innerHTML = rows.map(d => {
    const isUp = +d.close >= +d.open;
    return `<tr>
      <td>${d.date}</td>
      <td>${fmt(d.open)}</td>
      <td class="td-up">${fmt(d.high)}</td>
      <td class="td-down">${fmt(d.low)}</td>
      <td class="${isUp ? 'td-up' : 'td-down'}">${fmt(d.close)}</td>
      <td>${d.volume ? (+d.volume / 1e6).toFixed(1) + 'M' : '—'}</td>
      <td>${d.ma_7  ? fmt(d.ma_7)  : '—'}</td>
      <td>${d.ma_30 ? fmt(d.ma_30) : '—'}</td>
    </tr>`;
  }).join('');

  const maxPage = Math.max(1, Math.ceil(tableSorted.length / PAGE_SIZE));
  document.getElementById('page-info').textContent = `Page ${tablePage} / ${maxPage}`;
  document.getElementById('btn-prev').disabled = tablePage <= 1;
  document.getElementById('btn-next').disabled = tablePage >= maxPage;
}

// ═══════════════════════════════════════════════════════════
//  PREDICT TAB — LINE CHART ONLY with CI band
// ═══════════════════════════════════════════════════════════
document.getElementById('predict-form').addEventListener('submit', async e => {
  e.preventDefault();
  const text = document.getElementById('btn-predict-text');
  const spin = document.getElementById('btn-predict-spinner');
  const btn  = document.getElementById('btn-predict-submit');
  text.classList.add('hidden');
  spin.classList.remove('hidden');
  btn.disabled = true;

  const body = {
    open:   +document.getElementById('inp-open').value,
    high:   +document.getElementById('inp-high').value,
    low:    +document.getElementById('inp-low').value,
    close:  +document.getElementById('inp-close').value,
    volume: +document.getElementById('inp-volume').value,
  };

  try {
    const r    = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();

    document.getElementById('result-price').textContent = fmt(data.predicted_price);
    document.getElementById('result-ci').textContent    =
      `95% Confidence: ${fmt(data.confidence_lower)} – ${fmt(data.confidence_upper)}`;
    document.getElementById('rf-val').textContent  = fmt(data.rf_prediction);
    document.getElementById('xgb-val').textContent = fmt(data.xgb_prediction);
    document.getElementById('ens-val').textContent = fmt(data.predicted_price);

    drawPredictLineChart(body, data);
  } catch (err) {
    document.getElementById('result-price').textContent = 'Error';
    document.getElementById('result-ci').textContent    = 'Could not reach API. Is the server running?';
    console.error(err);
  }

  text.classList.remove('hidden');
  spin.classList.add('hidden');
  btn.disabled = false;
});

function drawPredictLineChart(input, result) {
  if (predictChart) { predictChart.destroy(); predictChart = null; }

  // Last 30 days of history + 1 predicted point
  const recent   = historyData.slice(-30);
  const nextDate = nextBusinessDay(historyData[historyData.length - 1]?.date);

  // Historical close line
  const histDs = {
    label: 'Actual Close',
    data: recent.map(d => ({ x: toTs(d.date), y: +d.close })),
    borderColor:     C.white,
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    pointHoverRadius: 4,
    tension: 0.3,
    fill: false,
  };

  // Bridge from last actual → predicted
  const bridgeDs = {
    label: 'Predicted',
    data: [
      { x: toTs(recent[recent.length - 1].date), y: +recent[recent.length - 1].close },
      { x: toTs(nextDate), y: result.predicted_price },
    ],
    borderColor:          C.red,
    backgroundColor:      'transparent',
    borderWidth:          2.5,
    pointRadius:          [0, 8],
    pointBackgroundColor: C.red,
    pointBorderColor:     C.white,
    pointBorderWidth:     2,
    borderDash:           [6, 4],
    tension: 0,
    fill: false,
  };

  // Confidence interval shading — build fill area between upper & lower
  const ciUpperDs = {
    label: '95% Upper',
    data: recent.map(d => ({ x: toTs(d.date), y: null }))
          .concat([
            { x: toTs(recent[recent.length - 1].date), y: result.confidence_upper },
            { x: toTs(nextDate), y: result.confidence_upper },
          ]),
    borderColor:     'transparent',
    backgroundColor: 'rgba(230,57,70,0.15)',
    pointRadius:     0,
    fill: '+1',
    tension: 0,
  };

  const ciLowerDs = {
    label: '95% Lower',
    data: recent.map(d => ({ x: toTs(d.date), y: null }))
          .concat([
            { x: toTs(recent[recent.length - 1].date), y: result.confidence_lower },
            { x: toTs(nextDate), y: result.confidence_lower },
          ]),
    borderColor:     'rgba(230,57,70,0.3)',
    backgroundColor: 'transparent',
    borderWidth:     1,
    borderDash:      [3, 3],
    pointRadius:     0,
    tension: 0,
    fill: false,
  };

  const ctx = document.getElementById('predictChart').getContext('2d');
  predictChart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [histDs, bridgeDs, ciUpperDs, ciLowerDs] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111',
          borderColor: C.red,
          borderWidth: 1,
          callbacks: {
            title: items => new Date(items[0].parsed.x).toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' }),
            label: ctx => {
              const v = ctx.parsed.y;
              if (v == null) return null;
              return `${ctx.dataset.label}: $${v.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: timeX('day'),
        y: priceY,
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════
//  FORECAST TAB — LINE CHART with CI
// ═══════════════════════════════════════════════════════════
function drawForecastChart(horizon = 7) {
  if (forecastChart) { forecastChart.destroy(); forecastChart = null; }
  if (!forecastData.length) return;

  const slice  = forecastData.slice(0, horizon);
  const recent = historyData.slice(-30);

  const spanDays = horizon <= 7 ? 'day' : 'week';

  const histDs = {
    label: 'Historical Close',
    data: recent.map(d => ({ x: toTs(d.date), y: +d.close })),
    borderColor:     C.white,
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  };

  const fcDs = {
    label: 'Forecast',
    data: slice.map(d => ({ x: toTs(d.date), y: d.predicted })),
    borderColor:          C.red,
    backgroundColor:      'transparent',
    borderWidth:          2.5,
    pointRadius:          3,
    pointBackgroundColor: C.red,
    tension: 0.3,
    fill: false,
  };

  const ciUpperDs = {
    label: '95% Upper',
    data: slice.map(d => ({ x: toTs(d.date), y: d.upper_95 })),
    borderColor:     'rgba(230,57,70,0.4)',
    backgroundColor: 'rgba(230,57,70,0.12)',
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    tension: 0.3,
    fill: '+1',
  };

  const ciLowerDs = {
    label: '95% Lower',
    data: slice.map(d => ({ x: toTs(d.date), y: d.lower_95 })),
    borderColor:     'rgba(230,57,70,0.4)',
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  };

  const ctx = document.getElementById('forecastChart').getContext('2d');
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [histDs, fcDs, ciUpperDs, ciLowerDs] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: C.text, boxWidth: 12, filter: item => item.text !== '95% Lower' }
        },
        tooltip: {
          backgroundColor: '#111',
          borderColor: C.red,
          borderWidth: 1,
          callbacks: {
            title: items => new Date(items[0].parsed.x).toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' }),
            label: ctx => {
              const v = ctx.parsed.y;
              if (v == null) return null;
              return `${ctx.dataset.label}: $${v.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: timeX(spanDays),
        y: priceY,
      }
    }
  });

  // Date range subtitle
  if (slice.length) {
    const last = historyData[historyData.length - 1];
    const sub  = last ? `From ${last.date}` : '';
    document.getElementById('fc-subtitle').textContent = sub;
  }

  // Populate table
  const lastClose = historyData.length ? +historyData[historyData.length - 1].close : null;
  const tbody = document.getElementById('forecast-tbody');
  tbody.innerHTML = slice.map(d => {
    const chg = lastClose ? d.predicted - lastClose : 0;
    const pct = lastClose ? (chg / lastClose * 100).toFixed(2) : '—';
    const cls = chg >= 0 ? 'td-up' : 'td-down';
    return `<tr>
      <td>${d.date}</td>
      <td>Day ${d.horizon_days}</td>
      <td class="${cls}">${fmt(d.predicted)}</td>
      <td>${fmt(d.lower_95)}</td>
      <td>${fmt(d.upper_95)}</td>
      <td class="${cls}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)} (${pct}%)</td>
    </tr>`;
  }).join('');

  document.getElementById('forecast-loader').classList.add('hidden');
}

function setForecastHorizon(days) {
  fcHorizon = days;
  document.getElementById('btn-fc-7').classList.toggle('active',  days === 7);
  document.getElementById('btn-fc-30').classList.toggle('active', days === 30);
  document.getElementById('btn-fc-90').classList.toggle('active', days === 90);
  drawForecastChart(days);
}

// ═══════════════════════════════════════════════════════════
//  MODEL STATS TAB
// ═══════════════════════════════════════════════════════════
async function loadMetrics() {
  const data = await apiFetch('/api/metrics');
  if (!data) return;
  const tm = data.training_metrics || {};
  const te = data.test_metrics     || {};
  const fu = data.full_metrics     || {};

  const set = (id, val, d = 2) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val != null ? (+val).toFixed(d) : '—';
  };
  set('m-train-mae',  tm.ens_mae);
  set('m-train-rmse', tm.ens_rmse);
  set('m-val-r2',     tm.ens_r2, 4);
  set('m-test-mae',   te.ens_mae);
  set('m-test-rmse',  te.ens_rmse);
  set('m-test-r2',    te.ens_r2, 4);
  set('m-test-mape',  te.ens_mape);
  set('m-full-r2',    fu.full_r2, 4);
}

async function loadModelInfo() {
  const data = await apiFetch('/api/model-info');
  if (!data) return;
  const s = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v ?? '—'; };
  s('inf-type',        data.model_type);
  s('inf-features',    data.feature_count);
  s('inf-train',       data.train_size);
  s('inf-val',         data.val_size);
  s('inf-test',        data.test_size);
  s('inf-train-range', data.train_date_range ? data.train_date_range.join(' → ') : '—');
  s('inf-test-range',  data.test_date_range  ? data.test_date_range.join(' → ')  : '—');
}

async function loadTestPredictions() {
  const data = await apiFetch('/api/test-predictions');
  const loader = document.getElementById('avp-loader');
  if (!data || !data.length) { if (loader) loader.classList.add('hidden'); return; }

  if (avpChart) { avpChart.destroy(); avpChart = null; }
  const ctx = document.getElementById('avpChart').getContext('2d');

  avpChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Actual',
          data: data.map(d => ({ x: toTs(d.date), y: d.actual })),
          borderColor:     C.white,
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: 'Ensemble Predicted',
          data: data.map(d => ({ x: toTs(d.date), y: d.ens_pred })),
          borderColor:     C.red,
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          borderDash: [5, 3],
          tension: 0.3,
          fill: false,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: C.text, boxWidth: 12 } },
        tooltip: {
          backgroundColor: '#111',
          borderColor: C.red,
          borderWidth: 1,
          callbacks: {
            title: items => new Date(items[0].parsed.x).toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' }),
            label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`,
          }
        }
      },
      scales: {
        x: timeX('month'),
        y: priceY,
      }
    }
  });

  if (loader) loader.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ═══════════════════════════════════════════════════════════
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  document.getElementById('nav-' + tabId).classList.add('active');

  if (tabId === 'forecast' && forecastData.length) drawForecastChart(fcHorizon);
  if (tabId === 'model') {
    loadMetrics();
    loadModelInfo();
    loadTestPredictions();
  }
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    switchTab(item.dataset.tab);
    document.getElementById('sidebar').classList.remove('open');
  });
});

// ── Hamburger ─────────────────────────────────────────────
document.getElementById('hamburger').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
});

// ═══════════════════════════════════════════════════════════
//  PWA INSTALL
// ═══════════════════════════════════════════════════════════
let deferredPrompt = null;
window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('install-banner').classList.remove('hidden');
});
document.getElementById('btn-install').addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  document.getElementById('install-banner').classList.add('hidden');
});
document.getElementById('btn-dismiss').addEventListener('click', () => {
  document.getElementById('install-banner').classList.add('hidden');
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/service-worker.js').catch(() => {});
}

// ═══════════════════════════════════════════════════════════
//  API HELPERS & STATUS
// ═══════════════════════════════════════════════════════════
async function apiFetch(path) {
  try {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn('API fetch failed:', path, e.message);
    return null;
  }
}

async function checkStatus() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const r = await fetch(API + '/ping');
    if (r.ok) { dot.className = 'status-dot online'; text.textContent = 'API Online'; }
    else       throw new Error();
  } catch {
    dot.className = 'status-dot offline'; text.textContent = 'API Offline';
  }
}

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════
async function init() {
  checkStatus();
  setInterval(checkStatus, 30000);

  // Load all history from API
  const hist = await apiFetch('/api/history?limit=3903');
  if (hist && hist.length) {
    historyData = hist;

    // Update live price card
    const last = hist[hist.length - 1];
    const prev = hist[hist.length - 2];
    document.getElementById('live-price').textContent = fmt(last.close);
    if (prev) {
      const chg = +last.close - +prev.close;
      const pct = (chg / +prev.close * 100).toFixed(2);
      const el  = document.getElementById('live-change');
      el.textContent = `${chg >= 0 ? '+' : ''}${chg.toFixed(2)} (${pct}%)`;
      el.className   = 'live-change ' + (chg >= 0 ? 'up' : 'down');
    }

    // Initial dashboard render — default to "All" data so charts always show
    const allBtn = document.querySelector('.range-bar .range-btn[data-days="0"]');
    setDashRange(0, allBtn);
  } else {
    document.getElementById('candle-loader').classList.add('hidden');
    document.getElementById('line-loader').classList.add('hidden');
  }

  // Load forecast
  const fc = await apiFetch('/api/forecast');
  if (fc && fc.length) forecastData = fc;
}

init();
