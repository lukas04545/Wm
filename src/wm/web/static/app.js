/* FIFA 2026 Predictor — frontend app */

let oddsChart = null;
let eloChart = null;
let simData = null;
let groupData = null;
let eloData = null;
let statusData = null;
let sortCol = 'champion';
let sortDesc = true;

// ── Navigation ────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => { t.classList.remove('active'); t.classList.add('hidden'); });
    btn.classList.add('active');
    const tab = document.getElementById('tab-' + btn.dataset.tab);
    tab.classList.remove('hidden');
    tab.classList.add('active');

    // Lazy-load tab content
    if (btn.dataset.tab === 'odds' && !simData) loadOdds();
    if (btn.dataset.tab === 'groups' && !groupData) loadGroups();
    if (btn.dataset.tab === 'elo' && !eloData) loadElo();
    if (btn.dataset.tab === 'status') loadStatus();
  });
});

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// ── ODDS TAB ──────────────────────────────────────────────────────────────────
async function loadOdds() {
  try {
    simData = await fetchJSON('/api/simulation');
    renderOdds();
  } catch {
    show('odds-unavailable');
  } finally {
    hide('odds-loading');
  }
}

function renderOdds() {
  if (!simData) return;
  const rows = Object.entries(simData)
    .filter(([k]) => !k.startsWith('_'))
    .map(([team, d]) => ({ team, ...d }));

  rows.sort((a, b) => {
    const av = (a[sortCol] || 0), bv = (b[sortCol] || 0);
    return sortDesc ? bv - av : av - bv;
  });

  // Chart (top 16 by champion prob)
  const top16 = rows.slice(0, 16);
  renderOddsChart(top16);

  // Table
  const tbody = document.getElementById('odds-tbody');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const champ = ((r.champion || 0) * 100).toFixed(1);
    const final = ((r.final || 0) * 100).toFixed(1);
    const sf = ((r.semifinal || 0) * 100).toFixed(1);
    const qf = ((r.quarterfinal || 0) * 100).toFixed(1);
    const qual = ((r.group_qualified || 0) * 100).toFixed(1);
    const pts = (r.avg_group_points || 0).toFixed(2);
    const group = teamGroup(r.team);
    const cls = i === 0 ? 'top-1' : i < 3 ? 'top-3' : '';
    tbody.innerHTML += `<tr class="${cls}">
      <td>${i + 1}</td>
      <td><strong>${r.team}</strong></td>
      <td><span style="color:var(--text-muted)">${group}</span></td>
      <td>${probBar(champ, 25, '#f0b429')}</td>
      <td>${probBar(final, 50, '#58a6ff')}</td>
      <td>${probBar(sf, 100, '#58a6ff')}</td>
      <td>${probBar(qf, 100, '#8b949e')}</td>
      <td>${probBar(qual, 100, '#3fb950')}</td>
      <td style="font-variant-numeric:tabular-nums">${pts}</td>
    </tr>`;
  });

  // Finals panel
  const meta = simData['_meta'] || {};
  const finals = meta.top_final_pairings || [];
  const ul = document.getElementById('finals-list');
  ul.innerHTML = '';
  finals.slice(0, 8).forEach(([teams, prob], i) => {
    const cls = i === 0 ? 'gold' : '';
    ul.innerHTML += `<li>
      <div class="finals-rank ${cls}">${i + 1}</div>
      <div class="finals-matchup"><strong>${teams[0]}</strong> vs <strong>${teams[1]}</strong></div>
      <div class="finals-prob">${(prob * 100).toFixed(2)}%</div>
    </li>`;
  });

  show('odds-content');

  // Sort headers
  document.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      if (sortCol === th.dataset.col) sortDesc = !sortDesc;
      else { sortCol = th.dataset.col; sortDesc = true; }
      document.querySelectorAll('th.sortable').forEach(t => t.classList.remove('active-sort'));
      th.classList.add('active-sort');
      renderOdds();
    });
  });
}

function probBar(val, maxVal, color) {
  const pct = Math.min((parseFloat(val) / maxVal) * 100, 100);
  return `<div class="prob-bar">
    <div class="prob-fill" style="width:${pct}%;background:${color}"></div>
    <span class="prob-text">${val}%</span>
  </div>`;
}

function renderOddsChart(rows) {
  const labels = rows.map(r => r.team);
  const values = rows.map(r => ((r.champion || 0) * 100).toFixed(2));

  const colors = values.map((v, i) => {
    const h = 40 - i * 2;
    return `hsl(${h}, 90%, 55%)`;
  });

  const ctx = document.getElementById('oddsChart').getContext('2d');
  if (oddsChart) oddsChart.destroy();
  oddsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Championship probability (%)',
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.raw}% championship probability`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b949e', callback: v => v + '%' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e6edf3', font: { size: 12 } }
        }
      }
    }
  });
}

// ── GROUPS TAB ────────────────────────────────────────────────────────────────
async function loadGroups() {
  try {
    const [grps, elos, sim] = await Promise.all([
      fetchJSON('/api/groups'),
      fetchJSON('/api/elo').catch(() => ({ rankings: [] })),
      fetchJSON('/api/simulation').catch(() => null),
    ]);
    groupData = grps.groups;
    const eloMap = Object.fromEntries(elos.rankings.map(e => [e.team, e.elo]));
    const qualMap = sim
      ? Object.fromEntries(
          Object.entries(sim)
            .filter(([k]) => !k.startsWith('_'))
            .map(([t, d]) => [t, (d.group_qualified || 0) * 100])
        )
      : {};
    renderGroups(eloMap, qualMap);
  } catch (e) {
    document.getElementById('groups-grid').innerHTML = `<p style="color:var(--danger)">${e.message}</p>`;
  } finally {
    hide('groups-loading');
    show('groups-grid');
  }
}

function renderGroups(eloMap, qualMap) {
  const grid = document.getElementById('groups-grid');
  grid.innerHTML = '';
  Object.entries(groupData).forEach(([grp, teams]) => {
    const sorted = [...teams].sort((a, b) => (eloMap[b] || 1500) - (eloMap[a] || 1500));
    let html = `<div class="group-card">
      <div class="group-header">Group <span>${grp}</span></div>`;
    sorted.forEach(team => {
      const elo = eloMap[team] || 1500;
      const qual = qualMap[team];
      let qualHtml = '';
      if (qual !== undefined) {
        const cls = qual >= 65 ? 'qual-high' : qual >= 40 ? 'qual-mid' : 'qual-low';
        qualHtml = `<span class="group-qual-prob ${cls}">${qual.toFixed(0)}%</span>`;
      }
      html += `<div class="group-team">
        <span class="group-team-name">${team}</span>
        <span class="group-elo">Elo ${elo}</span>
        ${qualHtml}
      </div>`;
    });
    html += '</div>';
    grid.innerHTML += html;
  });
}

// ── PREDICT TAB ───────────────────────────────────────────────────────────────
async function initPredict() {
  try {
    const [status, teams] = await Promise.all([
      fetchJSON('/api/status'),
      fetchJSON('/api/teams'),
    ]);
    statusData = status;

    if (!status.models_loaded) {
      show('predict-unavailable');
      return;
    }

    const selHome = document.getElementById('sel-home');
    const selAway = document.getElementById('sel-away');

    teams.teams.forEach(t => {
      selHome.innerHTML += `<option value="${t.name}">${t.name} (Elo ${t.elo})</option>`;
      selAway.innerHTML += `<option value="${t.name}">${t.name} (Elo ${t.elo})</option>`;
    });

    // Default: Argentina vs Brazil
    const argIdx = teams.teams.findIndex(t => t.name === 'Argentina');
    const braIdx = teams.teams.findIndex(t => t.name === 'Brazil');
    if (argIdx >= 0) selHome.selectedIndex = argIdx;
    if (braIdx >= 0) selAway.selectedIndex = braIdx;
  } catch {
    show('predict-unavailable');
  }
}

async function runPredict() {
  const home = document.getElementById('sel-home').value;
  const away = document.getElementById('sel-away').value;
  const venue = document.getElementById('sel-venue').value;

  if (!home || !away) return;
  if (home === away) {
    alert('Please select two different teams.');
    return;
  }

  hide('predict-result');
  show('predict-loading');

  try {
    const data = await fetchJSON('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home, away, venue, neutral: true }),
    });
    renderPredictResult(data);
    show('predict-result');
  } catch (e) {
    document.getElementById('predict-result').innerHTML =
      `<div class="info-box" style="color:var(--danger)">⚠ ${e.message}</div>`;
    show('predict-result');
  } finally {
    hide('predict-loading');
  }
}

function renderPredictResult(d) {
  const el = document.getElementById('predict-result');

  // Scoreline chips sorted by probability
  const chips = d.top_scorelines
    .map(s => `<div class="scoreline-chip">
      ${s.home}–${s.away}<span class="sc-prob">${(s.prob * 100).toFixed(1)}%</span>
    </div>`)
    .join('');

  el.innerHTML = `
    <div class="result-header">
      <div class="result-teams">
        <strong>${d.home}</strong>
        <span class="vs">vs</span>
        <strong>${d.away}</strong>
      </div>
      <div class="result-elos">Elo: ${d.elo_home} — ${d.elo_away}</div>
    </div>
    <div class="result-probs">
      <div class="result-prob-cell home-win">
        <div class="result-prob-label">${d.home} win</div>
        <div class="result-prob-value">${d.p_home_win}%</div>
      </div>
      <div class="result-prob-cell draw-result">
        <div class="result-prob-label">Draw</div>
        <div class="result-prob-value">${d.p_draw}%</div>
      </div>
      <div class="result-prob-cell away-win">
        <div class="result-prob-label">${d.away} win</div>
        <div class="result-prob-value">${d.p_away_win}%</div>
      </div>
    </div>
    <div class="result-goals">
      <span>Expected goals: <strong>${d.lambda_home}</strong> — <strong>${d.lambda_away}</strong></span>
    </div>
    <div class="scoreline-grid">
      <h4>Top scorelines</h4>
      <div class="scorelines">${chips}</div>
    </div>`;
}

// ── ELO TAB ───────────────────────────────────────────────────────────────────
async function loadElo() {
  try {
    eloData = await fetchJSON('/api/elo');
    renderElo();
  } catch (e) {
    hide('elo-loading');
    document.getElementById('elo-content').innerHTML =
      `<div class="info-box">Elo data unavailable: ${e.message}</div>`;
    show('elo-content');
  }
}

function renderElo() {
  hide('elo-loading');
  const rows = eloData.rankings;
  const maxElo = rows[0]?.elo || 2100;
  const minElo = Math.min(...rows.map(r => r.elo));

  // Chart: top 20
  const top20 = rows.slice(0, 20);
  const ctx = document.getElementById('eloChart').getContext('2d');
  if (eloChart) eloChart.destroy();
  eloChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top20.map(r => r.team),
      datasets: [{
        label: 'Elo Rating',
        data: top20.map(r => r.elo),
        backgroundColor: top20.map((_, i) => `hsl(${140 - i * 5}, 70%, 50%)`),
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: {
          min: Math.max(minElo - 50, 1000),
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b949e' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e6edf3', font: { size: 12 } }
        }
      }
    }
  });

  const tbody = document.getElementById('elo-tbody');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const pct = ((r.elo - 1000) / (maxElo - 1000)) * 100;
    tbody.innerHTML += `<tr>
      <td>${i + 1}</td>
      <td>${r.team}</td>
      <td style="font-variant-numeric:tabular-nums;font-weight:600">${r.elo}</td>
      <td style="width:200px"><div class="elo-bar" style="width:${pct}%"></div></td>
    </tr>`;
  });
  show('elo-content');
}

// ── STATUS TAB ────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    statusData = await fetchJSON('/api/status');
    const cards = document.getElementById('status-cards');
    cards.innerHTML = `
      <div class="status-card">
        <div class="status-icon">📥</div>
        <div class="status-label">Match Data</div>
        <div class="${statusData.data_ingested ? 'status-ok' : 'status-warn'}">
          ${statusData.data_ingested ? '✓ Ingested' : 'Not downloaded'}
        </div>
      </div>
      <div class="status-card">
        <div class="status-icon">🤖</div>
        <div class="status-label">ML Models</div>
        <div class="${statusData.models_loaded ? 'status-ok' : 'status-warn'}">
          ${statusData.models_loaded ? '✓ Loaded' : 'Not trained'}
        </div>
      </div>
      <div class="status-card">
        <div class="status-icon">🎲</div>
        <div class="status-label">Simulation</div>
        <div class="${statusData.simulation_available ? 'status-ok' : 'status-warn'}">
          ${statusData.simulation_available ? '✓ Available' : 'Not run'}
        </div>
      </div>
      <div class="status-card">
        <div class="status-icon">🏟️</div>
        <div class="status-label">Teams Loaded</div>
        <div class="${statusData.n_teams === 48 ? 'status-ok' : 'status-warn'}">
          ${statusData.n_teams} / 48
        </div>
      </div>`;
  } catch { /* ignore */ }
}

// ── Quick simulate ────────────────────────────────────────────────────────────
async function quickSimulate() {
  const btn = document.querySelector('#odds-unavailable .btn-primary');
  const statusEl = document.getElementById('sim-status');
  btn.disabled = true;
  show('sim-status');
  statusEl.textContent = '⏳ Starting simulation…';

  try {
    const r = await fetchJSON('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ runs: 10000, seed: 42 }),
    });
    statusEl.textContent = '⏳ Running 10,000 simulations… this may take a minute.';

    // Poll until done
    const poll = setInterval(async () => {
      const s = await fetchJSON('/api/simulate/status').catch(() => null);
      if (s && !s.running && s.available) {
        clearInterval(poll);
        statusEl.textContent = '✓ Done! Loading results…';
        hide('odds-unavailable');
        show('odds-loading');
        simData = null;
        await loadOdds();
      } else if (s && !s.running && !s.available) {
        clearInterval(poll);
        statusEl.textContent = '⚠ Simulation failed — models may not be trained yet.';
        btn.disabled = false;
      }
    }, 2000);
  } catch (e) {
    statusEl.textContent = `⚠ ${e.message}`;
    btn.disabled = false;
  }
}

// ── Group → team lookup ───────────────────────────────────────────────────────
function teamGroup(team) {
  if (!groupData) return '?';
  for (const [g, teams] of Object.entries(groupData)) {
    if (teams.includes(team)) return g;
  }
  return '?';
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  await loadOdds();
  await loadGroups();
  await initPredict();
  await loadStatus();
})();
