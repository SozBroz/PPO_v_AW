/**
 * Live game watcher — polls /api/watch/state every 500ms and re-renders.
 */

let lastUpdatedAt = null;
let pollInterval  = null;
let consecutiveErrors = 0;

function formatFunds(f) {
  return '$' + Number(f).toLocaleString();
}

function setPollStatus(msg, ok = true) {
  const el = document.getElementById('poll-status');
  if (el) {
    el.textContent = msg;
    el.style.color = ok ? '#555' : '#ff9800';
  }
}

function updateGameInfo(state) {
  if (state.status === 'no_game') {
    document.getElementById('game-info').innerHTML = `
      <p class="status warn">${state.message}</p>
    `;
    document.getElementById('p0-info').textContent = '—';
    document.getElementById('p1-info').textContent = '—';
    document.getElementById('board-title').textContent = 'Board — no active game';
    return;
  }

  const active = state.active_player;
  const activeColor = active === 0 ? '#3388ff' : '#ff4444';

  document.getElementById('game-info').innerHTML = `
    <div style="font-size:0.85rem; line-height:1.9;">
      <div><b style="color:#e94560;">Map:</b> ${state.map_name || state.map_id || '—'}</div>
      <div><b style="color:#e94560;">Tier:</b> ${state.tier || '—'}</div>
      <div><b style="color:#e94560;">Turn:</b> ${state.turn ?? '—'}</div>
      <div><b style="color:#e94560;">Active:</b> <span style="color:${activeColor}; font-weight:bold;">P${active}</span></div>
      ${state.done
        ? `<div class="status ok" style="margin-top:8px;">Game Over — P${state.winner} wins!</div>`
        : ''}
    </div>
  `;

  function coPanel(co, funds) {
    if (!co) return '<span style="color:#555;">—</span>';
    const totalStars = co.scop_stars * 9000;
    const powerPct = totalStars > 0
      ? Math.min(100, Math.round((co.power_bar / totalStars) * 100))
      : 0;
    const powerLabel = co.scop_active
      ? '<span style="color:#ffd700; font-weight:bold;">SCOP Active!</span>'
      : co.cop_active
      ? '<span style="color:#ffd700; font-weight:bold;">COP Active!</span>'
      : `<span style="color:#888;">${powerPct}%</span>`;

    // Power bar visual
    const barFill = co.cop_active || co.scop_active ? '#ffd700' : '#0f3460';
    const barPct  = co.cop_active || co.scop_active ? 100 : powerPct;

    return `
      <div style="line-height:1.8;">
        <div><b style="color:#e0e0e0;">${co.name}</b></div>
        <div>Funds: <span style="color:#4caf50;">${formatFunds(funds)}</span></div>
        <div>Power: ${powerLabel}</div>
        <div style="background:#111; border-radius:2px; height:4px; margin-top:4px;">
          <div style="background:${barFill}; width:${barPct}%; height:100%; border-radius:2px; transition:width 0.3s;"></div>
        </div>
      </div>
    `;
  }

  document.getElementById('p0-info').innerHTML = coPanel(state.co_p0, (state.funds ?? [])[0] ?? 0);
  document.getElementById('p1-info').innerHTML = coPanel(state.co_p1, (state.funds ?? [])[1] ?? 0);

  const a = state.last_action;
  if (a) {
    const from = a.from ? a.from.join(',') : '';
    const to   = a.to   ? a.to.join(',')   : '';
    const tgt  = a.target ? ` → atk ${a.target.join(',')}` : '';
    document.getElementById('last-action').textContent =
      `Last: ${a.type}  [${from}]→[${to}]${tgt}`;
  } else {
    document.getElementById('last-action').textContent = '—';
  }

  document.getElementById('board-title').textContent =
    `${state.map_name || state.map_id || 'Board'} | Turn ${state.turn} | P${active}'s move`;
}

async function fetchAndRender() {
  try {
    const res = await fetch('/api/watch/state');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const state = await res.json();
    consecutiveErrors = 0;

    updateGameInfo(state);

    if (state.board) {
      // Only re-render if state actually changed (compare timestamp)
      if (state.updated_at !== lastUpdatedAt) {
        lastUpdatedAt = state.updated_at;
        const canvas = document.getElementById('board-canvas');

        // Highlight last action's destination tile
        let highlight = null;
        if (state.last_action?.to) highlight = state.last_action.to;

        renderBoard(canvas, state.board, highlight);
      }
    }

    const ts = state.updated_at
      ? new Date(state.updated_at * 1000).toLocaleTimeString()
      : '—';
    setPollStatus(`Last update: ${ts}`, true);
  } catch (e) {
    consecutiveErrors++;
    setPollStatus(`Poll error (${consecutiveErrors}): ${e.message}`, false);
    console.warn('[watch] poll error:', e);
  }
}

async function startGame() {
  const p0 = parseInt(document.getElementById('co-p0').value, 10);
  const p1 = parseInt(document.getElementById('co-p1').value, 10);

  try {
    const res = await fetch('/api/watch/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ co_p0: p0, co_p1: p1 }),
    });
    const data = await res.json();
    setPollStatus(`Started: ${data.cmd || '—'}`, true);
    // Give the subprocess a moment to write first state
    setTimeout(fetchAndRender, 1500);
  } catch (e) {
    setPollStatus(`Start failed: ${e.message}`, false);
  }
}

function startPolling() {
  fetchAndRender();
  pollInterval = setInterval(() => {
    if (document.getElementById('auto-refresh').checked) {
      fetchAndRender();
    }
  }, 500);
}

document.addEventListener('DOMContentLoaded', startPolling);
