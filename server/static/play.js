/**
 * Human vs bot — uses POST /play/api/new, /play/api/step, /play/api/cancel_selection.
 * Server is source of truth for legality; client sends structured kinds only.
 */
(function () {
  const canvas = document.getElementById('playCanvas');
  const statusEl = document.getElementById('playStatus');
  const btnStart = document.getElementById('btnStart');
  const btnCop = document.getElementById('btnCop');
  const btnScop = document.getElementById('btnScop');
  const btnEnd = document.getElementById('btnEnd');
  const btnWait = document.getElementById('btnWait');
  const btnCapture = document.getElementById('btnCapture');
  const btnJoin = document.getElementById('btnJoin');
  const btnDiveHide = document.getElementById('btnDiveHide');
  const powerHud = document.getElementById('powerHud');
  const buildModal = document.getElementById('buildModal');
  const buildModalTitle = document.getElementById('buildModalTitle');
  const buildModalFunds = document.getElementById('buildModalFunds');
  const buildUnitSelect = document.getElementById('buildUnitSelect');
  const buildModalBuild = document.getElementById('buildModalBuild');
  const buildModalCancel = document.getElementById('buildModalCancel');

  let sessionId = null;
  let last = null;
  /** @type {{ r: number, c: number } | null} */
  let buildPendingFactory = null;

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = 'status ' + (cls || 'warn');
  }

  function tileIncludes(list, r, c) {
    if (!list) return false;
    const rr = Number(r);
    const cc = Number(c);
    return list.some((p) => Number(p[0]) === rr && Number(p[1]) === cc);
  }

  /** True if [r,c] is the committed MOVE destination (JSON coords may be strings). */
  function sameMoveDest(mp, r, c) {
    if (!mp || mp.length < 2) return false;
    return Number(mp[0]) === Number(r) && Number(mp[1]) === Number(c);
  }

  /**
   * Map pointer to grid cell using the canvas's **displayed** size (CSS pixels).
   * Do not use AWBW_TILE_SCREEN_PX here — the bitmap may be scaled by layout/zoom.
   */
  function canvasCell(ev) {
    const board = last && last.board;
    if (!board || !board.width || !board.height) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const tileW = rect.width / board.width;
    const tileH = rect.height / board.height;
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    let c = Math.floor(x / tileW);
    let r = Math.floor(y / tileH);
    c = Math.max(0, Math.min(board.width - 1, c));
    r = Math.max(0, Math.min(board.height - 1, r));
    return { r, c };
  }

  function pctLabel(x) {
    const v = typeof x === 'number' ? x : 0;
    return `${Math.round(Math.min(1, Math.max(0, v)) * 100)}%`;
  }

  function syncPowerHud(p) {
    if (!p || !p.co_p0 || !p.co_p1 || !powerHud) return;
    powerHud.style.display = 'block';
    document.getElementById('hudCop0').textContent = pctLabel(p.co_p0.cop_pct);
    document.getElementById('hudScop0').textContent = pctLabel(p.co_p0.scop_pct);
    document.getElementById('hudCop1').textContent = pctLabel(p.co_p1.cop_pct);
    document.getElementById('hudScop1').textContent = pctLabel(p.co_p1.scop_pct);
  }

  function buildHighlights(p) {
    if (!p || !p.board) return null;
    if (p.active_player !== 0 || p.done) return null;
    const st = p.action_stage;
    if (st === 'SELECT') {
      return {
        selected: null,
        reachable: [],
        attack: [],
        repair: [],
        factory_build: p.factory_build_tiles || [],
      };
    }
    if (st === 'MOVE') {
      return {
        selected: p.selected_unit_pos,
        reachable: p.reachable_tiles || [],
        attack: [],
        repair: [],
      };
    }
    if (st === 'ACTION') {
      const unloadTiles = (p.unload_options || []).map((o) => o.target_pos);
      return {
        selected: p.selected_move_pos || p.selected_unit_pos,
        reachable: [],
        attack: p.attack_targets || [],
        repair: p.repair_targets || [],
        unload: unloadTiles,
      };
    }
    return null;
  }

  function draw() {
    if (!last || !last.board) return;
    const ov = buildHighlights(last);
    renderBoard(canvas, last.board, ov);
  }

  function syncGlobalButtons(p) {
    const lg = p.legal_global || {};
    const humanSelect = p.active_player === 0 && p.action_stage === 'SELECT' && !p.done;
    btnCop.disabled = !humanSelect || !lg.cop;
    btnScop.disabled = !humanSelect || !lg.scop;
    btnEnd.disabled = !humanSelect || !lg.end_turn;
    const opts = p.action_options || [];
    const inAction = p.active_player === 0 && p.action_stage === 'ACTION' && !p.done;
    const showWait = inAction && opts.includes('WAIT');
    btnWait.style.display = showWait ? 'inline-block' : 'none';
    btnWait.disabled = !showWait;
    if (btnCapture) {
      const showCap = inAction && opts.includes('CAPTURE');
      btnCapture.style.display = showCap ? 'inline-block' : 'none';
      btnCapture.disabled =
        !showCap || !p.selected_unit_pos || !p.selected_move_pos;
    }
    if (btnJoin) {
      const showJoin = inAction && opts.includes('JOIN');
      btnJoin.style.display = showJoin ? 'inline-block' : 'none';
      btnJoin.disabled =
        !showJoin || !p.selected_unit_pos || !p.selected_move_pos;
    }
    if (btnDiveHide) {
      const showDh = inAction && opts.includes('DIVE_HIDE');
      btnDiveHide.style.display = showDh ? 'inline-block' : 'none';
      btnDiveHide.disabled =
        !showDh || !p.selected_unit_pos || !p.selected_move_pos;
    }
  }

  function applyPayload(p) {
    last = p;
    sessionId = p.session_id || sessionId;
    if (!p.ok && p.error) {
      setStatus(p.error, 'err');
      if (p.board) draw();
      syncGlobalButtons(p);
      syncPowerHud(p);
      return;
    }
    if (p.done) {
      const w = p.winner;
      let msg = 'Game over.';
      if (w === 0) msg = 'You win.';
      else if (w === 1) msg = 'Bot wins.';
      else if (w === -1) msg = 'Draw / max turns.';
      setStatus(msg, w === 0 ? 'ok' : 'warn');
    } else if (p.active_player === 1) {
      setStatus('Bot thinking… (blocked on server until your turn)', 'warn');
    } else {
      const bm = typeof p.bot_mode === 'string' ? p.bot_mode : '';
      let extra = '';
      if (bm.startsWith('book+')) {
        const tail = bm.endsWith('ppo') ? 'PPO' : 'random legal';
        extra = ` | P1: data/opening_books/std_pool_precombat.jsonl while line matches, then ${tail}`;
      } else if (bm === 'random') {
        extra =
          ' | Bot: random legal moves (train → checkpoints/latest.zip for PPO)';
      }
      setStatus(
        `Your turn — ${p.action_stage} | Day ${p.turn} | Funds ${p.funds[0]}g (you) vs ${p.funds[1]}g${extra}`,
        'ok'
      );
    }
    draw();
    syncGlobalButtons(p);
    syncPowerHud(p);
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    return { res, data };
  }

  btnStart.addEventListener('click', async () => {
    setStatus('Starting…', 'warn');
    const { res, data } = await postJson('/play/api/new', {
      map_id: 123858,
      tier: 'T4',
      human_co_id: 14,
      bot_co_id: 14,
    });
    if (!res.ok) {
      setStatus(data.error || 'Failed to start (need checkpoint?)', 'err');
      return;
    }
    applyPayload(data);
  });

  async function step(body) {
    if (!sessionId) return;
    const { res, data } = await postJson('/play/api/step', { session_id: sessionId, ...body });
    applyPayload(data);
    if (!res.ok && data.error) {
      /* keep message */
    }
  }

  function syncBuildModalControls() {
    if (!buildUnitSelect || !buildModalBuild) return;
    const v = buildUnitSelect.value;
    buildModalBuild.disabled = !v;
  }

  function closeBuildModal() {
    buildPendingFactory = null;
    if (buildModal) buildModal.style.display = 'none';
    if (buildUnitSelect) {
      buildUnitSelect.innerHTML = '';
      const ph = document.createElement('option');
      ph.value = '';
      ph.disabled = true;
      ph.selected = true;
      ph.textContent = 'Choose a unit…';
      buildUnitSelect.appendChild(ph);
    }
    syncBuildModalControls();
  }

  function openBuildModal(r, c) {
    if (!buildModal || !buildModalTitle || !buildUnitSelect || !buildModalFunds) return;
    const rr = Number(r);
    const cc = Number(c);
    const menu = last.factory_build_menu || [];
    const entry = menu.find(
      (e) => e.pos && Number(e.pos[0]) === rr && Number(e.pos[1]) === cc
    );
    if (!entry || !entry.options.length) {
      const funds = last.funds && last.funds[0] !== undefined ? last.funds[0] : '?';
      const onList = tileIncludes(last.factory_build_tiles || [], rr, cc);
      if (onList) {
        setStatus(
          `No buildable units at this tile (funds ${funds}g — need empty owned base/airport/port, below unit cap). If this persists, hard-refresh the page.`,
          'warn'
        );
      } else {
        setStatus(`Nothing buildable at row ${rr}, col ${cc}.`, 'warn');
      }
      return;
    }
    buildPendingFactory = { r: rr, c: cc };
    buildModalTitle.textContent = `Build — row ${rr}, col ${cc}`;
    const fundsYou = last.funds && last.funds[0] !== undefined ? last.funds[0] : '—';
    buildModalFunds.textContent = `Your funds: ${fundsYou}g`;
    buildUnitSelect.innerHTML = '';
    const ph = document.createElement('option');
    ph.value = '';
    ph.disabled = true;
    ph.selected = true;
    ph.textContent = 'Choose a unit…';
    buildUnitSelect.appendChild(ph);
    for (const opt of entry.options) {
      const o = document.createElement('option');
      o.value = opt.unit_type;
      o.textContent = `${opt.unit_type} — ${opt.cost}g`;
      buildUnitSelect.appendChild(o);
    }
    syncBuildModalControls();
    buildModal.style.display = 'flex';
  }

  async function cancelSel() {
    closeBuildModal();
    if (!sessionId) return;
    const { data } = await postJson('/play/api/cancel_selection', { session_id: sessionId });
    applyPayload(data);
  }

  btnCop.addEventListener('click', () => step({ kind: 'cop' }));
  btnScop.addEventListener('click', () => step({ kind: 'scop' }));
  btnEnd.addEventListener('click', () => step({ kind: 'end_turn' }));
  btnWait.addEventListener('click', () => {
    if (!last || !last.selected_unit_pos || !last.selected_move_pos) return;
    step({
      kind: 'wait',
      unit_pos: last.selected_unit_pos,
      move_pos: last.selected_move_pos,
    });
  });

  if (btnCapture) {
    btnCapture.addEventListener('click', () => {
      if (!last || !last.selected_unit_pos || !last.selected_move_pos) return;
      const opts = last.action_options || [];
      if (!opts.includes('CAPTURE')) return;
      step({
        kind: 'capture',
        unit_pos: last.selected_unit_pos,
        move_pos: last.selected_move_pos,
      });
    });
  }

  if (btnJoin) {
    btnJoin.addEventListener('click', () => {
      if (!last || !last.selected_unit_pos || !last.selected_move_pos) return;
      const opts = last.action_options || [];
      if (!opts.includes('JOIN')) return;
      step({
        kind: 'join',
        unit_pos: last.selected_unit_pos,
        move_pos: last.selected_move_pos,
      });
    });
  }

  if (btnDiveHide) {
    btnDiveHide.addEventListener('click', () => {
      if (!last || !last.selected_unit_pos || !last.selected_move_pos) return;
      const opts = last.action_options || [];
      if (!opts.includes('DIVE_HIDE')) return;
      step({
        kind: 'dive_hide',
        unit_pos: last.selected_unit_pos,
        move_pos: last.selected_move_pos,
      });
    });
  }

  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    if (buildModal && buildModal.style.display === 'flex') {
      closeBuildModal();
      return;
    }
    cancelSel();
  });

  if (buildUnitSelect) {
    buildUnitSelect.addEventListener('change', () => syncBuildModalControls());
  }

  if (buildModalBuild) {
    buildModalBuild.addEventListener('click', async () => {
      if (!buildPendingFactory || !buildUnitSelect) return;
      const ut = buildUnitSelect.value;
      if (!ut) return;
      const { r: br, c: bc } = buildPendingFactory;
      closeBuildModal();
      await step({
        kind: 'build',
        factory_pos: [br, bc],
        unit_type: ut,
      });
    });
  }

  if (buildModalCancel) {
    buildModalCancel.addEventListener('click', () => closeBuildModal());
  }

  canvas.addEventListener('click', async (ev) => {
    if (!last || !sessionId || last.done) return;
    if (last.active_player !== 0) {
      setStatus("Bot's turn — wait for your phase.", 'warn');
      return;
    }
    const cell = canvasCell(ev);
    if (!cell) return;
    const { r, c } = cell;
    const st = last.action_stage;

    if (st === 'SELECT') {
      const fb = last.factory_build_tiles || [];
      const sel = last.selectable_unit_tiles || [];
      if (tileIncludes(fb, r, c)) {
        openBuildModal(r, c);
        return;
      }
      if (tileIncludes(sel, r, c)) {
        await step({ kind: 'select_unit', unit_pos: [r, c] });
        return;
      }
      if (sel.length === 0 && fb.length === 0) {
        setStatus('No units or factories to act on — use End Turn if legal.', 'warn');
        return;
      }
      await cancelSel();
      return;
    }

    if (st === 'MOVE') {
      if (!tileIncludes(last.reachable_tiles, r, c)) {
        await cancelSel();
        return;
      }
      const up = last.selected_unit_pos;
      if (!up) return;
      await step({ kind: 'move_unit', unit_pos: up, move_pos: [r, c] });
      return;
    }

    if (st === 'ACTION') {
      const up = last.selected_unit_pos;
      const mp = last.selected_move_pos;
      if (!up || !mp) return;
      const opts = last.action_options || [];

      if (opts.includes('ATTACK') && tileIncludes(last.attack_targets, r, c)) {
        await step({
          kind: 'attack',
          unit_pos: up,
          move_pos: mp,
          target_pos: [r, c],
        });
        return;
      }
      if (opts.includes('REPAIR') && tileIncludes(last.repair_targets, r, c)) {
        await step({
          kind: 'repair',
          unit_pos: up,
          move_pos: mp,
          target_pos: [r, c],
        });
        return;
      }
      if (opts.includes('CAPTURE') && sameMoveDest(mp, r, c)) {
        await step({ kind: 'capture', unit_pos: up, move_pos: mp });
        return;
      }
      if (opts.includes('LOAD') && sameMoveDest(mp, r, c)) {
        await step({ kind: 'load', unit_pos: up, move_pos: mp });
        return;
      }
      if (opts.includes('JOIN') && sameMoveDest(mp, r, c)) {
        await step({ kind: 'join', unit_pos: up, move_pos: mp });
        return;
      }
      if (opts.includes('UNLOAD')) {
        const uo = last.unload_options || [];
        const hit = uo.find(
          (o) => Number(o.target_pos[0]) === Number(r) && Number(o.target_pos[1]) === Number(c)
        );
        if (hit) {
          await step({
            kind: 'unload',
            unit_pos: up,
            move_pos: mp,
            target_pos: hit.target_pos,
            unit_type: hit.unit_type,
          });
          return;
        }
      }

      await cancelSel();
    }
  });
})();
