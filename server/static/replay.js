/**
 * Replay viewer — step through recorded game actions.
 * Expects game data at /replay/api/<game_idx>
 *
 * Record shape (when per-step frames are recorded by the env):
 *   {
 *     board:   { height, width, terrain },     // static, written once
 *     frames:  [
 *       { turn, active_player, action, board: { units, properties } },
 *       ...
 *     ],
 *     ...summary fields
 *   }
 * Older records that only carry summary fields fall back to the
 * "Full replay recording not available" message.
 */

let currentGame  = null;
let currentStep  = 0;
let playTimer    = null;
let isPlaying    = false;

async function loadGame(gameIdx) {
  try {
    const res = await fetch(`/replay/api/${gameIdx}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    currentGame = await res.json();
    currentStep = 0;
    renderStep();
  } catch (e) {
    document.getElementById('step-info').textContent = `Load error: ${e.message}`;
    console.error('[replay] loadGame error:', e);
  }
}

/** Combine a per-frame `board` (units/properties) with the static terrain stored at game root. */
function mergedBoard(frame) {
  const frameBoard = frame.board || {};
  const staticBoard = currentGame.board || {};
  return {
    height:     frameBoard.height     ?? staticBoard.height,
    width:      frameBoard.width      ?? staticBoard.width,
    terrain:    frameBoard.terrain    ?? staticBoard.terrain,
    units:      frameBoard.units      ?? [],
    properties: frameBoard.properties ?? [],
  };
}

/** Build a compact human-readable label for the action that produced a frame. */
function describeAction(action) {
  if (!action) return 'start';
  const parts = [action.type];
  if (action.unit_type) parts.push(action.unit_type);
  if (action.unit_pos)  parts.push(`from ${action.unit_pos.join(',')}`);
  if (action.move_pos)  parts.push(`→ ${action.move_pos.join(',')}`);
  if (action.target_pos) parts.push(`tgt ${action.target_pos.join(',')}`);
  return parts.join(' ');
}

function renderStep() {
  if (!currentGame) return;

  const canvas  = document.getElementById('board-canvas');
  const info    = document.getElementById('step-info');
  const frames  = currentGame.frames;

  if (frames && frames.length > 0) {
    const frame = frames[Math.min(currentStep, frames.length - 1)];
    renderBoard(canvas, mergedBoard(frame));
    const actionLabel = describeAction(frame.action);
    const playerLabel = frame.active_player != null ? `P${frame.active_player}` : '—';
    const fundsLabel = Array.isArray(frame.funds)
      ? ` | Funds P0 ${frame.funds[0].toLocaleString()}g / P1 ${frame.funds[1].toLocaleString()}g`
      : '';
    info.textContent =
      `Step ${currentStep + 1} / ${frames.length} | Turn ${frame.turn ?? '—'} | ` +
      `${playerLabel} | ${actionLabel}${fundsLabel} | Map ${currentGame.map_id ?? '—'}`;
  } else {
    if (currentGame.board) {
      renderBoard(canvas, currentGame.board);
    }
    info.textContent =
      `Map ${currentGame.map_id ?? '—'} | ${currentGame.tier ?? ''} | ` +
      `P${currentGame.winner} wins | ${currentGame.turns ?? '?'} turns — ` +
      `Full replay recording not available for this game.`;
  }
}

function stepForward() {
  if (!currentGame) return;
  const max = (currentGame.frames?.length ?? 1) - 1;
  if (currentStep < max) {
    currentStep++;
    renderStep();
  } else if (isPlaying) {
    togglePlay();  // stop at end
  }
}

function stepBack() {
  if (!currentGame) return;
  if (currentStep > 0) {
    currentStep--;
    renderStep();
  }
}

function togglePlay() {
  const btn = document.getElementById('play-btn');
  if (isPlaying) {
    clearInterval(playTimer);
    playTimer = null;
    isPlaying = false;
    if (btn) btn.textContent = '▶ Play';
  } else {
    isPlaying = true;
    if (btn) btn.textContent = '⏸ Pause';
    playTimer = setInterval(() => stepForward(), 800);
  }
}
