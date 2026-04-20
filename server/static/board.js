/**
 * AWBW-style board renderer (aligned with AWBW Replay Player conventions).
 *
 * - Logical tile size 16×16 (see DrawableTile.BASE_SIZE in third_party viewer).
 * - Integer DISPLAY_SCALE upscaling + imageSmoothingEnabled=false = crisp pixels
 *   like NearestNeighbourTextureStore in the desktop viewer.
 * - Terrain + units: PNGs under /static/awbw_textures/ (see tools/sync_awbw_textures.py;
 *   textures synced via tools/sync_awbw_textures.py from upstream Replay Player paths). Until manifest/images load,
 *   falls back to palette fills + procedural unit glyphs.
 *
 * Call renderBoard(canvas, boardData, highlightPos)
 * highlightPos: legacy [r,c] | play object { selected, reachable, attack, repair, unload, factory_build }
 */

/** Native AWBW / replay viewer tile resolution */
const LOGICAL_TILE = 16;
/** Integer scale for browser canvas (3 → 48 CSS px per tile, crisp) */
const DISPLAY_SCALE = 3;

/** Bitmap tile size in CSS px (logical 16 × DISPLAY_SCALE). Play mode hit-testing uses canvas display size instead — see play.js canvasCell. */
window.AWBW_TILE_SCREEN_PX = LOGICAL_TILE * DISPLAY_SCALE;

/** Base URL for synced Replay-Player textures (manifest at manifest.json). */
const AWBW_TEX_BASE = '/static/awbw_textures/';

let __awbwManifest = null;
/** @type {string[][]|null} rel path per [engineTypeId][player 0|1] */
let __awbwUnitRel = null;
/** @type {Map<string, HTMLImageElement>} */
const __awbwImgCache = new Map();
let __awbwLoadPromise = null;
let __awbwTexturesReady = false;
/** @type {{ canvas: HTMLCanvasElement, boardData: object, highlightPos: * }|null} */
let __awbwPendingRedraw = null;

function __awbwAssetUrl(rel) {
  return AWBW_TEX_BASE + String(rel).replace(/\\/g, '/');
}

function __awbwBuildUnitRelMap(manifest) {
  const m = [];
  for (const row of manifest.units || []) {
    const id = row.engineTypeId;
    const pl = row.player;
    if (!m[id]) m[id] = [];
    m[id][pl] = row.rel;
  }
  return m;
}

function __awbwLoadImage(rel) {
  const cached = __awbwImgCache.get(rel);
  if (cached && cached.complete && cached.naturalWidth > 0) return Promise.resolve(cached);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      __awbwImgCache.set(rel, img);
      resolve(img);
    };
    img.onerror = () => {
      __awbwImgCache.set(rel, img);
      resolve(null);
    };
    img.src = __awbwAssetUrl(rel);
  });
}

function __awbwStartTextureLoad() {
  if (__awbwLoadPromise) return __awbwLoadPromise;
  __awbwLoadPromise = (async () => {
    try {
      const resp = await fetch(__awbwAssetUrl('manifest.json'));
      if (!resp.ok) throw new Error(`manifest ${resp.status}`);
      __awbwManifest = await resp.json();
      __awbwUnitRel = __awbwBuildUnitRelMap(__awbwManifest);
      const rels = new Set(Object.values(__awbwManifest.terrainByAwbwId || {}));
      for (const row of __awbwManifest.units || []) rels.add(row.rel);
      await Promise.all([...rels].map((rel) => __awbwLoadImage(rel)));
      __awbwTexturesReady = true;
    } catch (_e) {
      __awbwTexturesReady = false;
    }
    const pending = __awbwPendingRedraw;
    if (pending && __awbwTexturesReady) {
      renderBoard(pending.canvas, pending.boardData, pending.highlightPos);
    }
  })();
  return __awbwLoadPromise;
}

/**
 * Preload manifest + all terrain/unit PNGs (optional; renderBoard triggers this too).
 * @returns {Promise<void>}
 */
function preloadAWBWBoardTextures() {
  return __awbwStartTextureLoad().then(() => {});
}

window.preloadAWBWBoardTextures = preloadAWBWBoardTextures;

function __awbwDrawSpriteInTile(ctx, img, c, r, T) {
  if (!img || !img.complete || img.naturalWidth <= 0) return;
  const x0 = c * T;
  const y0 = r * T;
  const iw = img.naturalWidth;
  const ih = img.naturalHeight;
  const scale = Math.min(T / iw, T / ih);
  const w = Math.max(1, Math.round(iw * scale));
  const h = Math.max(1, Math.round(ih * scale));
  const dx = x0 + Math.floor((T - w) / 2);
  const dy = y0 + T - h;
  ctx.drawImage(img, 0, 0, iw, ih, dx, dy, w, h);
}

function drawUnitMovedOverlay(ctx, x0, y0, moved, pal, T) {
  if (moved) {
    ctx.fillStyle = pal.s;
    ctx.fillRect(x0, y0, T, T);
  }
}

function drawUnitHpBar(ctx, x0, y0, hpVal, T) {
  // HP bar matches AWBW: bucket granularity only (display_hp * 10 = 10..100).
  // Snap defensively even if an unbucketed value sneaks through the JSON
  // payload so the viewer can never out-leak what a human sees in-game.
  // See server/write_watch_state.py::units_list and docs/hp_belief.md.
  const raw = (hpVal != null ? hpVal : 100);
  const bucket = Math.max(0, Math.min(10, Math.ceil(raw / 10)));
  const hp = bucket / 10;
  const bw = T - 4;
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(x0 + 2, y0 + T - 4, bw, 3);
  ctx.fillStyle = hp > 0.5 ? '#3dd65e' : hp > 0.25 ? '#e8c030' : '#e04040';
  ctx.fillRect(x0 + 2, y0 + T - 4, Math.max(1, Math.round(bw * hp)), 3);
}

/** Moved dim + HP strip (sprite mode: both after bitmap). */
function drawUnitChrome(ctx, x0, y0, moved, hpVal, pal, T) {
  drawUnitMovedOverlay(ctx, x0, y0, moved, pal, T);
  drawUnitHpBar(ctx, x0, y0, hpVal, T);
}

// ── Terrain (tuned toward AWBW/GBA readability, not board.js 2024 flat fills) ──
const TERRAIN_COLORS = {
  1: '#5a8f3d',   // plain
  2: '#5c4d3d',   // mountain
  3: '#1e4a1a',   // wood
  14: '#7a6a52', 15: '#7a6a52', 16: '#7a6a52', 17: '#7a6a52', // roads
  28: '#1c4a7a',  // sea
  29: '#c4b060',  // shoal
  30: '#143d6b',  // reef
  32: '#2d5a28',  // pipe seam (rough)
  33: '#6a6048',  // pipe
  34: '#9a9a82', 35: '#9a9a82', 36: '#9a9a82', 37: '#9a9a82', // neutral prop bases
  133: '#a67c00', // lab
};

function terrainColor(tid) {
  if (TERRAIN_COLORS[tid] !== undefined) return TERRAIN_COLORS[tid];
  if (tid >= 38 && tid <= 42) return '#2a6aa8';   // OS
  if (tid >= 43 && tid <= 47) return '#6a9ad8';   // BM
  if (tid >= 48 && tid <= 52) return '#2d8a32';   // GE
  if (tid >= 53 && tid <= 57) return '#c9a020';   // YC
  if (tid >= 58 && tid <= 62) return '#c03030';   // BH
  if (tid >= 63 && tid <= 160) return '#7a5090';
  if (tid >= 14 && tid <= 17) return '#7a6a52';
  return '#5a8f3d';
}

/** Player tint for unit voxels: dark body, mid, highlight */
function playerPalette(p) {
  if (p === 0) {
    return { d: '#0f2f5c', m: '#2566c4', l: '#9fd4ff', o: '#06182c', s: '#00000055' };
  }
  return { d: '#5c1010', m: '#c42828', l: '#ffc8c8', o: '#2c0606', s: '#00000055' };
}

/**
 * Unit glyphs: boxes [x, y, w, h, layer] in 0..16 logical px.
 * layer: d=dark, m=mid, l=light (mapped through palette)
 */
const DEFAULT_GLYPH = [
  [4, 6, 8, 8, 'm'],
  [6, 4, 4, 4, 'l'],
  [3, 3, 10, 2, 'd'],
];

const UNIT_GLYPHS = [
  // 0 Infantry
  [
    [6, 2, 4, 3, 'l'], [5, 5, 6, 6, 'm'], [5, 11, 2, 3, 'd'], [9, 11, 2, 3, 'd'],
    [4, 4, 1, 1, 'd'], [11, 4, 1, 1, 'd'],
  ],
  // 1 Mech — bulkier
  [
    [5, 1, 6, 4, 'l'], [4, 5, 8, 7, 'm'], [3, 12, 3, 3, 'd'], [10, 12, 3, 3, 'd'],
    [6, 8, 4, 2, 'd'],
  ],
  // 2 Recon
  [[3, 6, 10, 6, 'm'], [5, 4, 6, 3, 'l'], [2, 9, 3, 3, 'd'], [11, 9, 3, 3, 'd'], [7, 3, 2, 2, 'd']],
  // 3 Tank
  [[2, 5, 12, 7, 'm'], [4, 3, 8, 3, 'l'], [1, 8, 3, 4, 'd'], [12, 8, 3, 4, 'd'], [6, 10, 4, 2, 'd']],
  // 4 Md Tank
  [[1, 4, 14, 8, 'm'], [3, 2, 10, 3, 'l'], [0, 7, 4, 5, 'd'], [12, 7, 4, 5, 'd']],
  // 5 Neo
  [[1, 3, 14, 9, 'm'], [4, 1, 8, 3, 'l'], [6, 12, 4, 3, 'd']],
  // 6 Mega
  [[0, 3, 16, 9, 'm'], [2, 1, 12, 3, 'l'], [3, 12, 10, 3, 'd']],
  // 7 APC
  [[3, 5, 10, 7, 'm'], [5, 3, 6, 3, 'l'], [2, 9, 2, 3, 'd'], [12, 9, 2, 3, 'd']],
  // 8 Artillery
  [[4, 7, 8, 6, 'm'], [6, 3, 4, 5, 'l'], [7, 1, 2, 3, 'd'], [3, 10, 2, 4, 'd'], [11, 10, 2, 4, 'd']],
  // 9 Rockets
  [[3, 4, 10, 9, 'm'], [6, 1, 4, 4, 'l'], [2, 10, 3, 4, 'd'], [11, 10, 3, 4, 'd']],
  // 10 AA
  [[4, 6, 8, 7, 'm'], [6, 3, 4, 4, 'l'], [7, 1, 2, 3, 'd'], [5, 11, 6, 2, 'd']],
  // 11 Missiles
  [[5, 2, 6, 11, 'm'], [4, 1, 8, 2, 'l'], [3, 12, 4, 3, 'd'], [9, 12, 4, 3, 'd']],
  // 12 Fighter
  [[6, 8, 4, 6, 'm'], [4, 6, 8, 3, 'l'], [7, 4, 2, 3, 'd'], [5, 3, 6, 2, 'd']],
  // 13 Bomber
  [[3, 7, 10, 6, 'm'], [2, 5, 12, 3, 'l'], [7, 3, 2, 3, 'd']],
  // 14 Stealth
  [[4, 6, 8, 7, 'm'], [3, 4, 10, 3, 'l'], [7, 2, 2, 3, 'd'], [5, 12, 6, 1, 'd']],
  // 15 B-Copter
  [[4, 6, 8, 5, 'm'], [2, 8, 12, 2, 'l'], [7, 4, 2, 3, 'd'], [6, 11, 4, 3, 'd']],
  // 16 T-Copter
  [[5, 5, 6, 6, 'm'], [3, 7, 10, 2, 'l'], [7, 3, 2, 3, 'd']],
  // 17 Battleship
  [[1, 6, 14, 6, 'm'], [0, 8, 16, 3, 'd'], [4, 4, 8, 3, 'l']],
  // 18 Carrier
  [[0, 7, 16, 5, 'm'], [2, 5, 12, 3, 'l'], [3, 12, 10, 2, 'd']],
  // 19 Sub
  [[3, 7, 10, 5, 'm'], [5, 5, 6, 3, 'l'], [1, 9, 2, 3, 'd'], [13, 9, 2, 3, 'd']],
  // 20 Cruiser
  [[1, 6, 14, 5, 'm'], [0, 9, 16, 3, 'd'], [4, 4, 8, 3, 'l']],
  // 21 Lander
  [[2, 6, 12, 6, 'm'], [1, 9, 14, 3, 'd'], [5, 4, 6, 3, 'l']],
  // 22 Gunboat
  [[4, 7, 8, 5, 'm'], [3, 9, 10, 2, 'd'], [6, 5, 4, 3, 'l']],
  // 23 Black Boat
  [[3, 6, 10, 6, 'm'], [2, 9, 12, 2, 'd'], [5, 4, 6, 3, 'l']],
  // 24 BBomb — diamond
  [[4, 4, 8, 8, 'm'], [6, 2, 4, 3, 'l'], [6, 11, 4, 3, 'd']],
  // 25 Piperunner
  [[2, 5, 12, 7, 'm'], [4, 3, 8, 3, 'l'], [1, 9, 3, 3, 'd'], [12, 9, 3, 3, 'd']],
  // 26 Oozium
  [[3, 3, 10, 10, 'm'], [5, 5, 6, 6, 'd'], [6, 6, 4, 4, 'l']],
];

const UNIT_NAMES = [
  'Inf', 'Mch', 'Rcn', 'Tnk', 'MDT', 'Neo', 'Meg', 'APC',
  'Art', 'Rkt', 'AAA', 'Mis', 'Ftr', 'Bmb', 'Sth', 'BCp',
  'TCp', 'BtS', 'Car', 'Sub', 'Cru', 'Lnd', 'Gbt', 'BBt',
  'BBm', 'Pip', 'Ozm',
];

function drawUnitGlyph(ctx, x0, y0, typeId, player, moved, hpVal) {
  const pal = playerPalette(player);
  const boxes = UNIT_GLYPHS[typeId] || DEFAULT_GLYPH;
  // subtle ground shadow
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  for (const b of boxes) {
    ctx.fillRect(x0 + b[0] + 1, y0 + b[1] + 1, b[2], b[3]);
  }
  for (const b of boxes) {
    const layer = b[4];
    ctx.fillStyle = pal[layer] || pal.m;
    ctx.fillRect(x0 + b[0], y0 + b[1], b[2], b[3]);
  }
  // outline crispness
  ctx.strokeStyle = pal.o;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, LOGICAL_TILE - 1, LOGICAL_TILE - 1);

  drawUnitMovedOverlay(ctx, x0, y0, moved, pal, LOGICAL_TILE);

  // Tiny type label (1px font scale — bitmap style)
  const tag = UNIT_NAMES[typeId] ?? '?';
  ctx.font = 'bold 5px monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 2;
  const tx = x0 + LOGICAL_TILE / 2;
  const ty = y0 + LOGICAL_TILE - 1;
  ctx.strokeText(tag, tx, ty);
  ctx.fillText(tag, tx, ty);

  drawUnitHpBar(ctx, x0, y0, hpVal, LOGICAL_TILE);
}

function drawUnitLayer(ctx, x0, y0, typeId, player, moved, hpVal, useTextures, T) {
  const pal = playerPalette(player);
  const pIdx = Math.min(1, Math.max(0, player | 0));
  if (useTextures && __awbwUnitRel) {
    const row = __awbwUnitRel[typeId];
    const rel = row ? row[pIdx] : null;
    const img = rel ? __awbwImgCache.get(rel) : null;
    if (img && img.complete && img.naturalWidth > 0) {
      const c = Math.round(x0 / T);
      const r = Math.round(y0 / T);
      __awbwDrawSpriteInTile(ctx, img, c, r, T);
      drawUnitChrome(ctx, x0, y0, moved, hpVal, pal, T);
      return;
    }
  }
  drawUnitGlyph(ctx, x0, y0, typeId, player, moved, hpVal);
}

function strokeCell(ctx, r, c, color, lineW, T) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineW;
  ctx.strokeRect(c * T + 0.5, r * T + 0.5, T - 1, T - 1);
}

function fillCell(ctx, r, c, rgba, T) {
  ctx.fillStyle = rgba;
  ctx.fillRect(c * T, r * T, T, T);
}

/**
 * Draw the full board onto canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {Object} boardData
 * @param {Array|null|Object} highlightPos
 */
function renderBoard(canvas, boardData, highlightPos = null) {
  if (!boardData) return;

  __awbwPendingRedraw = { canvas, boardData, highlightPos };
  const useTextures = Boolean(__awbwTexturesReady && __awbwManifest);
  if (!useTextures) __awbwStartTextureLoad();

  const { height, width, terrain, units, properties } = boardData;
  const T = LOGICAL_TILE;
  const S = DISPLAY_SCALE;

  canvas.width = width * T * S;
  canvas.height = height * T * S;

  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(S, S);

  // ── Terrain ─────────────────────────────────────────────────────────────
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const tid = terrain[r][c];
      let drewTex = false;
      if (useTextures && __awbwManifest && __awbwManifest.terrainByAwbwId) {
        const rel = __awbwManifest.terrainByAwbwId[String(tid)];
        if (rel) {
          const img = __awbwImgCache.get(rel);
          if (img && img.complete && img.naturalWidth > 0) {
            __awbwDrawSpriteInTile(ctx, img, c, r, T);
            drewTex = true;
          }
        }
      }
      if (!drewTex) {
        ctx.fillStyle = terrainColor(tid);
        ctx.fillRect(c * T, r * T, T, T);
      }
      if (!useTextures) {
        ctx.strokeStyle = 'rgba(0,0,0,0.18)';
        ctx.lineWidth = 1;
        ctx.strokeRect(c * T + 0.5, r * T + 0.5, T - 1, T - 1);
      }
    }
  }

  // ── Properties ───────────────────────────────────────────────────────────
  if (properties) {
    for (const prop of properties) {
      const x = prop.col * T;
      const y = prop.row * T;
      if (prop.owner !== null && prop.owner !== undefined && prop.owner >= 0) {
        const pal = playerPalette(prop.owner);
        if (useTextures) {
          ctx.strokeStyle = pal.m;
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          ctx.moveTo(x + 0.5, y + 0.5);
          ctx.lineTo(x + 5.5, y + 0.5);
          ctx.lineTo(x + 0.5, y + 5.5);
          ctx.closePath();
          ctx.stroke();
          ctx.globalAlpha = 1;
        } else {
          ctx.fillStyle = pal.m;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + 7, y);
          ctx.lineTo(x, y + 7);
          ctx.closePath();
          ctx.fill();
        }
      }
      if (prop.is_hq) {
        if (useTextures) {
          ctx.strokeStyle = 'rgba(255, 215, 0, 0.9)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 1.5, y + 1.5, T - 3, T - 3);
          ctx.fillStyle = '#ffd700';
          ctx.font = `bold ${Math.round(T * 0.38)}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText('HQ', x + T / 2, y + T - 1);
        } else {
          ctx.fillStyle = 'rgba(255,215,0,0.45)';
          ctx.fillRect(x + 2, y + 2, T - 4, T - 4);
          ctx.fillStyle = '#ffd700';
          ctx.font = `bold ${Math.round(T * 0.45)}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('HQ', x + T / 2, y + T / 2);
        }
      }
      if (prop.is_lab) {
        ctx.strokeStyle = useTextures ? 'rgba(232, 192, 64, 0.75)' : '#e8c040';
        ctx.lineWidth = useTextures ? 1 : 1.5;
        ctx.strokeRect(x + 1, y + 1, T - 2, T - 2);
      }
      const capPts =
        prop.capture_points !== undefined && prop.capture_points !== null
          ? Number(prop.capture_points)
          : 20;
      if (capPts < 20) {
        ctx.save();
        ctx.fillStyle = useTextures ? 'rgba(255,255,255,0.95)' : 'rgba(20,24,40,0.92)';
        ctx.strokeStyle = 'rgba(0,0,0,0.45)';
        ctx.lineWidth = 2;
        ctx.font = `bold ${Math.max(6, Math.round(T * 0.34))}px monospace`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        const label = String(capPts);
        const tx = x + T - 2;
        const ty = y + 1;
        ctx.strokeText(label, tx, ty);
        ctx.fillText(label, tx, ty);
        ctx.restore();
      }
    }
  }

  // ── Highlights ───────────────────────────────────────────────────────────
  if (highlightPos) {
    const isLegacy =
      Array.isArray(highlightPos) &&
      highlightPos.length === 2 &&
      typeof highlightPos[0] === 'number';
    if (isLegacy) {
      strokeCell(ctx, highlightPos[0], highlightPos[1], '#ffee58', 2, T);
    } else if (typeof highlightPos === 'object') {
      const reach = highlightPos.reachable || [];
      const atk = highlightPos.attack || [];
      const rep = highlightPos.repair || [];
      const unload = highlightPos.unload || [];
      const factoryBuild = highlightPos.factory_build || [];
      for (const [r, c] of reach) fillCell(ctx, r, c, 'rgba(66,165,245,0.45)', T);
      for (const [r, c] of atk) fillCell(ctx, r, c, 'rgba(239,83,80,0.5)', T);
      for (const [r, c] of rep) fillCell(ctx, r, c, 'rgba(102,187,106,0.55)', T);
      for (const [r, c] of unload) fillCell(ctx, r, c, 'rgba(186,104,200,0.5)', T);
      for (const [r, c] of factoryBuild) fillCell(ctx, r, c, 'rgba(220, 200, 80, 0.4)', T);
    }
  }

  // ── Units (sprites when loaded, else glyphs) ─────────────────────────────
  if (units) {
    for (const unit of units) {
      const x0 = unit.col * T;
      const y0 = unit.row * T;
      drawUnitLayer(
        ctx,
        x0,
        y0,
        unit.type_id,
        unit.player,
        unit.moved,
        unit.hp ?? 100,
        useTextures,
        T,
      );
    }
  }

  // Play: selection ring on top
  if (
    highlightPos &&
    typeof highlightPos === 'object' &&
    highlightPos.selected &&
    highlightPos.selected.length === 2
  ) {
    const [sr, sc] = highlightPos.selected;
    strokeCell(ctx, sr, sc, '#ffeb3b', 2, T);
  }

  ctx.restore();
}
