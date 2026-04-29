"""
Checkpoint compatibility for MaskablePPO zips saved before encoder layout bumps.

Supported best-effort migrations:
* 62 spatial channels -> current layout by duplicating the legacy HP channel.
* 63/70 spatial channels -> current layout by copying matching leading planes
  and zero-initializing new planes.
* 17 scalars -> 16 scalars by deleting the former tier column from
  ``scalar_to_plane.weight``.

Optimizer moments are stripped whenever a migration is materialized. Adam moments
are unsafe across both channel expansion and scalar column deletion; restarting
optimizer state is clearer than silently misaligning slots. The optimizer
``param_groups`` are preserved so SB3's loader still sees the expected
``policy.optimizer`` parameter block.
"""
from __future__ import annotations

import gc
import io
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS

# Legacy layout: 28 unit + 1 HP + 15 terrain + 15 property + 3 capture = 62.
# Intermediate layouts:
#   63 = dual HP
#   70 = dual HP + influence + defense stars
# Current layout is imported from ``rl.encoder``.
_LEGACY_TIER_SCALAR_INDEX = 12


def expand_spatial_stem_in_channels_62_to_63(t: torch.Tensor) -> torch.Tensor:
    """Expand dim 1 from 62 → 63 (HP split); works for weights or Adam moments."""
    if t.shape[1] != 62:
        return t
    out = t.new_zeros(t.shape[0], 63, t.shape[2], t.shape[3])
    out[:, 0:28] = t[:, 0:28]
    out[:, 28] = t[:, 28]
    out[:, 29] = t[:, 28]
    out[:, 30:45] = t[:, 29:44]
    out[:, 45:60] = t[:, 44:59]
    out[:, 60:63] = t[:, 59:62]
    return out


def transplant_spatial_stem_to_current(t: torch.Tensor) -> torch.Tensor:
    """Expand a stem-like tensor's input channels to ``N_SPATIAL_CHANNELS``."""
    if t.shape[1] == N_SPATIAL_CHANNELS:
        return t
    if t.shape[1] == 62:
        t = expand_spatial_stem_in_channels_62_to_63(t)
    if t.shape[1] not in (63, 70):
        return t
    out = t.new_zeros(t.shape[0], N_SPATIAL_CHANNELS, t.shape[2], t.shape[3])
    n = min(t.shape[1], N_SPATIAL_CHANNELS)
    out[:, :n] = t[:, :n]
    return out


def transplant_scalar_to_plane_weight_to_current(t: torch.Tensor) -> torch.Tensor:
    """Delete the former tier scalar column from a 17-input scalar projection."""
    if t.dim() != 2 or t.shape[1] != 17 or N_SCALARS != 16:
        return t
    return torch.cat(
        [t[:, :_LEGACY_TIER_SCALAR_INDEX], t[:, _LEGACY_TIER_SCALAR_INDEX + 1 :]],
        dim=1,
    ).contiguous()


def _patch_policy_state_dict(sd: dict) -> None:
    for k, v in list(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if v.dim() == 4 and "stem.0.weight" in k:
            sd[k] = transplant_spatial_stem_to_current(v)
            continue
        if v.dim() == 2 and "scalar_to_plane.weight" in k:
            sd[k] = transplant_scalar_to_plane_weight_to_current(v)


def checkpoint_needs_spatial_stem_patch(ckpt_path: Path) -> bool:
    """True if ``policy.pth`` does not match current encoder parameter shapes."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        return False
    try:
        with zipfile.ZipFile(ckpt_path, "r") as zf:
            if "policy.pth" not in zf.namelist():
                return False
            sd = torch.load(
                io.BytesIO(zf.read("policy.pth")),
                map_location="cpu",
                weights_only=False,
            )
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False
    w = sd.get("features_extractor.stem.0.weight")
    if isinstance(w, torch.Tensor) and w.shape[1] != N_SPATIAL_CHANNELS:
        return True
    sw = sd.get("features_extractor.scalar_to_plane.weight")
    if isinstance(sw, torch.Tensor) and sw.shape[1] != N_SCALARS:
        return True
    return False


def materialize_sb3_zip_with_spatial_compat(ckpt_path: Path) -> tuple[Path, bool]:
    """
    If the zip uses legacy encoder shapes, copy to a temp zip with patched
    ``policy.pth`` and optimizer moments cleared. Otherwise return
    ``(ckpt_path, False)``.
    """
    ckpt_path = Path(ckpt_path).resolve()
    if not checkpoint_needs_spatial_stem_patch(ckpt_path):
        return ckpt_path, False

    repo_root = Path(__file__).resolve().parent.parent
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix="ckpt_encoder_", suffix=".zip", dir=str(tmp_dir))
    os.close(fd)
    out_path = Path(tmp_name)

    buf = io.BytesIO()
    with zipfile.ZipFile(ckpt_path, "r") as zin:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in zin.namelist():
                raw = zin.read(name)
                if name == "policy.pth":
                    sd = torch.load(
                        io.BytesIO(raw), map_location="cpu", weights_only=False
                    )
                    _patch_policy_state_dict(sd)
                    bio = io.BytesIO()
                    torch.save(sd, bio)
                    raw = bio.getvalue()
                elif name == "policy.optimizer.pth":
                    od = torch.load(
                        io.BytesIO(raw), map_location="cpu", weights_only=False
                    )
                    if isinstance(od, dict):
                        od = {**od, "state": {}}
                    bio = io.BytesIO()
                    torch.save(od, bio)
                    raw = bio.getvalue()
                zout.writestr(name, raw)

    out_path.write_bytes(buf.getvalue())
    print(
        f"[ckpt_compat] Patched encoder shapes for {ckpt_path.name} "
        f"(spatial->{N_SPATIAL_CHANNELS}, scalars->{N_SCALARS}); cleared optimizer moments"
    )
    return out_path, True


def _current_awbw_observation_space() -> gym.spaces.Dict:
    """Dict space matching ``AWBWEnv`` / ``encode_state`` (``N_SPATIAL_CHANNELS``)."""
    return gym.spaces.Dict(
        {
            "spatial": gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS),
                dtype=np.float32,
            ),
            "scalars": gym.spaces.Box(
                low=-1.0,
                high=10.0,
                shape=(N_SCALARS,),
                dtype=np.float32,
            ),
        }
    )


def _sync_loaded_model_observation_space(model) -> None:
    """
    Legacy zips deserialize stale ``observation_space`` shapes. After weights are
    transplanted, align SB3's space so ``predict`` accepts env observations.
    """
    obs_sp = getattr(model, "observation_space", None)
    if not isinstance(obs_sp, gym.spaces.Dict):
        return
    sp = obs_sp.spaces.get("spatial")
    if not isinstance(sp, gym.spaces.Box):
        return
    sc = obs_sp.spaces.get("scalars")
    if not isinstance(sc, gym.spaces.Box):
        return
    if sp.shape == (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS) and sc.shape == (N_SCALARS,):
        return
    fixed = _current_awbw_observation_space()
    model.observation_space = fixed
    pol = getattr(model, "policy", None)
    if pol is not None and hasattr(pol, "observation_space"):
        pol.observation_space = fixed


def _register_numpy2_unpickle_shim() -> None:
    """
    SB3 zips pickled on NumPy 2 reference ``numpy._core.numeric`` (etc.).
    NumPy 1.26 ships a stub ``numpy/_core`` package **without** those modules.
    Pre-seed ``sys.modules`` so ``cloudpickle`` and plain imports resolve them
    to the real ``numpy.core.*`` implementations.
    """
    import importlib
    import sys

    if getattr(_register_numpy2_unpickle_shim, "_done", False):
        return

    # NumPy 1.26 exposes ``np._core`` as a lazy alias even though
    # ``numpy._core.numeric`` is not importable — do not use ``hasattr(np, "_core")``.
    try:
        importlib.import_module("numpy._core.numeric")
        setattr(_register_numpy2_unpickle_shim, "_done", True)
        return
    except ModuleNotFoundError:
        pass

    setattr(_register_numpy2_unpickle_shim, "_done", True)
    importlib.import_module("numpy._core")
    import numpy.core.multiarray as _multiarray
    import numpy.core.numeric as _numeric
    import numpy.core.umath as _umath

    sys.modules["numpy._core.multiarray"] = _multiarray
    sys.modules["numpy._core.numeric"] = _numeric
    sys.modules["numpy._core.umath"] = _umath


# In-process reuse for eval scripts (symmetric / bo3) so each game does not
# re-open the same zip; key includes device.
_MODEL_LOAD_CACHE: dict[str, object] = {}


def clear_maskable_ppo_load_cache() -> None:
    """Drop cached models (e.g. between test cases)."""
    _MODEL_LOAD_CACHE.clear()


def _ppo_load_cache_key(path: Path, kwargs: dict) -> str:
    dev = kwargs.get("device", "cpu")
    return f"{path.resolve()}|{dev}"


def _is_eval_frozen_snapshot(path: Path) -> bool:
    """True if ``path`` was created by :func:`snapshot_eval_checkpoints` (no extra copy in load)."""
    try:
        p = path.resolve()
        repo_root = Path(__file__).resolve().parent.parent
        return (
            p.parent == repo_root / ".tmp"
            and p.name.startswith("eval_snap_")
            and p.suffix.lower() == ".zip"
        )
    except OSError:
        return False


def snapshot_eval_checkpoints(pairs: list[tuple[str, Path]]) -> tuple[str, tuple[Path, ...]]:
    """
    Copy each ``(label, src)`` to ``<repo>/.tmp/eval_snap_<run_id>_<label>.zip``.

    Call once at the start of symmetric / bo3 eval so **all** games read the same
    bytes even if the original paths (e.g. ``Z:\\checkpoints\\latest.zip``) are
    overwritten mid-run.
    """
    repo_root = Path(__file__).resolve().parent.parent
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"

    out: list[Path] = []
    for label, src in pairs:
        src = Path(src).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"checkpoint missing for eval snapshot: {src}")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:64]
        dst = tmp_dir / f"eval_snap_{run_id}_{safe_label}.zip"
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                shutil.copy2(src, dst)
                with zipfile.ZipFile(dst, "r") as zf:
                    zf.namelist()
                out.append(dst.resolve())
                break
            except (OSError, zipfile.BadZipFile, EOFError) as e:
                last_err = e
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
        else:
            raise RuntimeError(f"eval snapshot failed for {label!r}: {src}") from last_err

    return run_id, tuple(out)


def delete_eval_snapshots(paths: tuple[Path, ...]) -> None:
    """Remove snapshot zips created by :func:`snapshot_eval_checkpoints` (best effort)."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _stable_zip_copy(src: Path) -> tuple[Path, bool]:
    """
    Copy checkpoint to ``<repo>/.tmp/`` so SB3/torch read a stable local file.

    Avoids ``EOFError`` / truncated reads when the source is on a network share
    or concurrently overwritten. Set ``AWBW_CKPT_LOCAL_COPY=0`` to read ``src``
    directly (faster on local SSD; less safe on SMB).

    Skips copying when ``src`` is already an eval snapshot from
    :func:`snapshot_eval_checkpoints`.
    """
    src = src.resolve()
    if _is_eval_frozen_snapshot(src):
        return src, False

    raw = os.environ.get("AWBW_CKPT_LOCAL_COPY", "1").strip().lower()
    if raw in ("0", "false", "no"):
        return src, False

    repo_root = Path(__file__).resolve().parent.parent
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(3):
        fd, tmp_name = tempfile.mkstemp(prefix="ckpt_load_", suffix=".zip", dir=str(tmp_dir))
        os.close(fd)
        dst = Path(tmp_name)
        try:
            shutil.copy2(src, dst)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.namelist()
            return dst, True
        except (OSError, zipfile.BadZipFile, EOFError) as e:
            last_err = e
            try:
                dst.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"checkpoint copy failed after 3 attempts: {src}") from last_err


def load_maskable_ppo_compat(path: str | Path, *, cache: bool = False, **kwargs):
    """
    ``MaskablePPO.load`` with automatic encoder-shape migration when needed.

    Writes a temp zip under ``<repo>/.tmp/`` when patching; removes it after
    load (best effort on Windows).

    Copies the source zip to ``.tmp`` before load (unless ``AWBW_CKPT_LOCAL_COPY=0``)
    so SMB / concurrent writers do not cause ``EOFError`` during torch unpickle.

    Set ``cache=True`` to reuse the same in-memory model for repeated loads of
    the same path (eval scripts only; training should leave ``cache=False``).
    """
    _register_numpy2_unpickle_shim()
    from sb3_contrib import MaskablePPO  # type: ignore[import]

    p = Path(path).resolve()
    ckey = _ppo_load_cache_key(p, kwargs)
    if cache and ckey in _MODEL_LOAD_CACHE:
        return _MODEL_LOAD_CACHE[ckey]

    local_src, del_local = _stable_zip_copy(p)
    try:
        use_path, is_temp = materialize_sb3_zip_with_spatial_compat(local_src)
        try:
            model = MaskablePPO.load(str(use_path), **kwargs)
            _sync_loaded_model_observation_space(model)
            if cache:
                _MODEL_LOAD_CACHE[ckey] = model
            return model
        finally:
            if is_temp:
                gc.collect()
                try:
                    use_path.unlink(missing_ok=True)
                except OSError:
                    pass
    finally:
        if del_local:
            try:
                local_src.unlink(missing_ok=True)
            except OSError:
                pass
