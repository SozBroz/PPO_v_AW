"""
Checkpoint compatibility for MaskablePPO zips saved before encoder layout bumps.

Supported best-effort migrations:
* 62 spatial channels -> current layout by duplicating the legacy HP channel.
* 63/70 spatial channels -> current layout by copying matching leading planes
  and zero-initializing new planes.
* 17 scalars -> 16 scalars by deleting the former tier column from
  ``scalar_to_plane.weight``.

* Policy trunk → :class:`rl.network.AWBWNet` for candidate-MLP / factored-head training
  (:func:`scalpel_checkpoint_zip_to_awbw_net_state`, :func:`scalpel_policy_state_dict_to_awbw_net`).

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
from typing import Any

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


def scalpel_policy_state_dict_to_awbw_net(
    policy_sd: dict,
    *,
    hidden_size: int = 256,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Build an :class:`rl.network.AWBWNet` state dict from a MaskablePPO ``policy.pth`` map
    (``MultiInputPolicy`` + :class:`~rl.network.AWBWFeaturesExtractor` or
    :class:`~rl.network.AWBWCandidateFeaturesExtractor`).

    **Copied (when shapes match):**
    ``features_extractor.{stem,trunk_blocks,scalar_to_plane}`` → same paths on ``AWBWNet``;
    value input linear → ``value_head.0``: for the legacy extractor (144→``features_dim``),
    full ``fc.0`` when ``features_dim`` matches ``hidden_size``; for the candidate extractor
    (208→``features_dim``), rows ``[:hidden_size]`` and columns ``[:FUSED_CHANNELS]`` (board
    pooled features only; candidate slice omitted).

    **Left at fresh init:** all spatial policy conv heads (select/move/attack/…),
    ``linear_scalar_policy``, ``candidate_mlp``, and ``value_head.2`` (no compatible
    tensor in the SB3 critic head without retraining).

    ``policy_sd`` should already be encoder-patched (e.g. scalars 17→16, spatial width
    current) — use :func:`materialize_sb3_zip_with_spatial_compat` before loading bytes
    from disk, or :func:`_patch_policy_state_dict` on raw dicts.
    """
    from rl.network import AWBWNet, FUSED_CHANNELS

    net = AWBWNet(hidden_size=hidden_size)
    out = {k: v.clone() for k, v in net.state_dict().items()}
    fe = "features_extractor."
    copied: list[str] = []

    def _take(src_key: str, dst_key: str) -> None:
        t = policy_sd.get(src_key)
        if not isinstance(t, torch.Tensor):
            return
        if dst_key not in out:
            return
        if out[dst_key].shape != t.shape:
            return
        out[dst_key] = t.clone()
        copied.append(f"{src_key} -> {dst_key}")

    for ok in list(policy_sd.keys()):
        if not isinstance(policy_sd.get(ok), torch.Tensor):
            continue
        if ok.startswith(fe + "stem."):
            _take(ok, ok[len(fe) :])
        elif ok.startswith(fe + "trunk_blocks."):
            _take(ok, ok[len(fe) :])
        elif ok.startswith(fe + "scalar_to_plane."):
            _take(ok, ok[len(fe) :])

    fc0_w = policy_sd.get("features_extractor.fc.0.weight")
    fc0_b = policy_sd.get("features_extractor.fc.0.bias")
    if isinstance(fc0_w, torch.Tensor) and fc0_w.dim() == 2:
        in_f = int(fc0_w.shape[1])
        out_f = int(fc0_w.shape[0])
        v_w = out.get("value_head.0.weight")
        v_b = out.get("value_head.0.bias")
        if isinstance(v_w, torch.Tensor) and in_f == FUSED_CHANNELS and out_f == v_w.shape[0]:
            _take("features_extractor.fc.0.weight", "value_head.0.weight")
            _take("features_extractor.fc.0.bias", "value_head.0.bias")
        elif (
            isinstance(v_w, torch.Tensor)
            and isinstance(v_b, torch.Tensor)
            and in_f == FUSED_CHANNELS + 64
            and out_f >= int(v_w.shape[0])
        ):
            # AWBWCandidateFeaturesExtractor: fc sees [board_g | cand_g]; value head uses board_g only.
            rows = int(v_w.shape[0])
            cols = int(FUSED_CHANNELS)
            out["value_head.0.weight"] = fc0_w[:rows, :cols].clone()
            if isinstance(fc0_b, torch.Tensor) and int(fc0_b.shape[0]) >= rows:
                out["value_head.0.bias"] = fc0_b[:rows].clone()
            copied.append(
                f"features_extractor.fc.0.weight[:{rows},:{cols}] -> value_head.0 (candidate extractor)"
            )

    return out, copied


def scalpel_policy_state_dict_to_candidate_maskable_policy(
    legacy_policy_sd: dict,
    template_policy_sd: dict[str, torch.Tensor],
    *,
    copied_log: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Map a flat-action (35k) or legacy-feature MaskablePPO ``policy.pth`` dict onto the
    parameter tensors of a **template** :class:`~sb3_contrib.ppo_mask.ppo_mask.MaskablePPO`
    policy built with :class:`~rl.network.AWBWCandidateFeaturesExtractor` and
    ``gym.spaces.Discrete(MAX_CANDIDATES)``.

    Copies: CNN trunk + scalar plane where shapes match; first MLP block ``fc.0`` from
    legacy 144-wide input into the candidate 208-wide board/candidate split (board columns
    only, candidate columns zeroed); overlays ``fc.2`` and ``value_net`` where a
    leading sub-block matches. **action_net** (35000→4096) stays at template init.

    ``legacy_policy_sd`` must already be encoder-patched (see :func:`_patch_policy_state_dict`).
    """
    from rl.network import FUSED_CHANNELS

    copied: list[str] = copied_log if copied_log is not None else []
    out: dict[str, torch.Tensor] = {k: v.clone() for k, v in template_policy_sd.items()}

    fe = "features_extractor."
    for k, v in legacy_policy_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in out and out[k].shape == v.shape:
            out[k] = v.clone()
            copied.append(f"{k} (exact)")

    fc0_w = legacy_policy_sd.get(f"{fe}fc.0.weight")
    dst0 = out.get(f"{fe}fc.0.weight")
    if isinstance(fc0_w, torch.Tensor) and isinstance(dst0, torch.Tensor) and fc0_w.dim() == 2:
        in_f = int(fc0_w.shape[1])
        if in_f == int(FUSED_CHANNELS) and int(dst0.shape[1]) == int(FUSED_CHANNELS) + 64:
            rows = min(int(fc0_w.shape[0]), int(dst0.shape[0]))
            cols = int(FUSED_CHANNELS)
            out[f"{fe}fc.0.weight"][:rows, :cols] = fc0_w[:rows, :cols].clone()
            out[f"{fe}fc.0.weight"][:rows, cols:] = 0.0
            fb = legacy_policy_sd.get(f"{fe}fc.0.bias")
            if isinstance(fb, torch.Tensor) and fb.dim() == 1:
                out[f"{fe}fc.0.bias"][:rows] = fb[:rows].clone()
            copied.append(f"{fe}fc.0 legacy 144→208 board slice (rows={rows})")
        elif (
            in_f == int(FUSED_CHANNELS) + 64
            and int(dst0.shape[1]) == int(FUSED_CHANNELS) + 64
        ):
            rows = min(int(fc0_w.shape[0]), int(dst0.shape[0]))
            cols = min(in_f, int(dst0.shape[1]))
            out[f"{fe}fc.0.weight"][:rows, :cols] = fc0_w[:rows, :cols].clone()
            fb = legacy_policy_sd.get(f"{fe}fc.0.bias")
            if isinstance(fb, torch.Tensor) and fb.dim() == 1:
                out[f"{fe}fc.0.bias"][:rows] = fb[:rows].clone()
            copied.append(f"{fe}fc.0 candidate-shaped overlay (rows={rows}, cols={cols})")

    fc2_w = legacy_policy_sd.get(f"{fe}fc.2.weight")
    dst2 = out.get(f"{fe}fc.2.weight")
    if isinstance(fc2_w, torch.Tensor) and isinstance(dst2, torch.Tensor) and fc2_w.dim() == 2:
        r = min(int(fc2_w.shape[0]), int(dst2.shape[0]))
        c = min(int(fc2_w.shape[1]), int(dst2.shape[1]))
        out[f"{fe}fc.2.weight"][:r, :c] = fc2_w[:r, :c].clone()
        fb2 = legacy_policy_sd.get(f"{fe}fc.2.bias")
        db2 = out.get(f"{fe}fc.2.bias")
        if isinstance(fb2, torch.Tensor) and isinstance(db2, torch.Tensor):
            m = min(int(fb2.shape[0]), int(db2.shape[0]))
            out[f"{fe}fc.2.bias"][:m] = fb2[:m].clone()
        copied.append(f"{fe}fc.2 overlaid top-left {r}×{c}")

    s_vw = legacy_policy_sd.get("value_net.weight")
    d_vw = out.get("value_net.weight")
    if (
        isinstance(s_vw, torch.Tensor)
        and isinstance(d_vw, torch.Tensor)
        and s_vw.dim() == 2
        and d_vw.dim() == 2
    ):
        r = min(int(s_vw.shape[0]), int(d_vw.shape[0]))
        c = min(int(s_vw.shape[1]), int(d_vw.shape[1]))
        out["value_net.weight"][:r, :c] = s_vw[:r, :c].clone()
        s_vb = legacy_policy_sd.get("value_net.bias")
        d_vb = out.get("value_net.bias")
        if isinstance(s_vb, torch.Tensor) and isinstance(d_vb, torch.Tensor):
            m = min(int(s_vb.shape[0]), int(d_vb.shape[0]))
            out["value_net.bias"][:m] = s_vb[:m].clone()
        copied.append(f"value_net overlaid leading {r}×{c}")

    return out


def scalpel_checkpoint_zip_to_candidate_maskable_ppo_zip(
    ckpt_path: str | Path,
    dst_zip: str | Path,
    *,
    features_dim: int = 512,
    device: str | None = None,
) -> list[str]:
    """
    Read a MaskablePPO ``.zip``, transplant CNN + compatible MLP/value weights into a new
    model with candidate actions ``Discrete(MAX_CANDIDATES)``, and write ``dst_zip``.

    The output is suitable as ``latest.zip`` for async/sync training with the current env.
    """
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO  # type: ignore[import]
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]

    from rl.candidate_actions import CANDIDATE_FEATURE_DIM, MAX_CANDIDATES
    from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS
    from rl.network import AWBWCandidateFeaturesExtractor

    ckpt_path = Path(ckpt_path).resolve()
    dst_zip = Path(dst_zip).resolve()

    use_path, is_temp = materialize_sb3_zip_with_spatial_compat(ckpt_path)
    try:
        src_zip = use_path if is_temp else ckpt_path
        with zipfile.ZipFile(src_zip, "r") as zf:
            src_sd = torch.load(
                io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False
            )
        if not is_temp:
            _patch_policy_state_dict(src_sd)
    finally:
        if is_temp:
            try:
                use_path.unlink(missing_ok=True)
            except OSError:
                pass

    class _E(gym.Env):  # noqa: D401
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {
                    "spatial": spaces.Box(
                        -10.0,
                        10.0,
                        (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS),
                        np.float32,
                    ),
                    "scalars": spaces.Box(-1.0, 10.0, (N_SCALARS,), np.float32),
                    "candidate_features": spaces.Box(
                        -10.0,
                        10.0,
                        (MAX_CANDIDATES, CANDIDATE_FEATURE_DIM),
                        np.float32,
                    ),
                    "candidate_mask": spaces.Box(0, 1, (MAX_CANDIDATES,), np.int8),
                }
            )
            self.action_space = spaces.Discrete(MAX_CANDIDATES)

        def action_masks(self) -> np.ndarray:
            return np.ones(MAX_CANDIDATES, dtype=bool)

        def reset(self, seed=None, options=None):
            return {
                "spatial": np.zeros(
                    (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32
                ),
                "scalars": np.zeros((N_SCALARS,), dtype=np.float32),
                "candidate_features": np.zeros(
                    (MAX_CANDIDATES, CANDIDATE_FEATURE_DIM), dtype=np.float32
                ),
                "candidate_mask": np.zeros((MAX_CANDIDATES,), dtype=np.int8),
            }, {}

        def step(self, action):
            o, _ = self.reset()
            return o, 0.0, True, False, {}

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = ActionMasker(_E(), lambda e: e.action_masks())
    policy_kwargs = dict(
        features_extractor_class=AWBWCandidateFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=int(features_dim)),
        net_arch=[],
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=dev,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=1,
        gamma=0.99925,
        ent_coef=0.05,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=False,
    )
    env.close()

    copied: list[str] = []
    merged = scalpel_policy_state_dict_to_candidate_maskable_policy(
        src_sd, model.policy.state_dict(), copied_log=copied
    )
    model.policy.load_state_dict(merged, strict=True)
    align_maskable_ppo_observation_space_to_awbw_env(model)
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dst_zip.with_suffix("")))
    return copied


def load_policy_state_dict_from_zip_path(path: Path) -> dict:
    """Load ``policy.pth`` from an SB3 zip and apply encoder-compat patches in memory."""
    with zipfile.ZipFile(path, "r") as zf:
        if "policy.pth" not in zf.namelist():
            raise KeyError(f"no policy.pth in {path}")
        sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False)
    _patch_policy_state_dict(sd)
    return sd


def scalpel_checkpoint_zip_to_awbw_net_state(
    ckpt_path: str | Path,
    *,
    hidden_size: int = 256,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Read ``latest.zip`` (or any MaskablePPO zip), apply encoder migration when needed,
    return ``AWBWNet`` weights + list of copied tensor paths (for logging).
    """
    ckpt_path = Path(ckpt_path).resolve()
    use_path, is_temp = materialize_sb3_zip_with_spatial_compat(ckpt_path)
    try:
        src_zip = use_path if is_temp else ckpt_path
        with zipfile.ZipFile(src_zip, "r") as zf:
            sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False)
        if not is_temp:
            _patch_policy_state_dict(sd)
        out, copied = scalpel_policy_state_dict_to_awbw_net(sd, hidden_size=hidden_size)
        return out, copied
    finally:
        if is_temp:
            try:
                use_path.unlink(missing_ok=True)
            except OSError:
                pass

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


def _current_awbw_observation_space(*, with_candidates: bool = False) -> gym.spaces.Dict:
    """Dict space matching ``AWBWEnv`` / ``encode_state`` (``N_SPATIAL_CHANNELS``)."""
    spaces_map: dict[str, gym.Space] = {
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
    if with_candidates:
        from rl.candidate_actions import (  # noqa: PLC0415
            CANDIDATE_FEATURE_DIM,
            MAX_CANDIDATES,
        )

        spaces_map["candidate_features"] = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(MAX_CANDIDATES, CANDIDATE_FEATURE_DIM),
            dtype=np.float32,
        )
        spaces_map["candidate_mask"] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(MAX_CANDIDATES,),
            dtype=np.int8,
        )
    return gym.spaces.Dict(spaces_map)


def align_maskable_ppo_observation_space_to_awbw_env(model: Any) -> None:
    """Force Dict observation_space to match :class:`~rl.env.AWBWEnv` (candidate tables included).

    SB3 pickles stale shapes; async actors may also load an older ``_async_actor_skeleton.zip``
    whose policy class predates candidate observations.  The env always supplies
    ``candidate_features`` / ``candidate_mask``; the policy must declare those Box spaces
    or ``obs_to_tensor`` raises ``KeyError``.

    Also updates each features extractor's ``_observation_space`` (SB3 private field) so
    nothing keeps a stale Dict with only two keys.
    """
    fixed = _current_awbw_observation_space(with_candidates=True)
    model.observation_space = fixed
    pol = getattr(model, "policy", None)
    if pol is None:
        return
    pol.observation_space = fixed
    for attr in (
        "features_extractor",
        "pi_features_extractor",
        "vf_features_extractor",
    ):
        fe = getattr(pol, attr, None)
        if fe is not None and hasattr(fe, "_observation_space"):
            fe._observation_space = fixed


def _sync_loaded_model_observation_space(model) -> None:
    """
    Legacy zips deserialize stale ``observation_space`` shapes. After weights are
    transplanted, align SB3's space so ``predict`` accepts env observations.

    Candidate-action checkpoints ship four Dict keys; :func:`materialize_sb3_zip_with_spatial_compat`
    patches convolution weights but the pickled ``observation_space`` can still list
    old channel counts.  Replacing the Dict with spatial+scalars-only used to drop
    ``candidate_*`` keys and crash opponent ``predict`` in workers.
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
    with_candidates = (
        isinstance(obs_sp.spaces.get("candidate_features"), gym.spaces.Box)
        and isinstance(obs_sp.spaces.get("candidate_mask"), gym.spaces.Box)
    )
    fixed = _current_awbw_observation_space(with_candidates=with_candidates)
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
            align_maskable_ppo_observation_space_to_awbw_env(model)
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
