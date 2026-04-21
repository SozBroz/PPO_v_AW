"""
Checkpoint compatibility for MaskablePPO zips saved before the spatial encoder
bump from 62 → 63 channels (single HP → hp_lo / hp_hi). See ``rl/encoder.py``.

When loading, we duplicate the legacy HP channel into both new slots and shift
later channels; stem ``Conv2d`` weights and matching Adam moments are expanded
the same way.
"""
from __future__ import annotations

import gc
import io
import os
import tempfile
import zipfile
from pathlib import Path

import torch

# Legacy layout: 28 unit + 1 HP + 15 terrain + 15 property + 3 capture = 62
# Current:      28 unit + 2 HP + 15 terrain + 15 property + 3 capture = 63


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


def _patch_policy_state_dict(sd: dict) -> None:
    for k, v in list(sd.items()):
        if not isinstance(v, torch.Tensor) or v.dim() != 4:
            continue
        if v.shape[1] != 62:
            continue
        if "stem.0.weight" not in k:
            continue
        sd[k] = expand_spatial_stem_in_channels_62_to_63(v)


def _patch_optimizer_state_dict(od: dict) -> None:
    st = od.get("state")
    if not isinstance(st, dict):
        return
    for _pid, entry in st.items():
        if not isinstance(entry, dict):
            continue
        for kk, v in list(entry.items()):
            if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[1] == 62:
                entry[kk] = expand_spatial_stem_in_channels_62_to_63(v)


def checkpoint_needs_spatial_stem_patch(ckpt_path: Path) -> bool:
    """True if ``policy.pth`` has 62 input channels on the conv stem."""
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
    if w is None or not isinstance(w, torch.Tensor):
        return False
    return w.shape[1] == 62


def materialize_sb3_zip_with_spatial_compat(ckpt_path: Path) -> tuple[Path, bool]:
    """
    If the zip uses legacy 62-channel stems, copy to a temp zip with patched
    ``policy.pth`` / ``policy.optimizer.pth``. Otherwise return ``(ckpt_path, False)``.
    """
    ckpt_path = Path(ckpt_path).resolve()
    if not checkpoint_needs_spatial_stem_patch(ckpt_path):
        return ckpt_path, False

    repo_root = Path(__file__).resolve().parent.parent
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix="ckpt_spatial62_", suffix=".zip", dir=str(tmp_dir))
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
                    _patch_optimizer_state_dict(od)
                    bio = io.BytesIO()
                    torch.save(od, bio)
                    raw = bio.getvalue()
                zout.writestr(name, raw)

    out_path.write_bytes(buf.getvalue())
    print(
        f"[ckpt_compat] Patched 62->63 spatial stem for {ckpt_path.name} "
        f"(legacy single-HP checkpoint to dual HP layout)"
    )
    return out_path, True


def load_maskable_ppo_compat(path: str | Path, **kwargs):
    """
    ``MaskablePPO.load`` with automatic 62→63 spatial stem migration when needed.

    Writes a temp zip under ``<repo>/.tmp/`` when patching; removes it after
    load (best effort on Windows).
    """
    from sb3_contrib import MaskablePPO  # type: ignore[import]

    p = Path(path)
    use_path, is_temp = materialize_sb3_zip_with_spatial_compat(p)
    try:
        return MaskablePPO.load(str(use_path), **kwargs)
    finally:
        if is_temp:
            gc.collect()
            try:
                use_path.unlink(missing_ok=True)
            except OSError:
                pass
