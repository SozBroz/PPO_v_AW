"""
IMPALA-style decoupled actors + V-trace learner for AWBW.

Actors run ``AWBWEnv`` in independent processes (no SubprocVecEnv step barrier).
The learner consumes fixed-length unrolls from a bounded queue and applies
V-trace policy-gradient + value loss on ``MaskablePPO.policy``.

Weight sync: learner atomically writes ``checkpoints/async_policy_weights.pt``;
actors poll and reload ``policy.state_dict()`` when the version increases.
"""
from __future__ import annotations

import multiprocessing as mp
import math
import os
import queue
import random
import secrets
import sys
import tempfile
from contextlib import nullcontext
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from stable_baselines3.common.utils import explained_variance as sb3_explained_variance
from torch import nn
from torch.utils.checkpoint import checkpoint

from rl.network import AWBWCandidateFeaturesExtractor
from rl.vtrace import from_importance_weights

# region agent log
_AGENT_DEBUG_LOG_PATH = Path(__file__).parent.parent / "debug-a6d5a1.log"
_AGENT_DEBUG_SESSION_ID = "a6d5a1"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION_ID,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(__import__("json").dumps(payload, default=str) + "\n")
    except Exception:
        pass
# endregion

if TYPE_CHECKING:
    from rl.self_play import SelfPlayTrainer


def _nn_train_log_ratio_diagnostics(
    log_pi_minus_mu: torch.Tensor,
    *,
    rho_floor: float,
    rho_hi: float = 20.0,
    kl_sym_abs_cap: float | None = None,
) -> tuple[float, float, float | None, float, float, float]:
    """Learner-vs-behaviour log-prob diagnostics for ``nn_train`` (see learner callsite)."""
    lr = log_pi_minus_mu.detach().float().reshape(-1)
    if kl_sym_abs_cap is None:
        raw_sym = (os.environ.get("AWBW_ASYNC_NN_KL_DIAG_ABS") or "2.0").strip()
        try:
            kl_sym_abs_cap = float(raw_sym)
        except ValueError:
            kl_sym_abs_cap = 2.0
    kl_sym_abs_cap = max(float(kl_sym_abs_cap), 1e-6)

    lt_sym = lr.clamp(-kl_sym_abs_cap, kl_sym_abs_cap)
    approx_kl = float(torch.mean(torch.exp(lt_sym) - 1.0 - lt_sym).cpu())

    lr_v = lr.clamp(min=float(rho_floor), max=float(rho_hi))
    approx_kl_vtrace_log = float(
        torch.mean(torch.exp(lr_v) - 1.0 - lr_v).cpu()
    )

    raw_terms = torch.exp(lr) - 1.0 - lr
    rk = float(raw_terms.mean().cpu())
    approx_uncapped: float | None = rk if math.isfinite(rk) else None
    ratio_mean = float(lr.mean().cpu())
    frac_hi = float((lr >= float(rho_hi)).float().mean().cpu())
    frac_lo = float((lr <= float(rho_floor)).float().mean().cpu())
    return (
        approx_kl,
        approx_kl_vtrace_log,
        approx_uncapped,
        ratio_mean,
        frac_hi,
        frac_lo,
    )


def _async_actor_configure_cpu_parallelism() -> int:
    """
    Limit BLAS / PyTorch intra-op threads per **actor** process.

    Without this, each spawned actor may default to using all cores; ``N`` actors
    then oversubscribe the CPU and spend most wall time context-switching (looks
    like "only one worker moves"). Override via ``AWBW_ASYNC_ACTOR_THREADS``
    (default ``1``).
    """
    raw = (os.environ.get("AWBW_ASYNC_ACTOR_THREADS") or "1").strip()
    try:
        n = int(raw, 10)
    except ValueError:
        n = 1
    n = max(1, min(n, 256))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    return n


def _async_wants_cuda_opponent_infer() -> bool:
    """
    Opt-in CUDA checkpoint opponents inside async **actor** processes.

    Default off: those workers share the same physical GPU as the IMPALA learner; even
    short ``policy.cuda()`` forwards from several processes routinely exhaust ~12GB cards.
    """
    return (os.environ.get("AWBW_ASYNC_GPU_OPPONENTS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _os_transient_checkpoint_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        errno = getattr(exc, "errno", None)
        if errno in (11, 13, 16):  # EAGAIN, EACCESS, EBUSY (platform-dependent)
            return True
        winerr = getattr(exc, "winerror", None)
        # 5 ACCESS_DENIED / 32 SHARING_VIOLATION when replace races readers or Defender
        if winerr in (5, 32):
            return True
    return False


def _atomic_torch_save(obj: dict[str, Any], dest: Path) -> None:
    """Write checkpoint bytes then publish under ``dest``. Prefer atomic ``os.replace``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f"{dest.name}.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    tmp_str = str(tmp_path)
    dest_str = str(dest)

    def backoff(attempt: int) -> None:
        time.sleep(min(1.0, 0.03 * (1.35**attempt)))

    try:
        torch.save(obj, tmp_str)

        first_fail: OSError | None = None
        for attempt in range(50):
            try:
                os.replace(tmp_str, dest_str)
                return
            except OSError as e:
                first_fail = e
                if attempt < 49 and _os_transient_checkpoint_error(e):
                    backoff(attempt)
                    continue
                break

        # NTFS overlay-replace can still deny when many readers ping the path; renaming
        # the incumbent file frees the basename so ``tmp`` -> ``dest`` is a rename, not ReplaceFile onto busy target.
        if os.name != "nt" or not Path(dest_str).is_file():
            raise first_fail

        stash_str = str(dest.parent / f"{dest.name}.prior.{secrets.token_hex(8)}")
        stash_path = Path(stash_str)
        rotated = False
        for attempt in range(50):
            try:
                os.replace(dest_str, stash_str)
                rotated = True
                break
            except OSError as e:
                if attempt < 49 and _os_transient_checkpoint_error(e):
                    backoff(attempt)
                    continue
                raise first_fail from e
        if not rotated:
            raise first_fail

        try:
            last_fail: OSError | None = None
            for attempt in range(50):
                try:
                    os.replace(tmp_str, dest_str)
                    return
                except OSError as e:
                    last_fail = e
                    if attempt < 49 and _os_transient_checkpoint_error(e):
                        backoff(attempt)
                        continue
                    break
            err = last_fail if last_fail is not None else RuntimeError("replace failed")
            raise RuntimeError(
                "Could not publish async policy weights after rotating prior copy "
                f"(stash suffix .prior.<hex>); disk or AV may still be locking files"
            ) from err
        finally:
            try:
                if stash_path.is_file():
                    stash_path.unlink()
            except OSError:
                pass
    finally:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _policy_weight_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "async_policy_weights.pt"


def _fps_diag_file_enabled() -> bool:
    return (os.environ.get("AWBW_FPS_DIAG") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_WORKER_RSS_RESAMPLE_S = 5.0


def _unsigned32(n: int) -> int:
    return int(n) & 0xFFFFFFFF


def _describe_process_exitcode(exitcode: int | None) -> str:
    """Best-effort decode for child ``multiprocessing.Process.exitcode`` (esp. Windows NTSTATUS)."""
    if exitcode is None:
        return "exitcode=None (unexpected here)"
    u = _unsigned32(exitcode)
    bits = [f"raw={exitcode!r}", f"u32=0x{u:08X}"]
    # Common process exit / NT status codes (non-exhaustive; enough to spot OOM vs AV).
    known: dict[int, str] = {
        0x0: "OK",
        0xC0000005: "STATUS_ACCESS_VIOLATION (native bad pointer; often C extension / torch)",
        0xC00000FD: "STATUS_STACK_OVERFLOW",
        0xC0000017: "STATUS_NO_MEMORY (Windows allocator could not commit pages)",
        0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
        0xC000012D: "STATUS_COMMITMENT_LIMIT (commit charge / page file exhausted)",
        0xC0000409: "STATUS_STACK_BUFFER_OVERRUN / fast fail",
    }
    if u in known:
        bits.append(known[u])
    elif 1 <= exitcode <= 255:
        bits.append("small positive (often a Python or app exit() code)")
    return " | ".join(bits)


def _psutil_rss_hint(pid: int) -> str:
    try:
        import psutil

        p = psutil.Process(pid)
        with p.oneshot():
            mi = p.memory_info()
            rss = float(mi.rss) / (1024.0**2)
        return f"pid={pid} still visible to psutil, rss~{rss:.0f}MiB (stale if pid reused)"
    except Exception as exc:  # noqa: BLE001
        return f"pid={pid} not visible: {type(exc).__name__}: {exc!r}"


def _log_dead_actor_diagnostics(
    procs: list[mp.Process], dead_indices: list[int]
) -> str:
    """Join + exitcode decode + host RAM snapshot for the operator log."""
    lines: list[str] = []
    try:
        import psutil  # noqa: F401

        vm = psutil.virtual_memory()
        lines.append(
            f"host: ram_used_percent={vm.percent:.1f} available_mib~{vm.available / (1024**2):.0f} "
            f"total_mib~{vm.total / (1024**2):.0f} (psutil virtual_memory())"
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"host: psutil RAM snapshot failed: {type(exc).__name__}: {exc!r}")
    for i in dead_indices:
        p = procs[i]
        try:
            p.join(timeout=2.0)
        except OSError as exc:  # noqa: BLE001
            lines.append(f"actor[{i}]: join failed: {exc!r}")
        ec = p.exitcode
        desc = _describe_process_exitcode(ec)
        pid = getattr(p, "pid", None)
        if pid is not None:
            desc = f"pid={pid} | {desc} | {_psutil_rss_hint(int(pid))}"
        lines.append(f"actor[{i}]: {desc}")
    return "\n".join(lines)


def _actor_skeleton_zip(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "_async_actor_skeleton.zip"


def _actor_step(
    policy: Any,
    obs: dict[str, np.ndarray],
    mask: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    """Return action, value estimate, behaviour logprob."""
    import torch as th

    obs_tensor, _ = policy.obs_to_tensor(obs)
    if isinstance(obs_tensor, dict):
        obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
    else:
        obs_tensor = obs_tensor.to(device)
    mask_row = mask[None, :].astype(np.bool_, copy=False)
    with th.inference_mode():
        actions, values, log_prob = policy.forward(
            obs_tensor,
            deterministic=False,
            action_masks=mask_row,
        )
    act = int(actions.view(-1)[0].item())
    val = float(values.view(-1)[0].item())
    lp = float(log_prob.view(-1)[0].item())
    return act, val, lp


def _dual_gradient_env(env: Any) -> Any:
    """Return the underlying AWBWEnv when factories wrap it in ActionMasker."""
    return getattr(env, "env", env)


def _tag_dual_gradient_rollout_mode(dg_env: Any, mirror_mode: bool) -> None:
    """Record per-episode async dual-gradient labelling for ``game_log.jsonl``."""
    fn = getattr(dg_env, "set_async_rollout_mode", None)
    if callable(fn):
        fn("mirror" if mirror_mode else "hist")


def actor_process_main(
    actor_id: int,
    env_factory: Callable[[], Any],
    skeleton_zip: str,
    weight_path: str,
    rollout_queue: mp.Queue,
    stop_event: mp.Event,
    unroll_len: int,
    device_str: str,
    poll_s: float,
    dual_gradient_self_play: bool = False,
    dual_gradient_hist_prob: float = 0.0,
) -> None:
    """One actor: build env, roll unrolls, push dict chunks to rollout_queue."""
    # Hide GPUs from this process *before* importing torch. Otherwise each spawned
    # actor can initialize its own CUDA context on the same card as the learner;
    # six contexts routinely exhaust ~12GB Windows setups (driver reports 0 B free).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _actor_threads = _async_actor_configure_cpu_parallelism()

    from rl.ckpt_compat import align_maskable_ppo_observation_space_to_awbw_env, load_maskable_ppo_compat

    try:
        dev = torch.device(device_str)
        env = env_factory()
        model = load_maskable_ppo_compat(
            skeleton_zip,
            env=None,
            device=device_str,
            verbose=0,
            n_envs=1,
            n_steps=1,
            batch_size=1,
        )
        align_maskable_ppo_observation_space_to_awbw_env(model)
        try:
            torch.set_num_threads(_actor_threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        policy = model.policy
        policy.set_training_mode(False)

        local_ver = -1

        def _reload_if_needed() -> None:
            nonlocal local_ver
            p = Path(weight_path)
            if not p.is_file():
                return
            try:
                with p.open("rb") as fh:
                    blob = torch.load(fh, map_location="cpu", weights_only=False)
            except OSError:
                return
            v = int(blob.get("version", -1))
            if v <= local_ver:
                return
            sd = blob.get("policy")
            if isinstance(sd, dict):
                policy.load_state_dict(sd, strict=True)
                local_ver = v

        _reload_if_needed()
        align_maskable_ppo_observation_space_to_awbw_env(model)
        obs, _info = env.reset()
        dg_env = _dual_gradient_env(env)
        if dual_gradient_self_play:
            mirror_mode = bool(
                dual_gradient_hist_prob <= 0.0
                or random.random() >= float(dual_gradient_hist_prob)
            )
            if mirror_mode:
                obs = dg_env.active_seat_observation()
            use_mirror_dg = mirror_mode
            _tag_dual_gradient_rollout_mode(dg_env, mirror_mode)
        else:
            mirror_mode = False
            use_mirror_dg = False
        if not isinstance(obs, dict):
            raise TypeError("AWBWEnv must return dict observations")

        while not stop_event.is_set():
            _reload_if_needed()
            align_maskable_ppo_observation_space_to_awbw_env(model)
            spatial_chunks: list[np.ndarray] = []
            scalars_chunks: list[np.ndarray] = []
            cand_feat_chunks: list[np.ndarray] = []
            cand_mask_chunks: list[np.ndarray] = []
            mask_chunks: list[np.ndarray] = []
            actions_arr = np.zeros((unroll_len,), dtype=np.int64)
            rewards_arr = np.zeros((unroll_len,), dtype=np.float32)
            dones_arr = np.zeros((unroll_len,), dtype=np.float32)
            mu_logp_arr = np.zeros((unroll_len,), dtype=np.float32)

            last_done = False
            for t in range(unroll_len):
                if stop_event.is_set():
                    return
                mask = (
                    dg_env.active_seat_action_mask()
                    if use_mirror_dg
                    else env.action_masks()
                )
                act, _v, logp = _actor_step(policy, obs, mask, dev)
                if use_mirror_dg:
                    next_obs, rew, term, trunc, _info = dg_env.step_active_seat_once(act)
                else:
                    next_obs, rew, term, trunc, _info = env.step(act)
                done = bool(term or trunc)
                spatial_chunks.append(np.asarray(obs["spatial"], dtype=np.float32))
                scalars_chunks.append(np.asarray(obs["scalars"], dtype=np.float32))
                cand_feat_chunks.append(
                    np.asarray(obs["candidate_features"], dtype=np.float32)
                )
                cand_mask_chunks.append(
                    np.asarray(obs["candidate_mask"], dtype=np.int8)
                )
                mask_chunks.append(np.asarray(mask, dtype=np.bool_))
                actions_arr[t] = act
                rewards_arr[t] = float(rew)
                dones_arr[t] = 1.0 if done else 0.0
                mu_logp_arr[t] = logp
                last_done = done
                if done:
                    obs, _info = env.reset()
                    if dual_gradient_self_play:
                        mirror_mode = bool(
                            dual_gradient_hist_prob <= 0.0
                            or random.random() >= float(dual_gradient_hist_prob)
                        )
                        if mirror_mode:
                            obs = dg_env.active_seat_observation()
                        use_mirror_dg = mirror_mode
                        _tag_dual_gradient_rollout_mode(dg_env, mirror_mode)
                    else:
                        mirror_mode = False
                        use_mirror_dg = False
                else:
                    obs = (
                        dg_env.active_seat_observation()
                        if use_mirror_dg
                        else next_obs
                    )
                if not isinstance(obs, dict):
                    raise TypeError("AWBWEnv must return dict observations")

            bootstrap_mask = (
                dg_env.active_seat_action_mask()
                if use_mirror_dg
                else env.action_masks()
            ).astype(np.bool_, copy=False)

            chunk = {
                "actor_id": actor_id,
                "spatial": np.stack(spatial_chunks, axis=0),
                "scalars": np.stack(scalars_chunks, axis=0),
                "candidate_features": np.stack(cand_feat_chunks, axis=0),
                "candidate_mask": np.stack(cand_mask_chunks, axis=0),
                "mask": np.stack(mask_chunks, axis=0),
                "actions": actions_arr,
                "rewards": rewards_arr,
                "dones": dones_arr,
                "mu_logp": mu_logp_arr,
                "bootstrap_spatial": np.asarray(obs["spatial"], dtype=np.float32),
                "bootstrap_scalars": np.asarray(obs["scalars"], dtype=np.float32),
                "bootstrap_candidate_features": np.asarray(
                    obs["candidate_features"], dtype=np.float32
                ),
                "bootstrap_candidate_mask": np.asarray(
                    obs["candidate_mask"], dtype=np.int8
                ),
                "bootstrap_mask": bootstrap_mask,
                "tail_done": bool(last_done),
            }
            # Block until there is queue space (no timeout+sleep: that burned CPU/wakeups
            # and exaggerated "all actors idle" in Task Manager when the learner fell behind).
            while not stop_event.is_set():
                try:
                    rollout_queue.put(chunk, block=True, timeout=1.0)
                    break
                except queue.Full:
                    continue
    except Exception:
        tb = traceback.format_exc()
        msg = f"[async_impala] actor {actor_id} crashed (pid={os.getpid()}):\n{tb}"
        print(msg, file=sys.stdout, flush=True)
        try:
            print(msg, file=sys.stderr, flush=True)
        except OSError:
            pass
        raise


def _flatten_time_batch(
    spatial: torch.Tensor,
    scalars: torch.Tensor,
    candidate_features: torch.Tensor,
    candidate_mask: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """(T, B, ...) -> (T*B, ...)."""
    t, b = spatial.shape[:2]
    flat_sp = spatial.reshape(t * b, *spatial.shape[2:])
    flat_sc = scalars.reshape(t * b, scalars.shape[-1])
    flat_cf = candidate_features.reshape(t * b, *candidate_features.shape[2:])
    flat_cm = candidate_mask.reshape(t * b, candidate_mask.shape[-1])
    flat_m = mask.reshape(t * b, mask.shape[-1])
    flat_a = actions.reshape(t * b)
    obs = {
        "spatial": flat_sp,
        "scalars": flat_sc,
        "candidate_features": flat_cf,
        "candidate_mask": flat_cm,
    }
    return obs, flat_m, flat_a


_LEARNER_EVAL_NO_CHUNK = 2**30


def _learner_forward_chunk_cap(device: torch.device, trainer: SelfPlayTrainer) -> int:
    """CUDA: cap observations per ``evaluate_actions`` to limit peak activation VRAM."""
    t = getattr(trainer, "async_learner_forward_chunk", None)
    if t is not None and int(t) > 0:
        return max(1, int(t))
    if device.type != "cuda":
        return _LEARNER_EVAL_NO_CHUNK
    raw = (os.environ.get("AWBW_ASYNC_LEARNER_FORWARD_CHUNK") or "256").strip()
    try:
        c = int(raw, 10)
    except ValueError:
        c = 256
    if c <= 0:
        return _LEARNER_EVAL_NO_CHUNK
    return max(16, c)


def _learner_eval_actions_checkpoint() -> bool:
    """Trade extra compute for VRAM: checkpoint each microbatched ``evaluate_actions``."""
    return (os.environ.get("AWBW_ASYNC_LEARNER_CHECKPOINT") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _evaluate_actions_microbatched(
    pol: Any,
    flat_obs: dict[str, torch.Tensor],
    flat_actions: torch.Tensor,
    flat_mask: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chunked ``evaluate_actions``; on CUDA uses gradient checkpointing per chunk (see microbatch note)."""
    n = int(flat_actions.shape[0])
    dev = flat_actions.device
    if chunk <= 0 or n <= chunk:
        return pol.evaluate_actions(flat_obs, flat_actions, action_masks=flat_mask)
    val_parts: list[torch.Tensor] = []
    lp_parts: list[torch.Tensor] = []
    ent_parts: list[torch.Tensor] = []
    use_ckpt = _learner_eval_actions_checkpoint() and dev.type == "cuda"
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        spatial_s = flat_obs["spatial"][s:e]
        scalars_s = flat_obs["scalars"][s:e]
        cand_f_s = flat_obs["candidate_features"][s:e]
        cand_m_s = flat_obs["candidate_mask"][s:e]
        acts_s = flat_actions[s:e]
        mask_s = flat_mask[s:e]
        if use_ckpt:

            def _fwd(
                sp: torch.Tensor,
                sc: torch.Tensor,
                cf: torch.Tensor,
                cm: torch.Tensor,
                ac: torch.Tensor,
                ms: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
                o = {
                    "spatial": sp,
                    "scalars": sc,
                    "candidate_features": cf,
                    "candidate_mask": cm,
                }
                return pol.evaluate_actions(o, ac, action_masks=ms)

            v, lp, ent = checkpoint(
                _fwd,
                spatial_s,
                scalars_s,
                cand_f_s,
                cand_m_s,
                acts_s,
                mask_s,
                use_reentrant=False,
            )
        else:
            obs_c = {
                "spatial": spatial_s,
                "scalars": scalars_s,
                "candidate_features": cand_f_s,
                "candidate_mask": cand_m_s,
            }
            v, lp, ent = pol.evaluate_actions(obs_c, acts_s, action_masks=mask_s)
        val_parts.append(v)
        lp_parts.append(lp)
        if ent is not None:
            ent_parts.append(ent)
    values_f = torch.cat(val_parts, dim=0)
    logp_pi = torch.cat(lp_parts, dim=0)
    entropy: torch.Tensor | None = (
        torch.cat(ent_parts, dim=0) if ent_parts else None
    )
    return values_f, logp_pi, entropy


def run_impala_training(trainer: SelfPlayTrainer) -> None:
    from sb3_contrib import MaskablePPO  # type: ignore[import]
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]

    from rl.fleet_env import (
        new_checkpoint_stem_utc,
        prune_checkpoint_zip_curated,
        prune_checkpoint_zip_snapshots,
        sorted_checkpoint_zip_paths,
    )
    from rl.self_play import (
        LOGS_DIR,
        _append_fps_diag_line,
        _append_nn_train_line,
        _atomic_model_save,
        _main_proc_rss_mb,
        _make_env_factory,
        _mask_fn,
        _resolve_live_snapshot_pkl_path,
        _sum_python_children_rss_mb,
        _system_ram_used_pct,
        gpu_opponent_pool_enabled,
        gpu_opponent_pool_permits,
    )
    from rl.candidate_actions import MAX_CANDIDATES
    from rl.ckpt_compat import (
        align_maskable_ppo_observation_space_to_awbw_env,
        load_maskable_ppo_compat,
        scalpel_checkpoint_zip_to_candidate_maskable_ppo_zip,
    )
    from rl.network import ACTION_SPACE_SIZE

    mp_ctx = mp.get_context("spawn")
    stop_event = mp_ctx.Event()
    ckpt_dir = Path(trainer.checkpoint_dir)
    weight_path = _policy_weight_path(ckpt_dir)
    skel_zip = _actor_skeleton_zip(ckpt_dir)
    unroll = max(4, int(trainer.async_unroll_length))
    n_actors = max(1, int(trainer.n_envs))
    batch_segments = max(1, int(trainer.async_learner_batch) // unroll)
    q: mp.Queue = mp_ctx.Queue(maxsize=max(2, int(trainer.async_queue_max)))

    policy_kwargs = dict(
        features_extractor_class=AWBWCandidateFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[],
    )

    latest_path = ckpt_dir / "latest.zip"
    promoted_path = ckpt_dir / "promoted" / "best.zip"
    resume_path = latest_path
    dual_gradient_self_play = bool(getattr(trainer, "dual_gradient_self_play", False))
    dual_gradient_hist_prob = float(getattr(trainer, "dual_gradient_hist_prob", 0.0) or 0.0)
    if trainer.load_promoted and promoted_path.is_file():
        if not latest_path.is_file():
            resume_path = promoted_path
        elif promoted_path.stat().st_mtime > latest_path.stat().st_mtime:
            resume_path = promoted_path

    from multiprocessing import Manager

    # Subproc CUDA opponents (fixed-index or short cuda() bursts) share the same GPU as the
    # learner and routinely OOM ~12GB cards; default forces CPU opponents in actor procs.
    opponent_force_cpu = not _async_wants_cuda_opponent_infer()
    gpu_pool_manager: Any = None
    gpu_sem = None
    if opponent_force_cpu:
        print(
            "[async_impala] Async actors: opponent checkpoints forced to CPU "
            "(learner uses CUDA). Set AWBW_ASYNC_GPU_OPPONENTS=1 + hybrid pool for GPU opponents."
        )
    elif gpu_opponent_pool_enabled():
        base_perm = int(gpu_opponent_pool_permits())
        sub = int(getattr(trainer, "async_gpu_opponent_permits_subtract", 0) or 0)
        eff_perm = max(1, base_perm - max(0, sub))
        gpu_pool_manager = Manager()
        gpu_sem = gpu_pool_manager.BoundedSemaphore(eff_perm)
        print(
            f"[async_impala] GPU opponent semaphore (manager proxy): "
            f"{eff_perm} concurrent CUDA opponent forwards "
            f"(pool cap {base_perm}"
            f"{f', -{sub} for async learner headroom' if sub else ''})"
        )
    else:
        print(
            "[async_impala] AWBW_ASYNC_GPU_OPPONENTS=1 but AWBW_GPU_OPPONENT_POOL is off; "
            "actors use normal CUDA/CPU opponent routing (VRAM risk on one GPU)."
        )

    env_kw = dict(
        co_p0=trainer.co_p0,
        co_p1=trainer.co_p1,
        tier_name=trainer.tier_name,
        curriculum_broad_prob=trainer.curriculum_broad_prob,
        curriculum_tag=trainer.curriculum_tag,
        opponent_mix=trainer.opponent_mix,
        pool_from_fleet=trainer.pool_from_fleet,
        cold_opponent=trainer.cold_opponent,
        fleet_opponent_root=trainer.fleet_opponent_root,
        max_env_steps=trainer.max_env_steps,
        max_p1_microsteps=trainer.max_p1_microsteps,
        max_turns=getattr(trainer, "max_turns", None),
        opening_book_path=getattr(trainer, "opening_book_path", None),
        opening_book_seat=int(getattr(trainer, "opening_book_seat", 1) or 1),
        opening_book_prob=float(getattr(trainer, "opening_book_prob", 1.0) or 0.0),
        opening_book_strict_co=bool(getattr(trainer, "opening_book_strict_co", False)),
        opening_book_max_day=getattr(trainer, "opening_book_max_day", None),
        opening_book_seed=int(getattr(trainer, "opening_book_seed", 0) or 0),
        cop_disable_per_seat_p=getattr(trainer, "cop_disable_per_seat_p", None),
        opponent_pool_newest_k=int(getattr(trainer, "checkpoint_pool_size", 24) or 0),
    )
    n_live = len(getattr(trainer, "live_games_id", None) or [])
    # region agent log
    _agent_debug_log(
        "H1,H2,H6",
        "rl/async_impala.py:run_impala_training",
        "async trainer env/opening/spirit configuration before actor factories",
        {
            "n_actors": int(n_actors),
            "n_live": int(n_live),
            "live_games_id": list(getattr(trainer, "live_games_id", None) or []),
            "opening_book_path": env_kw.get("opening_book_path"),
            "opening_book_seat": env_kw.get("opening_book_seat"),
            "opening_book_prob": env_kw.get("opening_book_prob"),
            "opening_book_max_day": env_kw.get("opening_book_max_day"),
            "spirit_env": os.environ.get("AWBW_SPIRIT_BROKEN"),
            "training_backend": getattr(trainer, "training_backend", None),
            "curriculum_tag": getattr(trainer, "curriculum_tag", None),
        },
    )
    # endregion

    if dual_gradient_self_play:
        hp = float(dual_gradient_hist_prob)
        if hp > 0.0:
            print(
                f"[async_impala] dual-gradient: ~{100.0 * (1.0 - hp):.0f}% mirror self-play "
                f"(both seats, synced weights); ~{100.0 * hp:.0f}% learner vs historical "
                "checkpoint opponent (per-episode roulette).",
                flush=True,
            )
        else:
            print(
                "[async_impala] dual-gradient self-play enabled: both engine seats "
                "sample from the shared policy and contribute rollout rows.",
                flush=True,
            )

    load_common = {"env": None, "device": trainer.device}
    ckpt_phys: Path | None = None
    reload_kw: dict[str, Any] | None = None

    if resume_path.exists():
        reload_kw = {
            **load_common,
            "custom_objects": {"n_steps": unroll, "n_envs": 1},
        }
        model = load_maskable_ppo_compat(resume_path, **reload_kw)
        model.tensorboard_log = str(LOGS_DIR)
        skeleton_zip = str(resume_path.resolve())
        ckpt_phys = resume_path.resolve()
    elif trainer.bc_init is not None and trainer.bc_init.is_file():
        reload_kw = {
            **load_common,
            "custom_objects": {
                "n_steps": unroll,
                "n_envs": 1,
                "batch_size": trainer.batch_size,
            },
        }
        model = load_maskable_ppo_compat(trainer.bc_init, **reload_kw)
        model.tensorboard_log = str(LOGS_DIR)
        skeleton_zip = str(Path(trainer.bc_init).resolve())
        ckpt_phys = Path(trainer.bc_init).resolve()
    else:
        trainer.live_snapshot_dir.mkdir(parents=True, exist_ok=True)
        fkw0 = dict(env_kw)
        if n_live > 0:
            fkw0["opening_book_path"] = None
        factory0 = _make_env_factory(
            trainer.map_pool,
            str(ckpt_dir),
            worker_index=0,
            gpu_infer_semaphore=gpu_sem,
            opponent_force_cpu=opponent_force_cpu,
            **fkw0,
        )
        dummy_env = ActionMasker(factory0(), _mask_fn)
        model = MaskablePPO(
            "MultiInputPolicy",
            dummy_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=trainer.device,
            learning_rate=float(trainer.async_learning_rate),
            n_steps=unroll,
            batch_size=min(trainer.batch_size, unroll * n_actors),
            n_epochs=1,
            gamma=float(trainer.async_gamma),
            gae_lambda=0.95,
            ent_coef=float(trainer.ent_coef),
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=False,
            tensorboard_log=str(LOGS_DIR),
        )
        dummy_env.close()
        _atomic_model_save(model, skel_zip.parent / skel_zip.stem)
        skeleton_zip = str(skel_zip.resolve())
        ckpt_phys = None
        reload_kw = None

    # IMPALA envs emit candidate masks (MAX_CANDIDATES). A legacy flat zip still has
    # Discrete(ACTION_SPACE_SIZE) logits — scalpel once so actors' skeleton matches masks.
    if ckpt_phys is not None and reload_kw is not None:
        n_act = int(model.action_space.n)
        if n_act != int(MAX_CANDIDATES):
            if n_act != int(ACTION_SPACE_SIZE):
                raise RuntimeError(
                    "Async IMPALA uses candidate actions; expected a checkpoint with "
                    f"action_space.n={MAX_CANDIDATES} (candidate) or {ACTION_SPACE_SIZE} (legacy flat "
                    "warm-start). "
                    f"Got n={n_act} from {ckpt_phys}. Use scripts/scalpel_latest_to_candidate_ppo.py "
                    "or remove/replace this zip."
                )
            dst = (ckpt_dir / "_async_scalpeled_resume.zip").resolve()
            dev_s = (
                trainer.device
                if isinstance(trainer.device, str)
                else str(trainer.device)
            )
            print(
                f"[async_impala] legacy flat checkpoint (n={n_act}) -> scalpel candidate "
                f"(n={MAX_CANDIDATES}) -> {dst}",
                flush=True,
            )
            scalpel_checkpoint_zip_to_candidate_maskable_ppo_zip(
                ckpt_phys, dst, features_dim=512, device=dev_s
            )
            model = load_maskable_ppo_compat(dst, **reload_kw)
            model.tensorboard_log = str(LOGS_DIR)
            skeleton_zip = str(dst)

    align_maskable_ppo_observation_space_to_awbw_env(model)

    pol = model.policy
    # MaskablePPO.load restores policy.optimizer (full Adam state on CUDA). We used to
    # allocate a *second* Adam for the V-trace loop, doubling optimizer VRAM (~2x params).
    pol.optimizer = torch.optim.Adam(
        pol.parameters(), lr=float(trainer.async_learning_rate)
    )
    opt = pol.optimizer
    device = torch.device(model.device)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.amp.GradScaler("cpu")

    policy_version = 0
    _atomic_torch_save(
        {"version": policy_version, "policy": pol.state_dict()},
        weight_path,
    )

    trainer.live_snapshot_dir.mkdir(parents=True, exist_ok=True)
    seats = trainer.live_learner_seats or [0] * n_live
    factories: list[Callable[[], Any]] = []
    for i in range(n_actors):
        ekw = dict(env_kw)
        if i < n_live:
            ekw["opening_book_path"] = None
        if i < n_live:
            gid = int(trainer.live_games_id[i])
            spath = _resolve_live_snapshot_pkl_path(trainer.live_snapshot_dir, gid)
            factories.append(
                _make_env_factory(
                    trainer.map_pool,
                    str(ckpt_dir),
                    live_snapshot_path=spath,
                    live_games_id=gid,
                    live_learner_seat=seats[i],
                    worker_index=i,
                    gpu_infer_semaphore=gpu_sem,
                    opponent_force_cpu=opponent_force_cpu,
                    **ekw,
                )
            )
        else:
            factories.append(
                _make_env_factory(
                    trainer.map_pool,
                    str(ckpt_dir),
                    worker_index=i,
                    gpu_infer_semaphore=gpu_sem,
                    opponent_force_cpu=opponent_force_cpu,
                    **ekw,
                )
            )

    actor_device = "cpu"
    procs: list[mp.Process] = []
    for aid, factory in enumerate(factories):
        p = mp_ctx.Process(
            target=actor_process_main,
            name=f"awbw_actor_{aid}",
            args=(
                aid,
                factory,
                skeleton_zip,
                str(weight_path),
                q,
                stop_event,
                unroll,
                actor_device,
                0.05,
                dual_gradient_self_play,
                float(dual_gradient_hist_prob),
            ),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    pids = [getattr(x, "pid", None) for x in procs]
    print(
        f"[async_impala] actor processes started: n={n_actors} pids={pids!r} "
        f"(use Task Manager / logs below if a worker vanishes)",
        flush=True,
    )

    steps_done = 0
    learner_updates = 0
    t_start = time.time()
    cap = trainer.total_timesteps
    clip_rho = float(trainer.async_clip_rho)
    clip_pg = float(trainer.async_clip_pg_rho)
    gamma = float(trainer.async_gamma)
    vf_coef = float(trainer.async_vf_coef)
    ent_coef = float(trainer.ent_coef)
    max_grad_norm = float(trainer.async_max_grad_norm)
    rho_floor = float(trainer.async_log_rho_floor)

    next_save_at = int(trainer.save_every) if int(trainer.save_every) > 0 else None

    fps_diag_on = _fps_diag_file_enabled()
    diag_perf_t0 = time.perf_counter()
    diag_initial_rss_mb: float | None = None
    diag_rollout_seq = 0
    diag_last_worker_scan_mono = 0.0
    diag_cached_sum_worker_rss_mb = 0.0
    prev_learn_s: float | None = None

    learn_chunk = _learner_forward_chunk_cap(device, trainer)
    _roll = int(trainer.n_steps) * int(trainer.n_envs)
    _eff = int(unroll * batch_segments)
    _capped = (
        not bool(getattr(trainer, "async_learner_batch_explicit", True))
        and _eff < _roll
    )
    _chunk_note = ""
    if device.type == "cuda" and learn_chunk < _LEARNER_EVAL_NO_CHUNK // 2:
        _chunk_note = f" | learner_eval_chunk={learn_chunk}"
    print(
        f"[async_impala] IMPALA learner on {trainer.device} | actors={n_actors} "
        f"unroll={unroll} batch_segments={batch_segments} (~{_eff} transitions/update; "
        f"rollout={_roll}{', default cap (VRAM)' if _capped else ''}) "
        f"| queue_max={trainer.async_queue_max}{_chunk_note}"
        f"{' | fps_diag->logs/fps_diag.jsonl+stdout' if fps_diag_on else ''}"
        f" | nn_train->logs/nn_train.jsonl"
    )

    try:
        while cap is None or steps_done < cap:
            if not all(p.is_alive() for p in procs):
                dead = [i for i, p in enumerate(procs) if not p.is_alive()]
                report = _log_dead_actor_diagnostics(procs, dead)
                print(
                    "[async_impala] FATAL: one or more actor processes exited.\n" + report,
                    flush=True,
                )
                details = [f"actor[{i}].exitcode={procs[i].exitcode!r}" for i in dead]
                raise RuntimeError(
                    f"actor process(es) died: indices {dead} ({', '.join(details)}). "
                    "Diagnostics (exitcode decode, host RAM) were printed above as "
                    "[async_impala] FATAL. If u32=0xC0000017/0xC000012D, suspect host OOM or "
                    "pagefile; if STATUS_ACCESS_VIOLATION, suspect native (Cython/torch) bug."
                )

            t_collect0 = time.perf_counter()
            chunks: list[dict[str, Any]] = []
            while len(chunks) < batch_segments and not stop_event.is_set():
                try:
                    ch = q.get(timeout=1.0)
                    chunks.append(ch)
                except queue.Empty:
                    continue

            if not chunks:
                continue

            t = unroll
            b = len(chunks)
            spatial = np.stack([c["spatial"] for c in chunks], axis=1)
            scalars = np.stack([c["scalars"] for c in chunks], axis=1)
            cand_feat = np.stack([c["candidate_features"] for c in chunks], axis=1)
            cand_mask = np.stack([c["candidate_mask"] for c in chunks], axis=1)
            mask = np.stack([c["mask"] for c in chunks], axis=1)
            actions = np.stack([c["actions"] for c in chunks], axis=1)
            rewards = np.stack([c["rewards"] for c in chunks], axis=1)
            dones = np.stack([c["dones"] for c in chunks], axis=1)
            mu_logp = np.stack([c["mu_logp"] for c in chunks], axis=1)

            spatial_t = torch.as_tensor(spatial, device=device, dtype=torch.float32)
            scalars_t = torch.as_tensor(scalars, device=device, dtype=torch.float32)
            cand_feat_t = torch.as_tensor(cand_feat, device=device, dtype=torch.float32)
            cand_mask_t = torch.as_tensor(cand_mask, device=device, dtype=torch.float32)
            mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
            actions_t = torch.as_tensor(actions, device=device, dtype=torch.long)
            rewards_t = torch.as_tensor(rewards, device=device, dtype=torch.float32)
            dones_t = torch.as_tensor(dones, device=device, dtype=torch.float32)
            mu_logp_t = torch.as_tensor(mu_logp, device=device, dtype=torch.float32)

            flat_obs, flat_mask, flat_actions = _flatten_time_batch(
                spatial_t, scalars_t, cand_feat_t, cand_mask_t, mask_t, actions_t
            )
            discounts = gamma * (1.0 - dones_t)

            t_pre_learn = time.perf_counter()
            collect_s = max(0.0, t_pre_learn - t_collect0)
            steps_batch = int(t * b)
            env_steps_per_s_collect = (
                float(steps_batch) / collect_s if collect_s > 0 else 0.0
            )
            ppo_update_s_out: float | None = (
                float(prev_learn_s) if prev_learn_s is not None else None
            )
            env_steps_per_s_total = 0.0
            if (
                prev_learn_s is not None
                and collect_s + prev_learn_s > 0
                and steps_batch > 0
            ):
                env_steps_per_s_total = float(steps_batch) / (collect_s + prev_learn_s)

            t_learn0 = time.perf_counter()
            bootstrap_vals: list[float] = []
            for c in chunks:
                if c["tail_done"]:
                    bootstrap_vals.append(0.0)
                else:
                    bo = {
                        "spatial": torch.as_tensor(
                            c["bootstrap_spatial"][None, ...],
                            device=device,
                            dtype=torch.float32,
                        ),
                        "scalars": torch.as_tensor(
                            c["bootstrap_scalars"][None, ...],
                            device=device,
                            dtype=torch.float32,
                        ),
                        "candidate_features": torch.as_tensor(
                            c["bootstrap_candidate_features"][None, ...],
                            device=device,
                            dtype=torch.float32,
                        ),
                        "candidate_mask": torch.as_tensor(
                            c["bootstrap_candidate_mask"][None, ...],
                            device=device,
                            dtype=torch.float32,
                        ),
                    }
                    with torch.no_grad():
                        bv = pol.predict_values(bo)
                    bootstrap_vals.append(float(bv.view(-1)[0].item()))

            bootstrap_value = torch.as_tensor(bootstrap_vals, device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
            )
            with amp_ctx:
                values_f, logp_pi, entropy = _evaluate_actions_microbatched(
                    pol, flat_obs, flat_actions, flat_mask, learn_chunk
                )
                # Keep (T, B); do not squeeze the B=1 dim or V-trace shapes disagree with logp_pi_tb.
                values_tb = values_f.view(t, b)
                logp_pi_tb = logp_pi.view(t, b)
                entropy_tb = entropy.view(t, b) if entropy is not None else None

                log_rhos = (logp_pi_tb - mu_logp_t).clamp(min=rho_floor, max=20.0)
                vt = from_importance_weights(
                    log_rhos=log_rhos,
                    discounts=discounts,
                    rewards=rewards_t,
                    values=values_tb,
                    bootstrap_value=bootstrap_value,
                    clip_rho_threshold=clip_rho,
                    clip_pg_rho_threshold=clip_pg,
                )
                pg_adv = vt.pg_advantages
                vs = vt.vs

                pi_loss = -(logp_pi_tb * pg_adv).mean()
                v_loss = 0.5 * ((values_tb - vs) ** 2).mean()
                if entropy_tb is not None:
                    ent_bonus = entropy_tb.mean()
                else:
                    ent_bonus = torch.zeros((), device=device)
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent_bonus

            # Log-ratios: Schulman surrogate on symmetric ±KL cap (dash-friendly), V-trace
            # log-clip analogue, uncapped surrogate, and clip hit rates (staleness).
            with torch.no_grad():
                (
                    approx_kl_f,
                    approx_kl_vtrace_log_f,
                    approx_kl_uncapped_f,
                    log_ratio_mean_f,
                    log_rho_frac_at_hi_f,
                    log_rho_frac_at_lo_f,
                ) = _nn_train_log_ratio_diagnostics(
                    logp_pi_tb - mu_logp_t,
                    rho_floor=float(rho_floor),
                )

            pg_d = pg_adv.detach().flatten()
            advantage_mean_f = float(pg_d.mean().cpu())
            advantage_std_f = float(pg_d.std(unbiased=False).cpu())
            vt_d = vs.detach().flatten()
            return_mean_f = float(vt_d.mean().cpu())
            vals_np = values_tb.detach().cpu().numpy().reshape(-1)
            tgt_np = vt_d.cpu().numpy()
            ev_try = float(np.asarray(sb3_explained_variance(vals_np, tgt_np)).reshape(-1)[0])
            explained_var_f = ev_try if math.isfinite(ev_try) else None

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm_t = nn.utils.clip_grad_norm_(pol.parameters(), max_grad_norm)
            grad_norm_f = float(grad_norm_t.detach().cpu().item())
            scaler.step(opt)
            scaler.update()

            learn_s = max(0.0, time.perf_counter() - t_learn0)
            prev_learn_s = learn_s

            learner_updates += 1
            steps_done += t * b
            policy_version += 1

            tot_l = float(loss.detach().cpu())
            pi_l = float(pi_loss.detach().cpu())
            v_l = float(v_loss.detach().cpu())
            ent_m = float(ent_bonus.detach().cpu())
            # entropy_loss aligns with Stable-Baselines3 PPO/A2C: negative mean entropy
            # (`loss += ent_coef * entropy_loss`).
            entropy_loss_sb3 = -ent_m

            nn_row = {
                "schema_version": "1.1",
                "training_backend": "async",
                "learner_update": int(learner_updates),
                "total_timesteps": int(steps_done),
                "total_loss": tot_l,
                "policy_loss": pi_l,
                "value_loss": v_l,
                "entropy_mean": ent_m,
                "entropy_loss": entropy_loss_sb3,
                "entropy_coef": float(ent_coef),
                # Schulman surrogate on symmetric ±KL cap (SB3-ish scale via ``AWBW_ASYNC_NN_KL_DIAG_ABS``).
                "approx_kl": approx_kl_f,
                # Schulman surrogate on IMPALA/V-trace pre-exp clamp [rho_floor, +20].
                "approx_kl_vtrace_log": approx_kl_vtrace_log_f,
                # Raw async staleness (omit if non-finite when exp overflows).
                "approx_kl_uncapped": approx_kl_uncapped_f,
                "log_ratio_mean": log_ratio_mean_f,
                "log_rho_frac_at_hi": log_rho_frac_at_hi_f,
                "log_rho_frac_at_lo": log_rho_frac_at_lo_f,
                "explained_variance": explained_var_f,
                "grad_norm": grad_norm_f,
                "advantage_mean": advantage_mean_f,
                "advantage_std": advantage_std_f,
                "return_mean": return_mean_f,
                "machine_id": os.environ.get("AWBW_MACHINE_ID"),
            }
            nn_row = {k: v for k, v in nn_row.items() if v is not None}
            try:
                _append_nn_train_line(nn_row)
            except Exception:
                pass

            if fps_diag_on:
                if diag_initial_rss_mb is None:
                    diag_initial_rss_mb = _main_proc_rss_mb()
                now_mono = time.perf_counter()
                main_rss = _main_proc_rss_mb()
                base = diag_initial_rss_mb
                delta_mb = main_rss - base if base is not None else 0.0
                sys_pct = _system_ram_used_pct()
                if now_mono - diag_last_worker_scan_mono >= _WORKER_RSS_RESAMPLE_S:
                    diag_cached_sum_worker_rss_mb = _sum_python_children_rss_mb()
                    diag_last_worker_scan_mono = now_mono
                sum_worker = diag_cached_sum_worker_rss_mb
                diag_rollout_seq += 1
                t_elapsed = now_mono - diag_perf_t0
                json_row = {
                    "schema_version": "1.0",
                    "training_backend": "async",
                    "iteration": int(diag_rollout_seq),
                    "total_timesteps": int(steps_done),
                    "time_elapsed_s": float(t_elapsed),
                    "env_collect_s": float(collect_s),
                    "ppo_update_s": ppo_update_s_out,
                    "env_steps_per_s_collect": float(env_steps_per_s_collect),
                    "env_steps_per_s_total": float(env_steps_per_s_total),
                    "main_proc_rss_mb": float(main_rss),
                    "main_proc_rss_delta_mb": float(delta_mb),
                    "sum_worker_rss_mb": float(sum_worker),
                    "system_ram_used_pct": float(sys_pct),
                    "worker_step_time_p99_max_s": 0.0,
                    "worker_step_time_p99_min_s": 0.0,
                    "n_envs": int(n_actors),
                    "machine_id": os.environ.get("AWBW_MACHINE_ID"),
                    "learner_update": int(learner_updates),
                    "train_loss": tot_l,
                    "train_policy_loss": pi_l,
                    "train_value_loss": v_l,
                    "train_entropy_mean": ent_m,
                }
                try:
                    _append_fps_diag_line(json_row)
                except Exception:
                    pass
                print(
                    "[fps_diag] "
                    f"iter={diag_rollout_seq} "
                    f"env_steps_per_s_total={env_steps_per_s_total:.1f} "
                    f"env_steps_per_s_collect={env_steps_per_s_collect:.1f} "
                    f"env_collect_s={collect_s:.3f} "
                    f"ppo_update_s={ppo_update_s_out if ppo_update_s_out is not None else '-'} "
                    f"steps_done={steps_done:,} "
                    f"backend=async "
                    f"loss={tot_l:.4f} pi={pi_l:.4f} v={v_l:.4f} ent={ent_m:.4f}",
                    flush=True,
                )

            if learner_updates % int(trainer.async_weight_save_every) == 0:
                _atomic_torch_save(
                    {"version": policy_version, "policy": pol.state_dict()},
                    weight_path,
                )

            if next_save_at is not None and steps_done >= next_save_at:
                ckpt_stem = new_checkpoint_stem_utc()
                trainer._save_checkpoint_with_publish(  # noqa: SLF001
                    model,
                    ckpt_stem,
                    also_publish_as_latest=bool(
                        getattr(trainer, "publish_latest_each_save", True)
                    ),
                )
                print(f"[async_impala] Saved {ckpt_stem}.zip (env_steps~{steps_done:,})")
                while next_save_at is not None and steps_done >= next_save_at:
                    next_save_at += int(trainer.save_every)

            elapsed = time.time() - t_start
            rate = steps_done / elapsed if elapsed > 0 else 0.0
            trainer._write_trainer_status(steps_done=steps_done, rate=rate)  # noqa: SLF001

            if cap is not None and steps_done >= cap:
                break

            if learner_updates % 10 == 0:
                print(
                    f"[nn_train] backend=async upd={learner_updates} env_steps~{steps_done:,} "
                    f"loss={tot_l:.4f} pi={pi_l:.4f} v={v_l:.4f} ent={ent_m:.4f} "
                    f"~{rate:,.0f} env-steps/s",
                    flush=True,
                )

            if trainer.checkpoint_curate and learner_updates % 50 == 0:
                summary = prune_checkpoint_zip_curated(
                    trainer.checkpoint_dir,
                    k_newest=trainer.curator_k_newest,
                    m_top_winrate=trainer.curator_m_top_winrate,
                    d_diversity=trainer.curator_d_diversity,
                    verdicts_root=trainer.verdicts_root,
                    min_age_minutes=trainer.curator_min_age_minutes,
                    dry_run=False,
                )
                if summary["removed"]:
                    print(
                        f"[async_impala] curated pool: removed {len(summary['removed'])}"
                    )
            elif trainer.checkpoint_zip_cap > 0 and learner_updates % 50 == 0:
                pruned = prune_checkpoint_zip_snapshots(
                    trainer.checkpoint_dir, trainer.checkpoint_zip_cap
                )
                if pruned:
                    print(f"[async_impala] pruned {pruned} old checkpoint zips")

            tail = trainer.checkpoint_pool_size if trainer.checkpoint_pool_size > 0 else None
            all_ck = sorted_checkpoint_zip_paths(trainer.checkpoint_dir)
            trainer.checkpoints = all_ck[-tail:] if tail else all_ck

    except KeyboardInterrupt:
        print("\n[async_impala] KeyboardInterrupt - saving latest...")
        trainer._save_checkpoint_with_publish(model, "latest", also_publish_as_latest=False)  # noqa: SLF001
    else:
        print("\n[async_impala] Saving latest.zip after completed run...")
        trainer._save_checkpoint_with_publish(model, "latest", also_publish_as_latest=False)  # noqa: SLF001
    finally:
        stop_event.set()
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        if gpu_pool_manager is not None:
            try:
                gpu_pool_manager.shutdown()
            except Exception:
                pass
        _atomic_torch_save(
            {"version": policy_version, "policy": pol.state_dict()},
            weight_path,
        )
        if trainer._publisher is not None:  # noqa: SLF001
            drained = trainer._publisher.drain(timeout_s=trainer.publisher_drain_timeout_s)  # noqa: SLF001
            print(f"[async_impala] publisher drained {drained}")
            trainer._publisher.close()  # noqa: SLF001

    total_elapsed = time.time() - t_start
    print(
        f"\n[async_impala] Done. {steps_done:,} env steps in {total_elapsed/60:.1f}min "
        f"({learner_updates} learner updates)"
    )
