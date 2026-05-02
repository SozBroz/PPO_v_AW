"""Rolling CHUNK-game stats from logs/game_log.jsonl — **latest training session only**.

Sessions split when ``game_id`` drops between successive appended lines (trainer restart /
new SQLite counter — see ``rl/env.py`` ``_append_game_log_line``).

Dedupe: within the chosen session, **last JSON line wins per game_id** (same merge rule as before).

Cumulative P0 win rate: ``hist_p0_win_rate`` (session prefix through end of each chunk) and
``hist_p0_win_rate_decided_only`` (same prefix, decided games only) — printed as ``hist_p0`` /
``hist_dec`` and written to CSV.

Neutral-property rollup: mean ``property_pressure_end.neutral_income_properties`` (neutral
**income** tiles at episode end, per ``AWBWEnv._property_pressure_snapshot``), averaged only
for games with ``turns >= MIN_TURNS_NEUTRAL_INCOME_METRIC`` (default 15).

Milestone snapshots: ``neutral_income_remaining_by_day_{7,9,11,13,15}`` from game logs (first
time ``state.turn`` hits each day in ``GAME_LOG_NEUTRAL_INCOME_SNAPSHOT_DAYS``). Means include
only games with ``turns >= day`` *and* a non-null logged column.

Async IMPALA dual-gradient: ``async_rollout_mode`` ``mirror`` vs ``hist`` — rolling per-chunk counts
and mirror P0 win rate (chunk rows only).

**CO matchups:** full-scope aggregates by ordered engine pair ``p0_co_id`` vs ``p1_co_id`` (names fallback
when ids missing): P0 win rate and learner win rate (via ``agent_plays`` / ``learner_seat``). Printed at the
end of verbose output (top ``AWBW_CO_MATCHUP_PRINT_TOP`` pairs by volume) and written to
``logs/co_matchup_session_totals.csv`` (one row per pair, not rolled into chunk CSV).

``logs/nn_train.jsonl``: latest learner session (reset when ``learner_update`` or
``rollout_iteration`` drops). Rows are split into **the same number of bins as game chunks**
(equal-count contiguous learner slices) so rolling NN stats align by chunk index with
``game_chunk_rollups`` (not timestep-aligned).

Async IMPALA (``training_backend`` ``async``, schema ≥1.1): ``entropy_loss`` matches Stable-Baselines3 sign
(`-entropy_mean`; same sign convention as SB3 ``train/entropy_loss``). ``approx_kl`` applies the Schulman surrogate
after **symmetric** log-ratio clamp ``AWBW_ASYNC_NN_KL_DIAG_ABS`` wide (default half-width ``2.0``, PPO-ish scale).
``approx_kl_vtrace_log`` repeats the surrogate on IMPALA/V-trace pre-exp ``[rho_floor, +20]`` bounds;
``approx_kl_uncapped`` plus ``log_ratio_mean`` / ``log_rho_frac_at_*`` remain staleness gauges.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from statistics import correlation, mean, median
from typing import Any

REPO = Path(__file__).resolve().parent.parent
GAME_LOG = REPO / "logs" / "game_log.jsonl"
NN_TRAIN_LOG = REPO / "logs" / "nn_train.jsonl"
CSV_OUT = REPO / "logs" / "game_chunk_rollups.csv"
CO_MATCHUP_CSV_OUT = REPO / "logs" / "co_matchup_session_totals.csv"
CHUNK = 50
# Mean ``property_pressure_end.neutral_income_properties`` uses only rows with
# ``turns`` >= this threshold (short games omitted — snapshot often not comparable).
MIN_TURNS_NEUTRAL_INCOME_METRIC = 15

# Must match ``GAME_LOG_NEUTRAL_INCOME_SNAPSHOT_DAYS`` in ``rl/env.py``.
NEUTRAL_INCOME_SNAPSHOT_DAYS: tuple[int, ...] = (7, 9, 11, 13, 15)

# Means over slices of nn_train.jsonl (async IMPALA + sync SB3 where keys exist).
NN_TRAIN_METRIC_KEYS: tuple[str, ...] = (
    "total_loss",
    "policy_loss",
    "value_loss",
    "entropy_mean",
    "entropy_loss",
    "entropy_coef",
    "approx_kl",
    "approx_kl_vtrace_log",
    "approx_kl_uncapped",
    "log_ratio_mean",
    "log_rho_frac_at_hi",
    "log_rho_frac_at_lo",
    "explained_variance",
    "grad_norm",
    "advantage_mean",
    "advantage_std",
    "return_mean",
)


def nn_train_progress_key(record: dict) -> int | None:
    """Monotonic step index within a training session (async or sync)."""
    lu = record.get("learner_update")
    if lu is not None:
        try:
            return int(lu)
        except (TypeError, ValueError):
            pass
    ri = record.get("rollout_iteration")
    if ri is not None:
        try:
            return int(ri)
        except (TypeError, ValueError):
            pass
    return None


def _finite_scalar(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def split_nn_train_sessions(records: list[dict]) -> list[list[dict]]:
    sessions: list[list[dict]] = []
    cur: list[dict] = []
    prev_k: int | None = None
    for r in records:
        k = nn_train_progress_key(r)
        if prev_k is not None and k is not None and k < prev_k:
            if cur:
                sessions.append(cur)
            cur = []
        cur.append(r)
        if k is not None:
            prev_k = k
    if cur:
        sessions.append(cur)
    return sessions


def load_nn_train_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if nn_train_progress_key(r) is None:
                continue
            rows.append(r)
    return rows


def aggregate_nn_train_slice(rows: list[dict]) -> dict[str, float | int | None]:
    pk = [nn_train_progress_key(r) for r in rows]
    pk_clean = [x for x in pk if x is not None]
    out: dict[str, float | int | None] = {
        "nn_train_chunk_rows": len(rows),
        "nn_train_progress_lo": min(pk_clean) if pk_clean else None,
        "nn_train_progress_hi": max(pk_clean) if pk_clean else None,
    }
    for mk in NN_TRAIN_METRIC_KEYS:
        vals = []
        for r in rows:
            if mk not in r:
                continue
            if _finite_scalar(r[mk]):
                vals.append(float(r[mk]))
        out[f"nn_train_{mk}"] = mean(vals) if vals else None
        out[f"nn_train_{mk}_obs_n"] = len(vals)
    return out


def neutral_income_snapshot_by_day(record: dict, day: int) -> int | None:
    """Logged neutral income count at first ``turn == day`` snapshot."""
    key = f"neutral_income_remaining_by_day_{day}"
    v = record.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def neutral_income_properties_at_end(record: dict) -> int | None:
    """Episode-end neutral *income* property count from ``property_pressure_end``."""
    pp = record.get("property_pressure_end")
    if not isinstance(pp, dict):
        return None
    v = pp.get("neutral_income_properties")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _session_co_bucket_key(record: dict) -> tuple[str, tuple[int | str, int | str]]:
    """Prefer ``(p0_co_id, p1_co_id)``; else keyed by logged CO names (ordered P0→P1)."""
    ia, ib = record.get("p0_co_id"), record.get("p1_co_id")
    if ia is not None and ib is not None:
        try:
            return "co_id_pair", (int(ia), int(ib))
        except (TypeError, ValueError):
            pass
    return "co_name_pair", (
        str(record.get("p0_co") or "?"),
        str(record.get("p1_co") or "?"),
    )


def session_co_matchup_agg(rows: list[dict]) -> list[dict[str, Any]]:
    """
    Ordered **engine-seat** matchup counts over the full scoped window (chunk rollups untouched).

    ``p0_win_rate`` mirrors per-chunk accounting: numerator is ``winner == 0`` only;
    unresolved games shrink the rate implicitly. Learner wins use ``learner_seat`` else
    ``agent_plays`` vs ``winner`` among decided legs / decided-known legs as documented below.
    """
    buckets: dict[tuple[str, tuple[int | str, int | str]], dict[str, Any]] = {}

    for r in rows:
        key = _session_co_bucket_key(r)
        b = buckets.get(key)
        if b is None:
            cid0 = cid1 = None
            if key[0] == "co_id_pair":
                cid0 = int(key[1][0])
                cid1 = int(key[1][1])
            b = {
                "pair_kind": key[0],
                "p0_co_id": cid0,
                "p1_co_id": cid1,
                "p0_co_name": str(r.get("p0_co") or ""),
                "p1_co_name": str(r.get("p1_co") or ""),
                "n_games": 0,
                "p0_win_count": 0,
                "n_decided": 0,
                "p0_wins_among_decided": 0,
                "learner_win_count_all": 0,
                "n_decided_learner_seat_known": 0,
                "learner_wins_among_decided_seat_known": 0,
            }
            buckets[key] = b
        b["n_games"] = int(b["n_games"]) + 1
        w_raw = r.get("winner")

        learner_seat: int | None
        learner_seat_raw = r.get("learner_seat")
        if learner_seat_raw is None:
            learner_seat_raw = r.get("agent_plays")
        try:
            learner_seat = int(learner_seat_raw) if learner_seat_raw is not None else None
        except (TypeError, ValueError):
            learner_seat = None

        if w_raw == 0:
            b["p0_win_count"] = int(b["p0_win_count"]) + 1

        try:
            w = int(w_raw) if w_raw is not None else None
        except (TypeError, ValueError):
            w = None

        decided = w is not None and w in (0, 1)

        if decided:
            assert w is not None
            b["n_decided"] = int(b["n_decided"]) + 1
            if w == 0:
                b["p0_wins_among_decided"] = int(b["p0_wins_among_decided"]) + 1
            if learner_seat is not None and learner_seat in (0, 1):
                b["n_decided_learner_seat_known"] = int(b["n_decided_learner_seat_known"]) + 1
                if learner_seat == w:
                    b["learner_wins_among_decided_seat_known"] = (
                        int(b["learner_wins_among_decided_seat_known"]) + 1
                    )
                    b["learner_win_count_all"] = int(b["learner_win_count_all"]) + 1

    out_list: list[dict[str, Any]] = []
    for acc in buckets.values():
        ng = int(acc["n_games"])
        nd = int(acc["n_decided"])
        nlsk = int(acc["n_decided_learner_seat_known"])
        p0_wr = float(acc["p0_win_count"]) / float(ng) if ng else None
        p0_wd = float(acc["p0_wins_among_decided"]) / float(nd) if nd else None
        lwr = float(acc["learner_win_count_all"]) / float(ng) if ng else None
        lwrd = (
            float(acc["learner_wins_among_decided_seat_known"]) / float(nlsk)
            if nlsk
            else None
        )
        out_list.append(
            {
                "pair_kind": acc["pair_kind"],
                "p0_co_id": acc["p0_co_id"],
                "p1_co_id": acc["p1_co_id"],
                "p0_co_name": acc["p0_co_name"],
                "p1_co_name": acc["p1_co_name"],
                "n_games": ng,
                "n_decided": nd,
                "n_unresolved": ng - nd,
                "n_decided_learner_seat_known": nlsk,
                "p0_win_rate": p0_wr,
                "p0_win_rate_decided_only": p0_wd,
                "learner_win_rate": lwr,
                "learner_win_rate_decided_seat_known": lwrd,
            }
        )
    # Sort for stable readers: biggest sample first, tie-break IDs / names.
    out_list.sort(
        key=lambda d: (
            -int(d["n_games"]),
            str(d["pair_kind"]),
            str(d.get("p0_co_id")),
            str(d.get("p1_co_id")),
            str(d["p0_co_name"]),
            str(d["p1_co_name"]),
        ),
    )
    return out_list


def env_int_positive(raw: str | None, default: int) -> int:
    if not raw or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return default
    return max(v, 1)


def format_pct(x: float | None) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def get_last_game_id(filename: str | Path) -> int:
    """``game_id`` from the last parsable JSON line near EOF (global counter at file end).

    Does not load the whole file into memory.
    """
    path = Path(filename)
    block_size = 131072
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            buf = b""
            pos = size
            while pos > 0:
                step = min(block_size, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
                lines = buf.split(b"\n")
                buf = lines[0]
                for raw in reversed(lines[1:]):
                    piece = raw.strip()
                    if not piece:
                        continue
                    try:
                        text = piece.decode("utf-8")
                        data = json.loads(text)
                        gid = data.get("game_id")
                        if gid is not None:
                            return int(gid)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                        continue
        raise ValueError("No valid game_id found scanning tail")
    except FileNotFoundError as e:
        raise FileNotFoundError(filename) from e


def _split_sessions_file_order(records: list[dict]) -> list[list[dict]]:
    sessions: list[list[dict]] = []
    cur: list[dict] = []
    prev_gid: int | None = None
    for r in records:
        gid = int(r["game_id"])
        if prev_gid is not None and gid < prev_gid:
            if cur:
                sessions.append(cur)
            cur = []
        cur.append(r)
        prev_gid = gid
    if cur:
        sessions.append(cur)
    return sessions


def load_records(path: Path) -> tuple[list[dict], int]:
    """Parse JSONL (skip blanks); return records and EOF ``game_id`` from last line."""
    rows: list[dict] = []
    eof_gid = -1
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = r.get("game_id")
            if gid is None:
                continue
            eof_gid = int(gid)
            rows.append(r)
    return rows, eof_gid


def main() -> None:
    verbose = "--csv-only" not in sys.argv

    raw_records, eof_game_id = load_records(GAME_LOG)
    if not raw_records:
        print("No rows.")
        return

    all_sessions = os.environ.get("AWBW_CHUNK_ANALYSIS_ALL_SESSIONS", "").strip() in (
        "1",
        "true",
        "yes",
    )

    if all_sessions:
        chosen = raw_records
        session_idx = None
        session_boundary_reason = "AWBW_CHUNK_ANALYSIS_ALL_SESSIONS set — entire log"
    else:
        sessions = _split_sessions_file_order(raw_records)
        chosen = sessions[-1]
        session_idx = len(sessions) - 1
        session_boundary_reason = (
            f"latest session only (segment {session_idx + 1}/{len(sessions)}, "
            "split on game_id decrease)"
        )

    last_by_gid: dict[int, dict] = {}
    for r in chosen:
        gid = int(r["game_id"])
        last_by_gid[gid] = r

    rows_sorted = [last_by_gid[k] for k in sorted(last_by_gid.keys())]
    n = len(rows_sorted)
    sess_gid_lo = int(rows_sorted[0]["game_id"]) if rows_sorted else 0
    sess_gid_hi = int(rows_sorted[-1]["game_id"]) if rows_sorted else 0

    def wr(chunk: list[dict]) -> dict:
        wins = sum(1 for r in chunk if r.get("winner") == 0)
        losses = sum(1 for r in chunk if r.get("winner") == 1)
        decided = [r for r in chunk if r.get("winner") in (0, 1)]
        decided_wr = (
            sum(1 for r in decided if r["winner"] == 0) / len(decided) if decided else None
        )
        unresolved = sum(1 for r in chunk if r.get("winner") not in (0, 1))
        trunc = sum(1 for r in chunk if r.get("truncated"))
        term = sum(1 for r in chunk if r.get("terminated"))
        agent = sum(1 for r in chunk if int(r.get("agent_plays", -1)) == 0)
        turns = [
            int(v)
            for r in chunk
            if (v := r.get("days", r.get("turns"))) is not None
        ]
        actions = [int(r["n_actions"]) for r in chunk if r.get("n_actions") is not None]
        opp_ckpt = sum(
            1 for r in chunk if str(r.get("opponent_type", "")).startswith("checkpoint")
        )
        opp_rand = sum(1 for r in chunk if r.get("opponent_type") == "random")
        inv = [int(r.get("invalid_action_count") or 0) for r in chunk]
        cap_p0 = [int(r.get("captures_completed_p0") or 0) for r in chunk]
        cap_p1 = [int(r.get("captures_completed_p1") or 0) for r in chunk]
        terr = [
            float(r["terrain_usage_p0"])
            for r in chunk
            if r.get("terrain_usage_p0") is not None
        ]
        fdeltas = []
        hp0 = []
        hp1 = []
        gold_p0 = []
        gold_p1 = []
        lu_p0 = []
        lu_p1 = []
        for r in chunk:
            fe = r.get("funds_end")
            if isinstance(fe, (list, tuple)) and len(fe) >= 2:
                fdeltas.append(float(fe[0]) - float(fe[1]))
            lh = r.get("losses_hp")
            if isinstance(lh, (list, tuple)) and len(lh) >= 2:
                hp0.append(float(lh[0]))
                hp1.append(float(lh[1]))
            gs = r.get("gold_spent")
            if isinstance(gs, (list, tuple)) and len(gs) >= 2:
                gold_p0.append(float(gs[0]))
                gold_p1.append(float(gs[1]))
            lu = r.get("losses_units")
            if isinstance(lu, (list, tuple)) and len(lu) >= 2:
                lu_p0.append(float(lu[0]))
                lu_p1.append(float(lu[1]))

        neutral_vals_ge15: list[float] = []
        games_turns_ge_min = 0
        for r in chunk:
            turns_i = int(r.get("days") or r.get("turns") or 0)
            if turns_i < MIN_TURNS_NEUTRAL_INCOME_METRIC:
                continue
            games_turns_ge_min += 1
            nv = neutral_income_properties_at_end(r)
            if nv is not None:
                neutral_vals_ge15.append(float(nv))

        out_snap: dict[str, float | int | None] = {}
        for day in NEUTRAL_INCOME_SNAPSHOT_DAYS:
            vals_d: list[float] = []
            for r in chunk:
                if int(r.get("days") or r.get("turns") or 0) < day:
                    continue
                sv = neutral_income_snapshot_by_day(r, day)
                if sv is not None:
                    vals_d.append(float(sv))
            out_snap[f"neutral_income_remaining_by_day_{day}_mean"] = (
                mean(vals_d) if vals_d else None
            )
            out_snap[f"neutral_income_remaining_by_day_{day}_n"] = len(vals_d)

        mirror_rs = [r for r in chunk if r.get("async_rollout_mode") == "mirror"]
        hist_rs = [r for r in chunk if r.get("async_rollout_mode") == "hist"]
        n_mirror = len(mirror_rs)
        n_hist = len(hist_rs)
        n_tagged = n_mirror + n_hist
        n_untagged = len(chunk) - n_tagged

        return {
            "n": len(chunk),
            "p0_win_rate": wins / len(chunk),
            "p0_win_rate_decided_only": decided_wr,
            "p1_win_rate": losses / len(chunk),
            "unresolved_rate": unresolved / len(chunk),
            "truncated_rate": trunc / len(chunk),
            "terminated_rate": term / len(chunk),
            "agent_p0_games": agent,
            "opp_checkpoint_games": opp_ckpt,
            "opp_random_games": opp_rand,
            "turns_mean": mean(turns) if turns else None,
            "turns_median": median(turns) if turns else None,
            "n_actions_mean": mean(actions) if actions else None,
            "invalid_actions_mean": mean(inv) if inv else None,
            "captures_p0_mean": mean(cap_p0) if cap_p0 else None,
            "captures_p1_mean": mean(cap_p1) if cap_p1 else None,
            "terrain_usage_p0_mean": mean(terr) if terr else None,
            "funds_p0_minus_p1_mean": mean(fdeltas) if fdeltas else None,
            "losses_hp_p0_mean": mean(hp0) if hp0 else None,
            "losses_hp_p1_mean": mean(hp1) if hp1 else None,
            "gold_spent_p0_mean": mean(gold_p0) if gold_p0 else None,
            "gold_spent_p1_mean": mean(gold_p1) if gold_p1 else None,
            "losses_units_p0_mean": mean(lu_p0) if lu_p0 else None,
            "losses_units_p1_mean": mean(lu_p1) if lu_p1 else None,
            "games_turns_ge_15_in_chunk": games_turns_ge_min,
            "neutral_income_remaining_mean_turns_ge_15": (
                mean(neutral_vals_ge15) if neutral_vals_ge15 else None
            ),
            "neutral_income_remaining_observations_turns_ge_15": len(neutral_vals_ge15),
            **out_snap,
            "async_rollout_mirror_games_in_chunk": n_mirror,
            "async_rollout_hist_games_in_chunk": n_hist,
            "async_rollout_untagged_games_in_chunk": n_untagged,
            "async_rollout_mirror_pct_of_mirror_hist": (
                float(n_mirror) / float(n_tagged) if n_tagged else None
            ),
            "async_rollout_hist_pct_of_mirror_hist": (
                float(n_hist) / float(n_tagged) if n_tagged else None
            ),
            "p0_win_rate_async_rollout_mirror": (
                sum(1 for r in mirror_rs if r.get("winner") == 0) / n_mirror if n_mirror else None
            ),
        }

    chunks: list[tuple[int, int, dict]] = []
    i = 0
    cum_p0_wins = 0
    cum_n = 0
    cum_decided_n = 0
    cum_p0_wins_decided = 0
    while i < n:
        chunk = rows_sorted[i : i + CHUNK]
        for r in chunk:
            cum_n += 1
            if r.get("winner") == 0:
                cum_p0_wins += 1
            if r.get("winner") in (0, 1):
                cum_decided_n += 1
                if r["winner"] == 0:
                    cum_p0_wins_decided += 1
        lo = int(chunk[0]["game_id"])
        hi = int(chunk[-1]["game_id"])
        s = wr(chunk)
        s["hist_p0_win_rate"] = cum_p0_wins / cum_n if cum_n else 0.0
        s["hist_p0_win_rate_decided_only"] = (
            cum_p0_wins_decided / cum_decided_n if cum_decided_n else None
        )
        chunks.append((lo, hi, s))
        i += CHUNK

    nn_raw_all = load_nn_train_records(NN_TRAIN_LOG)
    if all_sessions:
        nn_chosen = nn_raw_all
        nn_session_note = "full nn_train.jsonl (ALL_SESSIONS)"
    else:
        nn_sessions = split_nn_train_sessions(nn_raw_all)
        nn_chosen = nn_sessions[-1] if nn_sessions else []
        nn_session_note = (
            f"latest nn_train session — last of {len(nn_sessions)} file segment(s)"
            if nn_sessions
            else "no nn_train rows with learner_update / rollout_iteration"
        )

    nn_by_k: dict[int, dict] = {}
    for r in nn_chosen:
        k = nn_train_progress_key(r)
        if k is None:
            continue
        nn_by_k[k] = r
    nn_sorted = [nn_by_k[k] for k in sorted(nn_by_k.keys())]

    nc = len(chunks)
    nn_agg_chunks: list[dict[str, float | int | None]] = []
    if nc > 0:
        M = len(nn_sorted)
        for ci in range(nc):
            a = ci * M // nc
            b = (ci + 1) * M // nc
            nn_agg_chunks.append(aggregate_nn_train_slice(nn_sorted[a:b]))

    nn_csv_cols = ["nn_train_chunk_rows", "nn_train_progress_lo", "nn_train_progress_hi"]
    for mk in NN_TRAIN_METRIC_KEYS:
        nn_csv_cols.append(f"nn_train_{mk}")
        nn_csv_cols.append(f"nn_train_{mk}_obs_n")

    snap_cols: list[str] = []
    for d in NEUTRAL_INCOME_SNAPSHOT_DAYS:
        snap_cols.append(f"neutral_income_remaining_by_day_{d}_mean")
        snap_cols.append(f"neutral_income_remaining_by_day_{d}_n")

    csv_fieldnames = [
        "scope",
        "session_segment_index",
        "session_unique_games",
        "session_gid_lo",
        "session_gid_hi",
        "eof_file_game_id",
        "game_id_lo",
        "game_id_hi",
        "chunk_n_games",
        "p0_win_rate",
        "p0_win_rate_decided_only",
        "hist_p0_win_rate",
        "hist_p0_win_rate_decided_only",
        "p1_win_rate",
        "unresolved_rate",
        "truncated_rate",
        "terminated_rate",
        "agent_p0_games",
        "opp_checkpoint_games",
        "opp_random_games",
        "turns_mean",
        "turns_median",
        "n_actions_mean",
        "invalid_actions_mean",
        "captures_p0_mean",
        "captures_p1_mean",
        "terrain_usage_p0_mean",
        "funds_p0_minus_p1_mean",
        "losses_hp_p0_mean",
        "losses_hp_p1_mean",
        "gold_spent_p0_mean",
        "gold_spent_p1_mean",
        "losses_units_p0_mean",
        "losses_units_p1_mean",
        "games_turns_ge_15_in_chunk",
        "neutral_income_remaining_mean_turns_ge_15",
        "neutral_income_remaining_observations_turns_ge_15",
        *snap_cols,
        "async_rollout_mirror_games_in_chunk",
        "async_rollout_hist_games_in_chunk",
        "async_rollout_untagged_games_in_chunk",
        "async_rollout_mirror_pct_of_mirror_hist",
        "async_rollout_hist_pct_of_mirror_hist",
        "p0_win_rate_async_rollout_mirror",
        *nn_csv_cols,
    ]

    scope_tag = "all_sessions_merged" if all_sessions else "latest_session"
    seg_idx_csv = "" if session_idx is None else session_idx + 1

    co_agg_rows = session_co_matchup_agg(rows_sorted)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        w.writeheader()
        for ci, (lo, hi, s) in enumerate(chunks):
            nn_pack = nn_agg_chunks[ci] if ci < len(nn_agg_chunks) else {}
            row_out = {
                "scope": scope_tag,
                "session_segment_index": seg_idx_csv,
                "session_unique_games": n,
                "session_gid_lo": sess_gid_lo,
                "session_gid_hi": sess_gid_hi,
                "eof_file_game_id": eof_game_id,
                "game_id_lo": lo,
                "game_id_hi": hi,
                "chunk_n_games": s["n"],
                **{k: s[k] for k in s},
                **nn_pack,
            }
            w.writerow({k: row_out.get(k, "") for k in csv_fieldnames})

    co_match_csv_cols: tuple[str, ...] = (
        "scope",
        "session_segment_index",
        "session_unique_games",
        "session_gid_lo",
        "session_gid_hi",
        "eof_file_game_id",
        "pair_kind",
        "p0_co_id",
        "p1_co_id",
        "p0_co_name",
        "p1_co_name",
        "n_games",
        "n_decided",
        "n_unresolved",
        "n_decided_learner_seat_known",
        "p0_win_rate",
        "p0_win_rate_decided_only",
        "learner_win_rate",
        "learner_win_rate_decided_seat_known",
    )
    co_shared = {
        "scope": scope_tag,
        "session_segment_index": seg_idx_csv,
        "session_unique_games": n,
        "session_gid_lo": sess_gid_lo,
        "session_gid_hi": sess_gid_hi,
        "eof_file_game_id": eof_game_id,
    }
    CO_MATCHUP_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CO_MATCHUP_CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        wco = csv.DictWriter(f, fieldnames=list(co_match_csv_cols), extrasaction="ignore")
        wco.writeheader()
        for mr in co_agg_rows:
            csv_line = {**co_shared}
            for k in co_match_csv_cols:
                if k in co_shared:
                    continue
                v = mr.get(k)
                if k in ("p0_co_id", "p1_co_id") and v is None:
                    csv_line[k] = ""
                else:
                    csv_line[k] = "" if v is None else v
            wco.writerow({k: csv_line.get(k, "") for k in co_match_csv_cols})

    if verbose:
        tail_gid = get_last_game_id(GAME_LOG)
        print(f"game_log: {GAME_LOG}")
        print(f"EOF quick peek game_id (tail scan): {tail_gid}")
        print(session_boundary_reason + ".")
        print(
            f"Deduped within scope: {n} unique games (game_id {sess_gid_lo}..{sess_gid_hi}); "
            f"raw lines in scope: {len(chosen)}."
        )
        print(
            f"{NN_TRAIN_LOG.name}: {nn_session_note}; "
            f"deduped learner rows={len(nn_sorted)}"
            + (
                f" (progress {nn_train_progress_key(nn_sorted[0])}.."
                f"{nn_train_progress_key(nn_sorted[-1])})"
                if nn_sorted
                else ""
            )
        )
        print()

        print(f"Rolling ({CHUNK}-game, non-overlapping) chunks: {len(chunks)}")
        print()

        hdr = (
            "games       p0_wr  hist_p0  hist_dec  unresolved  trunc%  turns_avg  acts_avg  inv_avg  "
            "cap0/1_avg   terr_u   delta_funds   ckpt/rand\n"
            "            losses_hp_p0/p1_avg   gold_spent_p0/p1_avg   losses_units_p0/p1_avg (units destroyed)\n"
            "            hist_* = cumulative P0 win rate from session start through this chunk "
            "(hist_dec = decided games only)\n"
            f"            neutral income remaining (end state, turns>={MIN_TURNS_NEUTRAL_INCOME_METRIC} only): "
            "mean count | observed/total_ge_threshold\n"
            "            neutral_income_remaining_by_day_7/9/11/13/15: mean(n) per milestone turn "
            "(games must reach that turn)\n"
            "            async_rollout_mode: mirror vs hist counts; mirror share among tagged; P0 win rate on mirror\n"
            "            nn_train: same #bins as game chunks — mean learner metrics over contiguous nn_train rows "
            "(latest learner_update session). Async: approx_kl is symmetric KL-diag capped; approx_kl_vtrace_log mirrors "
            "IMPALA/V-trace log clamps; *_uncapped / log_ratio_mean / log_rho_frac_* for staleness."
        )
        print(hdr)
        print("-" * 132)
        for ci, (lo, hi, s) in enumerate(chunks):
            hp_hist = s["hist_p0_win_rate"]
            hp_dec = s["hist_p0_win_rate_decided_only"]
            hp_dec_s = f"{hp_dec:5.1%}" if hp_dec is not None else "   — "
            ck = s["opp_checkpoint_games"]
            rd = s["opp_random_games"]
            unf = s["unresolved_rate"]
            tr = s["truncated_rate"]
            tm = s["turns_mean"]
            am = s["n_actions_mean"]
            iv = s["invalid_actions_mean"]
            c0 = s["captures_p0_mean"]
            c1 = s["captures_p1_mean"]
            tu = s["terrain_usage_p0_mean"]
            fd = s["funds_p0_minus_p1_mean"]
            lh0 = s.get("losses_hp_p0_mean")
            lh1 = s.get("losses_hp_p1_mean")
            gs0 = s.get("gold_spent_p0_mean")
            gs1 = s.get("gold_spent_p1_mean")
            lu0 = s.get("losses_units_p0_mean")
            lu1 = s.get("losses_units_p1_mean")
            lh_s = (
                f"{lh0:,.0f}/{lh1:,.0f}"
                if lh0 is not None and lh1 is not None
                else "-/-"
            )
            gs_s = (
                f"{gs0:,.0f}/{gs1:,.0f}"
                if gs0 is not None and gs1 is not None
                else "-/-"
            )
            lu_s = (
                f"{lu0:,.0f}/{lu1:,.0f}"
                if lu0 is not None and lu1 is not None
                else "-/-"
            )
            print(
                f"{lo:4}-{hi:4}  {s['p0_win_rate']:6.1%}  {hp_hist:6.1%}  {hp_dec_s}    {unf:6.1%}    {tr:5.1%}   "
                f"{tm:6.1f}  {am:7.0f}  {iv:4.2f}   "
                f"{c0:4.1f}/{c1:4.1f}   {tu:6.2f}   {fd:9.0f}    {ck}/{rd}"
            )
            print(f"            {lh_s:<28} {gs_s:<28} {lu_s}")
            n15 = int(s.get("games_turns_ge_15_in_chunk") or 0)
            nobs = int(s.get("neutral_income_remaining_observations_turns_ge_15") or 0)
            nm = s.get("neutral_income_remaining_mean_turns_ge_15")
            if nm is not None and n15 > 0:
                neu_s = f"mean={nm:.2f}   ({nobs}/{n15} games with snapshot)"
            elif n15 > 0:
                neu_s = f"mean=n/a   (0/{n15} with property_pressure_end.neutral_income_properties)"
            else:
                neu_s = (
                    f"mean=n/a   (no games with turns>={MIN_TURNS_NEUTRAL_INCOME_METRIC} in chunk)"
                )
            print(f"            {neu_s}")
            parts_hi = []
            parts_lo = []
            for i, day in enumerate(NEUTRAL_INCOME_SNAPSHOT_DAYS):
                m = s.get(f"neutral_income_remaining_by_day_{day}_mean")
                nn = int(s.get(f"neutral_income_remaining_by_day_{day}_n") or 0)
                piece = (
                    f"d{day}={m:.2f}(n={nn})"
                    if m is not None
                    else f"d{day}=n/a(n={nn})"
                )
                (parts_hi if i < 3 else parts_lo).append(piece)
            print(f"            {'  '.join(parts_hi)}")
            print(f"            {'  '.join(parts_lo)}")
            nm = int(s.get("async_rollout_mirror_games_in_chunk") or 0)
            nh = int(s.get("async_rollout_hist_games_in_chunk") or 0)
            nu = int(s.get("async_rollout_untagged_games_in_chunk") or 0)
            pct_m = s.get("async_rollout_mirror_pct_of_mirror_hist")
            wr_m = s.get("p0_win_rate_async_rollout_mirror")
            if nm or nh:
                pm_s = f"{pct_m:.1%}" if pct_m is not None else "-"
                wm_s = f"{wr_m:.1%}" if wr_m is not None else "-"
                print(
                    f"            async: mirror={nm} hist={nh} untagged={nu} | "
                    f"mirror/(mirror+hist)={pm_s} | P0_wr_mirror={wm_s}"
                )
            else:
                print(
                    "            async: no mirror/hist-tagged games "
                    f"(untagged={nu}); mirror P0 win rate n/a"
                )

            nnp = nn_agg_chunks[ci] if ci < len(nn_agg_chunks) else {}
            nr = int(nnp.get("nn_train_chunk_rows") or 0)
            pl = nnp.get("nn_train_progress_lo")
            ph = nnp.get("nn_train_progress_hi")
            if nr > 0:
                bits = [
                    f"rows={nr}",
                    f"progress={pl}..{ph}",
                ]
                for mk in (
                    "total_loss",
                    "policy_loss",
                    "value_loss",
                    "entropy_mean",
                    "entropy_loss",
                    "entropy_coef",
                    "approx_kl",
                    "approx_kl_vtrace_log",
                    "approx_kl_uncapped",
                    "log_ratio_mean",
                    "log_rho_frac_at_hi",
                    "log_rho_frac_at_lo",
                    "explained_variance",
                    "grad_norm",
                    "advantage_mean",
                    "advantage_std",
                    "return_mean",
                ):
                    v = nnp.get(f"nn_train_{mk}")
                    if v is not None:
                        bits.append(f"{mk}={v:.6g}")
                    else:
                        bits.append(f"{mk}=n/a")
                print("            nn_train: " + " ".join(bits))
            else:
                print("            nn_train: no learner rows in this chunk bin")

        top_co_print = env_int_positive(os.environ.get("AWBW_CO_MATCHUP_PRINT_TOP"), 48)
        n_mirror_scope = sum(1 for r in rows_sorted if r.get("async_rollout_mode") == "mirror")
        n_hist_scope = sum(1 for r in rows_sorted if r.get("async_rollout_mode") == "hist")
        print()
        print(
            "=== Session CO matchup (ordered P0 seat vs P1 seat) "
            "— full-scope totals, not rolling ==="
        )
        print(
            "  p0_wr = P0 (Orange Star seat) wins / n games; learner_wr counts learner wins only "
            "on decided episodes, denominator still n games."
        )
        print(
            f"  Top {top_co_print} pairs by n_games (full table → {CO_MATCHUP_CSV_OUT.name}):"
        )
        for mr in co_agg_rows[:top_co_print]:
            id0, id1 = mr.get("p0_co_id"), mr.get("p1_co_id")
            id_txt = (
                f" (ids {id0} vs {id1})"
                if isinstance(id0, int) and isinstance(id1, int)
                else ""
            )
            print(
                f"  n={int(mr['n_games']):4}  "
                f"p0_wr={format_pct(mr.get('p0_win_rate'))}  "
                f"p0_wr_dec={format_pct(mr.get('p0_win_rate_decided_only'))}  "
                f"lr_wr={format_pct(mr.get('learner_win_rate'))}  "
                f"lr_wr_dec(sk)={format_pct(mr.get('learner_win_rate_decided_seat_known'))}  "
                f"|  {mr.get('p0_co_name')} vs {mr.get('p1_co_name')}{id_txt}"
            )
        if len(co_agg_rows) > top_co_print:
            tail = len(co_agg_rows) - top_co_print
            print(f"  … plus {tail} matchup row(s); see CSV for all pairs.")

        print()
        print(
            "=== Async ``async_rollout_mode`` counts (mirror / hist / untagged) "
            f"— deduped scope n={n} ==="
        )
        print(
            f"  mirror={n_mirror_scope}  hist={n_hist_scope}  "
            f"untagged={n - n_mirror_scope - n_hist_scope}"
        )

        if len(chunks) >= 2:
            first = chunks[0][2]
            last = chunks[-1][2]
            print()
            print("=== First chunk vs last chunk (delta = last - first) ===")
            keys = [
                "p0_win_rate",
                "hist_p0_win_rate",
                "hist_p0_win_rate_decided_only",
                "unresolved_rate",
                "truncated_rate",
                "turns_mean",
                "n_actions_mean",
                "invalid_actions_mean",
                "captures_p0_mean",
                "captures_p1_mean",
                "terrain_usage_p0_mean",
                "funds_p0_minus_p1_mean",
                "losses_hp_p0_mean",
                "losses_hp_p1_mean",
                "gold_spent_p0_mean",
                "gold_spent_p1_mean",
                "losses_units_p0_mean",
                "losses_units_p1_mean",
                "neutral_income_remaining_mean_turns_ge_15",
                "games_turns_ge_15_in_chunk",
                "neutral_income_remaining_observations_turns_ge_15",
            ]
            for d in NEUTRAL_INCOME_SNAPSHOT_DAYS:
                keys.append(f"neutral_income_remaining_by_day_{d}_mean")
            keys.extend(
                [
                    "async_rollout_mirror_pct_of_mirror_hist",
                    "async_rollout_hist_pct_of_mirror_hist",
                    "p0_win_rate_async_rollout_mirror",
                    "async_rollout_mirror_games_in_chunk",
                    "async_rollout_hist_games_in_chunk",
                ]
            )
            for mk in NN_TRAIN_METRIC_KEYS:
                keys.append(f"nn_train_{mk}")
            keys.extend(
                [
                    "nn_train_chunk_rows",
                    "nn_train_progress_lo",
                    "nn_train_progress_hi",
                ]
            )
            for k in keys:
                a, b = first.get(k), last.get(k)
                if a is None or b is None:
                    continue
                print(f"  {k:26}  {a:.4g} -> {b:.4g}  (delta {b-a:+.4g})")

        ot: dict[str, int] = {}
        for r in rows_sorted:
            t = r.get("opponent_type") or "?"
            ot[t] = ot.get(t, 0) + 1
        print()
        print("Scope opponent_type counts:", dict(sorted(ot.items(), key=lambda x: -x[1])))

        decided = [r for r in rows_sorted if r.get("winner") in (0, 1)]
        print(f"Decided games (winner 0/1): {len(decided)} / {n}")
        if decided:
            wr_all = sum(1 for r in decided if r["winner"] == 0) / len(decided)
            print(f"  P0 win rate over decided only: {wr_all:.1%}")

        xs = list(range(len(chunks)))
        ys = [c[2]["p0_win_rate"] for c in chunks]
        if len(xs) >= 3:
            r_chunk = correlation(xs, ys)
            print()
            print(
                f"Correlation(chunk_index vs p0_win_rate), non-overlapping 50s: {r_chunk:+.3f} "
                "(+1 strong up-trend, ~0 noise)"
            )

    print(f"Wrote Excel-ready CSV (UTF-8 BOM): {CSV_OUT}")
    print(f"Wrote session CO matchup totals (UTF-8 BOM): {CO_MATCHUP_CSV_OUT}")


if __name__ == "__main__":
    main()
