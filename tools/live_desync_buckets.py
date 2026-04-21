"""
Stable category ids for live ``desync_audit_amarriner_live`` first-divergence rows.

Used by ``tools/analyze_live_pool_overlap.py`` and category reports.
"""
from __future__ import annotations


def bucket_live_failure(
    msg: str, cls: str, action_kind: str | None = None
) -> str:
    """Map a live first-divergence row to a bucket id."""
    m = msg or ""
    if cls == "loader_error":
        return "loader_error"
    if m.startswith("live Load:") or "could not resolve cargo/transport" in m:
        return "live_load_pairing"
    if "Power without playerID" in m:
        return "power_json_shape"
    if "Supply without nested Move" in m:
        return "supply_json_shape"
    if m.startswith("Join: no path"):
        return "join_synth_no_path"
    if "Repair without Repair dict" in m:
        return "repair_json_shape"
    if "Capt (no path)" in m or "replay unit positions vs AWBW" in m:
        return "capt_position_drift"
    if "oracle_fire" in m or "drift_range_los_or_unmapped_co" in m:
        return "oracle_fire_drift"
    if "Move" in m and "no unit" in m.lower():
        return "move_no_unit"
    if "no repair-eligible" in m.lower() or ("Repair" in m and "ally" in m):
        return "repair_eligibility"
    if cls == "engine_bug":
        return "engine_bug"
    return "other"


# Human-readable titles for reports (bucket id -> line)
BUCKET_TITLE: dict[str, str] = {
    "loader_error": "Loader / JSON shape (e.g. AttackSeam keys)",
    "power_json_shape": "Power — missing playerID in live JSON",
    "live_load_pairing": "Live Load — cargo/transport id resolution",
    "supply_json_shape": "Supply — missing nested Move dict",
    "join_synth_no_path": "Join — synthesized Move, no path to partner",
    "repair_json_shape": "Repair — missing nested Repair dict",
    "capt_position_drift": "Capt — position / reachability drift",
    "oracle_fire_drift": "Fire — oracle_fire / range–LOS–CO drift",
    "move_no_unit": "Move — no unit at expected tile",
    "repair_eligibility": "Repair — eligibility / ally mismatch",
    "engine_bug": "Engine bug (classified)",
    "other": "Other / uncategorized",
}


def overlap_bucket_for_md(bucket_id: str) -> str:
    """Legacy ids used in live_pool_overlap_analysis.md tables."""
    legacy = {
        "loader_error": "not_covered_attackseam_loader",
        "power_json_shape": "not_covered_power_json_shape",
        "live_load_pairing": "not_covered_live_load_pairing",
        "supply_json_shape": "not_covered_supply_shape",
        "join_synth_no_path": "not_covered_join_synth_path",
        "repair_json_shape": "not_covered_repair_json_shape",
        "capt_position_drift": "not_covered_capt_position_drift",
        "oracle_fire_drift": "covered_by_fire_lane",
        "move_no_unit": "covered_by_move_lane",
        "repair_eligibility": "covered_by_repair_lane",
        "engine_bug": "engine_bug",
        "other": "not_covered_other",
    }
    return legacy.get(bucket_id, "not_covered_other")
