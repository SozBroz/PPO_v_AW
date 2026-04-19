"""CO rankings and map analysis API."""
import json
from flask import Blueprint, jsonify, request, current_app

bp = Blueprint("analysis", __name__, url_prefix="/api")


@bp.route("/co_rankings")
def co_rankings():
    """GET /api/co_rankings?map_id=133665&tier=T2"""
    data_dir = current_app.config["DATA_DIR"]
    rankings_path = data_dir / "co_rankings.json"

    if not rankings_path.exists():
        return jsonify({"error": "No rankings yet. Run: python train.py --rank"}), 404

    with open(rankings_path) as f:
        all_rankings = json.load(f)

    map_id = request.args.get("map_id")
    tier = request.args.get("tier")

    if map_id:
        data = all_rankings.get(map_id, {})
        if tier:
            return jsonify(data.get("by_tier", {}).get(tier, {}))
        return jsonify(data)

    return jsonify(all_rankings)


@bp.route("/map_features")
def map_features():
    data_dir = current_app.config["DATA_DIR"]
    path = data_dir / "map_features.json"
    if not path.exists():
        return jsonify({"error": "Run: python train.py --features"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@bp.route("/maps")
def map_list():
    data_dir = current_app.config["DATA_DIR"]
    pool_path = data_dir / "gl_map_pool.json"
    if not pool_path.exists():
        return jsonify({"error": "Map pool not found"}), 404
    with open(pool_path) as f:
        pool = json.load(f)
    return jsonify(pool)
