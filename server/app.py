"""Flask application factory.

To run the server from the project root:
    python -m server.app
    python -m server
    scripts\\run_play_server.cmd   (Windows)
    flask run --no-reload --port 5000   (uses .flaskenv when python-dotenv is installed)
"""
import sys
from pathlib import Path
from flask import Flask

# Fix imports when running as script (python server/app.py)
# This ensures 'server' package can be imported correctly
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def create_app():
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    app.config["ROOT"] = ROOT
    app.config["DATA_DIR"] = ROOT / "data"
    app.config["CHECKPOINT_DIR"] = ROOT / "checkpoints"

    from server.routes.watch import bp as watch_bp
    from server.routes.reward_audit import bp as reward_audit_bp
    from server.routes.replay import bp as replay_bp
    from server.routes.analysis import bp as analysis_bp
    from server.routes.game import bp as game_bp
    from server.step_visualization import bp as step_viz_bp
    from server.routes.nn_audit import bp as nn_audit_bp
    from server.routes.analyze import bp as analyze_bp
    from server.routes.your_games import bp as your_games_bp

    app.register_blueprint(watch_bp)
    app.register_blueprint(reward_audit_bp)
    app.register_blueprint(replay_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(game_bp)
    app.register_blueprint(step_viz_bp)
    app.register_blueprint(nn_audit_bp)
    app.register_blueprint(your_games_bp)
    app.register_blueprint(analyze_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    # use_reloader=False: the play-mode session dict lives in this process; the
    # Werkzeug reloader spawns a child and restarts it on every code change,
    # wiping in-memory sessions. Use flask run --no-reload for the same effect.
    app.run(debug=True, use_reloader=False, port=5000)
