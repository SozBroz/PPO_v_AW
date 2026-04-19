"""Flask application factory.

To run the server, use one of these methods from the project root:
    python -m server.app
    OR
    python server/app.py  (with sys.path fix below)
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
    from server.routes.replay import bp as replay_bp
    from server.routes.analysis import bp as analysis_bp
    from server.routes.game import bp as game_bp

    app.register_blueprint(watch_bp)
    app.register_blueprint(replay_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(game_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    # use_reloader=False: the play-mode session dict lives in this process; the
    # Werkzeug reloader spawns a child and restarts it on every code change,
    # wiping in-memory sessions. Use flask run --no-reload for the same effect.
    app.run(debug=True, use_reloader=False, port=5000)
