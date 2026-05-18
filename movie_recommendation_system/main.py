"""Application entry point.

Run the complete project with:
    python main.py
"""

from __future__ import annotations

import threading
import webbrowser

from flask import Flask

from env_config import load_env
from ui import MovieUI

load_env()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ai-lab-movie-recommender"
    MovieUI(app)
    return app


if __name__ == "__main__":
    flask_app = create_app()
    threading.Timer(1.2, lambda: webbrowser.open_new("http://127.0.0.1:5000")).start()
    # Spyder does not work well with Flask's automatic watchdog reloader.
    # use_reloader=False prevents the "SystemExit: 1" traceback in Spyder.
    flask_app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
