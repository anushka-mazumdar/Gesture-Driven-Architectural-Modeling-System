"""
webview_app: hosts the Three.js scene the renderer pushes into.

The package owns the web assets for the pywebview window. main.py
imports WEB_DIR / INDEX_HTML from here and opens the window around the
HTML page, letting the app's Three.js scene receive mesh/hand payloads
over the Python <-> JavaScript bridge (see render/threejs_renderer.py).
"""

import os

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")