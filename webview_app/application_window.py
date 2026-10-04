"""Single-window host for the drawing interface and Three.js scene."""


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700


def create_application_window(webview, api, renderer, on_closing):
    """Create and attach the application's one WebView window."""
    from webview_app import INDEX_HTML

    window = webview.create_window(
        "Gesture Modeling - 2D + 3D",
        INDEX_HTML,
        js_api=api,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
    )
    renderer.attach_window(window)
    window.events.closing += on_closing
    return window
