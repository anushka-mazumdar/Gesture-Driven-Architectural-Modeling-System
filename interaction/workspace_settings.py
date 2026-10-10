"""Python-owned settings exposed to the single-window workspace UI."""

import re


DEFAULT_STROKE_COLOR = (0x66 / 255.0, 0xBF / 255.0, 1.0)


class WorkspaceSettings:
    def __init__(self, snapping, renderer):
        self.snapping = snapping
        self.renderer = renderer
        self.stroke_color = DEFAULT_STROKE_COLOR
        self.snapping.enabled = bool(self.snapping.enabled)
        self.renderer.set_workspace_settings(
            snapping_enabled=self.snapping.enabled,
            stroke_color=self.stroke_color,
        )

    def set_snapping_enabled(self, enabled):
        """Update the existing snap engine gate and publish its UI state."""
        self.snapping.enabled = bool(enabled)
        self.renderer.set_workspace_settings(
            snapping_enabled=self.snapping.enabled
        )
        return self.snapping.enabled

    def set_stroke_color(self, color):
        """Choose the RGB color captured by strokes started from now on."""
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            return None
        self.stroke_color = tuple(
            int(color[index:index + 2], 16) / 255.0
            for index in (1, 3, 5)
        )
        self.renderer.set_workspace_settings(stroke_color=self.stroke_color)
        return self.stroke_color
