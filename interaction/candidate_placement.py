"""Construct a confirmed recommendation from its captured 2D sketch."""

from shapes.candidate_constructors import (
    CandidateConstructionError,
    CandidateParameterError,
    SketchError,
    build_candidate,
    derive_parameters_from_sketch,
)


class CandidatePlacement:
    """Manage pending previews and the active constructed candidate."""

    def __init__(self, renderer, selection=None):
        self.renderer = renderer
        self.selection = selection
        self.record = None
        self.preview_mesh = None
        self.active_object = None
        self._generation = None

    def begin(self, record, preview_mesh):
        """Remember a new original stroke and temporary mesh for its panel."""
        recommendation = getattr(record, "candidates", None)
        if (record is None or preview_mesh is None
                or preview_mesh not in self.renderer.objects
                or recommendation is None or recommendation.status != "ok"
                or not recommendation.candidate_ids):
            self.cancel_pending()
            return False
        if self.preview_mesh is not preview_mesh:
            self.cancel_pending()
        self.record = record
        self.preview_mesh = preview_mesh
        self._generation = self.renderer.candidate_generation
        return True

    def cancel_pending(self):
        """Discard the unconfirmed preview without removing older active objects."""
        if (self.selection is not None
                and self.selection.get_selected() is self.preview_mesh):
            self.selection.deselect_all()
        if (self.preview_mesh is not None
                and self.preview_mesh is not self.active_object):
            self.renderer.remove_object(self.preview_mesh)
        self._forget_pending()

    def _forget_pending(self):
        self.record = None
        self.preview_mesh = None
        self._generation = None

    def cancel(self):
        """Close the current recommendation and discard its unconfirmed preview."""
        self.renderer.set_candidate_recommendation(None)
        self.cancel_pending()

    def reset(self):
        """Remove this flow's pending and active objects and close its panel."""
        self.cancel_pending()
        if self.active_object is not None:
            self.renderer.remove_object(self.active_object)
            if (self.selection is not None
                    and self.selection.get_selected() is self.active_object):
                self.selection.deselect_all()
        self.active_object = None
        self.renderer.set_candidate_recommendation(None)

    def confirm(self, candidate_id):
        """Confirm the current choice once, then replace its preview in place."""
        if self.record is None:
            return False
        if (self.preview_mesh is None
                or self.preview_mesh not in self.renderer.objects
                or not self.renderer.has_candidate_panel
                or self._generation != self.renderer.candidate_generation):
            self.cancel_pending()
            return False

        record = self.record
        preview = self.preview_mesh
        if candidate_id not in record.candidates.candidate_ids:
            return False
        if not self.renderer.confirm_candidate(candidate_id):
            return False

        try:
            source_points = record.raw_points or record.points
            parameters = derive_parameters_from_sketch(
                candidate_id,
                source_points,
                source_sides=record.candidates.source_sides,
            )
            candidate_mesh = build_candidate(candidate_id, parameters)
        except (CandidateConstructionError, CandidateParameterError,
                SketchError, TypeError, ValueError):
            # Keep the current preview if its confirmed candidate cannot build.
            self.active_object = preview
            if self.selection is not None:
                self.selection.deselect_all()
                self.selection.selected_object = preview
                preview.selected = True
            self._forget_pending()
            return True

        for transform_name in ("position", "rotation", "scale"):
            source = getattr(preview, transform_name, None)
            target = getattr(candidate_mesh, transform_name, None)
            if source is not None and target is not None:
                target[...] = source
        candidate_mesh.shape_class = record.shape_class
        candidate_mesh.highlighted = bool(getattr(preview, "highlighted", False))

        if not self.renderer.replace_object(preview, candidate_mesh):
            self.cancel_pending()
            return True
        if self.selection is not None:
            self.selection.deselect_all()
            self.selection.selected_object = candidate_mesh
        candidate_mesh.selected = True

        self.active_object = candidate_mesh
        self.record = None
        self.preview_mesh = None
        self._generation = None
        return True
