from dataclasses import dataclass


@dataclass
class MeshRecommendation:
    """A concrete 3D build recipe decided from a classified stroke.

    'kind' is the mesh family the factory can construct ('polygon' for the
    closed family, 'ribbon' open family). The sizing params centralise the
    heuristics that previously lived inside Shape3DFactory._create_*; the
    factory now only assembles geometry from them.
    """
    kind: str = "polygon"         # "polygon" | "ribbon"
    extrude_depth: float = 60.0   # polygon extrusion thickness
    thickness: float = 20.0       # ribbon tube thickness


# closed-family build
def _closed_extrude(width, height):
    avg_size = (width + height) / 2.0
    return max(avg_size * 0.33, 20.0)


# open-family build
def _open_thickness(length):
    return max(length * 0.06, 12.0)


# shape label -> mesh family  (recommendation mapping)
_CLOSED_FAMILY  = {"circle", "ellipse", "triangle", "square", "rectangle",
                   "pentagon", "hexagon", "polygon"}
_OPEN_FAMILY    = {"line", "curve", "arc", "polyline"}


class ShapeRecommender:
    """Stage: map a classified stroke onto a MeshRecommendation.

    The only stage that decides *how* to build — which mesh family a label
    implies and what size. Everything below (Shape3DFactory) only obeys.
    Named N-gons beyond hexagon arrive as "<n>-gon" (see StrokeClassifier)
    rather than being enumerated here — matched by suffix instead.
    """

    def recommend(self, points, features, shape_class):
        kind = shape_class.kind.lower()
        if kind in _CLOSED_FAMILY or kind.endswith("-gon"):
            return MeshRecommendation(
                kind="polygon",
                extrude_depth=_closed_extrude(features.width, features.height),
                thickness=_open_thickness(features.length),   # unused for polygon
            )
        if kind in _OPEN_FAMILY:
            thickness = _open_thickness(features.length)
            return MeshRecommendation(
                kind="ribbon",
                extrude_depth=thickness,   # square cross-section -> tube-like
                thickness=thickness,
            )
        # unknown label — conservative default to a ribbon
        return MeshRecommendation(kind="ribbon", thickness=20.0, extrude_depth=20.0)