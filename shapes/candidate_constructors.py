"""Parameter schemas + procedural constructors for every taxonomy 3D candidate.

Each candidate ID from shapes/shape_taxonomy.json (``candidate_3d_families``)
has a CandidateSpec with:
  * an explicit, bounded parameter schema (validated by ``validate_parameters``),
  * a deterministic constructor (``build_candidate``) returning a CandidateMesh
    that has the same interface as render.primitives.PolygonMesh/RibbonMesh,
    so it serializes through render/mesh_bridge unchanged,
  * a documented sketch derivation (``derive_parameters_from_sketch``) that maps
    the original 2D stroke onto those parameters.

Geometry conventions are in shapes/candidate_geometry.py (+Y up, sketch read
as a plan-view footprint in XZ, bbox-centred). Nothing here selects a
candidate, places an object, or touches UI/gesture/classifier state.
"""

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from shapes import candidate_geometry as geo
from render.snap_anchors import build_snap_anchors
from shapes.stroke_analysis import _convex_hull


MIN_DIMENSION = 0.5
MAX_DIMENSION = 5000.0
MAX_COORDINATE = 10000.0
MAX_PATH_POINTS = 256
MAX_PROFILE_POINTS = 256
_DERIVED_POINT_LIMIT = 128

SOLID_COLOR = (0.55, 1.0, 0.65)   # matches PolygonMesh
SWEEP_COLOR = (0.4, 0.75, 1.0)    # matches RibbonMesh


class CandidateParameterError(ValueError):
    """Unknown candidate, or parameters that fail the candidate's schema."""


class CandidateConstructionError(ValueError):
    """Validated parameters that still could not produce geometry."""


class SketchError(ValueError):
    """A 2D sketch that cannot be measured (too few / non-finite points)."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str                      # float | int | path | polygon
    default: object = None         # None -> required
    minimum: float = None
    maximum: float = None
    description: str = ""

    @property
    def required(self):
        return self.default is None

    def to_dict(self):
        data = {"name": self.name, "type": self.kind, "required": self.required,
                "description": self.description}
        if self.kind in ("float", "int"):
            data.update(default=self.default, minimum=self.minimum, maximum=self.maximum)
        else:
            data.update(min_points=int(self.minimum), max_points=int(self.maximum),
                        coordinate_limit=MAX_COORDINATE)
        return data


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    params: tuple
    construct: object              # (MeshBuilder, params) -> scale tuple or None
    derive: object                 # (SketchMeasurements, source_sides) -> dict
    derivation: str
    check: object = None           # params -> error message or None
    param_names: tuple = field(init=False, default=())

    def __post_init__(self):
        object.__setattr__(self, "param_names", tuple(p.name for p in self.params))

    def to_dict(self):
        return {"id": self.id, "parameters": [p.to_dict() for p in self.params],
                "sketch_derivation": self.derivation}


def _dim(name, default, description, minimum=MIN_DIMENSION, maximum=MAX_DIMENSION):
    return ParamSpec(name, "float", float(default), minimum, maximum, description)


def _int(name, default, minimum, maximum, description):
    return ParamSpec(name, "int", default, minimum, maximum, description)


def _path(name="path"):
    return ParamSpec(name, "path", None, 2, MAX_PATH_POINTS,
                     "Open plan-view path as [[x, z], ...] in scene units")


def _polygon(name="profile"):
    return ParamSpec(name, "polygon", None, 3, MAX_PROFILE_POINTS,
                     "Simple closed plan-view footprint as [[x, z], ...]")


_RADIAL = _int("radial_segments", 24, 3, 64, "Facets around the circumference")
_TUBE_SEGMENTS = _int("radial_segments", 12, 3, 32, "Facets around the tube")
_WIDTH_SEGMENTS = _int("width_segments", 24, 8, 64, "Longitudinal facets")
_HEIGHT_SEGMENTS = _int("height_segments", 16, 4, 48, "Latitudinal facets")
_ARC_SEGMENTS = _int("segments", 32, 4, 128, "Path subdivisions along the arc")
_SWEEP_ANGLE = ParamSpec("sweep_angle", "float", 180.0, 1.0, 359.0,
                         "Arc sweep in degrees")
_SIDES = _int("sides", 8, 3, 64, "Regular polygon side count (from the source n-gon)")


# ---- parameter validation --------------------------------------------------

def _validate_number(spec, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CandidateParameterError(f"{spec.name} must be a number")
    if spec.kind == "int":
        if not isinstance(value, numbers.Integral):
            raise CandidateParameterError(f"{spec.name} must be an integer")
        value = int(value)
    else:
        value = float(value)
        if not math.isfinite(value):
            raise CandidateParameterError(f"{spec.name} must be finite")
    if not spec.minimum <= value <= spec.maximum:
        raise CandidateParameterError(
            f"{spec.name}={value} outside [{spec.minimum}, {spec.maximum}]")
    return value


def _read_points(spec, value):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise CandidateParameterError(f"{spec.name} must be a sequence of [x, z] points")
    if len(value) > spec.maximum + 1:
        raise CandidateParameterError(f"{spec.name} has more than {int(spec.maximum)} points")
    points = []
    for point in value:
        if isinstance(point, (str, bytes)) or not hasattr(point, "__len__") or len(point) != 2:
            raise CandidateParameterError(f"{spec.name} points must be [x, z] pairs")
        pair = []
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(coordinate, numbers.Real):
                raise CandidateParameterError(f"{spec.name} coordinates must be numbers")
            coordinate = float(coordinate)
            if not math.isfinite(coordinate) or abs(coordinate) > MAX_COORDINATE:
                raise CandidateParameterError(
                    f"{spec.name} coordinates must be finite and within ±{MAX_COORDINATE}")
            pair.append(coordinate)
        if not points or math.dist(points[-1], pair) > 1e-6:
            points.append(tuple(pair))
    return points


def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _is_simple(poly):
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(a1, a2, poly[j], poly[(j + 1) % n]):
                return False
    return True


def _validate_path(spec, value):
    points = _read_points(spec, value)
    if len(points) < 2:
        raise CandidateParameterError(f"{spec.name} needs at least 2 distinct points")
    if len(points) > spec.maximum:
        raise CandidateParameterError(f"{spec.name} has more than {int(spec.maximum)} points")
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if length < MIN_DIMENSION:
        raise CandidateParameterError(f"{spec.name} is shorter than {MIN_DIMENSION}")
    return tuple(points)


def _validate_polygon(spec, value):
    points = _read_points(spec, value)
    if len(points) > 1 and math.dist(points[0], points[-1]) <= 1e-6:
        points.pop()
    if not 3 <= len(points) <= spec.maximum:
        raise CandidateParameterError(
            f"{spec.name} needs 3..{int(spec.maximum)} distinct points")
    if abs(geo.signed_area(points)) < MIN_DIMENSION ** 2:
        raise CandidateParameterError(f"{spec.name} encloses no area")
    if not _is_simple(points):
        raise CandidateParameterError(f"{spec.name} must not self-intersect")
    try:
        geo.triangulate(points)
    except geo.GeometryError as exc:
        raise CandidateParameterError(f"{spec.name}: {exc}") from None
    return tuple(points)


_VALIDATORS = {"float": _validate_number, "int": _validate_number,
               "path": _validate_path, "polygon": _validate_polygon}


def get_candidate_spec(candidate_id):
    spec = CANDIDATE_SPECS.get(candidate_id) if isinstance(candidate_id, str) else None
    if spec is None:
        raise CandidateParameterError(f"Unknown 3D candidate {candidate_id!r}")
    return spec


def get_parameter_schema(candidate_id):
    """JSON-serializable parameter schema for a candidate."""
    return get_candidate_spec(candidate_id).to_dict()


def list_constructible_candidates():
    return sorted(CANDIDATE_SPECS)


def validate_parameters(candidate_id, params=None):
    """Return a complete, normalized parameter dict or raise CandidateParameterError.

    Missing optional parameters take their schema defaults; unknown keys,
    wrong types, non-finite values, out-of-range values, malformed paths or
    profiles, and cross-parameter conflicts are all rejected.
    """
    spec = get_candidate_spec(candidate_id)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise CandidateParameterError("parameters must be a dict")
    unknown = set(params) - set(spec.param_names)
    if unknown:
        raise CandidateParameterError(
            f"Unknown parameters for {candidate_id}: {sorted(map(str, unknown))}")

    validated = {}
    for param in spec.params:
        value = params.get(param.name, param.default)
        if value is None:
            raise CandidateParameterError(f"Missing required parameter {param.name!r}")
        validated[param.name] = _VALIDATORS[param.kind](param, value)
    if spec.check is not None:
        problem = spec.check(validated)
        if problem:
            raise CandidateParameterError(f"{candidate_id}: {problem}")
    return validated


# ---- construction ----------------------------------------------------------

class CandidateMesh:
    """A constructed candidate with the PolygonMesh/RibbonMesh object interface."""

    def __init__(self, candidate_id, parameters, vertices, normals, indices, color):
        self.candidate_id = candidate_id
        self.parameters = parameters
        self.vertices = vertices
        self.normals = normals
        self.indices = indices

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected = False
        self.highlighted = False
        self.color = color
        self.kind = "candidate"
        self.snap_anchors = build_snap_anchors(self.vertices, self.indices)

    def get_bounds(self):
        if len(self.vertices) == 0:
            return None
        mins = self.vertices.min(axis=0) + self.position
        maxs = self.vertices.max(axis=0) + self.position
        return {'min': mins, 'max': maxs, 'center': (mins + maxs) / 2}


def build_candidate(candidate_id, params=None):
    """Validate ``params`` and deterministically construct the candidate mesh."""
    validated = validate_parameters(candidate_id, params)
    spec = CANDIDATE_SPECS[candidate_id]
    builder = geo.MeshBuilder()
    try:
        scale = spec.construct(builder, validated) or (1.0, 1.0, 1.0)
        vertices, normals, indices = builder.finish(scale)
    except geo.GeometryError as exc:
        raise CandidateConstructionError(f"{candidate_id}: {exc}") from None
    color = SWEEP_COLOR if candidate_id.startswith("sweep.") else SOLID_COLOR
    return CandidateMesh(candidate_id, validated, vertices, normals, indices, color)


# ---- sketch measurement + derivation ---------------------------------------

@dataclass(frozen=True)
class SketchMeasurements:
    """Plan-view measurements of an original 2D stroke (screen px -> scene units).

    Sketch x maps to X and sketch y maps to Z; points are bbox-centred.
    """
    points: tuple          # centred, consecutive duplicates removed
    width: float           # X extent
    depth: float           # Z extent
    path_length: float
    chord: float           # start -> end distance

    @property
    def mean_size(self):
        return (self.width + self.depth) / 2.0

    @property
    def radius(self):
        return self.mean_size / 2.0

    @property
    def stroke_radius(self):
        # half of ShapeRecommender's open-stroke thickness heuristic
        return max(self.path_length * 0.06, 12.0) / 2.0


def measure_sketch(points):
    if isinstance(points, (str, bytes)) or not hasattr(points, "__len__"):
        raise SketchError("sketch must be a sequence of (x, y) points")
    try:
        pts = np.asarray([(float(p[0]), float(p[1])) for p in points], dtype=float)
    except (TypeError, ValueError, IndexError):
        raise SketchError("sketch points must be numeric (x, y) pairs") from None
    if len(pts) < 2 or not np.all(np.isfinite(pts)):
        raise SketchError("sketch needs at least 2 finite points")
    keep = [0] + [i for i in range(1, len(pts))
                  if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        raise SketchError("sketch needs at least 2 distinct points")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centred = np.clip(pts - (lo + hi) / 2.0, -MAX_COORDINATE, MAX_COORDINATE)
    return SketchMeasurements(
        points=tuple(map(tuple, centred.tolist())),
        width=float(hi[0] - lo[0]), depth=float(hi[1] - lo[1]),
        path_length=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
        chord=float(np.linalg.norm(pts[-1] - pts[0])),
    )


def _decimate(points, limit):
    if len(points) <= limit:
        return list(points)
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


def _sketch_path(sketch):
    return _decimate(list(sketch.points), _DERIVED_POINT_LIMIT)


def _sketch_profile(sketch):
    pts = list(sketch.points)
    if len(pts) > 2 and math.dist(pts[0], pts[-1]) <= 1e-6:
        pts.pop()
    pts = _decimate(pts, _DERIVED_POINT_LIMIT)
    try:
        return tuple(_validate_polygon(_polygon(), pts))
    except CandidateParameterError:
        return tuple(_convex_hull(pts))   # self-crossing air-draw -> hull fallback


def _fit_arc(sketch):
    """(radius, sweep_degrees) of the circle through start, mid-length, end."""
    pts = np.asarray(sketch.points)
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    a, b, c = pts[0], pts[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))], pts[-1]
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-9:
        return MAX_DIMENSION, 1.0
    sq = [p.dot(p) for p in (a, b, c)]
    ux = (sq[0] * (b[1] - c[1]) + sq[1] * (c[1] - a[1]) + sq[2] * (a[1] - b[1])) / d
    uy = (sq[0] * (c[0] - b[0]) + sq[1] * (a[0] - c[0]) + sq[2] * (b[0] - a[0])) / d
    centre = np.array([ux, uy])
    angle = [math.atan2(*(p - centre)[::-1]) for p in (a, b, c)]
    span = (angle[2] - angle[0]) % (2 * math.pi)
    mid = (angle[1] - angle[0]) % (2 * math.pi)
    sweep = span if mid <= span else 2 * math.pi - span
    return float(np.linalg.norm(a - centre)), math.degrees(sweep)


def _clamp_to_schema(spec, values):
    by_name = {p.name: p for p in spec.params}
    clamped = {}
    for name, value in values.items():
        param = by_name[name]
        if param.kind == "float":
            value = min(max(float(value), param.minimum), param.maximum)
        elif param.kind == "int":
            value = int(min(max(round(value), param.minimum), param.maximum))
        clamped[name] = value
    return clamped


def derive_parameters_from_sketch(candidate_id, points, source_sides=None):
    """Derive validated constructor parameters from the original 2D stroke.

    ``points`` are the stroke's (x, y) screen points; ``source_sides`` is the
    polygon side count from recommendation_api (used by candidates that need
    it). Dimensions are clamped into schema bounds, so any measurable sketch
    yields buildable parameters. Raises SketchError for unmeasurable input.
    """
    spec = get_candidate_spec(candidate_id)
    sketch = measure_sketch(points)
    values = spec.derive(sketch, source_sides)
    try:
        return validate_parameters(candidate_id, _clamp_to_schema(spec, values))
    except CandidateParameterError as exc:
        raise SketchError(f"sketch too degenerate for {candidate_id}: {exc}") from None


# ---- per-candidate construction + derivation --------------------------------

def _sides(source_sides):
    valid = (isinstance(source_sides, numbers.Integral) and not isinstance(source_sides, bool)
             and _SIDES.minimum <= source_sides <= _SIDES.maximum)
    return int(source_sides) if valid else _SIDES.default


def _elliptical_torus_check(p):
    if p["minor_radius"] >= min(p["major_radius_x"], p["major_radius_z"]):
        return "minor_radius must be smaller than both major radii"
    return None


def _torus_check(p):
    return None if p["minor_radius"] < p["major_radius"] else \
        "minor_radius must be smaller than major_radius"


def _arc_tube_check(p):
    return None if p["tube_radius"] < p["radius"] else \
        "tube_radius must be smaller than the arc radius"


def _arc_ribbon_check(p):
    return None if p["thickness"] / 2.0 < p["radius"] else \
        "thickness must be less than the arc diameter"


def _arc_tube_derive(s, _):
    radius, sweep = _fit_arc(s)
    return {"radius": radius, "sweep_angle": sweep,
            "tube_radius": min(s.stroke_radius, 0.9 * min(radius, MAX_DIMENSION))}


def _build_arc_tube(b, p):
    geo.sweep(b, geo.arc_path(p["radius"], p["sweep_angle"], p["segments"]),
              geo.circle_section(p["tube_radius"], p["radial_segments"]),
              closed=False)


_ARC_TUBE_PARAMS = (
    _dim("radius", 100.0, "Arc (centre-line) radius"), _SWEEP_ANGLE,
    _dim("tube_radius", 10.0, "Tube cross-section radius"),
    _ARC_SEGMENTS, _TUBE_SEGMENTS)


def _tube_on_path(smoothing_default):
    params = (_path(), _dim("radius", 10.0, "Tube cross-section radius"),
              _int("smoothing", smoothing_default, 0, 3,
                   "Chaikin corner-cutting iterations applied to the path"),
              _TUBE_SEGMENTS)

    def build(b, p):
        geo.sweep(b, geo.chaikin(p["path"], p["smoothing"]),
                  geo.circle_section(p["radius"], p["radial_segments"]),
                  closed=False)

    def derive(s, _):
        return {"path": _sketch_path(s), "radius": s.stroke_radius}
    return params, build, derive


def _beam_on_path(smoothing_default, ribbon=False):
    if ribbon:
        dims = (_dim("width", 60.0, "Vertical ribbon height"),
                _dim("thickness", 4.0, "Horizontal ribbon thickness"))
    else:
        dims = (_dim("width", 20.0, "Horizontal beam width"),
                _dim("thickness", 20.0, "Vertical beam depth"))
    params = (_path(),) + dims + (
        _int("smoothing", smoothing_default, 0, 3,
             "Chaikin corner-cutting iterations applied to the path"),)

    def build(b, p):
        u, v = (p["thickness"], p["width"]) if ribbon else (p["width"], p["thickness"])
        geo.sweep(b, geo.chaikin(p["path"], p["smoothing"]),
                  geo.rect_section(u, v), closed=False)

    def derive(s, _):
        size = 2.0 * s.stroke_radius
        if ribbon:
            return {"path": _sketch_path(s), "width": 3.0 * size,
                    "thickness": max(size * 0.2, MIN_DIMENSION)}
        return {"path": _sketch_path(s), "width": size, "thickness": size}
    return params, build, derive


def _rect(width, depth):
    return [(-width / 2, -depth / 2), (width / 2, -depth / 2),
            (width / 2, depth / 2), (-width / 2, depth / 2)]


def _spec(candidate_id, params, construct, derive, derivation, check=None):
    return CandidateSpec(candidate_id, tuple(params), construct, derive, derivation, check)


def _regular_specs(prefix, sides_fixed=None):
    """Prism / pyramid / bipyramid (+ cone) over a regular footprint."""
    sides_param = () if sides_fixed else (_SIDES,)

    def sides_of(p):
        return sides_fixed or p["sides"]

    def sides_value(source_sides):
        return {} if sides_fixed else {"sides": _sides(source_sides)}

    base = (_dim("radius", 50.0, "Footprint circumradius"),
            _dim("height", 100.0, "Height along +Y"))
    radius_text = ("radius = (sketch width + depth) / 4"
                   + ("" if sides_fixed else "; sides = source n-gon side count"))
    specs = [
        _spec(f"solid.{prefix}_prism", sides_param + base,
              lambda b, p: geo.prism(b, geo.regular_polygon(sides_of(p), p["radius"]), p["height"]),
              lambda s, n: {**sides_value(n), "radius": s.radius, "height": 2.0 * s.radius},
              f"{radius_text}; height = 2 * radius"),
        _spec(f"solid.{prefix}_pyramid", sides_param + base,
              lambda b, p: geo.pyramid(b, geo.regular_polygon(sides_of(p), p["radius"]), p["height"]),
              lambda s, n: {**sides_value(n), "radius": s.radius, "height": 2.0 * s.radius},
              f"{radius_text}; apex height = 2 * radius"),
        _spec(f"solid.{prefix}_bipyramid", sides_param + base,
              lambda b, p: geo.bipyramid(b, geo.regular_polygon(sides_of(p), p["radius"]), p["height"]),
              lambda s, n: {**sides_value(n), "radius": s.radius, "height": 3.0 * s.radius},
              f"{radius_text}; total apex-to-apex height = 3 * radius"),
    ]
    return specs


_SPECS = [
    # round
    _spec("solid.sphere",
          (_dim("radius", 50.0, "Sphere radius"), _WIDTH_SEGMENTS, _HEIGHT_SEGMENTS),
          lambda b, p: geo.sphere(b, p["radius"], p["width_segments"], p["height_segments"]),
          lambda s, _: {"radius": s.radius},
          "radius = (sketch width + depth) / 4"),
    _spec("solid.cylinder",
          (_dim("radius", 50.0, "Base radius"), _dim("height", 100.0, "Height along +Y"), _RADIAL),
          lambda b, p: geo.cylinder(b, p["radius"], p["height"], p["radial_segments"]),
          lambda s, _: ({"radius": s.radius, "height": 2.0 * s.radius}
                        if s.chord < 0.5 * s.path_length
                        else {"radius": s.stroke_radius, "height": s.chord}),
          "closed sketch: radius = (width + depth) / 4, height = 2 * radius; "
          "straight line: radius = stroke thickness / 2, height = line length"),
    _spec("solid.cone",
          (_dim("radius", 50.0, "Base radius"), _dim("height", 100.0, "Apex height along +Y"),
           _RADIAL),
          lambda b, p: geo.cone(b, p["radius"], p["height"], p["radial_segments"]),
          lambda s, _: {"radius": s.radius, "height": 2.0 * s.radius},
          "radius = (sketch width + depth) / 4; height = 2 * radius"),
    _spec("solid.hemisphere",
          (_dim("radius", 50.0, "Dome radius"), _WIDTH_SEGMENTS, _HEIGHT_SEGMENTS),
          lambda b, p: geo.hemisphere(b, p["radius"], p["width_segments"], p["height_segments"]),
          lambda s, _: {"radius": s.radius},
          "radius = (sketch width + depth) / 4 (flat base on the footprint)"),
    _spec("solid.torus",
          (_dim("major_radius", 40.0, "Centre-line radius"),
           _dim("minor_radius", 10.0, "Tube radius"), _TUBE_SEGMENTS,
           _int("tubular_segments", 32, 8, 128, "Facets around the ring")),
          lambda b, p: geo.torus(b, p["major_radius"], p["minor_radius"],
                                 p["radial_segments"], p["tubular_segments"]),
          lambda s, _: {"major_radius": 0.75 * s.radius, "minor_radius": 0.25 * s.radius},
          "outer radius = (width + depth) / 4: major = 0.75 * that, minor = 0.25 * that",
          _torus_check),

    # elliptical
    _spec("solid.ellipsoid",
          (_dim("radius_x", 60.0, "Semi-axis along X"), _dim("radius_y", 40.0, "Semi-axis along Y"),
           _dim("radius_z", 40.0, "Semi-axis along Z"), _WIDTH_SEGMENTS, _HEIGHT_SEGMENTS),
          lambda b, p: (geo.sphere(b, 1.0, p["width_segments"], p["height_segments"])
                        or (p["radius_x"], p["radius_y"], p["radius_z"])),
          lambda s, _: {"radius_x": s.width / 2, "radius_z": s.depth / 2,
                        "radius_y": min(s.width, s.depth) / 2},
          "radius_x = width / 2, radius_z = depth / 2, radius_y = min(radius_x, radius_z)"),
    _spec("solid.elliptical_prism",
          (_dim("radius_x", 60.0, "Semi-axis along X"), _dim("radius_z", 40.0, "Semi-axis along Z"),
           _dim("height", 80.0, "Height along +Y"), _RADIAL),
          lambda b, p: (geo.cylinder(b, 1.0, p["height"], p["radial_segments"])
                        or (p["radius_x"], 1.0, p["radius_z"])),
          lambda s, _: {"radius_x": s.width / 2, "radius_z": s.depth / 2,
                        "height": min(s.width, s.depth)},
          "radius_x = width / 2, radius_z = depth / 2, height = min(width, depth)"),
    _spec("solid.elliptical_cone",
          (_dim("radius_x", 60.0, "Semi-axis along X"), _dim("radius_z", 40.0, "Semi-axis along Z"),
           _dim("height", 80.0, "Apex height along +Y"), _RADIAL),
          lambda b, p: (geo.cone(b, 1.0, p["height"], p["radial_segments"])
                        or (p["radius_x"], 1.0, p["radius_z"])),
          lambda s, _: {"radius_x": s.width / 2, "radius_z": s.depth / 2,
                        "height": min(s.width, s.depth)},
          "radius_x = width / 2, radius_z = depth / 2, height = min(width, depth)"),
    _spec("solid.elliptical_dome",
          (_dim("radius_x", 60.0, "Semi-axis along X"), _dim("radius_z", 40.0, "Semi-axis along Z"),
           _dim("height", 40.0, "Dome height along +Y"), _WIDTH_SEGMENTS, _HEIGHT_SEGMENTS),
          lambda b, p: (geo.hemisphere(b, 1.0, p["width_segments"], p["height_segments"])
                        or (p["radius_x"], p["height"], p["radius_z"])),
          lambda s, _: {"radius_x": s.width / 2, "radius_z": s.depth / 2,
                        "height": min(s.width, s.depth) / 2},
          "radius_x = width / 2, radius_z = depth / 2, height = min(radius_x, radius_z)"),
    _spec("solid.elliptical_torus",
          (_dim("major_radius_x", 60.0, "Centre-line semi-axis along X"),
           _dim("major_radius_z", 40.0, "Centre-line semi-axis along Z"),
           _dim("minor_radius", 8.0, "Circular tube radius"),
           _int("tubular_segments", 48, 8, 128, "Facets around the ring"), _TUBE_SEGMENTS),
          lambda b, p: geo.sweep(b, geo.ellipse_path(p["major_radius_x"], p["major_radius_z"],
                                                     p["tubular_segments"]),
                                 geo.circle_section(p["minor_radius"], p["radial_segments"]),
                                 closed=True),
          lambda s, _: (lambda minor: {"minor_radius": minor,
                                       "major_radius_x": s.width / 2 - minor,
                                       "major_radius_z": s.depth / 2 - minor})(
              0.2 * min(s.width, s.depth) / 2),
          "outer extent matches the sketch: minor = 0.2 * min(width, depth) / 2, "
          "major_x = width / 2 - minor, major_z = depth / 2 - minor",
          _elliptical_torus_check),

    # square
    _spec("solid.cube", (_dim("size", 100.0, "Edge length"),),
          lambda b, p: geo.box(b, p["size"], p["size"], p["size"]),
          lambda s, _: {"size": s.mean_size},
          "size = (sketch width + depth) / 2"),
    _spec("solid.square_pyramid",
          (_dim("base_size", 100.0, "Base edge length"), _dim("height", 100.0, "Apex height")),
          lambda b, p: geo.pyramid(b, _rect(p["base_size"], p["base_size"]), p["height"]),
          lambda s, _: {"base_size": s.mean_size, "height": s.mean_size},
          "base_size = (sketch width + depth) / 2; height = base_size"),
    _spec("solid.square_prism",
          (_dim("base_size", 100.0, "Base edge length"), _dim("height", 200.0, "Height along +Y")),
          lambda b, p: geo.box(b, p["base_size"], p["height"], p["base_size"]),
          lambda s, _: {"base_size": s.mean_size, "height": 2.0 * s.mean_size},
          "base_size = (sketch width + depth) / 2; height = 2 * base_size"),

    # triangular
    _spec("solid.tetrahedron", (_dim("edge", 100.0, "Regular edge length"),),
          lambda b, p: geo.pyramid(b, geo.regular_polygon(3, p["edge"] / math.sqrt(3)),
                                   p["edge"] * math.sqrt(2.0 / 3.0)),
          lambda s, _: {"edge": s.mean_size},
          "edge = (sketch width + depth) / 2 (regular tetrahedron)"),
    _spec("solid.triangular_prism",
          (_dim("base_width", 100.0, "Triangle base width (X)"),
           _dim("base_depth", 90.0, "Triangle depth (Z)"), _dim("height", 100.0, "Height along +Y")),
          lambda b, p: geo.prism(b, geo.isosceles_triangle(p["base_width"], p["base_depth"]),
                                 p["height"]),
          lambda s, _: {"base_width": s.width, "base_depth": s.depth, "height": s.mean_size},
          "base_width = width, base_depth = depth, height = (width + depth) / 2"),
    _spec("solid.triangular_pyramid",
          (_dim("base_width", 100.0, "Triangle base width (X)"),
           _dim("base_depth", 90.0, "Triangle depth (Z)"), _dim("height", 100.0, "Apex height")),
          lambda b, p: geo.pyramid(b, geo.isosceles_triangle(p["base_width"], p["base_depth"]),
                                   p["height"]),
          lambda s, _: {"base_width": s.width, "base_depth": s.depth, "height": s.mean_size},
          "base_width = width, base_depth = depth, height = (width + depth) / 2"),
    _spec("solid.wedge",
          (_dim("width", 100.0, "Extent along X"), _dim("height", 60.0, "Back-face height"),
           _dim("depth", 100.0, "Ramp run along Z")),
          lambda b, p: geo.wedge(b, p["width"], p["height"], p["depth"]),
          lambda s, _: {"width": s.width, "depth": s.depth, "height": min(s.width, s.depth)},
          "width = sketch width, depth = sketch depth, height = min(width, depth)"),

    # rectangular
    _spec("solid.cuboid",
          (_dim("width", 150.0, "Extent along X"), _dim("height", 100.0, "Extent along Y"),
           _dim("depth", 100.0, "Extent along Z")),
          lambda b, p: geo.box(b, p["width"], p["height"], p["depth"]),
          lambda s, _: {"width": s.width, "depth": s.depth, "height": min(s.width, s.depth)},
          "width = sketch width, depth = sketch depth, height = min(width, depth)"),
    _spec("solid.rectangular_prism",
          (_dim("width", 150.0, "Extent along X"), _dim("depth", 100.0, "Extent along Z"),
           _dim("height", 300.0, "Extent along Y")),
          lambda b, p: geo.box(b, p["width"], p["height"], p["depth"]),
          lambda s, _: {"width": s.width, "depth": s.depth, "height": 2.0 * max(s.width, s.depth)},
          "width = sketch width, depth = sketch depth, height = 2 * max(width, depth)"),
    _spec("solid.rectangular_pyramid",
          (_dim("width", 150.0, "Base extent along X"), _dim("depth", 100.0, "Base extent along Z"),
           _dim("height", 120.0, "Apex height")),
          lambda b, p: geo.pyramid(b, _rect(p["width"], p["depth"]), p["height"]),
          lambda s, _: {"width": s.width, "depth": s.depth, "height": s.mean_size},
          "width = sketch width, depth = sketch depth, height = (width + depth) / 2"),

    # profile (irregular polygon)
    _spec("solid.profile_prism", (_polygon(), _dim("height", 60.0, "Extrusion height along +Y")),
          lambda b, p: geo.prism(b, list(p["profile"]), p["height"]),
          lambda s, _: {"profile": _sketch_profile(s), "height": s.mean_size / 2},
          "profile = centred sketch outline (convex hull if self-crossing); "
          "height = (width + depth) / 4"),
    _spec("solid.profile_pyramid", (_polygon(), _dim("height", 100.0, "Apex height")),
          lambda b, p: geo.pyramid(b, list(p["profile"]), p["height"]),
          lambda s, _: {"profile": _sketch_profile(s), "height": s.mean_size},
          "profile = centred sketch outline; apex over its area centroid at "
          "height = (width + depth) / 2"),
    _spec("solid.profile_loft",
          (_polygon(), _dim("height", 100.0, "Loft height"),
           ParamSpec("top_scale", "float", 0.5, 0.05, 1.0,
                     "Top outline scale relative to the base (about its centroid)")),
          lambda b, p: geo.pyramid(b, list(p["profile"]), p["height"], p["top_scale"]),
          lambda s, _: {"profile": _sketch_profile(s), "height": s.mean_size},
          "profile = centred sketch outline; height = (width + depth) / 2; top_scale = 0.5"),

    # linear sweeps
    _spec("sweep.rod",
          (_dim("length", 200.0, "Length along X"), _dim("radius", 10.0, "Rod radius"),
           _TUBE_SEGMENTS),
          lambda b, p: geo.sweep(b, [(-p["length"] / 2, 0.0), (p["length"] / 2, 0.0)],
                                 geo.circle_section(p["radius"], p["radial_segments"]),
                                 closed=False),
          lambda s, _: {"length": s.chord, "radius": s.stroke_radius},
          "length = start-to-end distance; radius = stroke thickness / 2 "
          "(thickness = max(0.06 * stroke length, 12))"),
    _spec("sweep.beam",
          (_dim("length", 200.0, "Length along X"), _dim("width", 20.0, "Horizontal width"),
           _dim("depth", 20.0, "Vertical depth")),
          lambda b, p: geo.sweep(b, [(-p["length"] / 2, 0.0), (p["length"] / 2, 0.0)],
                                 geo.rect_section(p["width"], p["depth"]),
                                 closed=False),
          lambda s, _: {"length": s.chord, "width": 2 * s.stroke_radius,
                        "depth": 2 * s.stroke_radius},
          "length = start-to-end distance; width = depth = stroke thickness"),

    # arcs
    _spec("sweep.arc_tube", _ARC_TUBE_PARAMS, _build_arc_tube, _arc_tube_derive,
          "radius + sweep_angle from the circle through the stroke's start, "
          "mid-length and end points; tube_radius = stroke thickness / 2", _arc_tube_check),
    _spec("sweep.torus_segment", _ARC_TUBE_PARAMS, _build_arc_tube, _arc_tube_derive,
          "same arc fit as sweep.arc_tube (a partial torus)", _arc_tube_check),
    _spec("sweep.arc_ribbon",
          (_dim("radius", 100.0, "Arc (centre-line) radius"), _SWEEP_ANGLE,
           _dim("width", 60.0, "Vertical ribbon height"),
           _dim("thickness", 4.0, "Horizontal ribbon thickness"), _ARC_SEGMENTS),
          lambda b, p: geo.sweep(b, geo.arc_path(p["radius"], p["sweep_angle"], p["segments"]),
                                 geo.rect_section(p["thickness"], p["width"]),
                                 closed=False),
          lambda s, _: (lambda arc: {"radius": arc[0], "sweep_angle": arc[1],
                                     "width": 6.0 * s.stroke_radius,
                                     "thickness": min(0.4 * s.stroke_radius, arc[0])})(
              _fit_arc(s)),
          "arc fit as sweep.arc_tube; width = 3 * stroke thickness, "
          "thickness = 0.2 * stroke thickness", _arc_ribbon_check),
]

_SPECS += _regular_specs("pentagonal", 5)
_SPECS += _regular_specs("hexagonal", 6)
_SPECS += _regular_specs("regular")
_SPECS.append(_spec(
    "solid.regular_cone",
    (_SIDES, _dim("radius", 50.0, "Footprint circumradius"),
     _dim("height", 100.0, "Height along +Y"),
     ParamSpec("top_scale", "float", 0.0, 0.0, 0.95,
               "0 = pointed apex; > 0 truncates into a frustum")),
    lambda b, p: geo.pyramid(b, geo.regular_polygon(p["sides"], p["radius"]),
                             p["height"], p["top_scale"]),
    lambda s, n: {"sides": _sides(n), "radius": s.radius, "height": 2.0 * s.radius},
    "radius = (width + depth) / 4; sides = source n-gon side count; height = 2 * radius"))

for _cid, _smoothing in (("sweep.bent_tube", 1), ("sweep.curved_tube", 3),
                         ("sweep.path_tube", 0)):
    _params, _build, _derive = _tube_on_path(_smoothing)
    _SPECS.append(_spec(_cid, _params, _build, _derive,
                        f"path = centred stroke points (<= {_DERIVED_POINT_LIMIT}); "
                        f"radius = stroke thickness / 2; smoothing = {_smoothing}"))
for _cid, _smoothing, _ribbon in (("sweep.polyline_beam", 0, False),
                                  ("sweep.curved_beam", 3, False),
                                  ("sweep.freeform_beam", 1, False),
                                  ("sweep.swept_ribbon", 1, True)):
    _params, _build, _derive = _beam_on_path(_smoothing, _ribbon)
    _SPECS.append(_spec(_cid, _params, _build, _derive,
                        f"path = centred stroke points (<= {_DERIVED_POINT_LIMIT}); "
                        + ("ribbon height = 3 * stroke thickness, thickness = 0.2 * that; "
                           if _ribbon else "width = depth = stroke thickness; ")
                        + f"smoothing = {_smoothing}"))

CANDIDATE_SPECS = {spec.id: spec for spec in _SPECS}
if len(CANDIDATE_SPECS) != len(_SPECS):
    raise RuntimeError("duplicate candidate constructor IDs")
