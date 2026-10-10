import math
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from shapes.stroke_analysis import simplify_to_count


@dataclass
class ShapeClass:
    """Result of stroke classification: a 2D shape label + confidence.

    Labels stay geometric (straight_line/polyline/arc/curve/freeform_path and
    circle/ellipse/triangle/square/
    rectangle/pentagon/hexagon/"<n>-gon"/polygon); mapping a label onto a
    concrete 3D build recipe is the recommender's job.
    """
    kind: str = "polygon"
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)


# curvature-based candidates (circle / ellipse) -----------------------------
# Only compete when the contour did NOT simplify down to a small number of
# sharp corners (see _CORNER_SHAPE_MAX below) — a clean quad/pentagon/hexagon
# must never lose to a "roundness" score computed on the same few points.
_CIRCLE_ROUND_MIN   = 0.60   # roundness  >= this to even consider "circle"
_CIRCLE_FILL_MIN    = 0.55   # bbox_fill  >= this to even consider "circle"
_CIRCLE_ASPECT_MAX  = 1.15   # circle needs a near-square bbox; wider -> ellipse
_ELLIPSE_ROUND_MIN  = 0.35
_ELLIPSE_FILL_MIN   = 0.45
_CORNER_SHAPE_MAX   = 6       # n in [3, this] -> treat as a cornered polygon,
                               # not a curve...
_CORNER_ROUNDNESS_MAX = 0.96   # ...unless roundness is almost perfect. A
                               # regular hexagon tops out at ~0.907 roundness
                               # (a pentagon ~0.865), so this sits safely
                               # above every named polygon but below a real
                               # circle/near-circle — lets a coarsely
                               # simplified circle (which can collapse to as
                               # few as 6 corners) still win as "circle"
                               # instead of "hexagon".

# corner-count driven candidates (polygon family) ---------------------------
_QUAD_BBOX_FILL   = 0.70   # a 4-corner contour needs this much bbox fill to
                           # read as a clean square/rectangle rather than a
                           # skewed quadrilateral (-> irregular polygon)
_QUAD_NEAR_N      = (4, 5, 6)  # corner counts eligible for a square/rectangle
                               # reading — see the air-draw note below
_SQUARE_ASPECT    = 1.18   # aspect_ratio below this (at N==4) -> square
_REGULARITY_MIN   = 0.55   # side-length uniformity needed to call an N-gon
                           # "regular" (named); below this it stays a
                           # generic/irregular polygon
_NGON_SOLIDITY_MIN = 0.90  # a concave contour (e.g. an L-shape) can still
                           # have uniform-ish side lengths — solidity (area /
                           # hull area) catches the concavity side-length
                           # regularity alone misses

# air-draw tolerance: forcing the natural corner list down to a named
# k-gon -----------------------------------------------------------------
# Air-drawn (no-surface) strokes routinely pick up one stray corner along
# an otherwise-straight edge (hand wobble mid-edge reads as an extra
# vertex), so a genuine triangle/pentagon/hexagon often measures one or two
# corners high. `simplify_to_count` can force the natural corner list down
# to any smaller k, but that's only trustworthy if the vertices it merged
# away were near-collinear noise, not real corners — measured by how much
# enclosed-triangle area got discarded to reach k, relative to the shape's
# total area.
_FORCE_AREA_FRAC_SCALE = 0.12  # discarded-area / shape-area at or above this
                               # -> forced fit fully distrusted (trust -> 0)

# safe fallback: always available so *something* wins the vote
_POLYGON_FLOOR = 0.35

# open family (1.4) --------------------------------------------------------
# straightness (endpoint_gap / length) drives "line"; the rest is keyed off
# the interior-corner list of features.approx_points (min_keep=2 for an
# open stroke, so both endpoints always survive — only interior points are
# ever pruned) and a circle fit on the raw stroke points.
#
# A coarsely-simplified circular arc naturally reduces to just a handful of
# interior points too (same as a circle collapsing to ~6-7 points in the
# closed family), so corner count alone can't tell "a few real polyline
# corners" from "an arc, just coarsely sampled" — polyline and arc are left
# to compete on confidence instead of gating each other out, mirroring how
# roundness overrides raw corner count for the closed family.
_LINE_STRAIGHT_MIN  = 0.92   # <=1 interior corner + straightness >= this -> straight_line
_LINE_STRAIGHT_SOFT = 0.80   # softer bar, still fires but at reduced confidence
_LINE_MAX_VERTICES  = 3      # endpoints + at most one (near-collinear) interior point

_POLYLINE_MAX_INTERIOR = 6   # more interior corners than this reads as a
                             # scribbled freeform path, not a deliberate polyline
_POLYLINE_MAX_VERTICES = 8
_SHARP_ANGLE_MAX = 150.0    # interior turn angle below this -> a real corner;
                            # gates "polyline" so a smooth curve's uniformly-
                            # sized (but gently bent) segments don't win it on
                            # segment-length regularity alone

_ARC_FIT_MAX  = 0.10   # mean radial residual / fitted radius, at/above this
                       # -> not a clean circular arc
_ARC_MIN_SPAN = 15.0   # degrees; below this the curvature is negligible/
                       # noise-dominated, too little to call "arc" reliably
_ARC_MAX_SPAN = 340.0  # near-full-loop opens read as "curve" instead of "arc"

_FREEFORM_FLOOR = 0.30  # safe fallback: always available, like _POLYGON_FLOOR
_CURVE_MIN_BEND = 0.08   # below this, a smooth stroke is effectively straight
_FREEFORM_MIN_CHAOS = 0.55  # sharp turns / irregular segments above this -> freeform

# Confidence/ambiguity policy. Closed ML predictions require both a strong
# top probability and separation from the runner-up; deterministic evidence
# uses its score and (for open strokes) the winning-vs-runner-up score gap.
_CLOSED_ML_MIN_CONFIDENCE = 0.55
_CLOSED_ML_MIN_MARGIN = 0.10
_CLOSED_FALLBACK_MIN_CONFIDENCE = 0.55
_OPEN_MIN_CONFIDENCE = 0.60
_OPEN_MIN_MARGIN = 0.04


class StrokeClassifier:
    """Stage: classify a stroke into a 2D shape family.

    Closed strokes use the trained Random Forest for its supported classes
    when its model and feature metadata are valid. Existing deterministic
    closed-shape rules remain the fallback and handle labels outside the
    model's class set. Open strokes use deterministic geometric thresholds.

    The deterministic fallback scores closed shapes against every applicable
    candidate (circle,
    ellipse, square, rectangle, triangle, pentagon, hexagon, a named
    "<n>-gon" for larger regular polygons, and a generic "polygon" catch-
    all for anything irregular) and the highest-confidence candidate wins —
    so e.g. a wobbly square that only scores 0.33 loses to an ellipse
    reading of 0.77 on the same stroke. Shapes that don't match any named
    family fall back to the old "polygon" behaviour and still build fine.
    """

    def __init__(self, model_path=None, metadata_path=None, use_ml_model=True):
        default_model = (Path(__file__).resolve().parent.parent / "models" /
                         "shape_random_forest_final.joblib")
        self.model_path = Path(model_path) if model_path is not None else default_model
        self.metadata_path = (Path(metadata_path) if metadata_path is not None
                              else self.model_path.with_name(
                                  self.model_path.stem + "_metadata.json"))
        self.use_ml_model = use_ml_model
        self._ml_model = None
        self._ml_feature_extractor = None
        self._ml_class_labels = ()
        self._ml_load_checked = False
        self._ml_fallback_logged = False

    def classify(self, points, features, closed):
        if closed:
            try:
                fallback = self._classify_closed(features)
                # Square/rectangle are backed by direct contour geometry
                # (convex corners, near-right angles, and opposite-edge
                # agreement). Keep that strong deterministic result instead
                # of letting the RF override it or reject it as ambiguous.
                if fallback.kind in {"square", "rectangle"}:
                    return fallback
                # Preserve explicit larger N-gon classifications. A generic
                # polygon with 3-6 corners may still be a supported shape
                # (for example, a rotated square), so let the model classify it.
                corner_count, _ = _polygon_signature(features.approx_points)
                if (fallback.kind.endswith("-gon")
                        or (fallback.kind == "polygon"
                            and corner_count not in (3, 4, 5, 6))):
                    return _confidence_decision(
                        fallback, _CLOSED_FALLBACK_MIN_CONFIDENCE)
                predicted = self._classify_closed_ml(features)
                if predicted is not None:
                    return _confidence_decision(
                        predicted, _CLOSED_ML_MIN_CONFIDENCE,
                        _CLOSED_ML_MIN_MARGIN,
                    )
                return _confidence_decision(
                    fallback, _CLOSED_FALLBACK_MIN_CONFIDENCE)
            except Exception:
                return ShapeClass("uncertain", 0.0)
        try:
            return self._classify_open(points, features)
        except Exception:
            # Malformed or non-finite open input must still produce a stable
            # API result; it must not affect the closed-shape model path.
            return ShapeClass("uncertain", 0.0)

    def _load_ml_model(self):
        """Lazily load the model and validate its saved feature/class metadata."""
        if self._ml_load_checked:
            return self._ml_model
        self._ml_load_checked = True
        if not self.use_ml_model:
            return None

        try:
            from tools.ml_features import FEATURE_COUNT, FEATURE_NAMES, extract_features
            import joblib

            with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if (metadata.get("feature_dimension") != FEATURE_COUNT
                    or FEATURE_COUNT != 143
                    or metadata.get("feature_names") != list(FEATURE_NAMES)):
                raise ValueError("Random Forest feature metadata/order mismatch")

            model = joblib.load(self.model_path)
            if getattr(model, "n_features_in_", None) != FEATURE_COUNT:
                raise ValueError("Random Forest feature dimension mismatch")

            model_labels = tuple(str(label) for label in model.classes_)
            metadata_labels = metadata.get("class_labels")
            expected_labels = {
                "circle", "ellipse", "triangle", "square", "rectangle",
                "pentagon", "hexagon",
            }
            if (not isinstance(metadata_labels, list)
                    or set(metadata_labels) != expected_labels
                    or set(metadata_labels) != set(model_labels)
                    or len(metadata_labels) != len(model_labels)):
                raise ValueError("Random Forest class metadata mismatch")

            self._ml_model = model
            self._ml_feature_extractor = extract_features
            self._ml_class_labels = model_labels
        except Exception as error:
            self._log_ml_fallback(f"could not load/validate model: {error}")
        return self._ml_model

    def _classify_closed_ml(self, features):
        model = self._load_ml_model()
        if model is None:
            return None
        try:
            vector = self._ml_feature_extractor(features.points)
            probability_row = model.predict_proba([vector])[0]
            probabilities = [float(value) for value in probability_row]
            if (len(probabilities) != len(self._ml_class_labels)
                    or not all(math.isfinite(value) and 0.0 <= value <= 1.0
                               for value in probabilities)):
                raise ValueError("Invalid Random Forest probabilities")

            predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            probability_map = {
                label: probability
                for label, probability in zip(self._ml_class_labels, probabilities)
            }
            predicted_label = self._ml_class_labels[predicted_index]
            return ShapeClass(
                predicted_label,
                probabilities[predicted_index],
                probability_map,
            )
        except Exception as error:
            self._log_ml_fallback(f"feature extraction/inference failed: {error}")
            return None

    def _log_ml_fallback(self, message):
        if not self._ml_fallback_logged:
            logging.getLogger(__name__).warning(
                "Using deterministic closed-shape classifier fallback: %s", message
            )
            self._ml_fallback_logged = True

    def _classify_closed(self, features):
        # Geometry must be a real (non-degenerate) loop to trust the ratio
        # tests. Scale-invariant: works on raw pixel or normalized points.
        usable = (features.point_count >= 3 and features.perimeter > 0
                  and features.area > 0)
        if not usable:
            return ShapeClass("polygon", _POLYGON_FLOOR)

        # Hand-drawn and rotated quadrilaterals often simplify to five or six
        # corners (a little wrist wobble adds vertices), while their axis-
        # aligned bounding boxes can make a square look rectangular. A strong
        # four-corner fit is more direct evidence than the RF's learned guess;
        # use it before the model can turn a clear square/rectangle into an
        # uncertain result.
        quadrilateral = _recognize_quadrilateral(features)
        if quadrilateral is not None:
            return quadrilateral

        candidates = {}
        n, regularity = _polygon_signature(features.approx_points)
        cornered = (3 <= n <= _CORNER_SHAPE_MAX
                    and features.roundness < _CORNER_ROUNDNESS_MAX)

        # -- curvature-based --------------------------------------------------
        # Suppressed once the contour cleanly simplifies to a small polygon —
        # a clean square must never lose to a "roundness" score on those same
        # few corners.
        if not cornered:
            if (features.roundness >= _CIRCLE_ROUND_MIN
                    and features.bbox_fill >= _CIRCLE_FILL_MIN
                    and features.aspect_ratio <= _CIRCLE_ASPECT_MAX):
                # cubed so a merely-good roundness (e.g. a regular octagon,
                # which also reads as very round) doesn't outscore a
                # legitimate "<n>-gon" reading — only near-perfect roundness
                # commands a high circle confidence
                candidates["circle"] = _clamp(0.3 + 0.7 * features.roundness ** 3, low=0.5)

            if (features.roundness >= _ELLIPSE_ROUND_MIN
                    and features.bbox_fill >= _ELLIPSE_FILL_MIN):
                candidates["ellipse"] = _clamp(
                    0.3 + 0.45 * features.roundness + 0.25 * features.bbox_fill, low=0.4)

        # -- corner-count driven (polygon family) ------------------------------
        # bbox_fill/aspect_ratio are computed on the FULL point set, not the
        # simplified corner list, so a quad reading stays reliable even when
        # n overshoots 4 by a stray corner — no forced re-fit needed.
        if n in _QUAD_NEAR_N and features.bbox_fill >= _QUAD_BBOX_FILL:
            squareness = _clamp(1.0 - (features.aspect_ratio - 1.0) /
                                (_SQUARE_ASPECT - 1.0))
            base = 0.5 + 0.4 * features.bbox_fill
            if squareness > 0.0:
                candidates["square"] = _clamp(base * squareness, low=0.4)
            if squareness < 1.0:
                candidates["rectangle"] = _clamp(base * (1.0 - squareness), low=0.4)

        # Triangle/pentagon/hexagon don't have a bbox_fill-style shortcut,
        # so test each by forcing the natural corner list down to its
        # target vertex count and trusting the result in proportion to how
        # little shape area that forcing had to discard (see
        # _FORCE_AREA_FRAC_SCALE) — a natural exact match (n == target)
        # costs nothing to force and always gets full trust.
        tri = _forced_fit(features, 3, n)
        if tri is not None:
            reg, trust = tri
            # named regardless of how equilateral it is — there is no
            # separate "irregular triangle" bucket
            candidates["triangle"] = _clamp((0.55 + 0.35 * reg) * trust,
                                            low=0.5 * trust)

        for target, label in ((5, "pentagon"), (6, "hexagon")):
            fit = _forced_fit(features, target, n)
            if fit is None:
                continue
            reg, trust = fit
            if reg >= _REGULARITY_MIN and features.solidity >= _NGON_SOLIDITY_MIN:
                candidates[label] = _clamp((0.4 + 0.55 * reg) * trust, low=0.4 * trust)
            else:
                candidates[label] = _clamp((0.2 + 0.3 * reg) * trust)

        if n >= 7:
            label = f"{n}-gon"
            if regularity >= _REGULARITY_MIN and features.solidity >= _NGON_SOLIDITY_MIN:
                candidates[label] = _clamp(0.4 + 0.55 * regularity, low=0.4)
            else:
                candidates[label] = _clamp(0.2 + 0.3 * regularity)

        # -- irregular polygon: always a safe fallback -------------------------
        candidates["polygon"] = (_clamp(_POLYGON_FLOOR + 0.3 * (1.0 - regularity))
                                 if n >= 3 else _POLYGON_FLOOR)

        label = max(candidates, key=candidates.get)
        return ShapeClass(label, candidates[label])

    def _classify_open(self, points, features):
        """Open-family candidates: straight_line, polyline, arc, curve, freeform_path.

        The corner-count/regularity signals come from features.approx_points
        (the noise-robust, VW-simplified, normalized corner list); the
        circle fit for "arc" runs on the raw `points` instead, since a
        shallow/short arc can simplify down to as few as 3-4 corner points —
        too sparse for a meaningful fit — while the original capture is
        usually dense enough. Every applicable candidate is scored and the
        highest-confidence one wins, same voting scheme as the closed
        family — "freeform_path" is the always-available safe fallback.
        """
        if features.length <= 1e-9 or not math.isfinite(features.length):
            return ShapeClass("uncertain", 0.0)

        pts = features.approx_points
        n, seg_regularity = _open_signature(pts)
        interior = max(0, n - 2)
        angles = _interior_turn_angles(pts)
        n_sharp = sum(1 for a in angles if a < _SHARP_ANGLE_MAX)

        candidates = {}

        # -- straight_line: very straight, essentially no interior structure
        if n <= _LINE_MAX_VERTICES and features.straightness >= _LINE_STRAIGHT_MIN:
            candidates["straight_line"] = _clamp(0.6 + 0.4 * features.straightness, low=0.9)
        elif n <= _LINE_MAX_VERTICES and features.straightness >= _LINE_STRAIGHT_SOFT:
            candidates["straight_line"] = _clamp(0.4 + 0.3 * features.straightness, low=0.4)

        # -- polyline: at least one real (sharp) corner, and few enough of
        # them to read as deliberate. A coarsely-simplified arc can *also*
        # land in this corner-count range with moderately sharp per-point
        # angles (see the module note above), so this competes against
        # "arc" on confidence rather than being gated out by it -----------
        if n_sharp >= 1 and interior <= _POLYLINE_MAX_INTERIOR and n <= _POLYLINE_MAX_VERTICES:
            candidates["polyline"] = _clamp(0.45 + 0.45 * seg_regularity, low=0.4)

        # -- arc: a genuine circle fits the raw stroke well, sweeping one
        # direction across a plausible span ---------------------------------
        arc = _fit_arc(points)
        if arc is not None:
            rel_error, span, monotonic = arc
            if (rel_error <= _ARC_FIT_MAX and monotonic
                    and _ARC_MIN_SPAN <= span <= _ARC_MAX_SPAN):
                candidates["arc"] = _clamp(
                    0.55 + 0.45 * (1.0 - rel_error / _ARC_FIT_MAX), low=0.55)

        # -- curve: generic smooth bend, including S-curves which are not
        # well represented by a single circle -----------------------------
        curviness = 1.0 - features.straightness
        smooth_turn_fraction = _smooth_turn_fraction(features.points)
        if (curviness >= _CURVE_MIN_BEND
                and smooth_turn_fraction >= 0.40
                and features.straightness < _LINE_STRAIGHT_MIN):
            candidates["curve"] = _clamp(
                0.70 + 0.25 * smooth_turn_fraction, low=0.70)

        # -- freeform path: always a safe fallback, grows with how chaotic
        # the corner list is (many real corners relative to how many
        # interior points survived, low segment regularity, or an
        # implausibly high corner count) ------------------------------
        chaos = (n_sharp / interior) if interior else 0.0
        overflow = 0.05 * max(0, n - _POLYLINE_MAX_VERTICES)
        candidates["freeform_path"] = _clamp(
            _FREEFORM_FLOOR + 0.2 * chaos + 0.15 * (1.0 - seg_regularity) + overflow)

        # A deliberately kinked or highly irregular path should not be
        # outvoted by a low-confidence smooth-shape candidate. The threshold
        # is based on sharp-turn density and segment-length regularity.
        if (chaos >= _FREEFORM_MIN_CHAOS
                and seg_regularity < 0.65
                and n > _POLYLINE_MAX_VERTICES):
            return _confidence_decision(
                ShapeClass("freeform_path", candidates["freeform_path"]),
                _OPEN_MIN_CONFIDENCE,
            )

        label = max(candidates, key=candidates.get)
        runner_up = max((score for name, score in candidates.items()
                         if name != label), default=0.0)
        return _confidence_decision(
            ShapeClass(label, candidates[label]),
            _OPEN_MIN_CONFIDENCE,
            _OPEN_MIN_MARGIN,
            margin=candidates[label] - runner_up,
        )


def _forced_fit(features, target, natural_n):
    """(regularity, trust) for reading this contour as a `target`-gon.

    `natural_n` is the corner count the area-threshold simplification
    settled on by itself. When it already equals `target`, this is a free,
    fully-trusted exact match. When it's higher, force the corner list down
    to `target` vertices and trust the result in proportion to how little
    shape area that forcing discarded — a stray, near-collinear air-draw
    vertex costs almost nothing to merge away, while forcing away an
    actually-sharp corner costs a lot (and should lose the confidence vote
    to whatever candidate matches the natural corner count instead).
    Returns None when `natural_n` is below `target` — you can't force
    vertices back in that were never there.
    """
    if natural_n < target:
        return None
    if natural_n == target:
        _, reg = _polygon_signature(features.approx_points)
        return reg, 1.0
    forced_pts, removed_area = simplify_to_count(features.approx_points, target, True)
    _, reg = _polygon_signature(forced_pts)
    frac = removed_area / features.area if features.area > 1e-9 else 1.0
    trust = _clamp(1.0 - frac / _FORCE_AREA_FRAC_SCALE)
    return reg, trust


def _polygon_signature(approx_points):
    """(vertex_count, regularity) from a Visvalingam-Whyatt corner list.

    `approx_points` may carry a duplicate closing point (first == last) when
    computed on a snapped/closed loop — that duplicate is stripped before
    counting. Regularity is 1 - the coefficient of variation of the
    resulting side lengths (1.0 == perfectly regular, 0.0 == wildly uneven).
    """
    pts = approx_points
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    n = len(pts)
    if n < 3:
        return n, 0.0

    sides = [math.hypot(b[0] - a[0], b[1] - a[1])
             for a, b in zip(pts, pts[1:] + pts[:1])]
    mean = sum(sides) / n
    if mean <= 1e-9:
        return n, 0.0

    variance = sum((s - mean) ** 2 for s in sides) / n
    cv = math.sqrt(variance) / mean
    return n, max(0.0, min(1.0, 1.0 - cv))


def _recognize_quadrilateral(features):
    """Return a square/rectangle only when a closed contour has a clean quad fit.

    The fit is rotation-invariant: it measures side lengths and corner angles
    from simplified vertices instead of relying on the screen-aligned bbox.
    Up to four small extra vertices may be merged only when their removed area
    is a small fraction of the full sketch area.
    """
    points = list(features.approx_points)
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    natural_count = len(points)
    if natural_count < 4 or natural_count > 8:
        return None

    trust = 1.0
    if natural_count > 4:
        points, removed_area = simplify_to_count(points, 4, True)
        trust = max(0.0, 1.0 - removed_area / max(features.area, 1e-9)
                    / _FORCE_AREA_FRAC_SCALE)
    if len(points) != 4 or trust < 0.72 or features.solidity < 0.88:
        return None

    sides = []
    crosses = []
    deviations = []
    for index, current in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % 4]
        to_previous = (previous[0] - current[0], previous[1] - current[1])
        to_following = (following[0] - current[0], following[1] - current[1])
        previous_length = math.hypot(*to_previous)
        following_length = math.hypot(*to_following)
        if min(previous_length, following_length) <= 1e-9:
            return None
        cosine = _clamp(
            (to_previous[0] * to_following[0]
             + to_previous[1] * to_following[1])
            / (previous_length * following_length), -1.0, 1.0,
        )
        angle = math.degrees(math.acos(cosine))
        deviations.append(abs(90.0 - angle))
        sides.append(following_length)
        next_point = points[(index + 2) % 4]
        crosses.append(
            (following[0] - current[0]) * (next_point[1] - following[1])
            - (following[1] - current[1]) * (next_point[0] - following[0])
        )

    # Reject concave, strongly skewed, or self-crossing fits.
    same_turn_direction = (all(value > 1e-9 for value in crosses)
                           or all(value < -1e-9 for value in crosses))
    if (not same_turn_direction or max(deviations) > 27.0
            or sum(deviations) / 4.0 > 17.0):
        return None

    # Opposite edges should have similar lengths for a perspective-tolerant
    # rectangle. Pair averages also make the resulting ratio independent of
    # rotation and starting corner.
    pair_a = (sides[0] + sides[2]) / 2.0
    pair_b = (sides[1] + sides[3]) / 2.0
    if min(pair_a, pair_b) <= 1e-9:
        return None
    opposite_error = max(
        abs(sides[0] - sides[2]) / pair_a,
        abs(sides[1] - sides[3]) / pair_b,
    )
    if opposite_error > 0.35:
        return None

    side_ratio = max(pair_a, pair_b) / min(pair_a, pair_b)
    angle_quality = max(0.0, 1.0 - sum(deviations) / (4.0 * 45.0))
    edge_quality = max(0.0, 1.0 - opposite_error / 0.5)
    confidence = _clamp(0.70 + 0.15 * angle_quality
                        + 0.10 * edge_quality + 0.05 * trust,
                        low=0.0, high=0.98)
    label = "square" if side_ratio <= _SQUARE_ASPECT else "rectangle"
    return ShapeClass(label, confidence)


def _open_signature(approx_points):
    """(vertex_count, regularity) for an OPEN corner list.

    Unlike _polygon_signature (a closed loop, wrapping last back to first),
    an open stroke's n vertices make n-1 segments with no wraparound.
    """
    n = len(approx_points)
    if n < 2:
        return n, 0.0
    sides = [math.hypot(b[0] - a[0], b[1] - a[1])
             for a, b in zip(approx_points, approx_points[1:])]
    mean = sum(sides) / len(sides)
    if mean <= 1e-9:
        return n, 0.0
    variance = sum((s - mean) ** 2 for s in sides) / len(sides)
    cv = math.sqrt(variance) / mean
    return n, max(0.0, min(1.0, 1.0 - cv))


def _interior_turn_angles(approx_points):
    """Interior turn angle (degrees) at each non-endpoint vertex.

    The angle is measured between the vectors back to the previous point
    and forward to the next, both anchored at the vertex — 180 degrees
    means the path runs straight through it, a small angle means a sharp
    direction reversal. The two endpoints have no "turn" and are excluded.
    """
    n = len(approx_points)
    angles = []
    for i in range(1, n - 1):
        a, b, c = approx_points[i - 1], approx_points[i], approx_points[i + 1]
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        d1 = math.hypot(*v1)
        d2 = math.hypot(*v2)
        if d1 < 1e-9 or d2 < 1e-9:
            angles.append(180.0)
            continue
        cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
        angles.append(math.degrees(math.acos(cosv)))
    return angles


def _smooth_turn_fraction(points):
    """Fraction of small, non-zero direction changes along a sampled stroke.

    A smooth non-circular curve distributes its turn across many adjacent
    segments. A polyline concentrates most turning at a few corners, while
    tracking jitter tends to create large alternating changes. This statistic
    complements the simplified-corner count without replacing it.
    """
    if len(points) < 4:
        return 0.0
    headings = [math.atan2(b[1] - a[1], b[0] - a[0])
                for a, b in zip(points, points[1:])]
    changes = []
    for first, second in zip(headings, headings[1:]):
        delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
        degrees = abs(math.degrees(delta))
        if degrees >= 0.5:
            changes.append(degrees)
    if not changes:
        return 0.0
    return sum(1 for degrees in changes if degrees <= 12.0) / len(changes)


def _solve3(matrix, vec):
    """Solve a 3x3 linear system via Cramer's rule. None if singular."""
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det3(matrix)
    if abs(d) < 1e-9:
        return None

    def replaced(col, vals):
        m = [row[:] for row in matrix]
        for i in range(3):
            m[i][col] = vals[i]
        return m

    return tuple(det3(replaced(col, vec)) / d for col in range(3))


def _fit_circle(points):
    """Algebraic (Kasa) least-squares circle fit. None if too few points
    or numerically degenerate (near-collinear -> no stable circle)."""
    if len(points) < 3:
        return None
    sx = sy = sxx = syy = sxy = sxz = syz = sz = 0.0
    for x, y in points:
        z = -(x * x + y * y)
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
        sxz += x * z
        syz += y * z
        sz += z
    n = float(len(points))
    sol = _solve3([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]],
                  [sxz, syz, sz])
    if sol is None:
        return None
    d, e, f = sol
    cx, cy = -d / 2.0, -e / 2.0
    r2 = cx * cx + cy * cy - f
    if r2 <= 1e-9:
        return None
    return cx, cy, math.sqrt(r2)


def _fit_arc(points):
    """Fit a circular arc to an open corner-point sequence.

    Returns (relative_error, span_degrees, monotonic) or None when there
    are too few points for a non-trivial fit (3 points always fit a circle
    exactly, which proves nothing), or the points are too close to
    collinear for a stable fit — that itself is evidence the segment isn't
    curved, not an arc-detection failure.

    relative_error is the mean radial residual as a fraction of the
    fitted radius (scale-invariant). monotonic means the points sweep
    around the fitted centre in one consistent direction, ruling out
    S-curves and reversals that a bare residual check wouldn't catch.
    """
    if len(points) < 4:
        return None
    fit = _fit_circle(points)
    if fit is None:
        return None
    cx, cy, r = fit
    if r <= 1e-9:
        return None

    residuals = [abs(math.hypot(x - cx, y - cy) - r) for x, y in points]
    rel_error = (sum(residuals) / len(residuals)) / r

    angles = [math.atan2(y - cy, x - cx) for x, y in points]
    deltas = []
    for a0, a1 in zip(angles, angles[1:]):
        delta = a1 - a0
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        deltas.append(delta)
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)
    total = positive + negative
    monotonic = total == 0 or max(positive, negative) / total >= 0.85
    span = math.degrees(abs(sum(deltas)))
    return rel_error, span, monotonic


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _confidence_decision(result, min_confidence, min_margin=None, margin=None):
    """Return a confident result or the shared ``uncertain`` API result.

    Random Forest results carry their class probabilities, so their margin is
    derived directly from those values. Deterministic open results pass the
    gap between their top two geometric evidence scores.
    """
    confidence = result.confidence
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        return ShapeClass("uncertain", 0.0, result.probabilities)

    evidence_margin = margin
    probabilities = result.probabilities or {}
    if probabilities:
        values = list(probabilities.values())
        if (len(values) < 2
                or not all(isinstance(value, (int, float))
                           and math.isfinite(value) and 0.0 <= value <= 1.0
                           for value in values)):
            return ShapeClass("uncertain", _clamp(confidence), probabilities)
        ordered = sorted((float(value) for value in values), reverse=True)
        evidence_margin = ordered[0] - ordered[1]
        confidence = ordered[0]

    ambiguous = (min_margin is not None
                 and (evidence_margin is None
                      or not math.isfinite(evidence_margin)
                      or evidence_margin < min_margin))
    if confidence < min_confidence or ambiguous:
        return ShapeClass("uncertain", _clamp(confidence), probabilities)
    return ShapeClass(result.kind, _clamp(confidence), probabilities)
