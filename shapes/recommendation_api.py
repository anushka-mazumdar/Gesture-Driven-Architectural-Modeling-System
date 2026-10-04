"""Shared recommendation API: classifier output -> ranked 3D candidates.

One entry point (``get_recommendations``) that every consumer — the stroke
pipeline today, the WebView/UI layer later — calls to turn a classified stroke
into a bounded, ordered list of 3D candidate families. It only reads the
taxonomy; it never classifies, builds meshes, or changes the runtime
MeshRecommendation produced by ShapeRecommender.
"""

import math
import re
from dataclasses import dataclass, field

from shapes.shape_taxonomy import get_3d_candidates_for_shape, resolve_shape


STATUS_OK          = "ok"           # supported label with >= 1 candidate
STATUS_UNCERTAIN   = "uncertain"    # classifier withheld a decision
STATUS_UNSUPPORTED = "unsupported"  # label not registered in the taxonomy
STATUS_INVALID     = "invalid"      # input could not be read as a label

_NGON_RE = re.compile(r"^(\d+)-gon$")


@dataclass(frozen=True)
class CandidateRecommendation:
    """One 3D candidate for a classified stroke, in taxonomy rank order."""
    id: str
    label: str
    category: str
    rank: int                         # 1 = default / most preferred
    construction_status: str = "candidate"
    requires_source_sides: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "rank": self.rank,
            "construction_status": self.construction_status,
            "requires_source_sides": self.requires_source_sides,
        }


@dataclass(frozen=True)
class RecommendationResult:
    """Structured, JSON-serializable recommendation for one classified stroke.

    ``candidates`` is empty for every status other than ``ok``; callers can
    rely on ``has_candidates`` / ``default_candidate_id`` without re-checking
    the status.
    """
    status: str
    classifier_label: str = None
    shape_id: str = None
    display_label: str = None
    category: str = None              # closed | open | classification_state
    confidence: float = None
    source_sides: int = None          # polygon side count when known
    candidates: tuple = field(default_factory=tuple)

    @property
    def has_candidates(self):
        return bool(self.candidates)

    @property
    def default_candidate_id(self):
        return self.candidates[0].id if self.candidates else None

    @property
    def candidate_ids(self):
        return [candidate.id for candidate in self.candidates]

    def to_dict(self):
        return {
            "status": self.status,
            "classifier_label": self.classifier_label,
            "shape_id": self.shape_id,
            "display_label": self.display_label,
            "category": self.category,
            "confidence": self.confidence,
            "source_sides": self.source_sides,
            "default_candidate_id": self.default_candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _read_label_and_confidence(shape_result, confidence):
    """Accept a ShapeClass-like object, a mapping, or a bare label string."""
    if isinstance(shape_result, str):
        label = shape_result
    elif isinstance(shape_result, dict):
        label = shape_result.get("kind", shape_result.get("label"))
        if confidence is None:
            confidence = shape_result.get("confidence")
    else:
        label = getattr(shape_result, "kind", None)
        if confidence is None:
            confidence = getattr(shape_result, "confidence", None)
    return label, confidence


def _sanitize_confidence(confidence):
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(confidence):
        return None
    return max(0.0, min(1.0, float(confidence)))


def _source_sides(label, shape):
    match = _NGON_RE.match(label)
    if match:
        return int(match.group(1))
    sides = shape.get("geometry", {}).get("nominal_sides")
    return sides if isinstance(sides, int) else None


def get_recommendations(shape_result, confidence=None, max_candidates=None):
    """Return a RecommendationResult for a classifier output.

    ``shape_result`` may be a ShapeClass, a ``{"kind"/"label", "confidence"}``
    mapping, or a label string. ``max_candidates`` optionally truncates the
    ranked list (must be a positive int). Never raises for bad classifier
    input — unreadable, uncertain, and unsupported labels return a result with
    no candidates and the matching status.
    """
    if max_candidates is not None and (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int) or max_candidates < 1):
        raise ValueError("max_candidates must be a positive integer or None")

    label, confidence = _read_label_and_confidence(shape_result, confidence)
    confidence = _sanitize_confidence(confidence)
    if not isinstance(label, str) or not label.strip():
        return RecommendationResult(status=STATUS_INVALID, confidence=confidence)

    normalized = label.strip().lower()
    shape = resolve_shape(normalized)
    if shape is None:
        return RecommendationResult(status=STATUS_UNSUPPORTED,
                                    classifier_label=normalized,
                                    confidence=confidence)

    common = dict(
        classifier_label=normalized,
        shape_id=shape["id"],
        display_label=shape["display_label"],
        category=shape["category"],
        confidence=confidence,
    )
    if shape["category"] == "classification_state":
        return RecommendationResult(status=STATUS_UNCERTAIN, **common)

    raw_candidates = get_3d_candidates_for_shape(normalized)
    if max_candidates is not None:
        raw_candidates = raw_candidates[:max_candidates]
    candidates = tuple(
        CandidateRecommendation(
            id=candidate["id"],
            label=candidate["label"],
            category=candidate["category"],
            rank=rank,
            construction_status=candidate.get("construction_status", "candidate"),
            requires_source_sides=bool(candidate.get("requires_source_sides", False)),
        )
        for rank, candidate in enumerate(raw_candidates, start=1)
    )
    status = STATUS_OK if candidates else STATUS_UNSUPPORTED
    return RecommendationResult(status=status,
                                source_sides=_source_sides(normalized, shape),
                                candidates=candidates, **common)
