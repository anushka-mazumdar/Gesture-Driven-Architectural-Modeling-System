from dataclasses import dataclass

from shapes.stroke_analysis   import analyze
from shapes.stroke_normalization import preprocess
from shapes.closure_detector  import ClosureDetector
from shapes.stroke_classifier import StrokeClassifier
from shapes.shape_recommender import ShapeRecommender
from shapes.candidate_constructors import (
    CandidateConstructionError,
    CandidateParameterError,
    SketchError,
    build_candidate,
    derive_parameters_from_sketch,
)


@dataclass
class ConversionResult:
    """Everything the pipeline produced for one stroke."""
    record: object        # StrokeRecord (raw data + back-filled metadata)
    mesh: object          # built PolygonMesh / RibbonMesh, or None
    closed: bool
    features: object      # StrokeFeatures on the ORIGINAL points (physical sizing)
    norm_features: object # StrokeFeatures on resampled+normalized points (robust classify)
    shape_class: object   # ShapeClass
    recommendation: object  # MeshRecommendation
    candidates: object = None  # RecommendationResult (ranked 3D candidates)


class StrokePipeline:
    """Orchestrates the staged conversion of a captured stroke:

        capture -> preprocess(normalize+resample) -> analyze -> closed?
                -> classify -> recommend -> factory.build

    Preprocessing/analysis feed classification with robust, density- and
    scale-invariant features. Closure detection and meshing still use the
    original captured points, so on-screen behaviour is unchanged.
    """

    def __init__(self, closure=None, classifier=None, recommender=None,
                 factory=None, preprocessor=None, resample_n=64):
        self.closure      = closure     or ClosureDetector()
        self.classifier   = classifier  or StrokeClassifier()
        self.recommender  = recommender or ShapeRecommender()
        self.factory      = factory     # must provide build(recommendation, points)
        self.preprocessor = preprocessor or preprocess
        self.resample_n   = resample_n
        self.conversions  = []          # history of ConversionResults

    def convert(self, record):
        """Run the full pipeline on a finished StrokeRecord.

        Back-fills the record's metadata so the raw data is preserved AND
        enriched, then builds the mesh through the factory. Returns
        ConversionResult or None if the stroke has no points to analyse.
        """
        if record is None or not record.points:
            return None

        params = self._fit(record.points)

        mesh = None
        if self.factory is not None:
            mesh = self.factory.build(params.recommendation, params.points)
            # The legacy factory builds a closed circle as a thin extrusion,
            # which reads visually as a ring. Circle already has a bounded,
            # taxonomy-ordered recommendation list; use its configured
            # default candidate (Sphere) for the initial scene object while
            # leaving all other closed-shape construction unchanged.
            if (params.closed and params.shape_class.kind.lower() == "circle"
                    and params.candidates is not None
                    and params.candidates.status == "ok"):
                default_id = params.candidates.default_candidate_id
                if default_id:
                    try:
                        candidate_params = derive_parameters_from_sketch(
                            default_id, params.points,
                            source_sides=params.candidates.source_sides,
                        )
                        mesh = build_candidate(default_id, candidate_params)
                    except (CandidateConstructionError, CandidateParameterError,
                            SketchError, TypeError, ValueError):
                        # Preserve a working closed-profile mesh if candidate
                        # derivation fails for a malformed or degenerate loop.
                        pass

        # preserve + enrich the raw record
        record.points = params.points          # sealed loop when closed
        record.closed = params.closed
        record.features = params.features            # raw/physical for sizing
        record.norm_features = params.norm_features  # normalized for classifying
        record.shape_class = params.shape_class
        record.recommendation = params.recommendation
        record.candidates = params.candidates

        result = ConversionResult(
            record=record, mesh=mesh, closed=params.closed,
            features=params.features, norm_features=params.norm_features,
            shape_class=params.shape_class, recommendation=params.recommendation,
            candidates=params.candidates,
        )
        self.conversions.append(result)
        return result

    def classify_stroke(self, points):
        """Analyse + classify a point list without building a mesh.

        Returns (closed, features, norm_features, shape_class, recommendation).
        """
        params = self._fit(points)
        return (params.closed, params.features, params.norm_features,
                params.shape_class, params.recommendation)

    # kept as `classify` for 1.1 back-compat convenience
    classify = classify_stroke

    def _fit(self, points):
        points, closed = self.closure.snap(points)      # seals the loop
        features = analyze(points)                      # physical (for sizing)
        norm_points = self.preprocessor(points, n=self.resample_n)
        norm_features = analyze(norm_points)            # robust (for classifying)
        shape_class = self.classifier.classify(points, norm_features, closed)
        recommendation = self.recommender.recommend(points, features, shape_class)
        candidates = self.recommender.recommend_candidates(shape_class)
        return _FitParams(points, closed, features, norm_features, shape_class,
                          recommendation, candidates)


class _FitParams:
    __slots__ = ("points", "closed", "features", "norm_features",
                 "shape_class", "recommendation", "candidates")

    def __init__(self, points, closed, features, norm_features, shape_class,
                 recommendation, candidates=None):
        self.points = points
        self.closed = closed
        self.features = features
        self.norm_features = norm_features
        self.shape_class = shape_class
        self.recommendation = recommendation
        self.candidates = candidates
