"""Read-only access to the centralized supported-shape taxonomy.

The taxonomy describes current classifier labels and current 3D mesh families.
It is read by shapes.recommendation_api; it does not classify or build meshes.
"""

import copy
import json
import re
from functools import lru_cache
from pathlib import Path


_TAXONOMY_PATH = Path(__file__).with_name("shape_taxonomy.json")
_VALID_CATEGORIES = {"closed", "open", "classification_state"}


@lru_cache(maxsize=1)
def _read_and_validate_taxonomy():
    with _TAXONOMY_PATH.open("r", encoding="utf-8") as taxonomy_file:
        taxonomy = json.load(taxonomy_file)

    families = taxonomy.get("candidate_families")
    candidate_3d_families = taxonomy.get("candidate_3d_families")
    classes = taxonomy.get("classes")
    if (not isinstance(families, list)
            or not isinstance(candidate_3d_families, list)
            or not isinstance(classes, list)):
        raise ValueError(
            "Taxonomy must define candidate_families, candidate_3d_families, "
            "and classes arrays"
        )

    family_ids = [family.get("id") for family in families]
    candidate_3d_ids = [family.get("id") for family in candidate_3d_families]
    class_ids = [shape.get("id") for shape in classes]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("Taxonomy candidate family IDs must be unique")
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("Taxonomy shape IDs must be unique")
    if len(candidate_3d_ids) != len(set(candidate_3d_ids)):
        raise ValueError("Taxonomy 3D candidate IDs must be unique")

    labels = []
    known_family_ids = set(family_ids)
    for shape in classes:
        if shape.get("category") not in _VALID_CATEGORIES:
            raise ValueError(f"Unsupported taxonomy category for {shape.get('id')!r}")
        exact_labels = shape.get("classifier_labels", [])
        if not isinstance(exact_labels, list):
            raise ValueError(f"classifier_labels must be a list for {shape.get('id')!r}")
        labels.extend(label.lower() for label in exact_labels)
        pattern = shape.get("classifier_label_pattern")
        if pattern is not None:
            re.compile(pattern)
        family_refs = shape.get("candidate_family_ids", [])
        if any(family_id not in known_family_ids for family_id in family_refs):
            raise ValueError(f"Unknown candidate family referenced by {shape.get('id')!r}")
        candidate_refs = shape.get("candidate_3d_family_ids", [])
        if (not isinstance(candidate_refs, list)
                or any(candidate_id not in candidate_3d_ids
                       for candidate_id in candidate_refs)):
            raise ValueError(f"Unknown 3D candidate referenced by {shape.get('id')!r}")
    if len(labels) != len(set(labels)):
        raise ValueError("Exact classifier labels must resolve to only one taxonomy entry")

    return taxonomy


def load_taxonomy():
    """Return a detached copy of the validated taxonomy document."""
    return copy.deepcopy(_read_and_validate_taxonomy())


def get_shape_by_id(shape_id):
    """Return a detached shape definition by stable ID, or ``None``."""
    for shape in _read_and_validate_taxonomy()["classes"]:
        if shape["id"] == shape_id:
            return copy.deepcopy(shape)
    return None


def resolve_shape(classifier_label):
    """Resolve an existing classifier label to taxonomy metadata.

    Dynamic classifier labels such as ``7-gon`` and ``12-gon`` resolve to the
    bounded ``other_regular_polygon`` entry while retaining their concrete
    label for display. Returns ``None`` when the label is not registered.
    """
    if not isinstance(classifier_label, str):
        return None
    normalized = classifier_label.strip().lower()
    if not normalized:
        return None

    for shape in _read_and_validate_taxonomy()["classes"]:
        if normalized in (label.lower() for label in shape.get("classifier_labels", [])):
            resolved = copy.deepcopy(shape)
            resolved["resolved_classifier_label"] = normalized
            resolved["display_label"] = shape["label"]
            return resolved

        pattern = shape.get("classifier_label_pattern")
        if pattern and re.fullmatch(pattern, normalized):
            resolved = copy.deepcopy(shape)
            resolved["resolved_classifier_label"] = normalized
            display_pattern = shape.get("display_label_pattern", shape["label"])
            sides = normalized.split("-", 1)[0]
            resolved["display_label"] = display_pattern.format(sides=sides)
            return resolved
    return None


def get_candidate_family(family_id):
    """Return a detached candidate-family definition by stable ID, or ``None``."""
    for family in _read_and_validate_taxonomy()["candidate_families"]:
        if family["id"] == family_id:
            return copy.deepcopy(family)
    return None


def get_3d_candidates_for_shape(classifier_label):
    """Return allowed 3D candidate families for a classifier output label.

    Results preserve the mapping order stored in the taxonomy. Unknown labels
    and the ``uncertain``/``unknown`` classification state return an empty list.
    This function only reads metadata; it does not construct or select objects.
    """
    shape = resolve_shape(classifier_label)
    if shape is None:
        return []
    candidates_by_id = {
        candidate["id"]: candidate
        for candidate in _read_and_validate_taxonomy()["candidate_3d_families"]
    }
    return [copy.deepcopy(candidates_by_id[candidate_id])
            for candidate_id in shape.get("candidate_3d_family_ids", [])]


def get_3d_candidate_ids_for_shape(classifier_label):
    """Return stable candidate IDs for a classifier label, or an empty list."""
    return [candidate["id"]
            for candidate in get_3d_candidates_for_shape(classifier_label)]
