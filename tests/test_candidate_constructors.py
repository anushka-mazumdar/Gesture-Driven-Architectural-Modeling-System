"""Tests for candidate parameter schemas, constructors, and sketch derivation."""

import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from render.mesh_bridge import serialize_mesh
from shapes.candidate_constructors import (
    CANDIDATE_SPECS,
    MAX_PATH_POINTS,
    CandidateMesh,
    CandidateParameterError,
    SketchError,
    build_candidate,
    derive_parameters_from_sketch,
    get_parameter_schema,
    list_constructible_candidates,
    validate_parameters,
)
from shapes.shape_taxonomy import get_3d_candidate_ids_for_shape, load_taxonomy


PATH = [(0, 0), (100, 0), (100, 80), (200, 80)]
PROFILE = [(0, 0), (100, 0), (100, 50), (50, 30), (0, 50)]


def required_params(candidate_id):
    params = {}
    for param in CANDIDATE_SPECS[candidate_id].params:
        if param.kind == "path":
            params[param.name] = PATH
        elif param.kind == "polygon":
            params[param.name] = PROFILE
    return params


def welded_edges(mesh):
    """Edge-use counts after merging coincident vertices (flat faces duplicate them)."""
    keys = [tuple(np.round(v, 3)) for v in mesh.vertices]
    ids = {}
    weld = [ids.setdefault(key, len(ids)) for key in keys]
    edges = Counter()
    for i in range(0, len(mesh.indices), 3):
        tri = [weld[k] for k in mesh.indices[i:i + 3]]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[(min(a, b), max(a, b))] += 1
    return edges


def signed_volume(mesh):
    v = mesh.vertices.astype(float)
    tri = np.asarray(mesh.indices).reshape(-1, 3)
    return np.einsum("ij,ij->i", v[tri[:, 0]], np.cross(v[tri[:, 1]], v[tri[:, 2]])).sum() / 6


def circle_sketch(radius=50, n=72):
    return [(200 + radius * math.cos(2 * math.pi * k / n),
             200 + radius * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


def rect_sketch(w, h):
    corners = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
    return [(x0 + (x1 - x0) * t / 10, y0 + (y1 - y0) * t / 10)
            for (x0, y0), (x1, y1) in zip(corners, corners[1:]) for t in range(10)] + [(0, 0)]


def ngon_sketch(sides, radius=60):
    return [(radius * math.cos(2 * math.pi * k / sides), radius * math.sin(2 * math.pi * k / sides))
            for k in range(sides + 1)]


# sketch that the taxonomy maps onto each candidate (first matching label wins)
SKETCHES = {
    "circle": circle_sketch(), "ellipse": [(120 * math.cos(t), 60 * math.sin(t))
                                           for t in np.linspace(0, 2 * math.pi, 73)],
    "triangle": ngon_sketch(3), "square": rect_sketch(100, 100),
    "rectangle": rect_sketch(200, 80), "pentagon": ngon_sketch(5),
    "hexagon": ngon_sketch(6), "9-gon": ngon_sketch(9), "polygon": PROFILE + [PROFILE[0]],
    "straight_line": [(i * 10, 5) for i in range(30)],
    "polyline": PATH,
    "arc": [(100 * math.cos(math.radians(d)), 100 * math.sin(math.radians(d)))
            for d in range(0, 121, 4)],
    "curve": [(x, 40 * math.sin(x / 30)) for x in range(0, 300, 5)],
    "freeform_path": [(0, 0), (40, 30), (80, -10), (120, 50), (160, 0)],
}


class RegistryTest(unittest.TestCase):
    def test_registry_covers_exactly_the_taxonomy_candidates(self):
        taxonomy_ids = {c["id"] for c in load_taxonomy()["candidate_3d_families"]}
        self.assertEqual(set(list_constructible_candidates()), taxonomy_ids)
        self.assertEqual(len(taxonomy_ids), 45)

    def test_source_sides_flag_matches_schema(self):
        for candidate in load_taxonomy()["candidate_3d_families"]:
            with self.subTest(candidate=candidate["id"]):
                has_sides = "sides" in CANDIDATE_SPECS[candidate["id"]].param_names
                self.assertEqual(has_sides, bool(candidate.get("requires_source_sides")))

    def test_schemas_are_json_serializable_and_documented(self):
        for candidate_id in CANDIDATE_SPECS:
            with self.subTest(candidate=candidate_id):
                schema = json.loads(json.dumps(get_parameter_schema(candidate_id)))
                self.assertEqual(schema["id"], candidate_id)
                self.assertTrue(schema["sketch_derivation"])
                for param in schema["parameters"]:
                    if param["type"] in ("float", "int"):
                        self.assertLessEqual(param["minimum"], param["default"])
                        self.assertLessEqual(param["default"], param["maximum"])


class ConstructionTest(unittest.TestCase):
    def test_every_candidate_builds_valid_closed_geometry(self):
        for candidate_id in CANDIDATE_SPECS:
            with self.subTest(candidate=candidate_id):
                mesh = build_candidate(candidate_id, required_params(candidate_id))
                self.assertIsInstance(mesh, CandidateMesh)
                self.assertEqual(mesh.vertices.shape[1], 3)
                self.assertEqual(mesh.normals.shape, mesh.vertices.shape)
                self.assertEqual(len(mesh.indices) % 3, 0)
                self.assertGreater(len(mesh.indices), 0)
                self.assertTrue(all(0 <= i < len(mesh.vertices) for i in mesh.indices))
                self.assertTrue(np.all(np.isfinite(mesh.vertices)))
                np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-4)
                bounds = mesh.get_bounds()
                np.testing.assert_allclose(bounds["center"], 0.0, atol=1e-3)
                self.assertGreater(signed_volume(mesh), 0.0)   # outward winding
                self.assertTrue(all(count == 2 for count in welded_edges(mesh).values()),
                                "mesh is not watertight")

    def test_mesh_matches_existing_object_and_bridge_interface(self):
        payload = json.loads(json.dumps(serialize_mesh(build_candidate("solid.cube"))))
        self.assertEqual(len(payload["vertices"]), len(payload["normals"]))
        self.assertEqual(len(payload["vertices"]) % 3, 0)
        self.assertEqual(payload["kind"], "candidate")
        mesh = build_candidate("sweep.rod")
        for attr in ("position", "rotation", "scale", "selected", "highlighted", "color"):
            self.assertTrue(hasattr(mesh, attr), attr)
        mesh.position[0] += 5.0
        self.assertAlmostEqual(float(mesh.get_bounds()["center"][0]), 5.0, places=3)

    def test_construction_is_deterministic(self):
        for candidate_id in CANDIDATE_SPECS:
            with self.subTest(candidate=candidate_id):
                a = build_candidate(candidate_id, required_params(candidate_id))
                b = build_candidate(candidate_id, required_params(candidate_id))
                np.testing.assert_array_equal(a.vertices, b.vertices)
                self.assertEqual(a.indices, b.indices)

    def test_explicit_dimensions_are_honoured(self):
        def extent(mesh):
            return mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)

        np.testing.assert_allclose(extent(build_candidate("solid.cube", {"size": 80})), 80, atol=1e-3)
        np.testing.assert_allclose(
            extent(build_candidate("solid.cuboid", {"width": 150, "height": 40, "depth": 90})),
            [150, 40, 90], atol=1e-3)
        np.testing.assert_allclose(
            extent(build_candidate("solid.sphere", {"radius": 30})), 60, atol=1e-2)
        np.testing.assert_allclose(
            extent(build_candidate("solid.cylinder", {"radius": 25, "height": 70})),
            [50, 70, 50], atol=1e-2)
        np.testing.assert_allclose(
            extent(build_candidate("sweep.beam", {"length": 300, "width": 20, "depth": 10})),
            [300, 10, 20], atol=1e-3)
        self.assertAlmostEqual(signed_volume(build_candidate("solid.cube", {"size": 10})), 1000, 1)

    def test_caller_parameters_are_not_mutated(self):
        params = {"path": list(PATH), "radius": 5}
        mesh = build_candidate("sweep.path_tube", params)
        self.assertEqual(params, {"path": list(PATH), "radius": 5})
        self.assertEqual(mesh.parameters["radial_segments"], 12)


class ValidationTest(unittest.TestCase):
    def assertRejected(self, candidate_id, params):
        with self.assertRaises(CandidateParameterError):
            validate_parameters(candidate_id, params)
        with self.assertRaises(CandidateParameterError):
            build_candidate(candidate_id, params)

    def test_unknown_candidate_and_bad_containers(self):
        for bad_id in ("solid.dodecahedron", "", None, 3):
            with self.subTest(candidate=bad_id):
                self.assertRejected(bad_id, {})
        self.assertRejected("solid.cube", [("size", 10)])
        self.assertRejected("solid.cube", {"size": 10, "colour": "red"})

    def test_every_numeric_parameter_rejects_bad_values(self):
        for candidate_id, spec in CANDIDATE_SPECS.items():
            base = required_params(candidate_id)
            for param in spec.params:
                if param.kind not in ("float", "int"):
                    continue
                bad = [param.minimum - 1, param.maximum + 1, math.nan, math.inf,
                       True, "10", None if param.required else [1]]
                if param.kind == "int":
                    bad.append(param.default + 0.5)
                for value in bad:
                    with self.subTest(candidate=candidate_id, param=param.name, value=value):
                        self.assertRejected(candidate_id, {**base, param.name: value})

    def test_path_and_profile_validation(self):
        self.assertRejected("sweep.path_tube", {})                         # required
        self.assertRejected("sweep.path_tube", {"path": [(0, 0)]})
        self.assertRejected("sweep.path_tube", {"path": [(0, 0), (0, 0)]})
        self.assertRejected("sweep.path_tube", {"path": [(0, 0), (0.1, 0)]})   # too short
        self.assertRejected("sweep.path_tube", {"path": [(0, 0), (1, math.nan)]})
        self.assertRejected("sweep.path_tube", {"path": [(0, 0), (1e9, 0)]})
        self.assertRejected("sweep.path_tube", {"path": [(0, 0, 0), (1, 1, 1)]})
        self.assertRejected("sweep.path_tube", {"path": "0,0 1,1"})
        self.assertRejected("sweep.path_tube",
                            {"path": [(i, i % 2) for i in range(MAX_PATH_POINTS + 5)]})
        self.assertRejected("solid.profile_prism", {"profile": [(0, 0), (10, 0)]})
        self.assertRejected("solid.profile_prism", {"profile": [(0, 0), (10, 0), (20, 0)]})
        self.assertRejected("solid.profile_prism",                          # bow-tie
                            {"profile": [(0, 0), (10, 10), (10, 0), (0, 10)]})
        closed = validate_parameters("solid.profile_prism", {"profile": PROFILE + [PROFILE[0]]})
        self.assertEqual(len(closed["profile"]), len(PROFILE))

    def test_cross_parameter_constraints(self):
        self.assertRejected("solid.torus", {"major_radius": 10, "minor_radius": 10})
        self.assertRejected("solid.elliptical_torus",
                            {"major_radius_x": 50, "major_radius_z": 5, "minor_radius": 6})
        self.assertRejected("sweep.arc_tube", {"radius": 5, "tube_radius": 6})
        self.assertRejected("sweep.torus_segment", {"radius": 5, "tube_radius": 5})
        self.assertRejected("sweep.arc_ribbon", {"radius": 2, "thickness": 5})

    def test_defaults_fill_optional_parameters(self):
        self.assertEqual(validate_parameters("solid.cube"), {"size": 100.0})
        self.assertEqual(validate_parameters("solid.regular_prism", {"sides": 9})["sides"], 9)


class SketchDerivationTest(unittest.TestCase):
    def test_roadmap_examples(self):
        cube = derive_parameters_from_sketch("solid.cube", rect_sketch(100, 100))
        self.assertAlmostEqual(cube["size"], 100.0, places=3)

        cuboid = derive_parameters_from_sketch("solid.cuboid", rect_sketch(200, 80))
        self.assertAlmostEqual(cuboid["width"], 200.0, places=3)
        self.assertAlmostEqual(cuboid["depth"], 80.0, places=3)
        self.assertAlmostEqual(cuboid["height"], 80.0, places=3)

        sphere = derive_parameters_from_sketch("solid.sphere", circle_sketch(50))
        self.assertAlmostEqual(sphere["radius"], 50.0, delta=0.5)
        cylinder = derive_parameters_from_sketch("solid.cylinder", circle_sketch(50))
        self.assertAlmostEqual(cylinder["radius"], 50.0, delta=0.5)
        self.assertAlmostEqual(cylinder["height"], 100.0, delta=1.0)

    def test_line_arc_and_ngon_derivations(self):
        rod = derive_parameters_from_sketch("sweep.rod", SKETCHES["straight_line"])
        self.assertAlmostEqual(rod["length"], 290.0, places=3)
        self.assertAlmostEqual(rod["radius"], max(290 * 0.06, 12) / 2, places=3)
        line_cylinder = derive_parameters_from_sketch("solid.cylinder", SKETCHES["straight_line"])
        self.assertAlmostEqual(line_cylinder["height"], 290.0, places=3)

        arc = derive_parameters_from_sketch("sweep.arc_tube", SKETCHES["arc"])
        self.assertAlmostEqual(arc["radius"], 100.0, delta=0.5)
        self.assertAlmostEqual(arc["sweep_angle"], 120.0, delta=1.0)
        reflex = [(100 * math.cos(math.radians(d)), 100 * math.sin(math.radians(d)))
                  for d in range(0, 271, 5)]
        self.assertAlmostEqual(
            derive_parameters_from_sketch("sweep.arc_tube", reflex)["sweep_angle"], 270.0, delta=1.0)

        self.assertEqual(derive_parameters_from_sketch(
            "solid.regular_prism", ngon_sketch(9), source_sides=9)["sides"], 9)
        for bad_sides in (None, 2, 1000, 7.5, True):
            self.assertEqual(derive_parameters_from_sketch(
                "solid.regular_prism", ngon_sketch(9), source_sides=bad_sides)["sides"], 8)

    def test_every_recommended_candidate_derives_and_builds_from_its_source_sketches(self):
        covered = set()
        for label, sketch in SKETCHES.items():
            sides = int(label.split("-")[0]) if label.endswith("-gon") else None
            for candidate_id in get_3d_candidate_ids_for_shape(label):
                with self.subTest(label=label, candidate=candidate_id):
                    params = derive_parameters_from_sketch(candidate_id, sketch, sides)
                    mesh = build_candidate(candidate_id, params)
                    self.assertGreater(signed_volume(mesh), 0.0)
                    covered.add(candidate_id)
        mapped_ids = {
            candidate_id
            for shape in load_taxonomy()["classes"]
            for candidate_id in shape["candidate_3d_family_ids"]
        }
        self.assertEqual(covered, mapped_ids)
        self.assertTrue(mapped_ids.issubset(CANDIDATE_SPECS))

    def test_self_crossing_profile_falls_back_to_hull(self):
        bowtie = [(0, 0), (100, 100), (100, 0), (0, 100), (0, 0)]
        params = derive_parameters_from_sketch("solid.profile_prism", bowtie)
        build_candidate("solid.profile_prism", params)

    def test_huge_sketch_is_clamped_into_bounds(self):
        params = derive_parameters_from_sketch("solid.cube", rect_sketch(50000, 50000))
        self.assertEqual(params["size"], 5000.0)

    def test_invalid_sketches_raise_sketch_error(self):
        for bad in ([], [(0, 0)], [(1, 1), (1, 1)], [(0, math.nan), (1, 1)],
                    [("a", "b"), (1, 1)], "abc", None, [(0,), (1,)]):
            with self.subTest(sketch=bad), self.assertRaises(SketchError):
                derive_parameters_from_sketch("solid.cube", bad)
        with self.assertRaises(SketchError):   # collinear: no area for a profile
            derive_parameters_from_sketch("solid.profile_prism", [(0, 0), (10, 0), (20, 0)])
        with self.assertRaises(CandidateParameterError):
            derive_parameters_from_sketch("solid.unknown", rect_sketch(10, 10))


if __name__ == "__main__":
    unittest.main()
