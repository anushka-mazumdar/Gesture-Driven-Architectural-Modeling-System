import math
import unittest

import numpy as np

from shapes import candidate_geometry as geo
from shapes.candidate_constructors import (
    build_candidate,
    derive_parameters_from_sketch,
)


OPEN_SKETCHES = {
    "straight_line": [(0, 0), (50, 2), (100, 0), (150, -2), (200, 0)],
    "polyline": [(0, 0), (100, 0), (100, 80), (200, 80), (250, 30)],
    "arc": [(100 * math.cos(math.radians(angle)),
             100 * math.sin(math.radians(angle))) for angle in range(-90, 91, 10)],
    "curve": [(index * 20, 35 * math.sin(index * math.pi / 5)) for index in range(11)],
    "freeform_path": [(0, 0), (30, 40), (70, 15), (110, 65), (160, 20)],
}

OPEN_CANDIDATES = {
    "straight_line": "sweep.rod",
    "polyline": "sweep.polyline_beam",
    "arc": "sweep.arc_tube",
    "curve": "sweep.curved_tube",
    "freeform_path": "sweep.path_tube",
}


def sweep_path_and_section_count(candidate_id, params):
    if candidate_id in {"sweep.rod", "sweep.beam"}:
        length = params["length"]
        return [(-length / 2, 0), (length / 2, 0)]
    if candidate_id in {"sweep.arc_tube", "sweep.torus_segment", "sweep.arc_ribbon"}:
        path = geo.arc_path(params["radius"], params["sweep_angle"], params["segments"])
        return path
    path = params["path"]
    return geo.chaikin(path, params["smoothing"])


class OpenCandidateTopologyTests(unittest.TestCase):
    def test_open_shape_candidates_keep_distinct_path_ends_and_no_loop_seam(self):
        for shape, candidate_id in OPEN_CANDIDATES.items():
            with self.subTest(shape=shape, candidate=candidate_id):
                params = derive_parameters_from_sketch(candidate_id, OPEN_SKETCHES[shape])
                mesh = build_candidate(candidate_id, params)
                path = sweep_path_and_section_count(candidate_id, params)
                radial = params.get("radial_segments", 8)
                self.assertGreater(len(path), 1)
                self.assertGreater(math.dist(path[0], path[-1]), 1.0)
                if candidate_id == "sweep.arc_tube":
                    self.assertLess(params["sweep_angle"], 359.0)

                # Geo.sweep lays its longitudinal cross-sections first. A
                # triangle joining both endpoint rings would be the unwanted
                # final-to-first seam; side faces must only join neighbors.
                if len(path) > 2:
                    first_ring = set(range(radial))
                    end_start = (len(path) - 1) * radial
                    last_ring = set(range(end_start, end_start + radial))
                    for offset in range(0, len(mesh.indices), 3):
                        triangle = set(mesh.indices[offset:offset + 3])
                        self.assertFalse(triangle & first_ring and triangle & last_ring)

                first_center = mesh.vertices[:radial].mean(axis=0)
                end_start = (len(path) - 1) * radial
                last_center = mesh.vertices[end_start:end_start + radial].mean(axis=0)
                self.assertGreater(float(np.linalg.norm(first_center - last_center)), 1.0)

    def test_open_arc_candidates_are_partial_sweeps_not_tori(self):
        sketch = OPEN_SKETCHES["arc"]
        for candidate_id in ("sweep.arc_tube", "sweep.torus_segment", "sweep.arc_ribbon"):
            with self.subTest(candidate=candidate_id):
                params = derive_parameters_from_sketch(candidate_id, sketch)
                mesh = build_candidate(candidate_id, params)
                self.assertLess(params["sweep_angle"], 359.0)
                path = sweep_path_and_section_count(candidate_id, params)
                self.assertGreater(math.dist(path[0], path[-1]), 1.0)
                self.assertGreater(len(mesh.indices), 0)


if __name__ == "__main__":
    unittest.main()
