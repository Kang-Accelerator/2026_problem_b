import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from competition_b.code.problem2_model import (
    has_observation_axis_symmetry,
    parse_case_and_geometry,
    parse_config,
    reflect_about_observation_axis,
    sample_exact_source_points,
    symmetry_preserving_subsample,
)
from competition_b.code.problem2_score import certify_receive, point_to_triangle_distance, score_candidate
from competition_b.code.problem2_solve import _cell_stencil, _json_safe, _plot_case, _plot_point_groups, _region_topology, point_in_candidate_region, solve_case
from competition_b.code.problem2_solve import score_margin


class Problem2ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data" / "problem2_cases.json"
        cls.cases = json.loads(path.read_text(encoding="utf-8"))
        cls.normal_case = next(item for item in cls.cases if item["case_id"] == "p2_normal_center")
        cls.normal_result = solve_case(cls.normal_case)

    def test_all_required_case_categories_are_present(self):
        ids = {case["case_id"] for case in self.cases}
        self.assertGreaterEqual(len(ids), 12)
        for required in {"p2_normal_center", "p2_wrap_around", "p2_external_tangent", "p2_invalid_outward_ray", "p2_near_branch", "p2_direction_upper_bound", "p2_target_shift"}:
            self.assertIn(required, ids)

    def test_empty_case_reports_unknown_not_fake_optimum(self):
        case = next(item for item in self.cases if item["case_id"] == "p2_invalid_outward_ray")
        result = solve_case(case)
        self.assertEqual(result["best_single_point"]["position"], None)
        self.assertEqual(result["candidate_region"]["recommendation_status"], "empty")
        self.assertEqual(result["candidate_region_receive_verification"]["status"], "not_available")

    def test_certification_status_is_explicit(self):
        config = parse_config(self.normal_case)
        sources = sample_exact_source_points(config)
        outer = np.asarray(self.normal_result["possible_source_set"]["convex_outer_vertices"], dtype=float)
        status = certify_receive(config, outer, config.s1, sources, subdivision_budget=20)
        self.assertIn(status["status"], {"passed", "violated", "unknown"})
        self.assertEqual(status["continuous_worst_case_certified"], status["status"] == "passed")

    def test_point_to_triangle_distance_and_cell_stencil(self):
        triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        self.assertAlmostEqual(point_to_triangle_distance((0.5, 0.5), triangle), 0.0)
        stencil = _cell_stencil([0.0, 2.0, 0.0, 2.0])
        self.assertEqual(stencil.shape, (9, 2))
        self.assertTrue(any(np.allclose(point, [1.0, 1.0]) for point in stencil))
        for corner in ([0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]):
            self.assertTrue(any(np.allclose(point, corner) for point in stencil))

    def test_json_report_replaces_nonfinite_numbers_with_null(self):
        safe = _json_safe({"bad": float("inf"), "nested": [float("nan"), np.float64(1.25)]})
        self.assertEqual(safe, {"bad": None, "nested": [None, 1.25]})
        self.assertEqual(json.dumps(safe, allow_nan=False), '{"bad": null, "nested": [null, 1.25]}')

    def test_plot_separates_seed_nonseed_and_region_stencil_points(self):
        result = {
            "candidate_region": {
                "J_best_passed_search_sample": 0.1,
                "search_seed_epsilon_J": 0.02,
                "evaluated_points": [
                    {"x": 0.0, "y": 0.0, "j_lambda": 0.11},
                    {"x": 1.0, "y": 0.0, "j_lambda": 0.14},
                    {"x": 2.0, "y": 0.0, "j_lambda": 0.105},
                ],
            },
            "candidate_score_table": [
                {"x": 0.0, "y": 0.0, "cert_status": "passed"},
                {"x": 1.0, "y": 0.0, "cert_status": "passed"},
            ],
        }
        groups = _plot_point_groups(result)
        self.assertEqual([(row["x"], row["y"]) for row in groups["near_optimal_seed"]], [(0.0, 0.0)])
        self.assertEqual([(row["x"], row["y"]) for row in groups["non_near_search"]], [(1.0, 0.0)])
        self.assertEqual([(row["x"], row["y"]) for row in groups["region_stencil"]], [(2.0, 0.0)])

    def test_plot_writes_jpg_and_pdf_but_not_png(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _plot_case(self.normal_case, self.normal_result, output)
            stem = output / "problem2_p2_normal_center"
            self.assertTrue(stem.with_suffix(".jpg").exists())
            self.assertTrue(stem.with_suffix(".pdf").exists())
            self.assertFalse(stem.with_suffix(".png").exists())

    def test_near_branch_sampling_and_search_keep_complete_mirror_pairs(self):
        case = next(item for item in self.cases if item["case_id"] == "p2_near_branch")
        config, sources, _, _ = parse_case_and_geometry(case)
        self.assertTrue(has_observation_axis_symmetry(config))
        search_sources = symmetry_preserving_subsample(config, sources, config.search_source_limit)
        self.assertEqual(len(search_sources), config.search_source_limit)

        def assert_mirror_closed(points):
            keys = {(round(float(point[0]), 6), round(float(point[1]), 6)) for point in points}
            for point in points:
                mirror = reflect_about_observation_axis(config, point)
                self.assertIn((round(float(mirror[0]), 6), round(float(mirror[1]), 6)), keys)

        assert_mirror_closed(search_sources)
        result = solve_case(case)
        rows = result["candidate_score_table"]
        self.assertEqual(len(rows), len({(round(float(row["x"]), 6), round(float(row["y"]), 6)) for row in rows}))
        self.assertEqual(result["diagnostics"]["symmetry_audit"]["status"], "passed")
        self.assertEqual(result["diagnostics"]["symmetry_audit"]["missing_mirror_count"], 0)
        self.assertLessEqual(result["diagnostics"]["symmetry_audit"]["max_mirror_J_abs_difference"], 1e-10)
        assert_mirror_closed([(row["x"], row["y"]) for row in rows])
        lookup = {(round(float(row["x"]), 6), round(float(row["y"]), 6)): row for row in rows}
        for row in rows:
            mirror = reflect_about_observation_axis(config, (row["x"], row["y"]))
            partner = lookup[(round(float(mirror[0]), 6), round(float(mirror[1]), 6))]
            self.assertEqual(row.get("cert_status"), partner.get("cert_status"))
            if math.isfinite(float(row.get("j_lambda", float("inf")))) and math.isfinite(float(partner.get("j_lambda", float("inf")))):
                self.assertAlmostEqual(float(row["j_lambda"]), float(partner["j_lambda"]), places=10)
            self.assertIn(row.get("point_origin"), {"coarse_grid_formula", "local_cross_formula"})

        groups = _plot_point_groups(result)
        plotted_search = groups["near_optimal_seed"] + groups["non_near_search"]
        assert_mirror_closed([(row["x"], row["y"]) for row in plotted_search])
        for row in result["candidate_region"]["evaluated_points"]:
            self.assertIn(row.get("point_origin"), {
                "coarse_grid_formula", "local_cross_formula", "search_point_formula", "adaptive_cell_9_point_stencil",
            })

    def test_topology_preserves_components_and_holes_without_internal_edges(self):
        origin = np.array([0.0, 0.0])
        two_components = _region_topology({(0, 0), (2, 0)}, origin, 1.0)
        self.assertEqual(two_components["components"], 2)
        self.assertEqual(len(two_components["boundary_segments"]), 8)
        ring = {(i, j) for i in range(3) for j in range(3)} - {(1, 1)}
        topology = _region_topology(ring, origin, 1.0)
        self.assertEqual(topology["components"], 1)
        self.assertEqual(topology["holes"], 1)
        self.assertEqual(len(topology["boundary_segments"]), 16)

    def test_normal_case_region_excludes_known_bad_points_and_checks_each_cell(self):
        region = self.normal_result["candidate_region"]
        config, sources, outer, _ = parse_case_and_geometry(self.normal_case)
        search_indices = np.linspace(0, len(sources) - 1, min(len(sources), config.search_source_limit), dtype=int)
        search_sources = sources[np.unique(search_indices)]
        region_indices = np.linspace(0, len(search_sources) - 1, min(len(search_sources), config.region_source_limit), dtype=int)
        region_sources = search_sources[np.unique(region_indices)]
        bad_score = score_candidate(config, outer, (0.0, 0.0), region_sources)["j_lambda"]
        bad_receive_margin = score_margin(config, np.asarray((700.0, -350.0)), region_sources)

        self.assertGreater(bad_score, float(region["score_threshold_J"]))
        self.assertGreater(bad_receive_margin, 0.0)
        self.assertFalse(point_in_candidate_region(region, (0.0, 0.0)))
        self.assertFalse(point_in_candidate_region(region, (700.0, -350.0)))
        self.assertGreater(len(region["cells"]), 0)
        self.assertEqual(region["receive_verification"]["status"], "passed")
        self.assertTrue(region["receive_verification"]["continuous_over_detector_cells"])
        self.assertEqual(region["score_verification"]["status"], "sampled_passed")
        self.assertFalse(region["score_verification"]["continuous_over_detector_cells"])
        threshold = float(region["score_threshold_J"])
        for cell in region["cells"]:
            self.assertLessEqual(cell["j_max"], threshold + 1e-12)
            self.assertEqual(cell["stencil_point_count"], 9)
            self.assertEqual(cell["receive_cell_status"], "passed")
            self.assertGreaterEqual(cell["size_m"], region["min_cell_size_m"] - 1e-12)
            self.assertLessEqual(cell["size_m"], region["base_cell_size_m"] + 1e-12)

    def test_translation_rotation_invariance_of_distance_geometry(self):
        source = np.array([1000.0, 200.0])
        candidate = np.array([700.0, 800.0])
        angle = math.radians(37.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        shift = np.array([2300.0, -1700.0])
        transformed_source = rotation @ source + shift
        transformed_candidate = rotation @ candidate + shift
        self.assertAlmostEqual(np.linalg.norm(source - candidate), np.linalg.norm(transformed_source - transformed_candidate), places=9)


if __name__ == "__main__":
    unittest.main()
