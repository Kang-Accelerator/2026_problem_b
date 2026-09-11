import json
import math
from pathlib import Path
import unittest

import numpy as np

from competition_b.code.problem2_model import (
    build_conservative_outer_polygon,
    exact_source_contains,
    parse_config,
    per_angle_distance_intervals,
    sample_exact_source_points,
)


class Problem2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data" / "problem2_cases.json"
        cls.cases = json.loads(path.read_text(encoding="utf-8"))

    def test_wrap_case_has_source_samples_and_angle_intervals(self):
        case = next(item for item in self.cases if item["case_id"] == "p2_wrap_around")
        config = parse_config(case)
        intervals = per_angle_distance_intervals(config)
        sources = sample_exact_source_points(config)
        self.assertTrue(any(item["has_interval"] for item in intervals))
        self.assertGreater(len(sources), 0)
        self.assertTrue(all(exact_source_contains(config, point) for point in sources))

    def test_external_outward_ray_is_not_replaced_by_center_ray_guess(self):
        case = next(item for item in self.cases if item["case_id"] == "p2_invalid_outward_ray")
        config = parse_config(case)
        intervals = per_angle_distance_intervals(config)
        self.assertFalse(any(item["has_interval"] for item in intervals))
        self.assertEqual(len(sample_exact_source_points(config)), 0)

    def test_conservative_outer_is_labeled_and_contains_sampled_sources(self):
        case = next(item for item in self.cases if item["case_id"] == "p2_normal_center")
        config = parse_config(case)
        outer = build_conservative_outer_polygon(config)
        sources = sample_exact_source_points(config)
        self.assertGreaterEqual(len(outer), 3)
        self.assertGreater(len(sources), 0)
        from competition_b.code.problem1_geometry import point_in_convex_polygon

        self.assertTrue(all(point_in_convex_polygon(point, outer, eps=1e-5) for point in sources))

    def test_ray_interval_respects_distance_upper_bound(self):
        case = next(item for item in self.cases if item["case_id"] == "p2_external_tangent")
        config = parse_config(case)
        intervals = per_angle_distance_intervals(config)
        for item in intervals:
            if item["has_interval"]:
                self.assertLessEqual(item["d_max"], 1500.0 + 1e-8)
                self.assertLessEqual(item["d_min"], item["d_max"])


if __name__ == "__main__":
    unittest.main()
