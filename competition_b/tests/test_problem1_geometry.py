import math
import unittest

import numpy as np

from competition_b.code.problem1_geometry import (
    build_regular_ngon,
    clean_polygon,
    coverage_decision,
    minimum_enclosing_circle,
    point_in_convex_polygon,
    polygon_diameter,
    polygon_area,
    solve_region,
)


class Problem1GeometryTests(unittest.TestCase):
    def test_rectangle_diameter_and_mec(self):
        rect = np.array([[0, 0], [3, 0], [3, 4], [0, 4]], dtype=float)
        d, pair = polygon_diameter(rect)
        center, radius = minimum_enclosing_circle(rect)
        self.assertAlmostEqual(d, 5.0, places=10)
        self.assertAlmostEqual(2 * radius, 5.0, places=10)
        self.assertTrue(coverage_decision(d, radius, radius=10, n=3600)["coverage_possible"])
        self.assertEqual(polygon_area(rect), 12.0)
        self.assertIsNotNone(pair)

    def test_equilateral_triangle_is_not_coverable_by_diameter_circle(self):
        tri = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]], dtype=float)
        d, _ = polygon_diameter(tri)
        _, radius = minimum_enclosing_circle(tri)
        self.assertAlmostEqual(d, 1.0, places=10)
        self.assertGreater(2 * radius, d)
        self.assertFalse(coverage_decision(d, radius, radius=10, n=3600)["coverage_possible"])

    def test_obtuse_triangle_uses_longest_side_circle(self):
        tri = np.array([[0, 0], [1.9, 0], [0.95, math.sqrt(0.0975)]], dtype=float)
        d, _ = polygon_diameter(tri)
        _, radius = minimum_enclosing_circle(tri)
        self.assertAlmostEqual(d, 1.9, places=8)
        self.assertAlmostEqual(2 * radius, d, places=8)

    def test_segment_and_near_collinear_mec(self):
        pts = np.array([[0, 0], [3, 4]], dtype=float)
        d, _ = polygon_diameter(pts)
        center, radius = minimum_enclosing_circle(pts)
        self.assertAlmostEqual(d, 5.0)
        self.assertAlmostEqual(radius, 2.5)
        nearly = np.array([[0, 0], [1, 1e-13], [2, 0]], dtype=float)
        center, radius = minimum_enclosing_circle(nearly)
        self.assertAlmostEqual(radius, 1.0, places=8)
        self.assertTrue(np.allclose(center, [1, 0], atol=1e-8))
        degenerate = clean_polygon(np.array([[0, 0], [1, 0], [2, 0]], dtype=float))
        self.assertEqual(len(degenerate), 2)
        self.assertTrue(np.allclose(degenerate, [[0, 0], [2, 0]]))

    def test_regular_ngon_is_ccw_and_circumscribed(self):
        poly = build_regular_ngon(1800, 12)
        self.assertGreater(polygon_area(poly), 0)
        self.assertAlmostEqual(np.linalg.norm(poly[0]), 1800 / math.cos(math.pi / 12), places=10)

    def test_bearing_wrap_and_source_containment(self):
        source = np.array([100.0, 100.0])
        region = solve_region(
            [{"x": -900, "y": 100, "bearing_deg": 0.4}, {"x": 100, "y": -900, "bearing_deg": 89.6}],
            radius=1800,
            delta_deg=1,
            n=360,
        )
        self.assertGreater(region["num_vertices"], 0)
        self.assertTrue(point_in_convex_polygon(source, np.asarray(region["vertices"]), eps=1e-6))

    def test_clearability_fields_follow_geometry_threshold(self):
        tiny = solve_region([], radius=10, n=96)
        self.assertTrue(tiny["guaranteed_clearable"])
        self.assertTrue(tiny["ready_to_clear"])
        self.assertNotIn("quality_grade", tiny)


if __name__ == "__main__":
    unittest.main()
