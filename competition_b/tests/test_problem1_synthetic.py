import math
import json
import random
import time
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from competition_b.code.problem1_geometry import point_in_convex_polygon, solve_case


def synthetic_case(rng: random.Random, index: int, n: int = 360) -> tuple[dict, np.ndarray]:
    angle = rng.random() * 2 * math.pi
    source_radius = math.sqrt(rng.random()) * 1700.0
    source = np.array([source_radius * math.cos(angle), source_radius * math.sin(angle)])
    points = []
    count = 2 + rng.randrange(5)
    for _ in range(count):
        while True:
            sensor_angle = rng.random() * 2 * math.pi
            sensor_radius = math.sqrt(rng.random()) * 1800.0
            x = sensor_radius * math.cos(sensor_angle)
            y = sensor_radius * math.sin(sensor_angle)
            distance = math.hypot(x - source[0], y - source[1])
            if 5.0 < distance <= 1500.0:
                break
        true_bearing = math.degrees(math.atan2(source[1] - y, source[0] - x)) % 360.0
        measured = (true_bearing + rng.uniform(-1.0, 1.0)) % 360.0
        points.append({"x": x, "y": y, "bearing_deg": measured})
    return {"case_id": f"synthetic_{index:05d}", "R": 1800.0, "delta_deg": 1.0, "N": n, "points": points}, source


class Problem1SyntheticTests(unittest.TestCase):
    def test_json_cases_cover_point_counts_and_consistent_sources(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "problem1_cases.json"
        cases = json.loads(data_path.read_text(encoding="utf-8"))
        new_cases = [case for case in cases if case["case_id"].startswith("new_")]
        self.assertEqual(len(cases), 34)
        self.assertEqual(len(new_cases), 20)
        self.assertEqual(Counter(len(case["points"]) for case in new_cases), Counter({2: 12, 3: 6, 4: 2}))
        counts = {len(case["points"]) for case in cases}
        self.assertTrue({2, 3, 4, 5, 6, 8, 10}.issubset(counts))

        required_fields = {
            "region_type", "is_bounded", "num_vertices", "vertices", "area_m2",
            "perimeter_m", "diameter_m", "diameter_pair", "mec_center", "mec_radius_m",
            "coverage_possible", "coverage_gap_m", "eps_cov_m", "coverage_status",
            "guaranteed_clearable", "ready_to_clear", "flags",
        }
        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                result = solve_case(case)
                main = result["main"]
                self.assertTrue(required_fields.issubset(main))
                self.assertNotIn("quality_grade", main)
                self.assertEqual(main["region_type"], "disk_clipped")
                self.assertIsInstance(main["flags"], list)
                self.assertIn("coverage_possible", main)
                self.assertIn("guaranteed_clearable", main)
                self.assertIn("ready_to_clear", main)
                if main["num_vertices"] == 0:
                    self.assertIsNone(main["mec_radius_m"])
                else:
                    self.assertIsNotNone(main["mec_radius_m"])
                if "true_source" in case:
                    self.assertTrue(main["vertices"])
                    source = case["true_source"]
                    sx, sy = float(source["x"]), float(source["y"])
                    self.assertTrue(point_in_convex_polygon((sx, sy), main["vertices"], eps=2e-6))
                    for point in case["points"]:
                        true_bearing = math.degrees(math.atan2(sy - point["y"], sx - point["x"])) % 360.0
                        bearing_error = abs((point["bearing_deg"] - true_bearing + 180.0) % 360.0 - 180.0)
                        self.assertLessEqual(bearing_error, case["delta_deg"] + 1e-6)

    def test_acute_three_point_counterexample(self):
        case = {
            "case_id": "acute_counterexample",
            "true_source": {"x": 809.621, "y": 34.927},
            "R": 1800.0,
            "delta_deg": 1.0,
            "N": 3600,
            "points": [
                {"x": -90.694, "y": 321.580, "bearing_deg": 341.7157},
                {"x": -181.879, "y": -587.247, "bearing_deg": 33.0308},
                {"x": 784.393, "y": -772.866, "bearing_deg": 87.4642},
            ],
        }
        result = solve_case(case)["main"]
        self.assertEqual(result["num_vertices"], 3)
        self.assertFalse(result["coverage_possible"])
        self.assertLess(result["coverage_gap_m"], 0.0)
        self.assertGreater(-result["coverage_gap_m"], 10.0 * result["eps_cov_m"])
        source = case["true_source"]
        for point in case["points"]:
            distance = math.hypot(source["x"] - point["x"], source["y"] - point["y"])
            self.assertLessEqual(distance, 1500.0 + 1e-6)

    def test_10000_consistent_observations_contain_source(self):
        rng = random.Random(20260910)
        failures = []
        start = time.perf_counter()
        for index in range(10000):
            case, source = synthetic_case(rng, index, n=360)
            for point in case["points"]:
                distance = math.hypot(source[0] - point["x"], source[1] - point["y"])
                self.assertGreater(distance, 5.0)
                self.assertLessEqual(distance, 1500.0 + 1e-6)
            result = solve_case(case)["main"]
            if not result["vertices"] or not point_in_convex_polygon(source, result["vertices"], eps=2e-6):
                failures.append((index, result["flags"]))
                if len(failures) >= 5:
                    break
        elapsed = time.perf_counter() - start
        self.assertEqual(failures, [], msg=f"failures={failures}, elapsed_s={elapsed:.3f}")


if __name__ == "__main__":
    unittest.main()
