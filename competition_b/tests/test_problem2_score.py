import unittest

import numpy as np

from competition_b.code.problem2_model import build_conservative_outer_polygon, parse_config
from competition_b.code.problem2_score import certify_receive_cell, classify_observation, posterior_for_scenario, receive_margin


def simple_case():
    return {
        "case_id": "score_case",
        "S1": {"x": 0.0, "y": 0.0},
        "theta1_deg": 0.0,
        "delta_deg": 1.0,
        "target_region": {"cx": 0.0, "cy": 0.0, "R": 1800.0},
        "receive_range": [1000.0, 1500.0],
        "near_threshold_m": 5.0,
        "budget_B_m": 3000.0,
        "source_grid": {"n_angle": 9, "n_distance": 9},
        "candidate_grid": {"step_m": 300.0, "max_candidates": 80},
        "error_grid_deg": [-1.0, 0.0, 1.0],
        "ngon_edges": 96,
        "certification": {"subdivision_budget": 100},
        "objective": {"lambda": 0.0, "R_ref_m": 1500.0, "B_ref_m": 1500.0, "epsilon_J": 0.02},
    }


class Problem2ScoreTests(unittest.TestCase):
    def test_observation_branch_boundaries(self):
        self.assertEqual(classify_observation((0, 0), (3, 4), 1000, 5), "near")
        self.assertEqual(classify_observation((0, 0), (6, 8), 10, 5), "direction")
        self.assertEqual(classify_observation((0, 0), (6, 8), 9.99, 5), "no_signal")

    def test_same_source_has_different_branch_for_explicit_rho_scenarios(self):
        config = parse_config(simple_case())
        outer = build_conservative_outer_polygon(config)
        source = np.array([1200.0, 0.0])
        candidate = np.array([1200.0, 1300.0])
        direction = posterior_for_scenario(config, outer, source, candidate, 0.0, rho_scenario=1500.0)
        no_signal = posterior_for_scenario(config, outer, source, candidate, 0.0, rho_scenario=1200.0)
        self.assertEqual(direction["branch"], "direction")
        self.assertEqual(no_signal["branch"], "no_signal")
        self.assertEqual(direction["rho_mode"], "explicit_scenario")
        self.assertIsNone(no_signal["bearing_deg"])

    def test_no_signal_does_not_create_a_fake_bearing(self):
        config = parse_config(simple_case())
        outer = build_conservative_outer_polygon(config)
        scenario = posterior_for_scenario(config, outer, np.array([1000.0, 0.0]), np.array([0.0, 1200.0]), 0.0)
        self.assertEqual(scenario["branch"], "no_signal")
        self.assertIsNone(scenario["bearing_deg"])
        self.assertEqual(len(scenario["posterior_vertices"]), 0)

    def test_near_branch_does_not_create_a_fake_bearing(self):
        config = parse_config(simple_case())
        outer = build_conservative_outer_polygon(config)
        scenario = posterior_for_scenario(config, outer, np.array([1000.0, 0.0]), np.array([1003.0, 4.0]), 0.0)
        self.assertEqual(scenario["branch"], "near")
        self.assertIsNone(scenario["bearing_deg"])
        self.assertGreater(len(scenario["posterior_vertices"]), 0)

    def test_direction_posterior_uses_1500_metre_upper_bound(self):
        config = parse_config(simple_case())
        outer = build_conservative_outer_polygon(config)
        source = np.array([900.0, 0.0])
        candidate = np.array([0.0, 400.0])
        scenario = posterior_for_scenario(config, outer, source, candidate, 0.0)
        self.assertEqual(scenario["branch"], "direction")
        self.assertIsNotNone(scenario["bearing_deg"])
        self.assertGreater(len(scenario["posterior_vertices"]), 0)
        self.assertIsNotNone(scenario["posterior_rmin_m"])

    def test_unknown_radius_is_not_silently_replaced_by_lower_bound(self):
        config = parse_config(simple_case())
        outer = build_conservative_outer_polygon(config)
        source = np.array([1200.0, 0.0])
        # 该候选点到源点距离约为 1345 米：
        # rho=1200 时无信号，rho=1500 时有方向；未知 rho 时应为 uncertain。
        candidate = np.array([0.0, 700.0])
        self.assertEqual(
            posterior_for_scenario(config, outer, source, candidate, 0.0, rho_scenario=1200.0)["branch"],
            "no_signal",
        )
        self.assertEqual(
            posterior_for_scenario(config, outer, source, candidate, 0.0, rho_scenario=1500.0)["branch"],
            "direction",
        )
        self.assertEqual(
            posterior_for_scenario(config, outer, source, candidate, 0.0)["branch"],
            "uncertain",
        )

    def test_receive_margin_uses_first_distance_conditioning(self):
        config = parse_config(simple_case())
        sources = np.array([[900.0, 0.0], [1400.0, 0.0]])
        margin = receive_margin(config, np.array([0.0, 0.0]), sources)
        self.assertLessEqual(margin, 1e-9)

    def test_receive_cell_rejects_rectangle_with_a_bad_corner(self):
        config = parse_config(simple_case())
        outer = build_conservative_outer_polygon(config)
        sources = np.array([[1000.0, 0.0]])
        result = certify_receive_cell(config, outer, [0.0, 1400.0, -400.0, 400.0], sources, subdivision_budget=20)
        self.assertEqual(result["status"], "violated")
        self.assertFalse(result["continuous_detector_cell_certified"])


if __name__ == "__main__":
    unittest.main()
