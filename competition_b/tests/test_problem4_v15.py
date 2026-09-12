import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from competition_b.code.problem3_solver import BearingObservation
from competition_b.tests.test_problem4_v14 import (
    DirectionalSourceSimulator,
    FakeSimulator,
    v14,
)


def load_v15():
    path = Path(__file__).resolve().parents[1] / "code" / "problem4_baseline-v1.5.py"
    spec = importlib.util.spec_from_file_location(
        "competition_b.code.problem4_baseline_v15", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v15 = load_v15()


def route_length(points):
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


class Problem4V15Tests(unittest.TestCase):
    def make_solver_with_origin_bearings(self, bearings, **overrides):
        cfg = v15.Problem4Config(
            channel_count=max(10, len(bearings)),
            source_count_min=1,
            source_count_max=max(10, len(bearings)),
            polygon_edges=64,
            **overrides,
        )
        solver = v15.Problem4BaselineSolver(cfg, FakeSimulator())
        for channel, bearing in enumerate(bearings, start=1):
            state = solver.channels[channel]
            state.status = "DETECTED"
            state.observations.append(BearingObservation((0.0, 0.0), bearing, 0))
        return solver

    def test_all_rotated_mirrored_routes_use_same_points_and_length(self):
        cfg = v15.Problem4Config()
        default = v15.build_direction_skeleton_ordered(cfg, 0, 1)
        default_set = set(default)
        default_length = route_length(default)
        for start in range(cfg.direction_inner_count):
            for step in (-1, 1):
                route = v15.build_direction_skeleton_ordered(cfg, start, step)
                self.assertEqual(set(route), default_set)
                self.assertAlmostEqual(route_length(route), default_length, places=7)
                self.assertEqual(len(route), 28)

    def test_clustered_origin_bearings_can_trigger_mirrored_prefix(self):
        solver = self.make_solver_with_origin_bearings(
            [115.0, 120.0, 125.0, 130.0],
            adaptive_prefix_enabled=True,
        )
        default_points = set(solver.full_direction_skeleton)
        default_length = route_length(solver.full_direction_skeleton)
        solver._adapt_coverage_prefix_after_origin()
        diagnostics = solver.strategy_diagnostics
        self.assertTrue(diagnostics["adaptive_prefix_considered"])
        self.assertTrue(diagnostics["adaptive_prefix_replanned"])
        self.assertEqual(diagnostics["adaptive_prefix_start_index"], 0)
        self.assertEqual(diagnostics["adaptive_prefix_step"], -1)
        self.assertGreaterEqual(diagnostics["adaptive_prefix_estimated_saved_s"], 10.0)
        self.assertEqual(set(solver.full_direction_skeleton), default_points)
        self.assertAlmostEqual(route_length(solver.full_direction_skeleton), default_length, places=7)

    def test_too_few_origin_directions_keep_v14_order(self):
        solver = self.make_solver_with_origin_bearings([10.0, 80.0, 200.0])
        before = list(solver.coverage_route)
        solver._adapt_coverage_prefix_after_origin()
        self.assertEqual(solver.coverage_route, before)
        self.assertFalse(solver.strategy_diagnostics["adaptive_prefix_replanned"])

    def test_switch_disables_prefix_replanning(self):
        solver = self.make_solver_with_origin_bearings(
            [175.0, 180.0, 185.0, 190.0],
            adaptive_prefix_enabled=False,
        )
        before = list(solver.coverage_route)
        solver._adapt_coverage_prefix_after_origin()
        self.assertEqual(solver.coverage_route, before)
        self.assertFalse(solver.strategy_diagnostics["adaptive_prefix_considered"])

    def test_no_source_state_machine_keeps_28_point_certificate(self):
        cfg = v15.Problem4Config(channel_count=1, source_count_min=1, source_count_max=1)
        result = v15.Problem4BaselineSolver(cfg, FakeSimulator()).run()
        self.assertEqual(result["strategy"], "problem4_ring28_verified_clear_v1.5")
        self.assertEqual(result["phase4"]["visited_coverage_points"], 28)
        self.assertTrue(result["directional_coverage"]["certified"])
        self.assertEqual(result["time_breakdown"]["N_f"], 0)
        history = result["diagnostics"]["coverage_prefix_history"]
        self.assertEqual(len(history), 28)
        self.assertEqual([row["visit"] for row in history], list(range(1, 29)))
        self.assertTrue(all(row["known_source_count"] == 0 for row in history))
        self.assertAlmostEqual(result["time_reconciliation_error_s"], 0.0, places=8)

    def test_v15_default_route_matches_v14_before_origin_feedback(self):
        cfg14 = v14.Problem4Config()
        cfg15 = v15.Problem4Config()
        self.assertEqual(
            v14.build_direction_skeleton(cfg14),
            v15.build_direction_skeleton(cfg15),
        )

    def test_verified_boundary_clear_accepts_19_2m_region_and_executes_once(self):
        cfg = v15.Problem4Config(
            channel_count=1,
            source_count_min=1,
            source_count_max=1,
            polygon_edges=64,
        )
        simulator = DirectionalSourceSimulator({1: ((0.0, 0.0), (1.0, 0.0))})
        solver = v15.Problem4BaselineSolver(cfg, simulator)
        state = solver.channels[1]
        state.status = "DETECTED"
        state.observations.extend(
            [
                BearingObservation((-100.0, 0.0), 0.0),
                BearingObservation((0.0, -100.0), 90.0),
            ]
        )
        polygon = np.array(
            [[-19.2, 0.0], [0.0, -19.2], [19.2, 0.0], [0.0, 19.2]],
            dtype=float,
        )
        solver._safe_clear_point = lambda ignored: None
        with patch.object(v15, "localization_polygon", return_value=polygon):
            options = solver._directional_service_options(1)
            self.assertEqual(len(options), 1)
            self.assertEqual(options[0]["role"], "verified_boundary_clear")
            action = {
                **options[0],
                "kind": options[0]["action"],
                "channel": 1,
                "committed_skeleton_index": None,
            }
            self.assertTrue(solver._execute_directional_service(action))
        self.assertEqual(solver.strategy_diagnostics["verified_boundary_clear_attempts"], 1)
        self.assertEqual(solver.strategy_diagnostics["verified_boundary_clear_successes"], 1)
        self.assertEqual(solver.strategy_diagnostics["verified_boundary_clear_failures"], 0)

    def test_verified_boundary_clear_rejects_region_beyond_19_5m_limit(self):
        solver = self.make_solver_with_origin_bearings([0.0, 90.0])
        polygon = np.array(
            [[-19.6, 0.0], [0.0, -19.6], [19.6, 0.0], [0.0, 19.6]],
            dtype=float,
        )
        with patch.object(v15, "localization_polygon", return_value=polygon):
            self.assertIsNone(
                solver._verified_boundary_clear_point(solver.channels[1])
            )

    def test_failed_verified_boundary_clear_is_not_retried_on_committed_edge(self):
        solver = self.make_solver_with_origin_bearings([0.0, 90.0])
        state = solver.channels[1]
        polygon = np.array(
            [[-19.2, 0.0], [0.0, -19.2], [19.2, 0.0], [0.0, 19.2]],
            dtype=float,
        )
        action = {
            "kind": "clear",
            "channel": 1,
            "point": (0.0, 0.0),
            "role": "verified_boundary_clear",
            "committed_skeleton_index": 1,
        }
        with patch.object(v15, "localization_polygon", return_value=polygon), patch.object(
            solver, "clear", return_value=False
        ):
            self.assertTrue(solver._execute_directional_service(action))
            self.assertIsNone(solver._verified_boundary_clear_point(state))
        self.assertIn("PROBLEM4_VERIFIED_BOUNDARY_CLEAR_FAILED", state.flags)
        self.assertEqual(solver.strategy_diagnostics["verified_boundary_clear_attempts"], 1)
        self.assertEqual(solver.strategy_diagnostics["verified_boundary_clear_failures"], 1)


if __name__ == "__main__":
    unittest.main()
