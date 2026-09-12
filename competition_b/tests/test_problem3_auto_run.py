import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from competition_b.code.problem3_solver import Config
from competition_b.problem3_auto_run import (
    main,
    offline_preflight,
    run_automatic_session,
)
from competition_b.tests.test_problem3_solver import FakeSimulator


class Problem3AutoRunTests(unittest.TestCase):
    def test_preflight_has_no_simulator_dependency(self):
        report = offline_preflight(Config())
        self.assertTrue(report["passed"])
        self.assertTrue(report["coverage"]["certified"])
        self.assertTrue(report["second_measurement_guarantee"]["passed"])

    def test_preflight_cli_never_constructs_live_client(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "competition_b.problem3_auto_run.SimulatorClient",
            side_effect=AssertionError("不应构造网络客户端"),
        ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = main(
                [
                    "--preflight",
                    "--config",
                    "competition_b/data/problem3_config.json",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "problem3_auto_preflight.json").exists())

    def test_missing_ready_confirmation_never_constructs_live_client(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "competition_b.problem3_auto_run.SimulatorClient",
            side_effect=AssertionError("不应构造网络客户端"),
        ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = main(
                [
                    "--robot-id",
                    "test-team",
                    "--config",
                    "competition_b/data/problem3_config.json",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 2)

    def test_automatic_session_needs_no_operator_actions_after_enter(self):
        cfg = Config(
            channel_count=2,
            source_count_min=1,
            source_count_max=1,
            polygon_edges=128,
        )
        fake = FakeSimulator({1: (600.0, 100.0)}, receive_radius=1000.0)
        result = run_automatic_session(
            cfg,
            fake,
            None,
            {"case_id": "offline-synthetic", "test_mode_label": "test"},
        )
        self.assertTrue(result["completion_certificate"])
        self.assertEqual(result["cleared_count"], 1)
        self.assertEqual(result["control_mode"], "automatic")
        self.assertEqual(result["operator_actions_required_after_enter"], 0)
        self.assertTrue(fake.exited)


if __name__ == "__main__":
    unittest.main()
