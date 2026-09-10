from __future__ import annotations

import json
import tempfile
import time
import unittest
from io import BytesIO
from urllib.error import HTTPError, URLError

from competition_b.action_logger import ActionLogger
from competition_b.simulator_client import (
    RuntimeBudgetExceeded,
    SimulatorClient,
    SimulatorClientError,
)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RawResponse(FakeResponse):
    def __init__(self, body: bytes, status: int = 200):
        self.status = status
        self._body = body


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ClientTests(unittest.TestCase):
    def make_client(self, opener):
        self.log_dir = tempfile.TemporaryDirectory()
        return SimulatorClient(
            base_url="http://simulator",
            robot_id="TEAM-TEST",
            opener=opener,
            max_network_retries=2,
            logger=ActionLogger(f"{self.log_dir.name}/test_actions.jsonl"),
        )

    def test_normal_sequence_and_clear_does_not_switch_channel(self):
        opener = FakeOpener([
            FakeResponse({"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 1200}),
            FakeResponse({"accepted": True, "virtual_time_s": 105, "measure_result": "direction", "svd_deg": 90.0}),
            FakeResponse({"accepted": True, "virtual_time_s": 110, "clear_result": "success"}),
        ])
        client = self.make_client(opener)
        client.enter()
        client.measure(300, 400, 2)
        client.clear(300, 400, 7)
        self.assertEqual(client.state.current_measure_channel, 2)
        self.assertEqual(client.state.position, (300.0, 400.0))
        self.assertEqual(client.state.last_virtual_time_s, 110.0)
        self.assertEqual(client.state.cleared_channels, {7})

    def test_measure_result_variants_are_returned(self):
        for result in ("direction", "near", "no_signal"):
            opener = FakeOpener([
                FakeResponse({"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 1200}),
                FakeResponse({"accepted": True, "virtual_time_s": 5, "measure_result": result}),
            ])
            client = self.make_client(opener)
            client.enter()
            response = client.measure(0, 0, 1)
            self.assertEqual(response["measure_result"], result)

    def test_transport_retry_reuses_identical_request(self):
        opener = FakeOpener([
            URLError("temporary failure"),
            FakeResponse({"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 1200}),
        ])
        client = self.make_client(opener)
        client.enter()
        self.assertEqual(len(opener.requests), 2)
        first = json.loads(opener.requests[0].data.decode("utf-8"))
        second = json.loads(opener.requests[1].data.decode("utf-8"))
        self.assertEqual(first, second)

    def test_rejected_response_does_not_update_state(self):
        opener = FakeOpener([
            FakeResponse({"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 1200}),
            FakeResponse({"accepted": False, "virtual_time_s": 0}),
        ])
        client = self.make_client(opener)
        client.enter()
        with self.assertRaises(SimulatorClientError):
            client.measure(100, 100, 3)
        self.assertEqual(client.state.position, (0.0, 0.0))
        self.assertEqual(client.state.last_virtual_time_s, 0.0)

    def test_new_actions_have_distinct_request_ids(self):
        opener = FakeOpener([
            FakeResponse({"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 1200}),
            FakeResponse({"accepted": True, "virtual_time_s": 5, "measure_result": "near"}),
            FakeResponse({"accepted": True, "virtual_time_s": 10, "measure_result": "no_signal"}),
        ])
        client = self.make_client(opener)
        client.enter()
        client.measure(0, 0, 1)
        client.measure(1, 1, 2)
        ids = [json.loads(request.data.decode("utf-8"))["request_id"] for request in opener.requests]
        self.assertEqual(len(ids), len(set(ids)))

    def test_non_json_response_is_reported_after_bounded_retry(self):
        opener = FakeOpener([
            RawResponse(b"not-json"),
            RawResponse(b"not-json"),
            RawResponse(b"not-json"),
        ])
        client = self.make_client(opener)
        with self.assertRaises(SimulatorClientError):
            client.enter()
        self.assertEqual(len(opener.requests), 3)
        payloads = [json.loads(request.data.decode("utf-8")) for request in opener.requests]
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])

    def test_explicit_http_rejection_is_not_retried(self):
        error = HTTPError(
            url="http://simulator/enter",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=BytesIO(b'{"accepted": false, "virtual_time_s": 0}'),
        )
        opener = FakeOpener([error])
        client = self.make_client(opener)
        with self.assertRaises(SimulatorClientError):
            client.enter()
        self.assertEqual(len(opener.requests), 1)

    def test_real_time_budget_stops_new_action(self):
        opener = FakeOpener([
            FakeResponse({"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 1200}),
        ])
        client = self.make_client(opener)
        client.enter()
        client._real_deadline = time.monotonic() - 1
        with self.assertRaises(RuntimeBudgetExceeded):
            client.measure(0, 0, 1)
        self.assertEqual(len(opener.requests), 1)


if __name__ == "__main__":
    unittest.main()
